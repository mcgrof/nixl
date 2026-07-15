/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// End-to-end integration test for the XNVME_KV plugin through the NIXL agent
// API against a real NVMe KV namespace.
//
// Gated on the NIXL_XNVME_KV_DEVICE environment variable (e.g. /dev/ng4n1);
// exits with the meson skip code (77) when unset so CI without a KV device
// passes cleanly.
//
// Covers: backend creation, DRAM_SEG + OBJ_SEG registration with 128-bit
// keys, patterned WRITE (batch 4x larger than the queue depth), queryMem in
// actual mode (present and missing keys), READ-back with byte comparison,
// missing-key READ failure, and both progress modes (caller-driven and
// progress thread).

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "nixl.h"
#include "nixl_descriptors.h"
#include "nixl_params.h"

namespace {

constexpr int num_objects = 32; // 4x the backend queue depth below
constexpr size_t value_len = 1024; // below vfio-user-kvssd's 4096-byte max
constexpr auto poll_timeout = std::chrono::seconds(30);

int failures = 0;

void
check(bool ok, const std::string &what) {
    if (ok) {
        printf("  OK: %s\n", what.c_str());
    } else {
        printf("  FAIL: %s\n", what.c_str());
        failures++;
    }
}

std::string
objectKeyHex(int idx, bool missing = false) {
    // 32 hex chars -> 128-bit binary key.
    char buf[33];
    snprintf(buf, sizeof(buf), "%030x%02x", missing ? 0xdead : 0xbeef, idx & 0xff);
    return buf;
}

nixl_status_t
pollXfer(nixlAgent &agent, nixlXferReqH *req) {
    const auto deadline = std::chrono::steady_clock::now() + poll_timeout;
    for (;;) {
        nixl_status_t status = agent.getXferStatus(req);
        if (status != NIXL_IN_PROG)
            return status;
        if (std::chrono::steady_clock::now() > deadline)
            return NIXL_ERR_UNKNOWN;
        std::this_thread::yield();
    }
}

void
fillPattern(std::vector<uint8_t> &buf, int idx) {
    for (size_t j = 0; j < buf.size(); j++)
        buf[j] = static_cast<uint8_t>(idx * 7 + j * 13 + 1);
}

int
runSuite(const char *device, bool use_prog_thread) {
    printf("=== progress thread %s ===\n", use_prog_thread ? "enabled" : "disabled");

    nixlAgentConfig cfg;
    cfg.useProgThread = use_prog_thread;
    cfg.pthrDelay = 100; // us

    const std::string agent_name = "XnvmeKvTester";
    nixlAgent agent(agent_name, cfg);

    nixlBackendH *backend = nullptr;
    nixl_b_params_t params = {
        {"device_name", device},
        {"num_tasks", "8"}, // small on purpose: the 32-object batch must park and resume
        {"query_mem_mode", "actual"},
    };
    nixl_status_t status = agent.createBackend("XNVME_KV", params, backend);
    check(status == NIXL_SUCCESS && backend, "createBackend(XNVME_KV)");
    if (status != NIXL_SUCCESS)
        return 1;

    nixl_opt_args_t extra;
    extra.backends = {backend};

    // Buffers + registrations.
    std::vector<std::vector<uint8_t>> bufs(num_objects, std::vector<uint8_t>(value_len));
    nixl_reg_dlist_t dram_list(DRAM_SEG);
    nixl_reg_dlist_t obj_list(OBJ_SEG);
    for (int i = 0; i < num_objects; i++) {
        fillPattern(bufs[i], i);
        dram_list.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(bufs[i].data()),
                                       value_len,
                                       0,
                                       ""));
        obj_list.addDesc(nixlBlobDesc(0, value_len, static_cast<uint64_t>(i), objectKeyHex(i)));
    }
    check(agent.registerMem(dram_list, &extra) == NIXL_SUCCESS, "registerMem(DRAM_SEG)");
    check(agent.registerMem(obj_list, &extra) == NIXL_SUCCESS,
          "registerMem(OBJ_SEG, 128-bit hex keys)");

    nixl_xfer_dlist_t local(DRAM_SEG);
    nixl_xfer_dlist_t remote(OBJ_SEG);
    for (int i = 0; i < num_objects; i++) {
        local.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(bufs[i].data()), value_len, 0));
        remote.addDesc(nixlBasicDesc(0, value_len, static_cast<uint64_t>(i)));
    }

    // WRITE: DRAM -> KV STORE, batch larger than queue depth.
    nixlXferReqH *req = nullptr;
    status = agent.createXferReq(NIXL_WRITE, local, remote, agent_name, req, &extra);
    check(status == NIXL_SUCCESS, "createXferReq(NIXL_WRITE)");
    if (status == NIXL_SUCCESS) {
        status = agent.postXferReq(req);
        check(status == NIXL_SUCCESS || status == NIXL_IN_PROG, "postXferReq(WRITE)");
        check(pollXfer(agent, req) == NIXL_SUCCESS, "WRITE batch of 32 completed");
        agent.releaseXferReq(req);
    }

    // queryMem(actual): all stored keys must exist.
    {
        std::vector<nixl_query_resp_t> resp;
        status = agent.queryMem(obj_list, resp, &extra);
        bool all = status == NIXL_SUCCESS && resp.size() == num_objects;
        if (all)
            for (auto &r : resp)
                all = all && r.has_value();
        check(all, "queryMem(actual): all 32 keys exist");
    }

    // queryMem(actual) on a missing key: engaged=false but query succeeds.
    {
        nixl_reg_dlist_t missing_list(OBJ_SEG);
        missing_list.addDesc(nixlBlobDesc(0, value_len, 9999, objectKeyHex(77, true)));
        std::vector<nixl_query_resp_t> resp;
        status = agent.queryMem(missing_list, resp, &extra);
        check(status == NIXL_SUCCESS && resp.size() == 1 && !resp[0].has_value(),
              "queryMem(actual): missing key reported absent, not an error");
    }

    // READ back into cleared buffers and compare every byte.
    for (auto &buf : bufs)
        std::fill(buf.begin(), buf.end(), 0);
    status = agent.createXferReq(NIXL_READ, local, remote, agent_name, req, &extra);
    check(status == NIXL_SUCCESS, "createXferReq(NIXL_READ)");
    if (status == NIXL_SUCCESS) {
        status = agent.postXferReq(req);
        check(status == NIXL_SUCCESS || status == NIXL_IN_PROG, "postXferReq(READ)");
        check(pollXfer(agent, req) == NIXL_SUCCESS, "READ batch of 32 completed");
        agent.releaseXferReq(req);

        bool match = true;
        std::vector<uint8_t> expect(value_len);
        for (int i = 0; i < num_objects; i++) {
            fillPattern(expect, i);
            if (memcmp(bufs[i].data(), expect.data(), value_len) != 0) {
                match = false;
                break;
            }
        }
        check(match, "all 32 values byte-identical after READ");
    }

    // READ of a missing key must fail the request.
    {
        nixl_reg_dlist_t missing_reg(OBJ_SEG);
        missing_reg.addDesc(nixlBlobDesc(0, value_len, 9999, objectKeyHex(78, true)));
        check(agent.registerMem(missing_reg, &extra) == NIXL_SUCCESS,
              "registerMem(missing-key OBJ desc)");

        nixl_xfer_dlist_t one_local(DRAM_SEG);
        one_local.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(bufs[0].data()), value_len, 0));
        nixl_xfer_dlist_t one_remote(OBJ_SEG);
        one_remote.addDesc(nixlBasicDesc(0, value_len, 9999));

        status = agent.createXferReq(NIXL_READ, one_local, one_remote, agent_name, req, &extra);
        check(status == NIXL_SUCCESS, "createXferReq(READ, missing key)");
        if (status == NIXL_SUCCESS) {
            status = agent.postXferReq(req);
            if (status == NIXL_IN_PROG)
                status = pollXfer(agent, req);
            check(status == NIXL_ERR_BACKEND, "READ of missing key fails with NIXL_ERR_BACKEND");
            agent.releaseXferReq(req);
        }
        agent.deregisterMem(missing_reg, &extra);
    }

    agent.deregisterMem(dram_list, &extra);
    agent.deregisterMem(obj_list, &extra);
    return 0;
}

} // namespace

int
main() {
    const char *device = getenv("NIXL_XNVME_KV_DEVICE");
    if (!device || !*device) {
        printf("NIXL_XNVME_KV_DEVICE not set; skipping XNVME_KV integration test\n");
        return 77; // meson skip
    }

    if (runSuite(device, false))
        return 1;
    if (runSuite(device, true))
        return 1;

    printf(failures ? "XNVME_KV integration: FAIL (%d)\n" : "XNVME_KV integration: PASS\n",
           failures);
    return failures ? 1 : 0;
}
