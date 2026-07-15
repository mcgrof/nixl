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

// Focused benchmark for the XNVME_KV plugin through the NIXL agent API.
//
// Sweeps STORE (NIXL_WRITE), RETRIEVE (NIXL_READ) and EXIST
// (queryMem(actual)) over queue depths, batch sizes and value sizes, in
// both progress modes, and reports throughput, IOPS, latency percentiles
// (p50/p95/p99/p99.9 per posted request) and CPU utilization as CSV on
// stdout.
//
// The maximum value size is discovered dynamically: the backend enforces the
// device's Identify (CSI=KV) value-size limit, so the bench probes downward
// from 4 MiB until a single-descriptor STORE is accepted. Nothing is assumed
// about the device's limits.
//
// Gated on NIXL_XNVME_KV_DEVICE (exit 77 = skip when unset).
//
// NOTE: results against QEMU/vfio-user-kvssd validate semantics, queueing
// and command patterns only. They are NOT representative of real KV SSD,
// CMX, or BlueField performance.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/resource.h>
#include <thread>
#include <vector>

#include "nixl.h"
#include "nixl_descriptors.h"
#include "nixl_params.h"

namespace {

using clk = std::chrono::steady_clock;

constexpr auto poll_timeout = std::chrono::seconds(60);
constexpr int target_requests = 512; // per cell
constexpr int warmup_requests = 32;

const char *g_device = nullptr;

double
cpuSeconds() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1e6 + ru.ru_stime.tv_sec +
        ru.ru_stime.tv_usec / 1e6;
}

std::string
keyHex(int idx) {
    char buf[33];
    snprintf(buf, sizeof(buf), "%024x%08x", 0xce, idx);
    return buf;
}

struct benchCell {
    const char *mode;
    const char *op;
    int qd;
    int batch;
    size_t vsize;
};

class benchRig {
public:
    benchRig(bool prog_thread, int qd, int batch, size_t vsize, bool query_actual)
        : batch_(batch),
          vsize_(vsize) {
        nixlAgentConfig cfg;
        cfg.useProgThread = prog_thread;
        cfg.pthrDelay = 10;

        agent_ = std::make_unique<nixlAgent>("XnvmeKvBench", cfg);
        nixl_b_params_t params = {
            {"device_name", g_device},
            {"num_tasks", std::to_string(qd)},
            {"query_mem_mode", query_actual ? "actual" : "assume_success"},
        };
        if (agent_->createBackend("XNVME_KV", params, backend_) != NIXL_SUCCESS || !backend_)
            throw std::runtime_error("createBackend failed");
        extra_.backends = {backend_};

        bufs_.resize(batch, std::vector<uint8_t>(vsize, 0x5a));
        dram_list_ = std::make_unique<nixl_reg_dlist_t>(DRAM_SEG);
        obj_list_ = std::make_unique<nixl_reg_dlist_t>(OBJ_SEG);
        local_ = std::make_unique<nixl_xfer_dlist_t>(DRAM_SEG);
        remote_ = std::make_unique<nixl_xfer_dlist_t>(OBJ_SEG);
        for (int i = 0; i < batch; i++) {
            auto addr = reinterpret_cast<uintptr_t>(bufs_[i].data());
            dram_list_->addDesc(nixlBlobDesc(addr, vsize, 0, ""));
            obj_list_->addDesc(nixlBlobDesc(0, vsize, static_cast<uint64_t>(i), keyHex(i)));
            local_->addDesc(nixlBasicDesc(addr, vsize, 0));
            remote_->addDesc(nixlBasicDesc(0, vsize, static_cast<uint64_t>(i)));
        }
        if (agent_->registerMem(*dram_list_, &extra_) != NIXL_SUCCESS ||
            agent_->registerMem(*obj_list_, &extra_) != NIXL_SUCCESS)
            throw std::runtime_error("registerMem failed");
    }

    ~benchRig() {
        agent_->deregisterMem(*dram_list_, &extra_);
        agent_->deregisterMem(*obj_list_, &extra_);
    }

    // One full transfer request; returns latency in microseconds, <0 on error.
    double
    xferOnce(nixl_xfer_op_t op) {
        nixlXferReqH *req = nullptr;
        if (agent_->createXferReq(op, *local_, *remote_, "XnvmeKvBench", req, &extra_) !=
            NIXL_SUCCESS)
            return -1.0;
        const auto t0 = clk::now();
        nixl_status_t status = agent_->postXferReq(req);
        const auto deadline = clk::now() + poll_timeout;
        while (status == NIXL_IN_PROG) {
            status = agent_->getXferStatus(req);
            if (clk::now() > deadline)
                break;
        }
        const auto t1 = clk::now();
        agent_->releaseXferReq(req);
        if (status != NIXL_SUCCESS)
            return -1.0;
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    double
    queryOnce() {
        std::vector<nixl_query_resp_t> resp;
        const auto t0 = clk::now();
        nixl_status_t status = agent_->queryMem(*obj_list_, resp, &extra_);
        const auto t1 = clk::now();
        if (status != NIXL_SUCCESS)
            return -1.0;
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    int batch_;
    size_t vsize_;

private:
    std::unique_ptr<nixlAgent> agent_;
    nixlBackendH *backend_ = nullptr;
    nixl_opt_args_t extra_;
    std::vector<std::vector<uint8_t>> bufs_;
    std::unique_ptr<nixl_reg_dlist_t> dram_list_, obj_list_;
    std::unique_ptr<nixl_xfer_dlist_t> local_, remote_;
};

double
percentile(std::vector<double> &sorted, double p) {
    if (sorted.empty())
        return 0;
    size_t idx = static_cast<size_t>(p * (sorted.size() - 1));
    return sorted[idx];
}

bool
runCell(const benchCell &cell, bool prog_thread) {
    const bool is_exist = strcmp(cell.op, "EXIST") == 0;
    const bool is_read = strcmp(cell.op, "RETRIEVE") == 0;

    try {
        benchRig rig(prog_thread, cell.qd, cell.batch, cell.vsize, is_exist);

        // Populate keys first (needed by RETRIEVE and EXIST; harmless for
        // STORE, and it doubles as warmup for the write path).
        for (int i = 0; i < 2; i++)
            if (rig.xferOnce(NIXL_WRITE) < 0)
                return false;

        auto once = [&]() -> double {
            if (is_exist)
                return rig.queryOnce();
            return rig.xferOnce(is_read ? NIXL_READ : NIXL_WRITE);
        };

        for (int i = 0; i < warmup_requests; i++)
            if (once() < 0)
                return false;

        std::vector<double> lat_us;
        lat_us.reserve(target_requests);
        const double cpu0 = cpuSeconds();
        const auto t0 = clk::now();
        for (int i = 0; i < target_requests; i++) {
            double us = once();
            if (us < 0)
                return false;
            lat_us.push_back(us);
        }
        const double wall_s = std::chrono::duration<double>(clk::now() - t0).count();
        const double cpu_s = cpuSeconds() - cpu0;

        std::sort(lat_us.begin(), lat_us.end());
        const double reqs = static_cast<double>(target_requests);
        const double iops = reqs * cell.batch / wall_s;
        const double mbps = is_exist ? 0.0 : iops * cell.vsize / (1024.0 * 1024.0);

        printf("%s,%s,%d,%d,%zu,%d,%.0f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n",
               cell.mode,
               cell.op,
               cell.qd,
               cell.batch,
               cell.vsize,
               target_requests,
               iops,
               mbps,
               percentile(lat_us, 0.50),
               percentile(lat_us, 0.95),
               percentile(lat_us, 0.99),
               percentile(lat_us, 0.999),
               100.0 * cpu_s / wall_s);
        fflush(stdout);
        return true;
    }
    catch (const std::exception &e) {
        fprintf(stderr, "cell failed: %s\n", e.what());
        return false;
    }
}

// Probe the largest single-descriptor value the backend accepts; the backend
// rejects anything above the device's Identify value-size limit with
// INVALID_PARAM before submission.
size_t
probeMaxValue() {
    for (size_t cand : {size_t(4) << 20, size_t(1) << 20, size_t(64) << 10, size_t(4) << 10,
                        size_t(512)}) {
        try {
            benchRig rig(false, 8, 1, cand, false);
            if (rig.xferOnce(NIXL_WRITE) >= 0)
                return cand;
        }
        catch (const std::exception &) {
        }
    }
    return 0;
}

} // namespace

int
main() {
    g_device = getenv("NIXL_XNVME_KV_DEVICE");
    if (!g_device || !*g_device) {
        printf("NIXL_XNVME_KV_DEVICE not set; skipping XNVME_KV benchmark\n");
        return 77;
    }

    const size_t max_value = probeMaxValue();
    if (!max_value) {
        fprintf(stderr, "could not find a working value size\n");
        return 1;
    }
    fprintf(stderr, "# discovered max supported value size: %zu bytes\n", max_value);

    printf("mode,op,qd,batch,vsize,requests,iops,MBps,p50_us,p95_us,p99_us,p999_us,cpu_pct\n");

    const int qds[] = {1, 2, 4, 8, 16, 32, 64, 128};
    const char *ops[] = {"STORE", "RETRIEVE", "EXIST"};

    // Queue-depth sweep at batch == qd, max value size, both progress modes.
    for (bool prog : {false, true}) {
        const char *mode = prog ? "thread" : "caller";
        for (const char *op : ops)
            for (int qd : qds)
                if (!runCell({mode, op, qd, qd, max_value}, prog))
                    return 1;
    }

    // Batch sweep (below / equal / above queue depth) at qd=32, caller mode.
    for (const char *op : ops)
        for (int batch : {8, 32, 128})
            if (!runCell({"caller", op, 32, batch, max_value}, false))
                return 1;

    // Value-size sweep at qd=32, batch=32, caller mode.
    for (const char *op : {"STORE", "RETRIEVE"})
        for (size_t shift : {8, 4, 2, 1})
            if (!runCell({"caller", op, 32, 32, max_value / shift}, false))
                return 1;

    fprintf(stderr, "# benchmark complete\n");
    return 0;
}
