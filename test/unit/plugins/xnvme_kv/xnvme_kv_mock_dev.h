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
#ifndef XNVME_KV_MOCK_DEV_H
#define XNVME_KV_MOCK_DEV_H

#include <cerrno>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <vector>

#include "xnvme_kv_dev.h"

// In-memory mock of the xNVMe device seam. Submissions queue up to
// `capacity` in-flight commands; poke() drains them FIFO against a
// std::map KV store and fires the completion callback, mimicking the
// real device's poke-driven completion model. All knobs are plain
// members, settable before (or between) operations.
//
// Thread-safety matches the real queue contract: submissions and poke are
// serialized by the progress engine, so no locking is needed except for
// the shared counters the tests read from another thread.
class nixlXnvmeKvMockDev : public nixlXnvmeKvDevIface {
public:
    // --- knobs ---
    uint32_t capacity = 8; // in-flight limit; beyond it submissions get -EAGAIN
    int eagainBudget = 0; // force the next N submissions to -EAGAIN
    int failSubmitAfter = -1; // >=0: the (N+1)-th submission fails permanently
    int permanentErr = -5; // -EIO
    bool forceCplError = false; // complete every command with forcedSc/forcedSct
    uint8_t forcedSc = XNVME_KV_SC_UNRECOVERED_ERR;
    uint8_t forcedSct = 1;
    bool reportRetrieveLen = false; // set cpl.cdw0 to the stored value length
    uint32_t forcedRetrieveCdw0 = 0; // nonzero: override cdw0 on retrieves
    nixl_status_t openResult = NIXL_SUCCESS;
    uint16_t kml = 16;
    uint32_t vml = 4096;
    std::array<uint8_t, 16> nguidVal = {};

    // --- observability ---
    std::map<std::string, std::vector<uint8_t>> store;
    int submitAttempts = 0;
    int submitAccepted = 0;
    int pokeCount = 0;

    nixl_status_t
    open(const std::string &uri,
         const std::string &be,
         uint32_t nsid_override,
         uint32_t num_tasks,
         nixlXnvmeKvCplCb cpl_cb) override {
        openedUri = uri;
        openedBe = be;
        openedNumTasks = num_tasks;
        nsid_ = nsid_override ? nsid_override : 1;
        cplCb_ = std::move(cpl_cb);
        return openResult;
    }

    void
    close() override {
        closed = true;
    }

    uint32_t
    nsid() const override {
        return nsid_;
    }

    uint16_t
    maxKeyLen() const override {
        return kml;
    }

    uint32_t
    maxValueLen() const override {
        return vml;
    }

    std::array<uint8_t, 16>
    nguid() const override {
        return nguidVal;
    }

    int
    submitStore(const void *key, uint8_t key_len, void *buf, uint32_t len, void *cookie) override {
        return enqueue(OP_STORE, key, key_len, buf, len, cookie);
    }

    int
    submitRetrieve(const void *key, uint8_t key_len, void *buf, uint32_t len, void *cookie)
        override {
        return enqueue(OP_RETRIEVE, key, key_len, buf, len, cookie);
    }

    int
    submitExist(const void *key, uint8_t key_len, void *cookie) override {
        return enqueue(OP_EXIST, key, key_len, nullptr, 0, cookie);
    }

    int
    poke() override {
        pokeCount++;
        int completed = 0;
        while (!inflight_.empty()) {
            pendingCmd cmd = inflight_.front();
            inflight_.pop_front();
            completeOne(cmd);
            completed++;
        }
        return completed;
    }

    uint32_t
    outstanding() const override {
        return static_cast<uint32_t>(inflight_.size());
    }

    std::string openedUri, openedBe;
    uint32_t openedNumTasks = 0;
    bool closed = false;

private:
    enum opKind { OP_STORE, OP_RETRIEVE, OP_EXIST };

    struct pendingCmd {
        opKind op;
        std::string key;
        void *buf;
        uint32_t len;
        void *cookie;
    };

    int
    enqueue(opKind op, const void *key, uint8_t key_len, void *buf, uint32_t len, void *cookie) {
        submitAttempts++;
        if (eagainBudget > 0) {
            eagainBudget--;
            return -EAGAIN;
        }
        if (failSubmitAfter >= 0 && submitAccepted >= failSubmitAfter)
            return permanentErr;
        if (inflight_.size() >= capacity)
            return -EAGAIN;
        inflight_.push_back(
            {op, std::string(static_cast<const char *>(key), key_len), buf, len, cookie});
        submitAccepted++;
        return 0;
    }

    void
    completeOne(const pendingCmd &cmd) {
        nixlXnvmeKvCpl cpl;

        if (forceCplError) {
            cpl.sc = forcedSc;
            cpl.sct = forcedSct;
            cplCb_(cmd.cookie, cpl);
            return;
        }

        auto it = store.find(cmd.key);
        switch (cmd.op) {
        case OP_STORE: {
            const auto *src = static_cast<const uint8_t *>(cmd.buf);
            store[cmd.key].assign(src, src + cmd.len);
            break;
        }
        case OP_RETRIEVE:
            if (it == store.end()) {
                cpl.sc = XNVME_KV_SC_KEY_NOT_EXISTS;
                cpl.sct = 1;
            } else {
                const size_t n = std::min<size_t>(cmd.len, it->second.size());
                memcpy(cmd.buf, it->second.data(), n);
                if (forcedRetrieveCdw0)
                    cpl.cdw0 = forcedRetrieveCdw0;
                else if (reportRetrieveLen)
                    cpl.cdw0 = static_cast<uint32_t>(it->second.size());
            }
            break;
        case OP_EXIST:
            if (it == store.end()) {
                cpl.sc = XNVME_KV_SC_KEY_NOT_EXISTS;
                cpl.sct = 1;
            }
            break;
        }
        cplCb_(cmd.cookie, cpl);
    }

    std::deque<pendingCmd> inflight_;
    nixlXnvmeKvCplCb cplCb_;
    uint32_t nsid_ = 1;
};

#endif // XNVME_KV_MOCK_DEV_H
