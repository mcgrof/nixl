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
#ifndef XNVME_NVM_MOCK_DEV_H
#define XNVME_NVM_MOCK_DEV_H

#include <cerrno>
#include <cstring>
#include <deque>
#include <vector>

#include "xnvme_nvm_dev.h"

// In-memory mock of the xNVMe NVM device seam: a byte array the size of
// the namespace. Submissions queue up to `capacity` in-flight commands;
// poke() completes them FIFO against the array and fires the completion
// callback, mimicking the real device's poke-driven completion model. All
// knobs are plain members, settable before (or between) operations.
class nixlXnvmeNvmMockDev : public nixlXnvmeNvmDevIface {
public:
    // --- geometry knobs (read at open) ---
    uint32_t lbaBytes = 512;
    uint64_t capacityBytes = 64 * 1024 * 1024;
    uint32_t idfyMdtsBytes = 0;
    uint32_t sysfsMaxHwBytes = 0;
    uint32_t sysfsMaxSegments = 0;
    uint32_t pageBytes = 4096;
    std::string serial = "MOCK0";
    std::string wwid = "mock.wwid.0";
    bool reportsBytes = false; // true mimics the io_uring/libaio bio paths
    nixl_status_t openResult = NIXL_SUCCESS;

    // --- queue knobs ---
    uint32_t capacity = 8; // in-flight limit; beyond it submissions get -EAGAIN
    int eagainBudget = 0; // force the next N submissions to -EAGAIN
    int failSubmitAfter = -1; // >=0: the (N+1)-th submission fails permanently
    int permanentErr = -EIO;
    bool forceCplError = false; // complete every command with forcedSc/forcedSct
    uint8_t forcedSc = 0x02; // Invalid Field
    uint8_t forcedSct = 0;
    int64_t shortBy = 0; // >0 with reportsBytes: report that many bytes fewer
    bool holdCompletions = false; // poke() completes nothing while set

    // --- observability ---
    std::vector<uint8_t> store;
    std::vector<std::pair<uint64_t, uint16_t>> submitted; // (slba, nlb) in order
    int submitAttempts = 0;
    int submitAccepted = 0;
    int pokeCount = 0;
    nixlXnvmeNvmOpenOpts openedOpts;
    std::string openedUri;
    bool closed = false;

    nixl_status_t
    open(const std::string &uri,
         const nixlXnvmeNvmOpenOpts &opts,
         nixlXnvmeNvmCplCb cpl_cb) override {
        openedUri = uri;
        openedOpts = opts;
        cplCb_ = std::move(cpl_cb);
        geo_.devType = "mock";
        geo_.nsid = opts.nsidOverride ? opts.nsidOverride : 1;
        geo_.lbaBytes = lbaBytes;
        geo_.capacityBytes = capacityBytes;
        geo_.idfyMdtsBytes = idfyMdtsBytes;
        geo_.xnvmeMdtsBytes = idfyMdtsBytes;
        geo_.sysfsMaxHwBytes = sysfsMaxHwBytes;
        geo_.sysfsMaxSegments = sysfsMaxSegments;
        geo_.pageBytes = pageBytes;
        geo_.serial = serial;
        geo_.model = "mock";
        geo_.firmware = "0";
        geo_.wwid = wwid;
        if (openResult == NIXL_SUCCESS) {
            store.assign(capacityBytes, 0);
        }
        return openResult;
    }

    void
    close() override {
        closed = true;
    }

    const nixlXnvmeNvmGeo &
    geo() const override {
        return geo_;
    }

    bool
    completionReportsBytes() const override {
        return reportsBytes;
    }

    int
    submitRead(uint64_t slba, uint16_t nlb, void *buf, void *cookie) override {
        return enqueue(true, slba, nlb, buf, cookie);
    }

    int
    submitWrite(uint64_t slba, uint16_t nlb, const void *buf, void *cookie) override {
        return enqueue(false, slba, nlb, const_cast<void *>(buf), cookie);
    }

    int
    poke() override {
        pokeCount++;
        if (holdCompletions) {
            return 0;
        }
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

private:
    struct pendingCmd {
        bool read;
        uint64_t slba;
        uint16_t nlb;
        void *buf;
        void *cookie;
    };

    int
    enqueue(bool read, uint64_t slba, uint16_t nlb, void *buf, void *cookie) {
        submitAttempts++;
        if (eagainBudget > 0) {
            eagainBudget--;
            return -EAGAIN;
        }
        if (failSubmitAfter >= 0 && submitAccepted >= failSubmitAfter) {
            return permanentErr;
        }
        if (inflight_.size() >= capacity) {
            return -EAGAIN;
        }
        inflight_.push_back({read, slba, nlb, buf, cookie});
        submitted.emplace_back(slba, nlb);
        submitAccepted++;
        return 0;
    }

    void
    completeOne(const pendingCmd &cmd) {
        nixlXnvmeNvmCpl cpl;
        const uint64_t bytes = (static_cast<uint64_t>(cmd.nlb) + 1) * lbaBytes;
        const uint64_t offset = cmd.slba * lbaBytes;

        if (forceCplError) {
            cpl.sc = forcedSc;
            cpl.sct = forcedSct;
            cplCb_(cmd.cookie, cpl);
            return;
        }
        if (offset + bytes > store.size()) {
            cpl.sc = 0x80; // LBA Out of Range
            cpl.sct = 0;
            cplCb_(cmd.cookie, cpl);
            return;
        }
        if (cmd.read) {
            memcpy(cmd.buf, store.data() + offset, bytes);
        } else {
            memcpy(store.data() + offset, cmd.buf, bytes);
        }
        if (reportsBytes) {
            cpl.result =
                shortBy > 0 && static_cast<uint64_t>(shortBy) < bytes ? bytes - shortBy : bytes;
        }
        cplCb_(cmd.cookie, cpl);
    }

    std::deque<pendingCmd> inflight_;
    nixlXnvmeNvmCplCb cplCb_;
    nixlXnvmeNvmGeo geo_;
};

#endif // XNVME_NVM_MOCK_DEV_H
