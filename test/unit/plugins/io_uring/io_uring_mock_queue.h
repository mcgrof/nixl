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
#ifndef IO_URING_MOCK_QUEUE_H
#define IO_URING_MOCK_QUEUE_H

#include <cerrno>
#include <cstring>
#include <deque>
#include <map>
#include <utility>
#include <vector>

#include "io_uring_queue.h"

// In-memory stand-in for the ring: a byte array the size of the target.
// Submissions queue up to `capacity` of them; poke() completes them in order
// against the array and fires the completion callback, so the caller-progress
// model the backend relies on is exercised the same way a real ring exercises
// it. Every knob is a plain member, settable before or between operations.
class nixlIoUringMockQueue : public nixlIoUringQueue {
public:
    // --- target facts, read at open ---
    uint32_t logicalBlockBytes = 512;
    uint64_t sizeBytes = 64 * 1024 * 1024;
    uint32_t maxHwBytes = 0;
    uint32_t memAlignBytes = 512;
    uint32_t offsetAlignBytes = 512;
    std::string serial = "MOCK0";
    std::string wwid = "mock.wwid.0";
    nixl_status_t openResult = NIXL_SUCCESS;

    // --- registration knobs ---
    // Non-zero stands in for a kernel that refuses the registration.
    int registerErr = 0;
    uint32_t maxSlots = 8;

    // --- ring knobs ---
    uint32_t capacity = 8; // in-flight limit; past it submissions get -EAGAIN
    int eagainBudget = 0; // force the next N submissions to -EAGAIN
    int failSubmitAfter = -1; // >=0: the (N+1)-th submission fails permanently
    int permanentErr = -EIO;
    int forceError = 0; // non-zero: complete every request with this errno
    int64_t shortBy = 0; // >0: report that many bytes fewer than asked
    bool holdCompletions = false; // poke() completes nothing while set
    int pokeErrorBudget = 0; // fail this many poke calls with pokeError
    int pokeError = -EIO;

    // --- observability ---
    std::vector<uint8_t> store;
    std::vector<std::pair<uint64_t, uint32_t>> submitted; // (offset, bytes) in order
    std::vector<uint32_t> submittedSlots; // slot per submitted request
    // Registered buffers: slot -> (base, len).
    std::map<uint32_t, std::pair<uint8_t *, size_t>> slots;
    int registerCalls = 0;
    int unregisterCalls = 0;
    int submitAttempts = 0;
    int submitAccepted = 0;
    int pokeCount = 0;
    nixlIoUringOpenOpts openedOpts;
    std::string openedPath;
    bool closed = false;

    nixl_status_t
    open(const std::string &path,
         const nixlIoUringOpenOpts &opts,
         nixlIoUringCplCb cpl_cb) override {
        openedPath = path;
        openedOpts = opts;
        cplCb_ = std::move(cpl_cb);
        target_.kind = nixlIoUringTarget::Kind::blockDevice;
        target_.sizeBytes = sizeBytes;
        target_.logicalBlockBytes = logicalBlockBytes;
        target_.maxHwBytes = maxHwBytes;
        target_.memAlignBytes = memAlignBytes;
        target_.offsetAlignBytes = offsetAlignBytes;
        target_.serial = serial;
        target_.model = "mock";
        target_.firmware = "0";
        target_.wwid = wwid;
        if (openResult == NIXL_SUCCESS) {
            store.assign(sizeBytes, 0);
        }
        return openResult;
    }

    void
    close() override {
        closed = true;
    }

    const nixlIoUringTarget &
    target() const override {
        return target_;
    }

    int
    registerHostBuffer(void *addr, size_t len, uint32_t &slot) override {
        if (registerErr) {
            return registerErr;
        }
        return addSlot(static_cast<uint8_t *>(addr), len, slot);
    }

    int
    unregisterBuffer(uint32_t slot) override {
        unregisterCalls++;
        return slots.erase(slot) ? 0 : -ENOENT;
    }

    int
    submit(bool write, uint64_t offset, uint32_t bytes, void *buf, uint32_t slot, void *cookie)
        override {
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
        inflight_.push_back({write, offset, bytes, buf, slot, cookie});
        submitted.emplace_back(offset, bytes);
        submittedSlots.push_back(slot);
        submitAccepted++;
        return 0;
    }

    int
    poke() override {
        pokeCount++;
        if (pokeErrorBudget > 0) {
            pokeErrorBudget--;
            return pokeError;
        }
        if (holdCompletions) {
            return 0;
        }
        int completed = 0;
        while (!inflight_.empty()) {
            pending cmd = inflight_.front();
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
    struct pending {
        bool write;
        uint64_t offset;
        uint32_t bytes;
        void *buf;
        uint32_t slot;
        void *cookie;
    };

    int
    addSlot(uint8_t *base, size_t len, uint32_t &slot) {
        registerCalls++;
        for (uint32_t s = 0; s < maxSlots; s++) {
            if (!slots.count(s)) {
                slots[s] = {base, len};
                slot = s;
                return 0;
            }
        }
        return -ENOSPC;
    }

    // Where a request's bytes live: a plain pointer, or a pointer inside a
    // registered host range.
    uint8_t *
    resolve(const pending &cmd, bool &ok) {
        ok = true;
        if (cmd.slot == IO_URING_NO_SLOT) {
            return static_cast<uint8_t *>(cmd.buf);
        }
        auto it = slots.find(cmd.slot);
        if (it == slots.end()) {
            ok = false;
            return nullptr;
        }
        const auto [base, len] = it->second;
        const auto raw = reinterpret_cast<uintptr_t>(cmd.buf);
        // A host slot must carry an address inside its registered range.
        // Anything else is a backend bug, and fails the request rather than
        // reading stray memory.
        const auto lo = reinterpret_cast<uintptr_t>(base);
        if (raw < lo || raw >= lo + len) {
            ok = false;
            return nullptr;
        }
        return static_cast<uint8_t *>(cmd.buf);
    }

    void
    completeOne(const pending &cmd) {
        if (forceError) {
            cplCb_(cmd.cookie, forceError);
            return;
        }
        if (cmd.offset + cmd.bytes > store.size()) {
            cplCb_(cmd.cookie, -EIO);
            return;
        }
        bool ok = true;
        uint8_t *mem = resolve(cmd, ok);
        if (!ok) {
            cplCb_(cmd.cookie, -EFAULT);
            return;
        }
        if (cmd.write) {
            memcpy(store.data() + cmd.offset, mem, cmd.bytes);
        } else {
            memcpy(mem, store.data() + cmd.offset, cmd.bytes);
        }
        const int32_t res = shortBy > 0 && static_cast<uint64_t>(shortBy) < cmd.bytes ?
            static_cast<int32_t>(cmd.bytes - shortBy) :
            static_cast<int32_t>(cmd.bytes);
        cplCb_(cmd.cookie, res);
    }

    std::deque<pending> inflight_;
    nixlIoUringCplCb cplCb_;
    nixlIoUringTarget target_;
};

#endif // IO_URING_MOCK_QUEUE_H
