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
#ifndef IO_URING_BACKEND_H
#define IO_URING_BACKEND_H

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "backend/backend_engine.h"
#include "io_uring_queue.h"

// Counters kept per backend instance. The byte counters only ever count
// requests that completed with exactly the length asked for.
struct nixlIoUringStats {
    uint64_t descsRequested = 0;
    uint64_t reqsSubmitted = 0;
    uint64_t reqsCompletedOk = 0;
    uint64_t reqsFailed = 0;
    uint64_t reqsShort = 0;
    uint64_t submitParked = 0;
    uint64_t submitPermanentErrors = 0;
    uint64_t bytesOkRead = 0;
    uint64_t bytesOkWrite = 0;
    uint32_t maxInflight = 0;
};

class nixlIoUringBackendReqH;

// One read or write of a transfer, laid out at prepXfer time and handed to
// the queue as the completion cookie. These live in
// nixlIoUringBackendReqH::tasks_, whose element addresses are published to
// in-flight requests, so that vector is never resized once submission starts.
struct nixlIoUringTask {
    nixlIoUringBackendReqH *reqH = nullptr;
    uint32_t descIndex = 0;
    uint32_t childIndex = 0;
    uint32_t bytes = 0;
    uint64_t offset = 0;
    // A plain pointer, or with a registered slot an address inside the
    // registered host range.
    void *buf = nullptr;
    uint32_t slot = IO_URING_NO_SLOT;
};

// How a payload buffer reaches the kernel. host passes a pointer with every
// request; fixed passes a slot in the ring's registered buffer table, which
// the kernel pins once at registration.
enum class nixlIoUringBufferMode { host, fixed };

class nixlIoUringBackendReqH : public nixlBackendReqH {
public:
    nixlIoUringBackendReqH() = default;
    nixlIoUringBackendReqH(const nixlIoUringBackendReqH &) = delete;
    nixlIoUringBackendReqH &
    operator=(const nixlIoUringBackendReqH &) = delete;

    bool
    allTasksCompleted() const noexcept {
        return allTasksCompleted_.load(std::memory_order_acquire);
    }

    nixl_status_t
    getOverallStatus() const noexcept {
        return overallStatus_;
    }

    uint32_t
    numTasks() const noexcept {
        return static_cast<uint32_t>(tasks_.size());
    }

private:
    friend class nixlIoUringEngine;
    friend class nixlIoUringProgressEngine;

    // Three distinct counts: what this post will issue, what reached the
    // queue so far, and what the queue completed.
    uint32_t totalTasks_ = 0;
    uint32_t submittedTasks_ = 0;
    uint32_t completedTasks_ = 0;
    std::atomic<bool> allTasksCompleted_{false};
    std::atomic<bool> cancelled_{false};
    nixl_status_t overallStatus_ = NIXL_IN_PROG;
    // Resume point for a request parked on a full ring.
    uint32_t nextTask_ = 0;
    bool isPending_ = false;
    nixl_xfer_op_t op_ = NIXL_WRITE;
    std::vector<nixlIoUringTask> tasks_;
};

class nixlIoUringProgressEngine;

class nixlIoUringEngine : public nixlBackendEngine {
public:
    explicit nixlIoUringEngine(const nixlBackendInitParams *init_params)
        : nixlIoUringEngine(init_params, nullptr) {}

    // Test seam: inject a queue implementation. A null queue means the real
    // liburing one.
    nixlIoUringEngine(const nixlBackendInitParams *init_params,
                      std::unique_ptr<nixlIoUringQueue> queue);
    ~nixlIoUringEngine() override;

    bool
    supportsRemote() const override {
        return false;
    }

    bool
    supportsLocal() const override {
        return true;
    }

    bool
    supportsNotif() const override {
        return false;
    }

    nixl_mem_list_t
    getSupportedMems() const override {
        return {BLK_SEG, DRAM_SEG};
    }

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

    nixl_status_t
    connect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    disconnect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    unloadMD(nixlBackendMD *input) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override {
        output = input;
        return NIXL_SUCCESS;
    }

    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;

    nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

    const nixlIoUringTarget &
    target() const;

    uint32_t
    maxIoBytes() const {
        return maxIoBytes_;
    }

    nixlIoUringStats
    stats() const;

    // The granularity a request's length must be a multiple of. statx reports
    // the buffer address and the file offset separately, and it is the offset
    // alignment that also governs length, so a target demanding a wider
    // address alignment does not thereby demand longer requests.
    static uint32_t
    lengthAlignment(const nixlIoUringTarget &target);

    // The request-size ceiling, exposed for tests: the smaller non-zero of
    // the kernel's max_hw_sectors_kb and the caller's max_io_bytes, rounded
    // down to the direct-I/O alignment. Where no source names a limit it
    // falls back to kFallbackMaxIoBytes.
    static uint32_t
    computeMaxIoBytes(const nixlIoUringTarget &target, uint32_t user_max_io_bytes);

    nixlIoUringBufferMode
    bufferMode() const {
        return bufferMode_;
    }

private:
    nixl_status_t
    parseInitParams(const nixl_b_params_t *custom_params);

    nixl_status_t
    initQueue();

    static constexpr uint32_t kDefaultNumTasks = 128;
    // A target that publishes no hardware limit still needs a request size.
    // This is the backend's own choice of one, not a claim about the device:
    // large enough to amortize a submission, small enough that a transfer
    // still splits into several requests and keeps the ring busy.
    static constexpr uint32_t kFallbackMaxIoBytes = 1u << 20;

    std::unique_ptr<nixlIoUringQueue> queue_;
    std::unique_ptr<nixlIoUringProgressEngine> engine_;
    // Raw view of engine_ for the completion callback: it stays valid while
    // the engine destructor drains in-flight requests, by which time the
    // unique_ptr already reads null.
    nixlIoUringProgressEngine *enginePtr_ = nullptr;

    std::string deviceName_;
    bool direct_ = true;
    uint32_t numTasks_ = kDefaultNumTasks;
    uint32_t userMaxIoBytes_ = 0;
    uint32_t forceMaxIoBytes_ = 0;
    uint32_t maxIoBytes_ = 0;
    std::string expectSerial_;
    std::string expectWwid_;
    nixlIoUringBufferMode bufferMode_ = nixlIoUringBufferMode::host;

    // BLK_SEG devId -> the path it was bound to at registration, so one devId
    // cannot be re-registered against a different target.
    mutable std::mutex devIdLock_;
    std::map<uint64_t, std::string> devIdPaths_;
};

#endif // IO_URING_BACKEND_H
