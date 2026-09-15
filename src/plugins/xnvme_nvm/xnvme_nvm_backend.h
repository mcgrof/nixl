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
#ifndef XNVME_NVM_BACKEND_H
#define XNVME_NVM_BACKEND_H

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "backend/backend_engine.h"
#include "xnvme_nvm_dev.h"

// Counters kept per backend instance. Byte counters only ever count
// commands that completed successfully with the expected length.
struct nixlXnvmeNvmStats {
    uint64_t descsRequested = 0;
    uint64_t cmdsSubmitted = 0;
    uint64_t cmdsCompletedOk = 0;
    uint64_t cmdsFailed = 0;
    uint64_t cmdsShort = 0;
    uint64_t submitParked = 0;
    uint64_t submitPermanentErrors = 0;
    uint64_t bytesOkRead = 0;
    uint64_t bytesOkWrite = 0;
    uint32_t maxInflight = 0;
};

class nixlXnvmeNvmBackendReqH;

// One NVMe command of a transfer request, laid out at prepXfer time and
// handed to the device layer as the completion cookie. Lives inside
// nixlXnvmeNvmBackendReqH::tasks_, whose address is published to in-flight
// commands, so that vector is never resized after submission starts.
struct nixlXnvmeNvmTask {
    nixlXnvmeNvmBackendReqH *reqH = nullptr;
    uint32_t descIndex = 0;
    uint32_t childIndex = 0;
    uint32_t bytes = 0;
    uint64_t slba = 0;
    uint16_t nlb = 0;
    // A plain host pointer, or with a registered slot either the address
    // inside a registered host buffer or the byte offset inside a dma-buf.
    void *buf = nullptr;
    uint32_t slot = XNVME_NVM_NO_SLOT;
};

// How host-side payload buffers reach the kernel.
enum class nixlXnvmeNvmBufferMode { host, fixed, dmabuf };

class nixlXnvmeNvmBackendReqH : public nixlBackendReqH {
public:
    nixlXnvmeNvmBackendReqH() = default;
    nixlXnvmeNvmBackendReqH(const nixlXnvmeNvmBackendReqH &) = delete;
    nixlXnvmeNvmBackendReqH &
    operator=(const nixlXnvmeNvmBackendReqH &) = delete;

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
    friend class nixlXnvmeNvmEngine;
    friend class nixlXnvmeNvmProgressEngine;

    // Three distinct counts: commands this post will issue, commands
    // handed to the device so far, and commands the device completed.
    uint32_t totalTasks_ = 0;
    uint32_t submittedTasks_ = 0;
    uint32_t completedTasks_ = 0;
    std::atomic<bool> allTasksCompleted_{false};
    std::atomic<bool> cancelled_{false};
    nixl_status_t overallStatus_ = NIXL_IN_PROG;
    // Resume point for a request parked on queue saturation.
    uint32_t nextTask_ = 0;
    bool isPending_ = false;
    nixl_xfer_op_t op_ = NIXL_WRITE;
    std::vector<nixlXnvmeNvmTask> tasks_;
};

class nixlXnvmeNvmProgressEngine;

class nixlXnvmeNvmEngine : public nixlBackendEngine {
public:
    explicit nixlXnvmeNvmEngine(const nixlBackendInitParams *init_params)
        : nixlXnvmeNvmEngine(init_params, nullptr) {}

    // Test seam: inject a device implementation. When dev is null the real
    // xNVMe-backed device is created.
    nixlXnvmeNvmEngine(const nixlBackendInitParams *init_params,
                       std::unique_ptr<nixlXnvmeNvmDevIface> dev);
    ~nixlXnvmeNvmEngine() override;

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
        // A GPU buffer is reachable only as a dma-buf: VRAM_SEG registers
        // through the same dma-buf path with the device pointer as its base.
        if (bufferMode_ == nixlXnvmeNvmBufferMode::dmabuf) {
            return {BLK_SEG, DRAM_SEG, VRAM_SEG};
        }
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

    // Device facts and the per-command ceiling the backend settled on.
    const nixlXnvmeNvmGeo &
    geo() const;

    uint32_t
    maxIoBytes() const {
        return maxIoBytes_;
    }

    nixlXnvmeNvmStats
    stats() const;

    // The ceiling rule, exposed for tests: the smallest non-zero of the
    // Identify MDTS, the 16-bit NLB field (always present), the kernel's
    // max_hw_sectors_kb (max_hw_dmabuf_sectors_kb for dma-buf payloads),
    // for passthrough the kernel's max_segments worth of host pages, and
    // the user's max_io_bytes, rounded down to a logical block.
    static uint32_t
    computeMaxIoBytes(const nixlXnvmeNvmGeo &geo,
                      bool passthrough,
                      nixlXnvmeNvmBufferMode mode,
                      uint32_t user_max_io_bytes);

    nixlXnvmeNvmBufferMode
    bufferMode() const {
        return bufferMode_;
    }

private:
    nixl_status_t
    parseInitParams(const nixl_b_params_t *custom_params);

    nixl_status_t
    initDevice();

    static constexpr uint32_t kDefaultNumTasks = 128;
    static constexpr const char *kDefaultAsync = "io_uring_cmd";

    std::unique_ptr<nixlXnvmeNvmDevIface> dev_;
    std::unique_ptr<nixlXnvmeNvmProgressEngine> engine_;
    // Raw view of engine_ for the device completion callback: it stays valid
    // while the engine destructor drains in-flight commands, when the
    // unique_ptr already reads null.
    nixlXnvmeNvmProgressEngine *enginePtr_ = nullptr;

    std::string deviceName_;
    std::string async_ = kDefaultAsync;
    bool direct_ = true;
    uint32_t nsid_ = 0;
    uint32_t numTasks_ = kDefaultNumTasks;
    uint32_t userMaxIoBytes_ = 0;
    uint32_t forceMaxIoBytes_ = 0;
    uint32_t maxIoBytes_ = 0;
    std::string expectSerial_;
    std::string expectWwid_;
    nixlXnvmeNvmBufferMode bufferMode_ = nixlXnvmeNvmBufferMode::host;
    // Largest dma-buf the kernel registers (rw-dmabuf v6 refuses more than
    // 1 GiB with EINVAL); refused here with a message instead.
    uint64_t maxDmabufBytes_ = uint64_t(1) << 30;

    // BLK_SEG devId -> path bound at registration, so one slot cannot be
    // re-registered against a different device.
    mutable std::mutex devIdLock_;
    std::map<uint64_t, std::string> devIdPaths_;
};

#endif // XNVME_NVM_BACKEND_H
