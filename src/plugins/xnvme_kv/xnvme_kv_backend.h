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
#ifndef XNVME_KV_BACKEND_H
#define XNVME_KV_BACKEND_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <sys/uio.h>
#include <vector>

#include "backend/backend_engine.h"
#include "xnvme_kv_dev.h"

// NVMe KV keys are at most 16 bytes (KV spec 1.3).
inline constexpr size_t XNVME_KV_MAX_OBJECT_KEY_LEN = 16;

struct nixlXnvmeKvKey {
    uint8_t key[XNVME_KV_MAX_OBJECT_KEY_LEN] = {};
    uint16_t keyLen = 0;
};

std::ostream &
operator<<(std::ostream &os, const nixlXnvmeKvKey &key);

class nixlXnvmeKvBackendReqH;

// Per-descriptor context handed to the device layer as the completion cookie.
// Lives inside nixlXnvmeKvBackendReqH::taskContexts_, whose address is
// published to in-flight commands — the vector must never be resized after
// submission starts.
struct nixlXnvmeKvTaskContext {
    nixlXnvmeKvBackendReqH *reqH = nullptr;
    int taskIndex = 0;
    bool isRetrieve = false;
    size_t expectedValueLen = 0;
};

// EXIST outcome for queryMem(actual).
enum nixlXnvmeKvExistResult : int {
    XNVME_KV_EXIST_UNKNOWN = 0,
    XNVME_KV_EXIST_YES,
    XNVME_KV_EXIST_NO,
};

class nixlXnvmeKvBackendReqH : public nixlBackendReqH {
public:
    explicit nixlXnvmeKvBackendReqH(int num_tasks) : totalTasks_(num_tasks) {}

    // Hand-written because atomics are not movable; only used while building
    // queryMem handle vectors, strictly before any command is submitted (the
    // vector is reserve()d so task-context addresses stay stable afterwards).
    nixlXnvmeKvBackendReqH(nixlXnvmeKvBackendReqH &&other) noexcept
        : totalTasks_(other.totalTasks_),
          submittedTasks_(other.submittedTasks_),
          completedTasks_(other.completedTasks_),
          allTasksCompleted_(other.allTasksCompleted_.load(std::memory_order_acquire)),
          cancelled_(other.cancelled_.load(std::memory_order_acquire)),
          overallStatus_(other.overallStatus_),
          existResult_(other.existResult_.load(std::memory_order_acquire)),
          isExistQuery_(other.isExistQuery_),
          ignoreNotFound_(other.ignoreNotFound_),
          nextDescriptorIndex_(other.nextDescriptorIndex_),
          isPending_(other.isPending_),
          storedOperation_(other.storedOperation_),
          valueIovecs_(std::move(other.valueIovecs_)),
          objectKeys_(std::move(other.objectKeys_)),
          taskContexts_(std::move(other.taskContexts_)) {}

    nixlXnvmeKvBackendReqH(const nixlXnvmeKvBackendReqH &) = delete;
    nixlXnvmeKvBackendReqH &
    operator=(const nixlXnvmeKvBackendReqH &) = delete;
    nixlXnvmeKvBackendReqH &
    operator=(nixlXnvmeKvBackendReqH &&) = delete;

    bool
    allTasksCompleted() const noexcept {
        return allTasksCompleted_.load(std::memory_order_acquire);
    }

    nixl_status_t
    getOverallStatus() const noexcept {
        return overallStatus_;
    }

private:
    friend class nixlXnvmeKvEngine;
    friend class nixlXnvmeKvProgressEngine;
    friend class nixlXnvmeKvNoThreadEngine;
    friend class nixlXnvmeKvThreadedEngine;

    int totalTasks_ = 0;
    int submittedTasks_ = 0;
    // Plain int: only mutated under the no-thread engine lock or on the
    // progress thread.
    int completedTasks_ = 0;
    std::atomic<bool> allTasksCompleted_{false};
    std::atomic<bool> cancelled_{false};
    nixl_status_t overallStatus_ = NIXL_IN_PROG;
    // EXIST outcome (nixlXnvmeKvExistResult), written by the completion path.
    std::atomic<int> existResult_{XNVME_KV_EXIST_UNKNOWN};
    bool isExistQuery_ = false;
    bool ignoreNotFound_ = false;
    // Resume point for partially submitted requests.
    int nextDescriptorIndex_ = 0;
    bool isPending_ = false;
    nixl_xfer_op_t storedOperation_ = NIXL_WRITE;

    std::vector<struct iovec> valueIovecs_;
    std::vector<nixlXnvmeKvKey> objectKeys_;
    std::vector<nixlXnvmeKvTaskContext> taskContexts_;
};

class nixlXnvmeKvProgressEngine;

class nixlXnvmeKvEngine : public nixlBackendEngine {
public:
    explicit nixlXnvmeKvEngine(const nixlBackendInitParams *init_params)
        : nixlXnvmeKvEngine(init_params, nullptr) {}

    // Test seam: inject a device implementation. When dev is null the real
    // xNVMe-backed device is created.
    nixlXnvmeKvEngine(const nixlBackendInitParams *init_params,
                      std::unique_ptr<nixlXnvmeKvDevIface> dev);

    ~nixlXnvmeKvEngine() override;

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
        return {OBJ_SEG, DRAM_SEG};
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

    // Decode a NIXL OBJ_SEG metaInfo into an NVMe KV key: an even-length
    // string of at most 32 hex characters decodes to binary; otherwise raw
    // bytes of at most 16 characters are used verbatim. Hex wins over raw.
    static bool
    convertToKvKey(const std::string &meta_info, nixlXnvmeKvKey &key);

    // Full key resolution: empty metaInfo falls back to the 8 raw
    // native-endian bytes of dev_id.
    static bool
    resolveKvKey(uint64_t dev_id, const std::string &meta_info, nixlXnvmeKvKey &key);

private:
    nixl_status_t
    parseInitParams(const nixl_b_params_t *custom_params);

    nixl_status_t
    initDevice();

    nixl_status_t
    createProgressEngine(bool enable_prog_thread, uint64_t pthr_delay_us);

    static constexpr uint32_t kDefaultNumTasks = 128;
    static constexpr const char *kDefaultBe = "io_uring_cmd";

    std::unique_ptr<nixlXnvmeKvDevIface> dev_;
    std::unique_ptr<nixlXnvmeKvProgressEngine> progressEngine_;
    // Raw view of progressEngine_ for the device completion callback: it
    // stays valid while the progress-engine destructor drains in-flight
    // commands, when the unique_ptr already reads null (reset() nulls the
    // stored pointer before destroying the object).
    nixlXnvmeKvProgressEngine *progressEnginePtr_ = nullptr;

    std::string deviceName_;
    std::string be_ = kDefaultBe;
    uint32_t nsid_ = 0;
    uint32_t numTasks_ = kDefaultNumTasks;
    std::string nguid_;
    bool queryMemAssumeSuccess_ = true;
    bool ignoreReadNotFound_ = false;
};

#endif // XNVME_KV_BACKEND_H
