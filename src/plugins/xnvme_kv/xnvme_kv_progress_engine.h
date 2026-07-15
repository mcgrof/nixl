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
#ifndef XNVME_KV_PROGRESS_ENGINE_H
#define XNVME_KV_PROGRESS_ENGINE_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "xnvme_kv_backend.h"

// Owns all interaction with the (single-owner) xNVMe queue. Two flavors:
//
// - nixlXnvmeKvNoThreadEngine: the caller's thread drives submission,
//   completion reaping and retries from checkXfer()/queryMem(); one mutex
//   serializes every queue operation.
//
// - nixlXnvmeKvThreadedEngine: a dedicated progress thread is the sole
//   caller of queue operations; producers only enqueue lightweight records
//   under a short-held mutex. checkXfer() is lock-free (atomics only).
//
// An xNVMe queue is not safe for concurrent use from unrelated threads, so
// exactly one of these disciplines is always in force.
class nixlXnvmeKvProgressEngine {
public:
    nixlXnvmeKvProgressEngine(nixlXnvmeKvDevIface *dev, bool ignore_read_not_found);
    virtual ~nixlXnvmeKvProgressEngine() = default;

    nixlXnvmeKvProgressEngine(const nixlXnvmeKvProgressEngine &) = delete;
    nixlXnvmeKvProgressEngine &
    operator=(const nixlXnvmeKvProgressEngine &) = delete;

    virtual nixl_status_t
    checkXfer(nixlXnvmeKvBackendReqH *req_h) const;

    virtual nixl_status_t
    postXfer(nixlXnvmeKvBackendReqH *req_h, nixl_xfer_op_t operation) const = 0;

    // Takes ownership of req_h; deletion may be deferred until all in-flight
    // commands have drained.
    virtual void
    cancelRequest(nixlXnvmeKvBackendReqH *req_h) const = 0;

    virtual nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const = 0;

    // Completion delivered by the device layer from inside poke().
    void
    onCompletion(void *cookie, const nixlXnvmeKvCpl &cpl) const;

protected:
    // Stop a request at its first permanent submission failure: unsubmitted
    // descriptors will never produce completions, so collapse the task count
    // to what is in flight plus the failed slot and let the rest drain.
    static void
    handleSubmissionFailure(nixlXnvmeKvBackendReqH *req_h, nixl_status_t status);

    // Submit (or resume submitting) all descriptors of a transfer request.
    // Returns true when the request needs no further submission attempts
    // (fully submitted or permanently failed), false to park it for retry.
    bool
    trySubmitRequest(nixlXnvmeKvBackendReqH *req_h) const;

    // Same contract for a single-EXIST queryMem handle.
    bool
    trySubmitExistTask(nixlXnvmeKvBackendReqH *req_h) const;

    // queryMem(actual) helpers shared by both engines.
    nixl_status_t
    prepareQueryHandles(const nixl_reg_dlist_t &descs,
                        std::vector<nixlXnvmeKvBackendReqH> &handles) const;

    template <typename ProgressFn>
    void
    waitForQueryCompletion(std::vector<nixlXnvmeKvBackendReqH> &handles,
                           ProgressFn &&progress) const {
        // No timeout by design: a bounded wait would let the handles go out
        // of scope while commands still reference their task contexts.
        for (;;) {
            bool all_done = true;
            for (auto &handle : handles)
                if (!handle.allTasksCompleted()) {
                    all_done = false;
                    break;
                }
            if (all_done)
                return;
            progress();
            std::this_thread::yield();
        }
    }

    nixl_status_t
    collectQueryResults(std::vector<nixlXnvmeKvBackendReqH> &handles,
                        std::vector<nixl_query_resp_t> &resp) const;

    static constexpr auto kDestructorDrainBudget = std::chrono::seconds(10);

    nixlXnvmeKvDevIface *dev_; // non-owning; the backend engine owns it
    bool ignoreReadNotFound_ = false;
    // Requests with unsubmitted descriptors, FIFO. No-thread engine: guarded
    // by lock_. Threaded engine: touched only by the progress thread.
    mutable std::deque<nixlXnvmeKvBackendReqH *> pendingRequests_;
};

class nixlXnvmeKvNoThreadEngine : public nixlXnvmeKvProgressEngine {
public:
    nixlXnvmeKvNoThreadEngine(nixlXnvmeKvDevIface *dev, bool ignore_read_not_found);
    ~nixlXnvmeKvNoThreadEngine() override;

    nixl_status_t
    checkXfer(nixlXnvmeKvBackendReqH *req_h) const override;

    nixl_status_t
    postXfer(nixlXnvmeKvBackendReqH *req_h, nixl_xfer_op_t operation) const override;

    void
    cancelRequest(nixlXnvmeKvBackendReqH *req_h) const override;

    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

private:
    // Reap completions and free cancelled requests that finished draining.
    // Returns the number of completions processed.
    int
    progress() const;

    void
    tryResumePendingRequests() const;

    // Serializes every device/queue operation and all request bookkeeping.
    mutable std::mutex lock_;
    // Cancelled requests with commands still in flight; freed by progress().
    mutable std::vector<nixlXnvmeKvBackendReqH *> cancelledRequests_;
};

class nixlXnvmeKvThreadedEngine : public nixlXnvmeKvProgressEngine {
public:
    nixlXnvmeKvThreadedEngine(nixlXnvmeKvDevIface *dev,
                              bool ignore_read_not_found,
                              std::chrono::microseconds thread_delay);
    ~nixlXnvmeKvThreadedEngine() override;

    // Base checkXfer is already lock-free; used as-is.

    nixl_status_t
    postXfer(nixlXnvmeKvBackendReqH *req_h, nixl_xfer_op_t operation) const override;

    void
    cancelRequest(nixlXnvmeKvBackendReqH *req_h) const override;

    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

private:
    struct pendingEntry {
        nixlXnvmeKvBackendReqH *reqH;
        nixl_xfer_op_t operation;
        bool isExistQuery;
    };

    void
    progressThreadFunc();

    void
    processCancellations(std::vector<nixlXnvmeKvBackendReqH *> &cancelled);

    std::chrono::microseconds threadDelay_;
    std::atomic<bool> threadStop_{false};
    std::thread progressThread_;

    // Producer side: postXfer/cancelRequest/queryMem push under queueMutex_;
    // the progress thread swaps the vectors out and drains them unlocked.
    mutable std::mutex queueMutex_;
    mutable std::atomic<bool> hasNewWork_{false};
    mutable std::vector<pendingEntry> vecA_, vecB_;
    mutable std::vector<pendingEntry> *producerVec_ = &vecA_;
    mutable std::vector<nixlXnvmeKvBackendReqH *> cancelledRequests_;
    // Cancelled-with-in-flight-commands requests owned by the progress
    // thread; freed once fully drained.
    std::vector<nixlXnvmeKvBackendReqH *> pendingDeletes_;

    mutable std::mutex wakeupMutex_;
    mutable std::condition_variable wakeup_;
};

#endif // XNVME_KV_PROGRESS_ENGINE_H
