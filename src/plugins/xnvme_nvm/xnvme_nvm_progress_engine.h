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
#ifndef XNVME_NVM_PROGRESS_ENGINE_H
#define XNVME_NVM_PROGRESS_ENGINE_H

#include <chrono>
#include <deque>
#include <mutex>
#include <vector>

#include "xnvme_nvm_backend.h"

// Owns all interaction with the single-owner xNVMe queue in the
// caller-progress discipline: the caller's thread drives submission,
// completion reaping and retries from checkXfer(), and one mutex serializes
// every queue operation and all request bookkeeping. A progress-thread
// discipline is a later addition; until then a NIXL progress thread is
// acknowledged with a warning and caller progress is used.
class nixlXnvmeNvmProgressEngine {
public:
    explicit nixlXnvmeNvmProgressEngine(nixlXnvmeNvmDevIface *dev);
    ~nixlXnvmeNvmProgressEngine();
    nixlXnvmeNvmProgressEngine(const nixlXnvmeNvmProgressEngine &) = delete;
    nixlXnvmeNvmProgressEngine &
    operator=(const nixlXnvmeNvmProgressEngine &) = delete;

    nixl_status_t
    postXfer(nixlXnvmeNvmBackendReqH *req_h, nixl_xfer_op_t operation) const;

    nixl_status_t
    checkXfer(nixlXnvmeNvmBackendReqH *req_h) const;

    // Takes ownership of req_h; deletion may be deferred until every
    // in-flight command has drained.
    void
    cancelRequest(nixlXnvmeNvmBackendReqH *req_h) const;

    // Completion delivered by the device layer from inside poke(), which
    // this engine only ever calls with lock_ held.
    void
    onCompletion(void *cookie, const nixlXnvmeNvmCpl &cpl) const;

    nixlXnvmeNvmStats
    stats() const;

private:
    // Stop a request at its first permanent submission failure: unsubmitted
    // commands never complete, so collapse the total to what is in flight
    // plus the failed slot and let the rest drain.
    static void
    handleSubmissionFailure(nixlXnvmeNvmBackendReqH *req_h, nixl_status_t status);

    // Submit (or resume submitting) the commands of a request. Returns true
    // when no further attempt is needed (fully submitted or permanently
    // failed), false to park it for retry.
    bool
    trySubmitRequest(nixlXnvmeNvmBackendReqH *req_h) const;

    // Reap completions and free cancelled requests that finished draining.
    // Returns the number of completions processed.
    int
    progress() const;

    void
    tryResumePendingRequests() const;

    static constexpr auto kDestructorDrainBudget = std::chrono::seconds(10);

    nixlXnvmeNvmDevIface *dev_; // non-owning; the backend engine owns it
    // Serializes every device/queue operation and all request bookkeeping.
    mutable std::mutex lock_;
    // Requests with unsubmitted commands, FIFO.
    mutable std::deque<nixlXnvmeNvmBackendReqH *> pendingRequests_;
    // Cancelled requests with commands still in flight; freed by progress().
    mutable std::vector<nixlXnvmeNvmBackendReqH *> cancelledRequests_;
    mutable nixlXnvmeNvmStats stats_;
};

#endif // XNVME_NVM_PROGRESS_ENGINE_H
