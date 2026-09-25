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
#ifndef IO_URING_PROGRESS_ENGINE_H
#define IO_URING_PROGRESS_ENGINE_H

#include <chrono>
#include <deque>
#include <mutex>
#include <vector>

#include "io_uring_backend.h"

// Owns submission and completion for the single ring, in a caller-progress
// discipline: the caller's own thread drives submission, reaping and retries
// out of checkXfer(), and one mutex serializes those against the request
// bookkeeping. Buffer registration goes straight to the queue and is
// serialized by the queue itself, not by this lock. A NIXL progress thread is
// acknowledged with a warning and caller progress is used instead.
class nixlIoUringProgressEngine {
public:
    explicit nixlIoUringProgressEngine(nixlIoUringQueue *queue);
    ~nixlIoUringProgressEngine();
    nixlIoUringProgressEngine(const nixlIoUringProgressEngine &) = delete;
    nixlIoUringProgressEngine &
    operator=(const nixlIoUringProgressEngine &) = delete;

    nixl_status_t
    postXfer(nixlIoUringBackendReqH *req_h, nixl_xfer_op_t operation) const;

    nixl_status_t
    checkXfer(nixlIoUringBackendReqH *req_h) const;

    // Takes ownership of req_h and does not return until the requests it
    // already submitted have completed, because those requests name the
    // caller's payload buffers and the caller is free to release them once
    // this returns.
    void
    cancelRequest(nixlIoUringBackendReqH *req_h) const;

    // Delivered by the queue from inside poke(), which this engine only ever
    // calls with lock_ held. res is the completion result: bytes transferred,
    // or a negative errno.
    void
    onCompletion(void *cookie, int32_t res) const;

    nixlIoUringStats
    stats() const;

private:
    // Stop a transfer at its first permanent submission failure: what was
    // never submitted will never complete, so collapse the total to what is in
    // flight plus the failed slot and let the rest drain.
    static void
    handleSubmissionFailure(nixlIoUringBackendReqH *req_h, nixl_status_t status);

    // Submit, or resume submitting, a transfer's requests. Returns true when
    // no further attempt is needed (fully submitted, or permanently failed),
    // false to park it for retry.
    bool
    trySubmitRequest(nixlIoUringBackendReqH *req_h) const;

    // Reap completions and free cancelled transfers that finished draining.
    // Returns the number of completions processed.
    int
    progress() const;

    // Reap completions with lock_ already held.
    int
    pokeLocked() const;

    void
    tryResumePendingRequests() const;

    static constexpr auto kDestructorDrainBudget = std::chrono::seconds(10);
    static constexpr auto kCancelDrainBudget = std::chrono::seconds(10);

    nixlIoUringQueue *queue_; // non-owning; the backend engine owns it
    mutable std::mutex lock_;
    // Transfers with unsubmitted requests, FIFO.
    mutable std::deque<nixlIoUringBackendReqH *> pendingRequests_;
    // Cancelled transfers with requests still in flight; freed by progress().
    mutable std::vector<nixlIoUringBackendReqH *> cancelledRequests_;
    // A permanent ring-level error makes every active and future transfer fail
    // closed. Their storage stays alive until accepted requests drain.
    mutable int queueError_ = 0;
    mutable nixlIoUringStats stats_;
};

#endif // IO_URING_PROGRESS_ENGINE_H
