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
#include "io_uring_progress_engine.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <thread>

#include "common/nixl_log.h"

nixlIoUringProgressEngine::nixlIoUringProgressEngine(nixlIoUringQueue *queue) : queue_(queue) {}

nixlIoUringProgressEngine::~nixlIoUringProgressEngine() {
    const auto deadline = std::chrono::steady_clock::now() + kDestructorDrainBudget;

    for (;;) {
        int did = progress();
        bool idle;
        {
            std::lock_guard<std::mutex> guard(lock_);
            idle = did <= 0 && cancelledRequests_.empty() && queue_->outstanding() == 0;
        }
        if (idle) {
            break;
        }
        if (std::chrono::steady_clock::now() > deadline) {
            // Deliberately leak: the kernel may still be writing into the
            // buffers those requests reference.
            NIXL_WARN << "IO_URING: requests did not drain within "
                      << kDestructorDrainBudget.count() << "s at engine destruction";
            break;
        }
        std::this_thread::yield();
    }
    if (!pendingRequests_.empty()) {
        NIXL_WARN << "IO_URING: " << pendingRequests_.size()
                  << " unreleased request handle(s) at engine destruction";
    }

    NIXL_INFO << "IO_URING: stats: descs_requested=" << stats_.descsRequested
              << " reqs_submitted=" << stats_.reqsSubmitted
              << " reqs_completed_ok=" << stats_.reqsCompletedOk
              << " reqs_failed=" << stats_.reqsFailed << " reqs_short=" << stats_.reqsShort
              << " submit_parked=" << stats_.submitParked
              << " submit_permanent_errors=" << stats_.submitPermanentErrors
              << " bytes_ok_read=" << stats_.bytesOkRead
              << " bytes_ok_write=" << stats_.bytesOkWrite
              << " max_inflight=" << stats_.maxInflight;
}

nixlIoUringStats
nixlIoUringProgressEngine::stats() const {
    std::lock_guard<std::mutex> guard(lock_);
    return stats_;
}

void
nixlIoUringProgressEngine::handleSubmissionFailure(nixlIoUringBackendReqH *req_h,
                                                   nixl_status_t status) {
    if (req_h->overallStatus_ == NIXL_IN_PROG) {
        req_h->overallStatus_ = status;
    }
    req_h->totalTasks_ = req_h->submittedTasks_ + 1;
    if (++req_h->completedTasks_ >= req_h->totalTasks_) {
        req_h->allTasksCompleted_.store(true, std::memory_order_release);
    }
}

void
nixlIoUringProgressEngine::onCompletion(void *cookie, int32_t res) const {
    auto *task = static_cast<nixlIoUringTask *>(cookie);
    nixlIoUringBackendReqH *req_h = task->reqH;
    const bool cancelled = req_h->cancelled_.load(std::memory_order_relaxed);
    const bool is_read = req_h->op_ == NIXL_READ;

    if (res < 0) {
        stats_.reqsFailed++;
        if (!cancelled) {
            NIXL_ERROR << "IO_URING: descriptor " << task->descIndex << " request "
                       << task->childIndex << " (offset=" << task->offset
                       << " bytes=" << task->bytes << ") failed: " << strerror(-res);
            if (req_h->overallStatus_ == NIXL_IN_PROG) {
                req_h->overallStatus_ = NIXL_ERR_BACKEND;
            }
        }
    } else if (static_cast<uint32_t>(res) != task->bytes) {
        // A read or write may legitimately transfer fewer bytes than asked
        // for, but the caller named an exact range, so here that is a failed
        // request and never a partial success.
        stats_.reqsShort++;
        if (!cancelled) {
            NIXL_ERROR << "IO_URING: descriptor " << task->descIndex << " request "
                       << task->childIndex << " transferred " << res << " of " << task->bytes
                       << " bytes";
            if (req_h->overallStatus_ == NIXL_IN_PROG) {
                req_h->overallStatus_ = NIXL_ERR_BACKEND;
            }
        }
    } else {
        stats_.reqsCompletedOk++;
        if (is_read) {
            stats_.bytesOkRead += task->bytes;
        } else {
            stats_.bytesOkWrite += task->bytes;
        }
    }

    if (++req_h->completedTasks_ >= req_h->totalTasks_) {
        if (!cancelled && req_h->overallStatus_ == NIXL_IN_PROG) {
            req_h->overallStatus_ = NIXL_SUCCESS;
        }
        req_h->allTasksCompleted_.store(true, std::memory_order_release);
    }
}

bool
nixlIoUringProgressEngine::trySubmitRequest(nixlIoUringBackendReqH *req_h) const {
    // FIFO fairness: a new transfer must not jump ahead of parked ones.
    if (!pendingRequests_.empty() && !req_h->isPending_) {
        req_h->nextTask_ = 0;
        return false;
    }

    const bool write = req_h->op_ != NIXL_READ;
    const uint32_t total = req_h->totalTasks_;

    for (uint32_t i = req_h->nextTask_; i < total; i++) {
        nixlIoUringTask *task = &req_h->tasks_[i];
        int err = queue_->submit(write, task->offset, task->bytes, task->buf, task->slot, task);
        if (err == -EAGAIN) {
            stats_.submitParked++;
            req_h->nextTask_ = i;
            return false;
        }
        if (err) {
            stats_.submitPermanentErrors++;
            NIXL_ERROR << "IO_URING: submission of descriptor " << task->descIndex << " request "
                       << task->childIndex << " failed permanently: " << strerror(-err);
            handleSubmissionFailure(req_h, NIXL_ERR_BACKEND);
            return true;
        }
        stats_.reqsSubmitted++;
        req_h->submittedTasks_++;
        req_h->nextTask_ = i + 1;
        const uint32_t inflight = queue_->outstanding();
        if (inflight > stats_.maxInflight) {
            stats_.maxInflight = inflight;
        }
    }
    return true;
}

int
nixlIoUringProgressEngine::pokeLocked() const {
    int did = queue_->poke();
    if (did < 0) {
        if (did == -EAGAIN || did == -EINTR || did == -EBUSY) {
            return 0;
        }
        NIXL_ERROR << "IO_URING: ring poke failed: " << strerror(-did);
        queueError_ = did;
    }
    return did;
}

int
nixlIoUringProgressEngine::progress() const {
    std::lock_guard<std::mutex> guard(lock_);

    const int did = pokeLocked();
    for (auto it = cancelledRequests_.begin(); it != cancelledRequests_.end();) {
        if ((*it)->allTasksCompleted()) {
            delete *it;
            it = cancelledRequests_.erase(it);
        } else {
            ++it;
        }
    }
    return did;
}

void
nixlIoUringProgressEngine::tryResumePendingRequests() const {
    std::lock_guard<std::mutex> guard(lock_);

    if (queueError_) {
        return;
    }

    while (!pendingRequests_.empty()) {
        nixlIoUringBackendReqH *req_h = pendingRequests_.front();
        pendingRequests_.pop_front();
        if (!trySubmitRequest(req_h)) {
            // Still saturated; keep FIFO order and retry later.
            pendingRequests_.push_front(req_h);
            break;
        }
        req_h->isPending_ = false;
    }
}

nixl_status_t
nixlIoUringProgressEngine::checkXfer(nixlIoUringBackendReqH *req_h) const {
    if (!req_h) {
        return NIXL_ERR_INVALID_PARAM;
    }
    progress();
    {
        std::lock_guard<std::mutex> guard(lock_);
        if (queueError_) {
            if (req_h->overallStatus_ == NIXL_IN_PROG) {
                req_h->overallStatus_ = NIXL_ERR_BACKEND;
            }
            return NIXL_ERR_BACKEND;
        }
    }
    tryResumePendingRequests();
    if (req_h->allTasksCompleted()) {
        return req_h->getOverallStatus();
    }
    return NIXL_IN_PROG;
}

nixl_status_t
nixlIoUringProgressEngine::postXfer(nixlIoUringBackendReqH *req_h, nixl_xfer_op_t operation) const {
    std::lock_guard<std::mutex> guard(lock_);

    if (queueError_) {
        req_h->overallStatus_ = NIXL_ERR_BACKEND;
        return NIXL_ERR_BACKEND;
    }

    // Reset for a (re)post: the same handle may be posted again once done.
    req_h->op_ = operation;
    req_h->totalTasks_ = req_h->numTasks();
    req_h->submittedTasks_ = 0;
    req_h->completedTasks_ = 0;
    req_h->nextTask_ = 0;
    req_h->isPending_ = false;
    req_h->overallStatus_ = NIXL_IN_PROG;
    req_h->allTasksCompleted_.store(false, std::memory_order_release);
    stats_.descsRequested += req_h->tasks_.empty() ? 0 : req_h->tasks_.back().descIndex + 1;

    const bool done = trySubmitRequest(req_h);

    // Queued submissions only reach the kernel on a poke; do it now so
    // requests progress even if the caller delays checkXfer. Completions may
    // fire inline here.
    int rc = queue_->poke();
    if (rc < 0) {
        if (rc == -EAGAIN || rc == -EINTR || rc == -EBUSY) {
            rc = 0;
        } else {
            NIXL_ERROR << "IO_URING: ring poke failed: " << strerror(-rc);
            queueError_ = rc;
            req_h->overallStatus_ = NIXL_ERR_BACKEND;
        }
    }

    if (queueError_) {
        return NIXL_ERR_BACKEND;
    }

    if (!done) {
        req_h->isPending_ = true;
        pendingRequests_.push_back(req_h);
        return NIXL_IN_PROG;
    }
    if (req_h->allTasksCompleted()) {
        return req_h->getOverallStatus();
    }
    return NIXL_IN_PROG;
}

void
nixlIoUringProgressEngine::cancelRequest(nixlIoUringBackendReqH *req_h) const {
    std::lock_guard<std::mutex> guard(lock_);

    if (req_h->isPending_) {
        auto it = std::find(pendingRequests_.begin(), pendingRequests_.end(), req_h);
        if (it != pendingRequests_.end()) {
            pendingRequests_.erase(it);
        }
        req_h->isPending_ = false;
        // The unsubmitted remainder will never produce completions.
        req_h->totalTasks_ = req_h->submittedTasks_;
    }

    // Nothing was ever submitted, or everything already completed: no request
    // references the handle, so free it now.
    if (req_h->completedTasks_ >= req_h->totalTasks_ ||
        (req_h->submittedTasks_ == 0 && req_h->completedTasks_ == 0)) {
        delete req_h;
        return;
    }

    // Requests still in flight name both the handle's tasks and the caller's
    // payload buffers, and cancellation is a request to stop rather than
    // evidence that DMA has ceased. The caller may release those buffers as
    // soon as this returns, so drain here rather than deferring.
    req_h->cancelled_.store(true, std::memory_order_release);
    const auto deadline = std::chrono::steady_clock::now() + kCancelDrainBudget;
    while (!req_h->allTasksCompleted()) {
        pokeLocked();
        if (std::chrono::steady_clock::now() > deadline) {
            // Deliberately leak the handle: its tasks are still published to
            // requests the kernel has not finished with.
            NIXL_ERROR << "IO_URING: a released transfer did not drain within "
                       << kCancelDrainBudget.count()
                       << "s; its payload buffers are still owned by the kernel";
            cancelledRequests_.push_back(req_h);
            return;
        }
        std::this_thread::yield();
    }
    delete req_h;
}
