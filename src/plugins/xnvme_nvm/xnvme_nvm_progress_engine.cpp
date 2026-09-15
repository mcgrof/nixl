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
#include "xnvme_nvm_progress_engine.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <thread>

#include "common/nixl_log.h"

nixlXnvmeNvmProgressEngine::nixlXnvmeNvmProgressEngine(nixlXnvmeNvmDevIface *dev) : dev_(dev) {}

nixlXnvmeNvmProgressEngine::~nixlXnvmeNvmProgressEngine() {
    const auto deadline = std::chrono::steady_clock::now() + kDestructorDrainBudget;

    for (;;) {
        int did = progress();
        bool idle;
        {
            std::lock_guard<std::mutex> guard(lock_);
            idle = did <= 0 && cancelledRequests_.empty() && dev_->outstanding() == 0;
        }
        if (idle) {
            break;
        }
        if (std::chrono::steady_clock::now() > deadline) {
            // Deliberately leak: the kernel may still DMA into the buffers
            // those requests reference.
            NIXL_WARN << "XNVME_NVM: requests did not drain within "
                      << kDestructorDrainBudget.count() << "s at engine destruction";
            break;
        }
        std::this_thread::yield();
    }
    if (!pendingRequests_.empty()) {
        NIXL_WARN << "XNVME_NVM: " << pendingRequests_.size()
                  << " unreleased request(s) at engine destruction";
    }

    NIXL_INFO << "XNVME_NVM: stats: descs_requested=" << stats_.descsRequested
              << " cmds_submitted=" << stats_.cmdsSubmitted
              << " cmds_completed_ok=" << stats_.cmdsCompletedOk
              << " cmds_failed=" << stats_.cmdsFailed << " cmds_short=" << stats_.cmdsShort
              << " submit_parked=" << stats_.submitParked
              << " submit_permanent_errors=" << stats_.submitPermanentErrors
              << " bytes_ok_read=" << stats_.bytesOkRead
              << " bytes_ok_write=" << stats_.bytesOkWrite
              << " max_inflight=" << stats_.maxInflight;
}

nixlXnvmeNvmStats
nixlXnvmeNvmProgressEngine::stats() const {
    std::lock_guard<std::mutex> guard(lock_);
    return stats_;
}

void
nixlXnvmeNvmProgressEngine::handleSubmissionFailure(nixlXnvmeNvmBackendReqH *req_h,
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
nixlXnvmeNvmProgressEngine::onCompletion(void *cookie, const nixlXnvmeNvmCpl &cpl) const {
    auto *task = static_cast<nixlXnvmeNvmTask *>(cookie);
    nixlXnvmeNvmBackendReqH *req_h = task->reqH;
    const bool cancelled = req_h->cancelled_.load(std::memory_order_relaxed);
    const bool is_read = req_h->op_ == NIXL_READ;

    if (!cpl.ok()) {
        stats_.cmdsFailed++;
        if (!cancelled) {
            const int err = cpl.kernelErrno();
            NIXL_ERROR << "XNVME_NVM: descriptor " << task->descIndex << " command "
                       << task->childIndex << " (slba=" << task->slba << " nlb=" << task->nlb
                       << ") failed: sc=" << std::hex << static_cast<unsigned>(cpl.sc)
                       << " sct=" << static_cast<unsigned>(cpl.sct) << std::dec
                       << (err ? " (" + std::string(strerror(-err)) + ")" : "");
            if (req_h->overallStatus_ == NIXL_IN_PROG) {
                req_h->overallStatus_ = NIXL_ERR_BACKEND;
            }
        }
    } else if (dev_->completionReportsBytes() && cpl.result != task->bytes) {
        // The bio path can complete with fewer bytes than asked; that is a
        // failed child, never a partial success.
        stats_.cmdsShort++;
        if (!cancelled) {
            NIXL_ERROR << "XNVME_NVM: descriptor " << task->descIndex << " command "
                       << task->childIndex << " transferred " << cpl.result << " of " << task->bytes
                       << " bytes";
            if (req_h->overallStatus_ == NIXL_IN_PROG) {
                req_h->overallStatus_ = NIXL_ERR_BACKEND;
            }
        }
    } else {
        stats_.cmdsCompletedOk++;
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
nixlXnvmeNvmProgressEngine::trySubmitRequest(nixlXnvmeNvmBackendReqH *req_h) const {
    // FIFO fairness: a new request must not jump ahead of parked ones.
    if (!pendingRequests_.empty() && !req_h->isPending_) {
        req_h->nextTask_ = 0;
        return false;
    }

    const bool is_read = req_h->op_ == NIXL_READ;
    const uint32_t total = req_h->totalTasks_;

    for (uint32_t t = req_h->nextTask_; t < total; t++) {
        nixlXnvmeNvmTask *task = &req_h->tasks_[t];
        int err = is_read ? dev_->submitRead(task->slba, task->nlb, task->buf, task->slot, task) :
                            dev_->submitWrite(task->slba, task->nlb, task->buf, task->slot, task);
        if (err == -EAGAIN) {
            stats_.submitParked++;
            req_h->nextTask_ = t;
            return false;
        }
        if (err) {
            stats_.submitPermanentErrors++;
            NIXL_ERROR << "XNVME_NVM: submission of descriptor " << task->descIndex << " command "
                       << task->childIndex << " failed permanently: " << err;
            handleSubmissionFailure(req_h, NIXL_ERR_BACKEND);
            return true;
        }
        stats_.cmdsSubmitted++;
        req_h->submittedTasks_++;
        req_h->nextTask_ = t + 1;
    }
    const uint32_t inflight = dev_->outstanding();
    if (inflight > stats_.maxInflight) {
        stats_.maxInflight = inflight;
    }
    return true;
}

int
nixlXnvmeNvmProgressEngine::progress() const {
    std::lock_guard<std::mutex> guard(lock_);

    int did = dev_->poke();
    if (did < 0) {
        NIXL_ERROR << "XNVME_NVM: queue poke failed: " << did;
    }

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
nixlXnvmeNvmProgressEngine::tryResumePendingRequests() const {
    std::lock_guard<std::mutex> guard(lock_);

    while (!pendingRequests_.empty()) {
        nixlXnvmeNvmBackendReqH *req_h = pendingRequests_.front();
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
nixlXnvmeNvmProgressEngine::checkXfer(nixlXnvmeNvmBackendReqH *req_h) const {
    if (!req_h) {
        return NIXL_ERR_INVALID_PARAM;
    }
    progress();
    tryResumePendingRequests();
    if (req_h->allTasksCompleted()) {
        return req_h->getOverallStatus();
    }
    return NIXL_IN_PROG;
}

nixl_status_t
nixlXnvmeNvmProgressEngine::postXfer(nixlXnvmeNvmBackendReqH *req_h,
                                     nixl_xfer_op_t operation) const {
    std::lock_guard<std::mutex> guard(lock_);

    // Reset for (re)post: the same handle may be posted again after DONE.
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

    // Queued submissions reach the kernel on the next poke; do it now so
    // commands progress even if the caller delays checkXfer. Completions
    // may fire inline here.
    int rc = dev_->poke();
    if (rc < 0) {
        NIXL_ERROR << "XNVME_NVM: queue poke failed: " << rc;
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
nixlXnvmeNvmProgressEngine::cancelRequest(nixlXnvmeNvmBackendReqH *req_h) const {
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

    // Nothing was ever submitted, or everything already completed: no
    // command references the handle, free it now.
    if (req_h->completedTasks_ >= req_h->totalTasks_ ||
        (req_h->submittedTasks_ == 0 && req_h->completedTasks_ == 0)) {
        delete req_h;
        return;
    }

    // Commands still in flight reference the handle's tasks; cancellation
    // is a request to stop, not evidence that DMA has ceased. Defer deletion
    // until progress() observes full drain.
    req_h->cancelled_.store(true, std::memory_order_release);
    cancelledRequests_.push_back(req_h);
}
