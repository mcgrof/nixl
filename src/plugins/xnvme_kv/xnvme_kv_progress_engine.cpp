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
#include "xnvme_kv_progress_engine.h"

#include <algorithm>
#include <cerrno>

#include "common/nixl_log.h"

nixlXnvmeKvProgressEngine::nixlXnvmeKvProgressEngine(nixlXnvmeKvDevIface *dev,
                                                     bool ignore_read_not_found)
    : dev_(dev),
      ignoreReadNotFound_(ignore_read_not_found) {}

nixl_status_t
nixlXnvmeKvProgressEngine::checkXfer(nixlXnvmeKvBackendReqH *req_h) const {
    if (!req_h)
        return NIXL_ERR_INVALID_PARAM;
    if (req_h->allTasksCompleted())
        return req_h->getOverallStatus();
    return NIXL_IN_PROG;
}

void
nixlXnvmeKvProgressEngine::handleSubmissionFailure(nixlXnvmeKvBackendReqH *req_h,
                                                   nixl_status_t status) {
    if (req_h->overallStatus_ == NIXL_IN_PROG)
        req_h->overallStatus_ = status;
    // Descriptors past this point are never submitted and will never
    // complete; the failed slot counts as completed so the handle can reach
    // the completion threshold once in-flight commands drain.
    req_h->totalTasks_ = req_h->submittedTasks_ + 1;
    if (++req_h->completedTasks_ >= req_h->totalTasks_)
        req_h->allTasksCompleted_.store(true, std::memory_order_release);
}

void
nixlXnvmeKvProgressEngine::onCompletion(void *cookie, const nixlXnvmeKvCpl &cpl) const {
    auto *task_ctx = static_cast<nixlXnvmeKvTaskContext *>(cookie);
    nixlXnvmeKvBackendReqH *req_h = task_ctx->reqH;
    // Relaxed: a cancellation racing this completion is reconciled by the
    // cancel bookkeeping; a missed recent flag is harmless here.
    const bool cancelled = req_h->cancelled_.load(std::memory_order_relaxed);

    if (!cancelled) {
        if (req_h->isExistQuery_) {
            if (cpl.ok()) {
                req_h->existResult_.store(XNVME_KV_EXIST_YES, std::memory_order_release);
            } else if (cpl.keyNotExists()) {
                req_h->existResult_.store(XNVME_KV_EXIST_NO, std::memory_order_release);
            } else {
                NIXL_ERROR << "XNVME_KV: EXIST failed: sc=" << std::hex
                           << static_cast<unsigned>(cpl.sc) << " sct="
                           << static_cast<unsigned>(cpl.sct) << std::dec;
                if (req_h->overallStatus_ == NIXL_IN_PROG)
                    req_h->overallStatus_ = NIXL_ERR_BACKEND;
            }
        } else if (cpl.ok()) {
            // Per the KV spec a Retrieve reports the actual value length in
            // completion dword 0. Devices that leave it 0 (vfio-user-kvssd)
            // cannot be length-checked.
            if (task_ctx->isRetrieve && cpl.cdw0 != 0 &&
                cpl.cdw0 != task_ctx->expectedValueLen)
                NIXL_WARN << "XNVME_KV: task " << task_ctx->taskIndex << " retrieved "
                          << cpl.cdw0 << " bytes, expected " << task_ctx->expectedValueLen;
        } else if (cpl.keyNotExists() && req_h->ignoreNotFound_) {
            NIXL_DEBUG << "XNVME_KV: task " << task_ctx->taskIndex
                       << " key not found; ignored per ignore_read_not_found";
        } else {
            NIXL_ERROR << "XNVME_KV: task " << task_ctx->taskIndex << " failed: sc="
                       << std::hex << static_cast<unsigned>(cpl.sc) << " sct="
                       << static_cast<unsigned>(cpl.sct) << std::dec;
            if (req_h->overallStatus_ == NIXL_IN_PROG)
                req_h->overallStatus_ = NIXL_ERR_BACKEND;
        }
    }

    if (++req_h->completedTasks_ >= req_h->totalTasks_) {
        if (!cancelled && req_h->overallStatus_ == NIXL_IN_PROG)
            req_h->overallStatus_ = NIXL_SUCCESS;
        req_h->allTasksCompleted_.store(true, std::memory_order_release);
    }
}

bool
nixlXnvmeKvProgressEngine::trySubmitRequest(nixlXnvmeKvBackendReqH *req_h) const {
    // FIFO fairness: a new request must not jump ahead of parked ones.
    if (!pendingRequests_.empty() && !req_h->isPending_) {
        req_h->nextDescriptorIndex_ = 0;
        return false;
    }

    const bool is_retrieve = (req_h->storedOperation_ == NIXL_READ);
    const uint32_t max_value_len = dev_->maxValueLen();
    const int desc_count = static_cast<int>(req_h->valueIovecs_.size());

    for (int i = req_h->nextDescriptorIndex_; i < desc_count; i++) {
        const struct iovec &iov = req_h->valueIovecs_[i];

        if (iov.iov_len > UINT32_MAX) {
            NIXL_ERROR << "XNVME_KV: descriptor " << i << " length " << iov.iov_len
                       << " exceeds the 32-bit KV value size field";
            handleSubmissionFailure(req_h, NIXL_ERR_INVALID_PARAM);
            return true;
        }
        if (max_value_len > 0 && iov.iov_len > max_value_len) {
            NIXL_ERROR << "XNVME_KV: descriptor " << i << " length " << iov.iov_len
                       << " exceeds device max value length " << max_value_len;
            handleSubmissionFailure(req_h, NIXL_ERR_INVALID_PARAM);
            return true;
        }

        nixlXnvmeKvTaskContext *task_ctx = &req_h->taskContexts_[i];
        task_ctx->reqH = req_h;
        task_ctx->taskIndex = i;
        task_ctx->isRetrieve = is_retrieve;
        task_ctx->expectedValueLen = iov.iov_len;

        const nixlXnvmeKvKey &key = req_h->objectKeys_[i];
        const auto len = static_cast<uint32_t>(iov.iov_len);
        int err = is_retrieve ?
            dev_->submitRetrieve(key.key, key.keyLen, iov.iov_base, len, task_ctx) :
            dev_->submitStore(key.key, key.keyLen, iov.iov_base, len, task_ctx);

        if (err == -EAGAIN) {
            // Queue saturated; remember where to resume and park the request.
            req_h->nextDescriptorIndex_ = i;
            return false;
        }
        if (err) {
            NIXL_ERROR << "XNVME_KV: submission of descriptor " << i
                       << " failed permanently: " << err;
            handleSubmissionFailure(req_h, NIXL_ERR_BACKEND);
            return true;
        }
        req_h->submittedTasks_++;
        req_h->nextDescriptorIndex_ = i + 1;
    }
    return true;
}

bool
nixlXnvmeKvProgressEngine::trySubmitExistTask(nixlXnvmeKvBackendReqH *req_h) const {
    if (!pendingRequests_.empty() && !req_h->isPending_)
        return false;

    nixlXnvmeKvTaskContext *task_ctx = &req_h->taskContexts_[0];
    task_ctx->reqH = req_h;
    task_ctx->taskIndex = 0;
    task_ctx->isRetrieve = false;
    task_ctx->expectedValueLen = 0;

    const nixlXnvmeKvKey &key = req_h->objectKeys_[0];
    int err = dev_->submitExist(key.key, key.keyLen, task_ctx);

    if (err == -EAGAIN)
        return false;
    if (err) {
        NIXL_ERROR << "XNVME_KV: EXIST submission failed permanently: " << err;
        handleSubmissionFailure(req_h, NIXL_ERR_BACKEND);
        return true;
    }
    req_h->submittedTasks_++;
    return true;
}

nixl_status_t
nixlXnvmeKvProgressEngine::prepareQueryHandles(const nixl_reg_dlist_t &descs,
                                               std::vector<nixlXnvmeKvBackendReqH> &handles) const {
    const int desc_count = descs.descCount();

    // reserve() is load-bearing: task-context addresses inside the handles
    // are published to in-flight commands, so the vector must never grow.
    handles.reserve(desc_count);
    for (int i = 0; i < desc_count; i++) {
        auto &req_h = handles.emplace_back(1);
        req_h.isExistQuery_ = true;
        req_h.taskContexts_.resize(1);

        nixlXnvmeKvKey key;
        if (!nixlXnvmeKvEngine::resolveKvKey(descs[i].devId, descs[i].metaInfo, key)) {
            NIXL_ERROR << "XNVME_KV: queryMem descriptor " << i << " has no derivable key";
            handleSubmissionFailure(&req_h, NIXL_ERR_INVALID_PARAM);
            continue;
        }
        req_h.objectKeys_.push_back(key);
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlXnvmeKvProgressEngine::collectQueryResults(std::vector<nixlXnvmeKvBackendReqH> &handles,
                                               std::vector<nixl_query_resp_t> &resp) const {
    size_t successful_queries = 0;

    for (auto &req_h : handles) {
        if (req_h.getOverallStatus() == NIXL_SUCCESS) {
            const int exist = req_h.existResult_.load(std::memory_order_acquire);
            if (exist == XNVME_KV_EXIST_YES) {
                resp.emplace_back(nixl_b_params_t{});
                successful_queries++;
                continue;
            }
            if (exist == XNVME_KV_EXIST_NO) {
                // A missing key is a successful query with a negative answer.
                resp.emplace_back(std::nullopt);
                successful_queries++;
                continue;
            }
        }
        resp.emplace_back(std::nullopt);
    }
    return successful_queries < handles.size() ? NIXL_ERR_BACKEND : NIXL_SUCCESS;
}

// ---------------------------------------------------------------------------
// Caller-progress engine
// ---------------------------------------------------------------------------

nixlXnvmeKvNoThreadEngine::nixlXnvmeKvNoThreadEngine(nixlXnvmeKvDevIface *dev,
                                                     bool ignore_read_not_found)
    : nixlXnvmeKvProgressEngine(dev, ignore_read_not_found) {}

nixlXnvmeKvNoThreadEngine::~nixlXnvmeKvNoThreadEngine() {
    const auto deadline = std::chrono::steady_clock::now() + kDestructorDrainBudget;

    for (;;) {
        int did = progress();
        bool idle;
        {
            std::lock_guard<std::mutex> guard(lock_);
            idle = did <= 0 && cancelledRequests_.empty() && dev_->outstanding() == 0;
        }
        if (idle)
            break;
        if (std::chrono::steady_clock::now() > deadline) {
            NIXL_WARN << "XNVME_KV: requests did not drain within "
                      << kDestructorDrainBudget.count() << "s at engine destruction";
            break;
        }
        std::this_thread::yield();
    }
    if (!pendingRequests_.empty())
        NIXL_WARN << "XNVME_KV: " << pendingRequests_.size()
                  << " unreleased request(s) at engine destruction";
}

int
nixlXnvmeKvNoThreadEngine::progress() const {
    std::lock_guard<std::mutex> guard(lock_);

    int did = dev_->poke();
    if (did < 0)
        NIXL_ERROR << "XNVME_KV: queue poke failed: " << did;

    // Free cancelled requests whose in-flight commands have drained.
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
nixlXnvmeKvNoThreadEngine::tryResumePendingRequests() const {
    std::lock_guard<std::mutex> guard(lock_);

    while (!pendingRequests_.empty()) {
        nixlXnvmeKvBackendReqH *req_h = pendingRequests_.front();
        pendingRequests_.pop_front();
        const bool done =
            req_h->isExistQuery_ ? trySubmitExistTask(req_h) : trySubmitRequest(req_h);
        if (!done) {
            // Still saturated; keep FIFO order and retry later.
            pendingRequests_.push_front(req_h);
            break;
        }
        // A finished submission attempt may have completed the handle (e.g.
        // a permanent failure with nothing in flight), at which point its
        // owner — a queryMem caller spinning on allTasksCompleted() without
        // this lock — may already have destroyed it. Do not touch req_h
        // here; stale isPending_ is reset on repost.
    }
}

nixl_status_t
nixlXnvmeKvNoThreadEngine::checkXfer(nixlXnvmeKvBackendReqH *req_h) const {
    if (!req_h)
        return NIXL_ERR_INVALID_PARAM;
    progress();
    tryResumePendingRequests();
    return nixlXnvmeKvProgressEngine::checkXfer(req_h);
}

nixl_status_t
nixlXnvmeKvNoThreadEngine::postXfer(nixlXnvmeKvBackendReqH *req_h,
                                    nixl_xfer_op_t operation) const {
    std::lock_guard<std::mutex> guard(lock_);

    // Reset for (re)post: the same handle may be posted again after DONE.
    req_h->storedOperation_ = operation;
    req_h->totalTasks_ = static_cast<int>(req_h->valueIovecs_.size());
    req_h->submittedTasks_ = 0;
    req_h->completedTasks_ = 0;
    req_h->nextDescriptorIndex_ = 0;
    req_h->isPending_ = false;
    req_h->overallStatus_ = NIXL_IN_PROG;
    req_h->allTasksCompleted_.store(false, std::memory_order_release);

    const bool done = trySubmitRequest(req_h);

    // With xNVMe's default command batching, queued submissions reach the
    // kernel on the next poke; do it now so commands progress even if the
    // caller delays checkXfer. Completions may fire inline here.
    int rc = dev_->poke();
    if (rc < 0)
        NIXL_ERROR << "XNVME_KV: queue poke failed: " << rc;

    if (!done) {
        req_h->isPending_ = true;
        pendingRequests_.push_back(req_h);
        return NIXL_IN_PROG;
    }
    if (req_h->allTasksCompleted())
        return req_h->getOverallStatus();
    return NIXL_IN_PROG;
}

void
nixlXnvmeKvNoThreadEngine::cancelRequest(nixlXnvmeKvBackendReqH *req_h) const {
    std::lock_guard<std::mutex> guard(lock_);

    if (req_h->isPending_) {
        auto it = std::find(pendingRequests_.begin(), pendingRequests_.end(), req_h);
        if (it != pendingRequests_.end())
            pendingRequests_.erase(it);
        req_h->isPending_ = false;
        // The unsubmitted remainder will never produce completions.
        req_h->totalTasks_ = req_h->submittedTasks_;
    }

    // Nothing was ever submitted (prepXfer'd but never posted, or fully
    // reaped): no command references the handle, free it now.
    if (req_h->completedTasks_ >= req_h->totalTasks_ ||
        (req_h->submittedTasks_ == 0 && req_h->completedTasks_ == 0)) {
        delete req_h;
        return;
    }

    // Commands still in flight reference the handle's task contexts; defer
    // deletion until progress() observes full drain.
    req_h->cancelled_.store(true, std::memory_order_release);
    cancelledRequests_.push_back(req_h);
}

nixl_status_t
nixlXnvmeKvNoThreadEngine::queryMem(const nixl_reg_dlist_t &descs,
                                    std::vector<nixl_query_resp_t> &resp) const {
    std::vector<nixlXnvmeKvBackendReqH> handles;

    prepareQueryHandles(descs, handles);

    {
        std::lock_guard<std::mutex> guard(lock_);
        for (auto &req_h : handles) {
            if (req_h.allTasksCompleted())
                continue; // key resolution already failed the handle
            if (!trySubmitExistTask(&req_h)) {
                req_h.isPending_ = true;
                pendingRequests_.push_back(&req_h);
            }
        }
        int rc = dev_->poke();
        if (rc < 0)
            NIXL_ERROR << "XNVME_KV: queue poke failed: " << rc;
    }

    waitForQueryCompletion(handles, [this] {
        progress();
        tryResumePendingRequests();
    });

    return collectQueryResults(handles, resp);
}

// ---------------------------------------------------------------------------
// Progress-thread engine
// ---------------------------------------------------------------------------

nixlXnvmeKvThreadedEngine::nixlXnvmeKvThreadedEngine(nixlXnvmeKvDevIface *dev,
                                                     bool ignore_read_not_found,
                                                     std::chrono::microseconds thread_delay)
    : nixlXnvmeKvProgressEngine(dev, ignore_read_not_found),
      threadDelay_(thread_delay) {
    progressThread_ = std::thread(&nixlXnvmeKvThreadedEngine::progressThreadFunc, this);
}

nixlXnvmeKvThreadedEngine::~nixlXnvmeKvThreadedEngine() {
    threadStop_.store(true, std::memory_order_release);
    wakeup_.notify_all();
    if (progressThread_.joinable())
        progressThread_.join();

    // Drain whatever is still in flight; after the thread has stopped this
    // destructor is the sole queue owner.
    const auto deadline = std::chrono::steady_clock::now() + kDestructorDrainBudget;
    while (dev_->outstanding() > 0) {
        int rc = dev_->poke();
        if (rc < 0)
            break;
        if (std::chrono::steady_clock::now() > deadline) {
            NIXL_WARN << "XNVME_KV: " << dev_->outstanding()
                      << " command(s) did not drain within " << kDestructorDrainBudget.count()
                      << "s at engine destruction";
            break;
        }
    }

    const bool drained = dev_->outstanding() == 0;
    for (auto *req_h : pendingDeletes_)
        if (req_h->allTasksCompleted() || drained)
            delete req_h;
        else
            NIXL_WARN << "XNVME_KV: leaking request with in-flight commands at destruction";
    for (auto &entry : vecA_)
        delete entry.reqH; // never submitted
    for (auto &entry : vecB_)
        delete entry.reqH; // never submitted
    for (auto *req_h : cancelledRequests_) {
        // The progress thread never processed these cancellations; reconcile
        // parked entries here so a fully drained queue frees them.
        if (req_h->isPending_) {
            auto it = std::find(pendingRequests_.begin(), pendingRequests_.end(), req_h);
            if (it != pendingRequests_.end())
                pendingRequests_.erase(it);
            req_h->isPending_ = false;
            req_h->totalTasks_ = req_h->submittedTasks_;
        }
        if (req_h->completedTasks_ >= req_h->totalTasks_ || drained)
            delete req_h;
        else
            NIXL_WARN << "XNVME_KV: leaking cancelled request with in-flight commands";
    }
    if (!pendingRequests_.empty())
        NIXL_WARN << "XNVME_KV: " << pendingRequests_.size()
                  << " unreleased request(s) at engine destruction";
}

nixl_status_t
nixlXnvmeKvThreadedEngine::postXfer(nixlXnvmeKvBackendReqH *req_h,
                                    nixl_xfer_op_t operation) const {
    // The handle is not yet visible to the progress thread; reset unlocked.
    req_h->storedOperation_ = operation;
    req_h->totalTasks_ = static_cast<int>(req_h->valueIovecs_.size());
    req_h->submittedTasks_ = 0;
    req_h->completedTasks_ = 0;
    req_h->nextDescriptorIndex_ = 0;
    req_h->isPending_ = false;
    req_h->overallStatus_ = NIXL_IN_PROG;
    req_h->allTasksCompleted_.store(false, std::memory_order_release);

    {
        std::lock_guard<std::mutex> guard(queueMutex_);
        producerVec_->push_back({req_h, operation, false});
    }
    hasNewWork_.store(true, std::memory_order_release);
    wakeup_.notify_one();
    return NIXL_IN_PROG;
}

void
nixlXnvmeKvThreadedEngine::cancelRequest(nixlXnvmeKvBackendReqH *req_h) const {
    {
        std::lock_guard<std::mutex> guard(queueMutex_);
        // Not yet picked up by the thread: remove and delete immediately.
        auto it = std::find_if(producerVec_->begin(),
                               producerVec_->end(),
                               [req_h](const pendingEntry &e) { return e.reqH == req_h; });
        if (it != producerVec_->end()) {
            producerVec_->erase(it);
            delete req_h;
            return;
        }
        // Set cancelled_ before publishing so racing completion callbacks
        // skip status bookkeeping from now on.
        req_h->cancelled_.store(true, std::memory_order_release);
        cancelledRequests_.push_back(req_h);
    }
    hasNewWork_.store(true, std::memory_order_release);
    wakeup_.notify_one();
}

nixl_status_t
nixlXnvmeKvThreadedEngine::queryMem(const nixl_reg_dlist_t &descs,
                                    std::vector<nixl_query_resp_t> &resp) const {
    std::vector<nixlXnvmeKvBackendReqH> handles;

    prepareQueryHandles(descs, handles);

    size_t enqueued = 0;
    {
        std::lock_guard<std::mutex> guard(queueMutex_);
        for (auto &req_h : handles) {
            if (req_h.allTasksCompleted())
                continue;
            producerVec_->push_back({&req_h, NIXL_READ, true});
            enqueued++;
        }
    }
    if (enqueued > 0) {
        hasNewWork_.store(true, std::memory_order_release);
        wakeup_.notify_one();
        waitForQueryCompletion(handles, [] {});
    }

    return collectQueryResults(handles, resp);
}

void
nixlXnvmeKvThreadedEngine::processCancellations(std::vector<nixlXnvmeKvBackendReqH *> &cancelled) {
    for (auto *req_h : cancelled) {
        if (req_h->isPending_) {
            auto it = std::find(pendingRequests_.begin(), pendingRequests_.end(), req_h);
            if (it != pendingRequests_.end())
                pendingRequests_.erase(it);
            req_h->isPending_ = false;
            // Only a parked request may have its task count collapsed: for a
            // fully submitted one, in-flight completions still count against
            // the original total (collapsing it would free the handle while
            // commands reference it).
            req_h->totalTasks_ = req_h->submittedTasks_;
        }
        if (req_h->completedTasks_ >= req_h->totalTasks_ ||
            (req_h->submittedTasks_ == 0 && req_h->completedTasks_ == 0))
            delete req_h;
        else
            pendingDeletes_.push_back(req_h);
    }
    cancelled.clear();
}

void
nixlXnvmeKvThreadedEngine::progressThreadFunc() {
    std::vector<pendingEntry> local_work;
    std::vector<nixlXnvmeKvBackendReqH *> local_cancelled;

    while (!threadStop_.load(std::memory_order_acquire)) {
        int did = dev_->poke();
        if (did < 0)
            NIXL_ERROR << "XNVME_KV: queue poke failed: " << did;

        // Free cancelled requests that have fully drained.
        for (auto it = pendingDeletes_.begin(); it != pendingDeletes_.end();) {
            if ((*it)->allTasksCompleted()) {
                delete *it;
                it = pendingDeletes_.erase(it);
            } else {
                ++it;
            }
        }

        // Resume partially submitted requests, FIFO. Pop before the attempt
        // and never touch the handle after a finished attempt: completing it
        // (permanent failure) releases its owner — a queryMem caller's stack
        // vector — immediately. Stale isPending_ is reset on repost.
        while (!pendingRequests_.empty()) {
            nixlXnvmeKvBackendReqH *req_h = pendingRequests_.front();
            pendingRequests_.pop_front();
            const bool done =
                req_h->isExistQuery_ ? trySubmitExistTask(req_h) : trySubmitRequest(req_h);
            if (!done) {
                pendingRequests_.push_front(req_h);
                break;
            }
        }

        bool had_work = false;
        if (hasNewWork_.exchange(false, std::memory_order_acquire)) {
            {
                std::lock_guard<std::mutex> guard(queueMutex_);
                std::vector<pendingEntry> *full = producerVec_;
                producerVec_ = (producerVec_ == &vecA_) ? &vecB_ : &vecA_;
                local_work.swap(*full);
                local_cancelled.swap(cancelledRequests_);
            }
            // Cancellations first so a cancelled-but-never-submitted entry in
            // local_work is recognized below.
            processCancellations(local_cancelled);

            for (auto &entry : local_work) {
                if (entry.reqH->cancelled_.load(std::memory_order_acquire)) {
                    // Cancelled after the swap: never submit it; collapsing
                    // the task count lets the deferred deletion free it.
                    entry.reqH->totalTasks_ = entry.reqH->submittedTasks_;
                    continue;
                }
                const bool done = entry.isExistQuery ? trySubmitExistTask(entry.reqH) :
                                                       trySubmitRequest(entry.reqH);
                if (!done) {
                    entry.reqH->isPending_ = true;
                    pendingRequests_.push_back(entry.reqH);
                }
            }
            local_work.clear();
            had_work = true;
        }

        const bool busy = did > 0 || had_work || dev_->outstanding() > 0 ||
            !pendingRequests_.empty() || !pendingDeletes_.empty();
        if (threadDelay_.count() == 0)
            continue; // explicit busy-spin configuration
        if (busy) {
            std::this_thread::sleep_for(threadDelay_);
        } else {
            std::unique_lock<std::mutex> lk(wakeupMutex_);
            wakeup_.wait_for(lk, threadDelay_, [this] {
                return threadStop_.load(std::memory_order_acquire) ||
                    hasNewWork_.load(std::memory_order_acquire);
            });
        }
    }
}
