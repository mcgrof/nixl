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
#include "io_uring_queue.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <mutex>

#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#include <liburing.h>

#include "common/nixl_log.h"

namespace {

using nixl_io_uring::diskSysfsDir;
using nixl_io_uring::readDiskLimits;

// Direct I/O constrains both the payload address and the file offset, and a
// filesystem may demand more than the logical block size of the disk under
// it. statx reports what it actually requires; st_blksize is the fallback
// where it does not.
void
setDirectAlignments(int fd, const struct stat &st, bool direct, nixlIoUringTarget &target) {
    target.memAlignBytes = target.logicalBlockBytes;
    target.offsetAlignBytes = target.logicalBlockBytes;
    if (!direct || !S_ISREG(st.st_mode)) {
        return;
    }

    const uint32_t fallback = static_cast<uint32_t>(std::max<uint64_t>(
        target.logicalBlockBytes, st.st_blksize > 0 ? st.st_blksize : target.logicalBlockBytes));
    target.memAlignBytes = fallback;
    target.offsetAlignBytes = fallback;
#ifdef STATX_DIOALIGN
    struct statx stx;
    memset(&stx, 0, sizeof(stx));
    if (statx(fd, "", AT_EMPTY_PATH | AT_STATX_DONT_SYNC, STATX_DIOALIGN, &stx) == 0 &&
        (stx.stx_mask & STATX_DIOALIGN)) {
        if (stx.stx_dio_mem_align) {
            target.memAlignBytes = stx.stx_dio_mem_align;
        }
        if (stx.stx_dio_offset_align) {
            target.offsetAlignBytes = stx.stx_dio_offset_align;
        }
    }
#endif
}

class nixlIoUringRing final : public nixlIoUringQueue {
public:
    nixlIoUringRing() = default;

    ~nixlIoUringRing() override {
        close();
    }

    nixl_status_t
    open(const std::string &path,
         const nixlIoUringOpenOpts &opts,
         nixlIoUringCplCb cpl_cb) override {
        int flags = O_RDWR | O_CLOEXEC;
        if (opts.direct) {
            flags |= O_DIRECT;
        }
        fd_ = ::open(path.c_str(), flags);
        if (fd_ < 0) {
            NIXL_ERROR << "IO_URING: open(" << path << ") failed: " << strerror(errno);
            return NIXL_ERR_BACKEND;
        }

        struct stat st;
        if (fstat(fd_, &st) != 0) {
            NIXL_ERROR << "IO_URING: fstat(" << path << ") failed: " << strerror(errno);
            close();
            return NIXL_ERR_BACKEND;
        }

        // The limits come from the disk under the target either way: its own,
        // for a block device, or the one the file's filesystem is mounted
        // from. A file needs no block device to be usable, so a lookup that
        // finds none is not an error.
        if (S_ISBLK(st.st_mode)) {
            target_.kind = nixlIoUringTarget::Kind::blockDevice;
            unsigned long long size = 0;
            if (ioctl(fd_, BLKGETSIZE64, &size) != 0) {
                NIXL_ERROR << "IO_URING: BLKGETSIZE64 on " << path
                           << " failed: " << strerror(errno);
                close();
                return NIXL_ERR_BACKEND;
            }
            target_.sizeBytes = size;
            readDiskLimits(diskSysfsDir(st.st_rdev), target_);
            if (target_.logicalBlockBytes == 0) {
                int lbs = 0;
                if (ioctl(fd_, BLKSSZGET, &lbs) != 0 || lbs <= 0) {
                    NIXL_ERROR << "IO_URING: BLKSSZGET on " << path
                               << " failed: " << strerror(errno);
                    close();
                    return NIXL_ERR_BACKEND;
                }
                target_.logicalBlockBytes = static_cast<uint32_t>(lbs);
            }
        } else if (S_ISREG(st.st_mode)) {
            target_.kind = nixlIoUringTarget::Kind::regularFile;
            // The addressable range is what the file already holds: this
            // backend writes into a range a caller registered, and never
            // extends the file to make one.
            target_.sizeBytes = static_cast<uint64_t>(st.st_size);
            readDiskLimits(diskSysfsDir(st.st_dev), target_);
            if (target_.logicalBlockBytes == 0) {
                target_.logicalBlockBytes = 512;
            }
        } else {
            NIXL_ERROR << "IO_URING: " << path << " is neither a block device nor a regular file";
            close();
            return NIXL_ERR_INVALID_PARAM;
        }
        setDirectAlignments(fd_, st, opts.direct, target_);

        int err = io_uring_queue_init(opts.numTasks, &ring_, 0);
        if (err < 0) {
            NIXL_ERROR << "IO_URING: io_uring_queue_init(" << opts.numTasks
                       << ") failed: " << strerror(-err);
            close();
            return NIXL_ERR_BACKEND;
        }
        ringUp_ = true;
        capacity_ = opts.numTasks;

        // A sparse table reserves the slot indices without pinning anything,
        // so registrations can come and go while the ring stays up.
        err = io_uring_register_buffers_sparse(&ring_, kSlots);
        if (err < 0) {
            NIXL_ERROR << "IO_URING: sparse buffer table of " << kSlots
                       << " slots refused: " << strerror(-err);
            close();
            return NIXL_ERR_BACKEND;
        }
        tableUp_ = true;

        cplCb_ = std::move(cpl_cb);
        return NIXL_SUCCESS;
    }

    void
    close() override {
        if (ringUp_) {
            NIXL_INFO << "IO_URING: queue stats: submitted=" << statSubmitted_
                      << " saturation_retries=" << statSaturated_
                      << " registrations=" << statRegistered_;
            if (tableUp_) {
                io_uring_unregister_buffers(&ring_);
            }
            io_uring_queue_exit(&ring_);
            ringUp_ = false;
            tableUp_ = false;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    const nixlIoUringTarget &
    target() const override {
        return target_;
    }

    int
    registerHostBuffer(void *addr, size_t len, uint32_t &slot) override {
        std::lock_guard<std::mutex> guard(slotLock_);
        int err = takeSlot(slot);
        if (err) {
            return err;
        }
        struct iovec iov = {addr, len};
        __u64 tag = 0;
        err = io_uring_register_buffers_update_tag(&ring_, slot, &iov, &tag, 1);
        if (err < 0) {
            freeSlot(slot);
            return err;
        }
        statRegistered_++;
        return 0;
    }

    int
    unregisterBuffer(uint32_t slot) override {
        std::lock_guard<std::mutex> guard(slotLock_);
        if (slot >= kSlots || !used_[slot]) {
            return -ENOENT;
        }
        struct iovec iov = {nullptr, 0};
        __u64 tag = 0;
        const int err = io_uring_register_buffers_update_tag(&ring_, slot, &iov, &tag, 1);

        // A failed update leaves the old registration installed, and the kernel
        // still holds the buffer. Retire the slot for the life of the ring so a
        // later registration cannot alias a mapping that is still live.
        if (err < 0) {
            return err;
        }
        freeSlot(slot);
        return 0;
    }

    int
    submit(bool write, uint64_t offset, uint32_t bytes, void *buf, uint32_t slot, void *cookie)
        override {
        // inflight_ counts both submission queue entries still held by
        // liburing and requests the kernel already has. Adding pendingSqes_
        // would count the queued ones twice and halve the usable depth.
        if (inflight_ >= capacity_) {
            statSaturated_++;
            return -EAGAIN;
        }
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring_);
        if (!sqe) {
            statSaturated_++;
            return -EAGAIN;
        }
        if (slot == IO_URING_NO_SLOT) {
            if (write) {
                io_uring_prep_write(sqe, fd_, buf, bytes, offset);
            } else {
                io_uring_prep_read(sqe, fd_, buf, bytes, offset);
            }
        } else if (write) {
            io_uring_prep_write_fixed(sqe, fd_, buf, bytes, offset, static_cast<int>(slot));
        } else {
            io_uring_prep_read_fixed(sqe, fd_, buf, bytes, offset, static_cast<int>(slot));
        }
        io_uring_sqe_set_data(sqe, cookie);
        pendingSqes_++;
        inflight_++;
        statSubmitted_++;
        return 0;
    }

    int
    poke() override {
        int submit_err = 0;
        if (pendingSqes_ > 0) {
            int rc = io_uring_submit(&ring_);
            if (rc < 0) {
                submit_err = rc;
            } else {
                // io_uring_submit() may take only a prefix of the queued
                // entries. The rest stay in liburing's submission queue and
                // must be retried on the next poke rather than forgotten.
                const uint32_t submitted = static_cast<uint32_t>(rc);
                if (submitted > pendingSqes_) {
                    NIXL_ERROR << "IO_URING: io_uring_submit returned " << submitted << " for "
                               << pendingSqes_ << " pending submission queue entries";
                    submit_err = -EIO;
                    pendingSqes_ = 0;
                } else {
                    pendingSqes_ -= submitted;
                }
            }
        }
        int completed = 0;
        struct io_uring_cqe *cqe;
        while (io_uring_peek_cqe(&ring_, &cqe) == 0) {
            void *cookie = io_uring_cqe_get_data(cqe);
            const int32_t res = cqe->res;
            io_uring_cqe_seen(&ring_, cqe);
            inflight_--;
            completed++;
            if (cplCb_) {
                cplCb_(cookie, res);
            }
        }
        // Reap before reporting a submission failure either way: completions
        // free ring capacity and may be what lets the next attempt through.
        // Only report the error when nothing at all moved.
        return completed > 0 ? completed : submit_err;
    }

    uint32_t
    outstanding() const override {
        return inflight_;
    }

private:
    static constexpr uint32_t kSlots = 256;

    int
    takeSlot(uint32_t &slot) {
        for (uint32_t s = 0; s < kSlots; s++) {
            if (!used_[s]) {
                used_[s] = true;
                slot = s;
                return 0;
            }
        }
        return -ENOSPC;
    }

    void
    freeSlot(uint32_t slot) {
        if (slot < kSlots) {
            used_[slot] = false;
        }
    }

    int fd_ = -1;
    struct io_uring ring_{};
    bool ringUp_ = false;
    bool tableUp_ = false;
    uint32_t capacity_ = 0;
    uint32_t inflight_ = 0;
    uint32_t pendingSqes_ = 0;
    // Registration runs on whatever thread the caller registers memory from,
    // while submission and completion run on its transfer threads, so the slot
    // table needs its own serialization.
    std::mutex slotLock_;
    bool used_[kSlots] = {};
    nixlIoUringCplCb cplCb_;
    nixlIoUringTarget target_;
    uint64_t statSubmitted_ = 0;
    uint64_t statSaturated_ = 0;
    uint64_t statRegistered_ = 0;
};

} // namespace

std::unique_ptr<nixlIoUringQueue>
nixlIoUringQueueCreate() {
    return std::make_unique<nixlIoUringRing>();
}
