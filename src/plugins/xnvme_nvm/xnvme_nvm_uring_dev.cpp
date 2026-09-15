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
#include "xnvme_nvm_dev.h"

#include <cerrno>
#include <cstring>
#include <map>
#include <vector>

#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#include <liburing.h>

#include "common/nixl_log.h"

// io_uring device for registered buffers: ordinary READ_FIXED/WRITE_FIXED
// on a block device or file opened here with O_DIRECT, with the buffer
// table holding either pinned host memory (IO_REGBUF_TYPE_UADDR) or dma-bufs
// (IO_REGBUF_TYPE_DMABUF). The dma-buf registration is the extended
// buffer-update ABI of the rw-dmabuf series; its structures are mirrored
// here because no released liburing or kernel header carries them, and a
// kernel without the series answers the update with -EINVAL, which is
// reported at registration time and never at I/O time.

namespace {

using nixl_xnvme_nvm::readSysfs;
using nixl_xnvme_nvm::readSysfsNumber;
using nixl_xnvme_nvm::sysfsNamesFor;

// Mirrors of include/uapi/linux/io_uring.h at rw-dmabuf-v6 / v7.3-rc3:
// io_uring_rsrc_update2 with the flags word, and the extended descriptor.
struct nixlUringRsrcUpdate2 {
    uint32_t offset;
    uint32_t flags;
    uint64_t data;
    uint64_t tags;
    uint32_t nr;
    uint32_t resv2;
};

struct nixlUringRegbufDesc {
    uint32_t type;
    uint32_t flags;
    uint64_t size;
    uint64_t uaddr;
    int32_t dmabuf_fd;
    int32_t target_fd;
    uint64_t resv[6];
};

constexpr uint32_t kRsrcUpdateExtended = 1u << 1; // IORING_RSRC_UPDATE_EXTENDED
constexpr uint32_t kRegbufTypeEmpty = 0; // IO_REGBUF_TYPE_EMPTY
constexpr uint32_t kRegbufTypeDmabuf = 2; // IO_REGBUF_TYPE_DMABUF
#ifndef IORING_REGISTER_BUFFERS_UPDATE
#define IORING_REGISTER_BUFFERS_UPDATE 16
#endif

int
uringRegister(int ring_fd, unsigned op, const void *arg, unsigned nr) {
    const long rc = syscall(__NR_io_uring_register, ring_fd, op, arg, nr);
    return rc < 0 ? -errno : static_cast<int>(rc);
}

class nixlXnvmeNvmUringDev : public nixlXnvmeNvmDevIface {
public:
    nixlXnvmeNvmUringDev() = default;

    ~nixlXnvmeNvmUringDev() override {
        close();
    }

    nixl_status_t
    open(const std::string &uri,
         const nixlXnvmeNvmOpenOpts &opts,
         nixlXnvmeNvmCplCb cpl_cb) override {
        int flags = O_RDWR | O_CLOEXEC;
        if (opts.direct) {
            flags |= O_DIRECT;
        }
        fd_ = ::open(uri.c_str(), flags);
        if (fd_ < 0) {
            NIXL_ERROR << "XNVME_NVM(uring): open(" << uri << ") failed: " << strerror(errno);
            return NIXL_ERR_BACKEND;
        }

        struct stat st;
        if (fstat(fd_, &st) != 0) {
            NIXL_ERROR << "XNVME_NVM(uring): fstat(" << uri << ") failed: " << strerror(errno);
            close();
            return NIXL_ERR_BACKEND;
        }
        if (S_ISBLK(st.st_mode)) {
            geo_.devType = "block_device";
            unsigned long long size = 0;
            int lbs = 0;
            if (ioctl(fd_, BLKGETSIZE64, &size) != 0 || ioctl(fd_, BLKSSZGET, &lbs) != 0 ||
                lbs <= 0) {
                NIXL_ERROR << "XNVME_NVM(uring): block geometry ioctls failed on " << uri << ": "
                           << strerror(errno);
                close();
                return NIXL_ERR_BACKEND;
            }
            geo_.capacityBytes = size;
            geo_.lbaBytes = static_cast<uint32_t>(lbs);
        } else if (S_ISREG(st.st_mode)) {
            geo_.devType = "fs_file";
            geo_.capacityBytes = static_cast<uint64_t>(st.st_size);
            geo_.lbaBytes = 512;
        } else {
            NIXL_ERROR << "XNVME_NVM(uring): " << uri
                       << " is neither a block device nor a regular file";
            close();
            return NIXL_ERR_INVALID_PARAM;
        }
        geo_.nsid = 1;
        const long page = sysconf(_SC_PAGESIZE);
        geo_.pageBytes = page > 0 ? static_cast<uint32_t>(page) : 4096u;

        std::string ctrl, ns;
        if (sysfsNamesFor(uri, ctrl, ns)) {
            const std::string cdir = "/sys/class/nvme/" + ctrl + "/";
            const std::string qdir = "/sys/block/" + ns + "/queue/";
            geo_.serial = readSysfs(cdir + "serial");
            geo_.model = readSysfs(cdir + "model");
            geo_.firmware = readSysfs(cdir + "firmware_rev");
            geo_.wwid = readSysfs("/sys/block/" + ns + "/wwid");
            const unsigned long nsid = readSysfsNumber("/sys/block/" + ns + "/nsid");
            if (nsid) {
                geo_.nsid = static_cast<uint32_t>(nsid);
            }
            const unsigned long kb = readSysfsNumber(qdir + "max_hw_sectors_kb");
            if (kb && kb < (1ul << 22)) {
                geo_.sysfsMaxHwBytes = static_cast<uint32_t>(kb) * 1024u;
            }
            const unsigned long segs = readSysfsNumber(qdir + "max_segments");
            if (segs && segs < (1ul << 20)) {
                geo_.sysfsMaxSegments = static_cast<uint32_t>(segs);
            }
            const unsigned long dkb = readSysfsNumber(qdir + "max_hw_dmabuf_sectors_kb");
            if (dkb && dkb < (1ul << 22)) {
                geo_.sysfsMaxHwDmabufBytes = static_cast<uint32_t>(dkb) * 1024u;
            }
        }

        int err = io_uring_queue_init(opts.numTasks, &ring_, 0);
        if (err < 0) {
            NIXL_ERROR << "XNVME_NVM(uring): io_uring_queue_init(" << opts.numTasks
                       << ") failed: " << strerror(-err);
            close();
            return NIXL_ERR_BACKEND;
        }
        ringUp_ = true;
        capacity_ = opts.numTasks;

        err = io_uring_register_buffers_sparse(&ring_, kSlots);
        if (err < 0) {
            NIXL_ERROR << "XNVME_NVM(uring): sparse buffer table of " << kSlots
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
            NIXL_INFO << "XNVME_NVM(uring): device stats: submitted=" << statSubmitted_
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

    const nixlXnvmeNvmGeo &
    geo() const override {
        return geo_;
    }

    bool
    completionReportsBytes() const override {
        return true;
    }

    bool
    supportsRegistration() const override {
        return true;
    }

    int
    registerHostBuffer(void *addr, size_t len, uint32_t &slot) override {
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
    registerDmabuf(int fd, size_t len, uint32_t &slot) override {
        int err = takeSlot(slot);
        if (err) {
            return err;
        }
        nixlUringRegbufDesc desc;
        memset(&desc, 0, sizeof(desc));
        desc.type = kRegbufTypeDmabuf;
        desc.dmabuf_fd = fd;
        desc.target_fd = fd_;
        nixlUringRsrcUpdate2 up;
        memset(&up, 0, sizeof(up));
        up.offset = slot;
        up.flags = kRsrcUpdateExtended;
        up.data = reinterpret_cast<uint64_t>(&desc);
        up.nr = 1;
        err = uringRegister(ring_.ring_fd, IORING_REGISTER_BUFFERS_UPDATE, &up, sizeof(up));
        if (err < 0) {
            freeSlot(slot);
            return err;
        }
        (void)len;
        dmabufSlots_[slot] = true;
        statRegistered_++;
        return 0;
    }

    int
    unregisterBuffer(uint32_t slot) override {
        if (slot >= kSlots || !used_[slot]) {
            return -ENOENT;
        }
        int err;
        if (dmabufSlots_.count(slot)) {
            nixlUringRegbufDesc desc;
            memset(&desc, 0, sizeof(desc));
            desc.type = kRegbufTypeEmpty;
            nixlUringRsrcUpdate2 up;
            memset(&up, 0, sizeof(up));
            up.offset = slot;
            up.flags = kRsrcUpdateExtended;
            up.data = reinterpret_cast<uint64_t>(&desc);
            up.nr = 1;
            err = uringRegister(ring_.ring_fd, IORING_REGISTER_BUFFERS_UPDATE, &up, sizeof(up));
            dmabufSlots_.erase(slot);
        } else {
            struct iovec iov = {nullptr, 0};
            __u64 tag = 0;
            err = io_uring_register_buffers_update_tag(&ring_, slot, &iov, &tag, 1);
        }
        freeSlot(slot);
        return err < 0 ? err : 0;
    }

    int
    submitRead(uint64_t slba, uint16_t nlb, void *buf, uint32_t slot, void *cookie) override {
        return submit(false, slba, nlb, buf, slot, cookie);
    }

    int
    submitWrite(uint64_t slba, uint16_t nlb, const void *buf, uint32_t slot, void *cookie)
        override {
        return submit(true, slba, nlb, const_cast<void *>(buf), slot, cookie);
    }

    int
    poke() override {
        if (pendingSqes_ > 0) {
            int rc = io_uring_submit(&ring_);
            if (rc < 0) {
                return rc;
            }
            pendingSqes_ = 0;
        }
        int completed = 0;
        struct io_uring_cqe *cqe;
        while (io_uring_peek_cqe(&ring_, &cqe) == 0) {
            nixlXnvmeNvmCpl cpl;
            void *cookie = io_uring_cqe_get_data(cqe);
            if (cqe->res < 0) {
                // Same convention as xNVMe: a kernel errno rides in sc under
                // status code type 7.
                cpl.sc = static_cast<uint8_t>(cqe->res);
                cpl.sct = XNVME_NVM_SCT_ERRNO;
            } else {
                cpl.result = static_cast<uint64_t>(cqe->res);
            }
            io_uring_cqe_seen(&ring_, cqe);
            inflight_--;
            completed++;
            if (cplCb_) {
                cplCb_(cookie, cpl);
            }
        }
        return completed;
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

    int
    submit(bool write, uint64_t slba, uint16_t nlb, void *buf, uint32_t slot, void *cookie) {
        if (inflight_ + pendingSqes_ >= capacity_) {
            statSaturated_++;
            return -EAGAIN;
        }
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring_);
        if (!sqe) {
            statSaturated_++;
            return -EAGAIN;
        }
        const uint64_t bytes = (static_cast<uint64_t>(nlb) + 1) * geo_.lbaBytes;
        const uint64_t off = slba * geo_.lbaBytes;
        if (slot == XNVME_NVM_NO_SLOT) {
            if (write) {
                io_uring_prep_write(sqe, fd_, buf, static_cast<unsigned>(bytes), off);
            } else {
                io_uring_prep_read(sqe, fd_, buf, static_cast<unsigned>(bytes), off);
            }
        } else {
            // For a host slot buf is a virtual address inside the registered
            // range; for a dma-buf slot it is the byte offset into the buffer.
            if (write) {
                io_uring_prep_write_fixed(
                    sqe, fd_, buf, static_cast<unsigned>(bytes), off, static_cast<int>(slot));
            } else {
                io_uring_prep_read_fixed(
                    sqe, fd_, buf, static_cast<unsigned>(bytes), off, static_cast<int>(slot));
            }
        }
        io_uring_sqe_set_data(sqe, cookie);
        pendingSqes_++;
        inflight_++;
        statSubmitted_++;
        return 0;
    }

    int fd_ = -1;
    struct io_uring ring_{};
    bool ringUp_ = false;
    bool tableUp_ = false;
    uint32_t capacity_ = 0;
    uint32_t inflight_ = 0;
    uint32_t pendingSqes_ = 0;
    bool used_[kSlots] = {};
    std::map<uint32_t, bool> dmabufSlots_;
    nixlXnvmeNvmCplCb cplCb_;
    nixlXnvmeNvmGeo geo_;
    uint64_t statSubmitted_ = 0;
    uint64_t statSaturated_ = 0;
    uint64_t statRegistered_ = 0;
};

} // namespace

std::unique_ptr<nixlXnvmeNvmDevIface>
nixlXnvmeNvmUringDevCreate() {
    return std::make_unique<nixlXnvmeNvmUringDev>();
}
