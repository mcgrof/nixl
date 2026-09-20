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

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <libxnvme.h>

#include "common/nixl_log.h"

namespace nixl_xnvme_nvm {

std::string
readSysfs(const std::string &path) {
    std::ifstream in(path);
    std::string value;
    if (!in) {
        return "";
    }
    std::getline(in, value);
    while (!value.empty() && (value.back() == '\n' || value.back() == ' ')) {
        value.pop_back();
    }
    return value;
}

// A sysfs decimal attribute, or 0 when absent or malformed.
unsigned long
readSysfsNumber(const std::string &path) {
    const std::string text = readSysfs(path);
    if (text.empty()) {
        return 0;
    }
    char *end = nullptr;
    const unsigned long value = strtoul(text.c_str(), &end, 10);
    return (end && *end == '\0') ? value : 0;
}

// Map /dev/ngXnY or /dev/nvmeXnY to its controller and namespace sysfs
// names. Anything else (files, other block devices) yields false.
bool
sysfsNamesFor(const std::string &uri, std::string &ctrl, std::string &ns) {
    const size_t slash = uri.find_last_of('/');
    const std::string base = slash == std::string::npos ? uri : uri.substr(slash + 1);
    unsigned c = 0, n = 0;
    char tail = 0;
    if (sscanf(base.c_str(), "ng%un%u%c", &c, &n, &tail) == 2 ||
        sscanf(base.c_str(), "nvme%un%u%c", &c, &n, &tail) == 2) {
        ctrl = "nvme" + std::to_string(c);
        ns = ctrl + "n" + std::to_string(n);
        return true;
    }
    return false;
}

} // namespace nixl_xnvme_nvm

namespace {

using nixl_xnvme_nvm::readSysfs;
using nixl_xnvme_nvm::readSysfsNumber;
using nixl_xnvme_nvm::sysfsNamesFor;

enum class devKind { unknown, nvmeNamespace, nvmeController, blockDevice, fsFile };

// xNVMe keeps its device-type enum in a private header, so classify the
// path the same way it does: a character device is an NVMe handle (a
// controller when the namespace id reads back as all ones), a block
// device or a regular file are the ordinary block I/O targets.
devKind
classify(const std::string &uri, uint32_t nsid) {
    struct stat st;
    if (stat(uri.c_str(), &st) != 0) {
        return devKind::unknown;
    }
    if (S_ISCHR(st.st_mode)) {
        return nsid == 0xFFFFFFFFu ? devKind::nvmeController : devKind::nvmeNamespace;
    }
    if (S_ISBLK(st.st_mode)) {
        return devKind::blockDevice;
    }
    if (S_ISREG(st.st_mode)) {
        return devKind::fsFile;
    }
    return devKind::unknown;
}

const char *
devKindName(devKind kind) {
    switch (kind) {
    case devKind::nvmeNamespace:
        return "nvme_namespace";
    case devKind::nvmeController:
        return "nvme_controller";
    case devKind::blockDevice:
        return "block_device";
    case devKind::fsFile:
        return "fs_file";
    default:
        return "unknown";
    }
}

void
setDirectAlignments(const std::string &uri,
                    const struct stat &st,
                    bool direct,
                    nixlXnvmeNvmGeo &geo) {
    geo.memAlignBytes = geo.lbaBytes;
    geo.offsetAlignBytes = geo.lbaBytes;
    if (!direct || !S_ISREG(st.st_mode)) {
        return;
    }

    // statx reports the actual filesystem O_DIRECT contract. st_blksize is
    // only a conservative fallback when STATX_DIOALIGN is unavailable.
    const uint32_t fallback = static_cast<uint32_t>(
        std::max<uint64_t>(geo.lbaBytes, st.st_blksize > 0 ? st.st_blksize : geo.lbaBytes));
    geo.memAlignBytes = fallback;
    geo.offsetAlignBytes = fallback;
#ifdef STATX_DIOALIGN
    struct statx stx;
    memset(&stx, 0, sizeof(stx));
    if (statx(AT_FDCWD, uri.c_str(), AT_STATX_DONT_SYNC, STATX_DIOALIGN, &stx) == 0 &&
        (stx.stx_mask & STATX_DIOALIGN)) {
        if (stx.stx_dio_mem_align) {
            geo.memAlignBytes = stx.stx_dio_mem_align;
        }
        if (stx.stx_dio_offset_align) {
            geo.offsetAlignBytes = stx.stx_dio_offset_align;
        }
    }
#endif
}

class nixlXnvmeNvmRealDev : public nixlXnvmeNvmDevIface {
public:
    nixlXnvmeNvmRealDev() = default;

    ~nixlXnvmeNvmRealDev() override {
        close();
    }

    nixl_status_t
    open(const std::string &uri,
         const nixlXnvmeNvmOpenOpts &opts,
         nixlXnvmeNvmCplCb cpl_cb) override {
        struct xnvme_opts xopts = xnvme_opts_default();

        async_ = opts.async;
        xopts.be = "linux";
        xopts.async = async_.c_str();
        xopts.rdwr = 1;
        xopts.direct = opts.direct ? 1 : 0;

        // xNVMe's default Linux admin interface issues NVMe ioctls. A
        // regular file (and a non-NVMe block device) instead needs the
        // file_as_ns identify shim to populate capacity and 512-byte LBA
        // geometry. Without it xnvme_dev_open() can return a handle whose
        // geometry is all zeroes, making the documented file path unusable.
        struct stat path_stat;
        std::string sysfs_ctrl, sysfs_ns;
        if (stat(uri.c_str(), &path_stat) == 0 &&
            (S_ISREG(path_stat.st_mode) ||
             (S_ISBLK(path_stat.st_mode) && !sysfsNamesFor(uri, sysfs_ctrl, sysfs_ns)))) {
            xopts.admin = "file_as_ns";
        }

        dev_ = xnvme_dev_open(uri.c_str(), &xopts);
        if (!dev_) {
            NIXL_ERROR << "XNVME_NVM: xnvme_dev_open(" << uri << ", async=" << async_
                       << ") failed: " << strerror(errno);
            return NIXL_ERR_BACKEND;
        }

        geo_.nsid = xnvme_dev_get_nsid(dev_);
        const devKind kind = classify(uri, geo_.nsid);
        geo_.devType = devKindName(kind);

        // Each interface only works on the handle type it was written for;
        // refuse the wrong pairing here rather than at the first command.
        const bool is_ns = kind == devKind::nvmeNamespace;
        const bool is_bio = kind == devKind::blockDevice || kind == devKind::fsFile;
        if (async_ == "io_uring_cmd" && !is_ns) {
            NIXL_ERROR << "XNVME_NVM: io_uring_cmd needs a /dev/ngXnY namespace handle; " << uri
                       << " is a " << geo_.devType;
            close();
            return NIXL_ERR_INVALID_PARAM;
        }
        if ((async_ == "io_uring" || async_ == "libaio") && !is_bio) {
            NIXL_ERROR << "XNVME_NVM: " << async_ << " needs a block device or file; " << uri
                       << " is a " << geo_.devType;
            close();
            return NIXL_ERR_INVALID_PARAM;
        }

        if (opts.nsidOverride && opts.nsidOverride != geo_.nsid) {
            NIXL_ERROR << "XNVME_NVM: " << uri << " is namespace " << geo_.nsid
                       << ", not the requested nsid " << opts.nsidOverride;
            close();
            return NIXL_ERR_INVALID_PARAM;
        }

        const struct xnvme_geo *xgeo = xnvme_dev_get_geo(dev_);
        geo_.lbaBytes = xgeo->lba_nbytes;
        geo_.capacityBytes = xgeo->tbytes;
        geo_.xnvmeMdtsBytes = xgeo->mdts_nbytes;
        if (geo_.lbaBytes == 0 || (geo_.lbaBytes & (geo_.lbaBytes - 1)) != 0) {
            NIXL_ERROR << "XNVME_NVM: unusable logical block size " << geo_.lbaBytes;
            close();
            return NIXL_ERR_BACKEND;
        }

        const long page = sysconf(_SC_PAGESIZE);
        geo_.pageBytes = page > 0 ? static_cast<uint32_t>(page) : 4096u;
        if (stat(uri.c_str(), &path_stat) == 0) {
            setDirectAlignments(uri, path_stat, opts.direct, geo_);
        }

        std::string ctrl, ns;
        if (sysfsNamesFor(uri, ctrl, ns)) {
            const std::string cdir = "/sys/class/nvme/" + ctrl + "/";
            const std::string qdir = "/sys/block/" + ns + "/queue/";
            geo_.serial = readSysfs(cdir + "serial");
            geo_.model = readSysfs(cdir + "model");
            geo_.firmware = readSysfs(cdir + "firmware_rev");
            geo_.wwid = readSysfs("/sys/block/" + ns + "/wwid");
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

        int err = xnvme_queue_init(dev_, opts.numTasks, 0, &queue_);
        if (err) {
            NIXL_ERROR << "XNVME_NVM: xnvme_queue_init(capacity=" << opts.numTasks
                       << ") failed: " << strerror(-err);
            close();
            return NIXL_ERR_BACKEND;
        }

        cplCb_ = std::move(cpl_cb);
        return NIXL_SUCCESS;
    }

    void
    close() override {

        if (queue_) {
            NIXL_INFO << "XNVME_NVM: device stats: submitted=" << statSubmitted_
                      << " saturation_retries=" << statSaturated_;
            int err = xnvme_queue_term(queue_);
            if (err) {
                NIXL_WARN << "XNVME_NVM: xnvme_queue_term failed: " << strerror(-err);
            }
            queue_ = nullptr;
        }
        if (dev_) {
            xnvme_dev_close(dev_);
            dev_ = nullptr;
        }
    }

    const nixlXnvmeNvmGeo &
    geo() const override {
        return geo_;
    }

    bool
    completionReportsBytes() const override {
        return async_ != "io_uring_cmd";
    }

    int
    submitRead(uint64_t slba, uint16_t nlb, void *buf, uint32_t slot, void *cookie) override {
        if (slot != XNVME_NVM_NO_SLOT) {
            return -ENOTSUP;
        }
        return submit(false, slba, nlb, buf, cookie);
    }

    int
    submitWrite(uint64_t slba, uint16_t nlb, const void *buf, uint32_t slot, void *cookie)
        override {
        if (slot != XNVME_NVM_NO_SLOT) {
            return -ENOTSUP;
        }
        return submit(true, slba, nlb, const_cast<void *>(buf), cookie);
    }

    int
    poke() override {
        tlsPokingDev_ = this;
        int ret = xnvme_queue_poke(queue_, 0);
        tlsPokingDev_ = nullptr;
        return ret;
    }

    uint32_t
    outstanding() const override {
        return queue_ ? xnvme_queue_get_outstanding(queue_) : 0;
    }

private:
    int
    submit(bool write, uint64_t slba, uint16_t nlb, void *buf, void *cookie) {
        struct xnvme_cmd_ctx *ctx = xnvme_queue_get_cmd_ctx(queue_);

        if (!ctx) {
            statSaturated_++;
            return -EAGAIN;
        }

        // Pooled contexts keep the previous command; start from zero so no
        // stale field survives into a strict controller's view.
        memset(&ctx->cmd, 0, sizeof(ctx->cmd));
        xnvme_cmd_ctx_set_cb(ctx, cplTrampoline, cookie);

        int err = write ? xnvme_nvm_write(ctx, geo_.nsid, slba, nlb, buf, nullptr) :
                          xnvme_nvm_read(ctx, geo_.nsid, slba, nlb, buf, nullptr);
        if (err) {
            xnvme_queue_put_cmd_ctx(queue_, ctx);
            // -EBUSY (in-flight == capacity) and -EAGAIN (no SQE) are the
            // same retry-later condition to the engine.
            if (err == -EBUSY || err == -EAGAIN) {
                statSaturated_++;
                return -EAGAIN;
            }
            return err;
        }
        statSubmitted_++;
        return 0;
    }

    static void
    cplTrampoline(struct xnvme_cmd_ctx *ctx, void *cb_arg) {
        // xNVMe only fires callbacks from within xnvme_queue_poke(), which is
        // always entered through poke() above, so the thread-local is valid.
        nixlXnvmeNvmRealDev *dev = tlsPokingDev_;
        nixlXnvmeNvmCpl cpl;

        cpl.sc = ctx->cpl.status.sc;
        cpl.sct = ctx->cpl.status.sct;
        cpl.result = ctx->cpl.result;

        xnvme_queue_put_cmd_ctx(ctx->async.queue, ctx);

        if (dev && dev->cplCb_) {
            dev->cplCb_(cb_arg, cpl);
        }
    }

    static thread_local nixlXnvmeNvmRealDev *tlsPokingDev_;

    struct xnvme_dev *dev_ = nullptr;
    struct xnvme_queue *queue_ = nullptr;
    nixlXnvmeNvmCplCb cplCb_;
    nixlXnvmeNvmGeo geo_;
    std::string async_;
    uint64_t statSubmitted_ = 0;
    uint64_t statSaturated_ = 0;
};

thread_local nixlXnvmeNvmRealDev *nixlXnvmeNvmRealDev::tlsPokingDev_ = nullptr;

} // namespace

std::unique_ptr<nixlXnvmeNvmDevIface>
nixlXnvmeNvmDevCreate() {
    return std::make_unique<nixlXnvmeNvmRealDev>();
}
