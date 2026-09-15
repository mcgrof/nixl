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
#include "xnvme_nvm_backend.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "common/nixl_log.h"
#include "xnvme_nvm_progress_engine.h"

namespace {

// A BLK_SEG registration: the byte range of the namespace a devId slot may
// address. len 0 means the whole namespace (NIXL itself stores such a
// registration with an unlimited range).
class nixlXnvmeNvmBlkMetadata : public nixlBackendMD {
public:
    nixlXnvmeNvmBlkMetadata(uint64_t dev_id, uint64_t offset, size_t len)
        : nixlBackendMD(true),
          devId(dev_id),
          offset(offset),
          len(len) {}

    uint64_t devId;
    uint64_t offset;
    size_t len;
};

bool
parseUnsigned(const std::string &text, unsigned long long max, unsigned long long &out) {
    try {
        size_t pos = 0;
        unsigned long long parsed = std::stoull(text, &pos, 0);
        if (pos != text.size() || parsed > max) {
            return false;
        }
        out = parsed;
        return true;
    }
    catch (const std::exception &) {
        return false;
    }
}

bool
parseBool(const std::string &text, bool &out) {
    if (text == "true") {
        out = true;
        return true;
    }
    if (text == "false") {
        out = false;
        return true;
    }
    return false;
}

} // namespace

uint32_t
nixlXnvmeNvmEngine::computeMaxIoBytes(const nixlXnvmeNvmGeo &geo,
                                      bool passthrough,
                                      uint32_t user_max_io_bytes) {
    uint64_t limit = UINT64_MAX;
    auto fold = [&limit](uint64_t candidate) {
        if (candidate != 0 && candidate < limit) {
            limit = candidate;
        }
    };
    fold(geo.idfyMdtsBytes);
    fold(static_cast<uint64_t>(65536) * geo.lbaBytes);
    fold(geo.sysfsMaxHwBytes);
    // A passthrough command maps the user buffer as it is: the kernel
    // refuses one that needs more segments than the queue allows instead
    // of splitting it, and an ordinary host buffer can need one segment
    // per page. The block paths split, so they are not bounded here.
    if (passthrough) {
        fold(static_cast<uint64_t>(geo.sysfsMaxSegments) * geo.pageBytes);
    }
    fold(user_max_io_bytes);
    // The NLB term is never zero, so a ceiling always exists.
    limit -= limit % geo.lbaBytes;
    if (limit < geo.lbaBytes) {
        limit = geo.lbaBytes;
    }
    return static_cast<uint32_t>(std::min<uint64_t>(limit, UINT32_MAX));
}

nixlXnvmeNvmEngine::nixlXnvmeNvmEngine(const nixlBackendInitParams *init_params,
                                       std::unique_ptr<nixlXnvmeNvmDevIface> dev)
    : nixlBackendEngine(init_params),
      dev_(std::move(dev)) {
    nixl_status_t status = parseInitParams(init_params->customParams);
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("XNVME_NVM: invalid backend parameters");
    }

    if (deviceName_.empty()) {
        throw std::runtime_error("XNVME_NVM: device_name is required");
    }

    if (!dev_) {
        dev_ = nixlXnvmeNvmDevCreate();
    }

    status = initDevice();
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("XNVME_NVM: device initialization failed");
    }

    if (init_params->enableProgTh) {
        NIXL_WARN << "XNVME_NVM: progress thread requested; this backend progresses from "
                     "checkXfer() only";
    }

    engine_ = std::make_unique<nixlXnvmeNvmProgressEngine>(dev_.get());
    enginePtr_ = engine_.get();

    const nixlXnvmeNvmGeo &g = dev_->geo();
    NIXL_INFO << "XNVME_NVM: initialized device_name=" << deviceName_ << " async=" << async_
              << " direct=" << (direct_ ? "true" : "false") << " dev_type=" << g.devType
              << " nsid=" << g.nsid << " serial='" << g.serial << "' model='" << g.model
              << "' firmware='" << g.firmware << "' wwid='" << g.wwid
              << "' lba_bytes=" << g.lbaBytes << " capacity_bytes=" << g.capacityBytes
              << " idfy_mdts_bytes=" << g.idfyMdtsBytes << " xnvme_mdts_bytes=" << g.xnvmeMdtsBytes
              << " sysfs_max_hw_bytes=" << g.sysfsMaxHwBytes
              << " sysfs_max_segments=" << g.sysfsMaxSegments << " page_bytes=" << g.pageBytes
              << " max_io_bytes=" << maxIoBytes_ << " num_tasks=" << numTasks_
              << " buffer_mode=host";
}

nixlXnvmeNvmEngine::~nixlXnvmeNvmEngine() {
    // The engine drains in-flight commands before the device and queue go
    // away; completions reaped during that drain still route via enginePtr_.
    engine_.reset();
    enginePtr_ = nullptr;
    if (dev_) {
        dev_->close();
    }
}

nixl_status_t
nixlXnvmeNvmEngine::parseInitParams(const nixl_b_params_t *custom_params) {
    if (!custom_params) {
        return NIXL_SUCCESS;
    }

    unsigned long long value = 0;

    if (auto it = custom_params->find("device_name"); it != custom_params->end()) {
        deviceName_ = it->second;
    }

    if (auto it = custom_params->find("async"); it != custom_params->end()) {
        static const char *const known[] = {
            "io_uring_cmd", "io_uring", "libaio", "posix", "thrpool"};
        const bool ok = std::any_of(
            std::begin(known), std::end(known), [&](const char *k) { return it->second == k; });
        if (!ok) {
            NIXL_ERROR << "XNVME_NVM: async must be one of io_uring_cmd, io_uring, libaio, "
                          "posix, thrpool; got '"
                       << it->second << "'";
            return NIXL_ERR_INVALID_PARAM;
        }
        async_ = it->second;
        if (async_ == "posix" || async_ == "thrpool") {
            NIXL_WARN << "XNVME_NVM: async=" << async_
                      << " is an emulated interface for development only";
        }
    }

    if (auto it = custom_params->find("direct"); it != custom_params->end()) {
        if (!parseBool(it->second, direct_)) {
            NIXL_ERROR << "XNVME_NVM: direct must be 'true' or 'false'";
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    if (auto it = custom_params->find("nsid"); it != custom_params->end()) {
        if (!parseUnsigned(it->second, UINT32_MAX, value)) {
            NIXL_ERROR << "XNVME_NVM: invalid nsid '" << it->second << "'";
            return NIXL_ERR_INVALID_PARAM;
        }
        nsid_ = static_cast<uint32_t>(value);
    }

    if (auto it = custom_params->find("num_tasks"); it != custom_params->end()) {
        // xNVMe queue capacity must be a power of two below 4096.
        if (!parseUnsigned(it->second, 2048, value) || value == 0 || (value & (value - 1)) != 0) {
            NIXL_ERROR << "XNVME_NVM: num_tasks must be a power of two in [1, 2048], got '"
                       << it->second << "'";
            return NIXL_ERR_INVALID_PARAM;
        }
        numTasks_ = static_cast<uint32_t>(value);
    }

    if (auto it = custom_params->find("max_io_bytes"); it != custom_params->end()) {
        if (!parseUnsigned(it->second, UINT32_MAX, value)) {
            NIXL_ERROR << "XNVME_NVM: invalid max_io_bytes '" << it->second << "'";
            return NIXL_ERR_INVALID_PARAM;
        }
        userMaxIoBytes_ = static_cast<uint32_t>(value);
    }

    if (auto it = custom_params->find("max_io_bytes_force"); it != custom_params->end()) {
        if (!parseUnsigned(it->second, UINT32_MAX, value)) {
            NIXL_ERROR << "XNVME_NVM: invalid max_io_bytes_force '" << it->second << "'";
            return NIXL_ERR_INVALID_PARAM;
        }
        forceMaxIoBytes_ = static_cast<uint32_t>(value);
    }

    if (auto it = custom_params->find("expect_serial"); it != custom_params->end()) {
        expectSerial_ = it->second;
    }
    if (auto it = custom_params->find("expect_wwid"); it != custom_params->end()) {
        expectWwid_ = it->second;
    }

    if (auto it = custom_params->find("buffer_mode");
        it != custom_params->end() && !it->second.empty() && it->second != "host") {
        // fixed: the pinned xNVMe never consumes opts.register_buffers.
        // dmabuf: needs the kernel lane's registration path (Phase 3).
        NIXL_ERROR << "XNVME_NVM: buffer_mode '" << it->second
                   << "' is not supported; only 'host' is";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlXnvmeNvmEngine::initDevice() {
    nixlXnvmeNvmOpenOpts opts;
    opts.async = async_;
    opts.direct = direct_;
    opts.nsidOverride = nsid_;
    opts.numTasks = numTasks_;

    nixl_status_t status =
        dev_->open(deviceName_, opts, [this](void *cookie, const nixlXnvmeNvmCpl &cpl) {
            if (enginePtr_) {
                enginePtr_->onCompletion(cookie, cpl);
            }
        });
    if (status != NIXL_SUCCESS) {
        return status;
    }

    const nixlXnvmeNvmGeo &g = dev_->geo();
    if (g.capacityBytes == 0 || g.lbaBytes == 0) {
        NIXL_ERROR << "XNVME_NVM: device reports capacity " << g.capacityBytes << " bytes and lba "
                   << g.lbaBytes << " bytes; unusable";
        dev_->close();
        return NIXL_ERR_BACKEND;
    }
    // Identity is checked before any command can be issued: an ephemeral
    // device name is not ownership evidence.
    if (!expectSerial_.empty() && g.serial != expectSerial_) {
        NIXL_ERROR << "XNVME_NVM: serial '" << g.serial << "' does not match expect_serial '"
                   << expectSerial_ << "'";
        dev_->close();
        return NIXL_ERR_INVALID_PARAM;
    }
    if (!expectWwid_.empty() && g.wwid != expectWwid_) {
        NIXL_ERROR << "XNVME_NVM: wwid '" << g.wwid << "' does not match expect_wwid '"
                   << expectWwid_ << "'";
        dev_->close();
        return NIXL_ERR_INVALID_PARAM;
    }
    if (expectSerial_.empty() && expectWwid_.empty()) {
        NIXL_WARN << "XNVME_NVM: no expect_serial/expect_wwid given; device identity unvalidated";
    }

    maxIoBytes_ = computeMaxIoBytes(g, async_ == "io_uring_cmd", userMaxIoBytes_);
    if (forceMaxIoBytes_) {
        // The NLB field is an encoding limit; everything else in the
        // ceiling is a limit the kernel or device enforces at its end, and
        // a caller with physically contiguous buffers may choose to test it.
        const uint64_t nlb_limit = static_cast<uint64_t>(65536) * g.lbaBytes;
        uint64_t forced = forceMaxIoBytes_ - forceMaxIoBytes_ % g.lbaBytes;
        if (forced == 0 || forced > nlb_limit) {
            NIXL_ERROR << "XNVME_NVM: max_io_bytes_force=" << forceMaxIoBytes_ << " is outside (0, "
                       << nlb_limit << "] for a " << g.lbaBytes << "-byte logical block";
            dev_->close();
            return NIXL_ERR_INVALID_PARAM;
        }
        if (forced > maxIoBytes_) {
            NIXL_WARN << "XNVME_NVM: max_io_bytes_force=" << forced
                      << " overrides the computed ceiling of " << maxIoBytes_
                      << "; the kernel or device may refuse such commands";
        }
        maxIoBytes_ = static_cast<uint32_t>(forced);
    }
    return NIXL_SUCCESS;
}

const nixlXnvmeNvmGeo &
nixlXnvmeNvmEngine::geo() const {
    return dev_->geo();
}

nixlXnvmeNvmStats
nixlXnvmeNvmEngine::stats() const {
    return engine_->stats();
}

nixl_status_t
nixlXnvmeNvmEngine::registerMem(const nixlBlobDesc &mem,
                                const nixl_mem_t &nixl_mem,
                                nixlBackendMD *&out) {
    out = nullptr;

    if (nixl_mem == DRAM_SEG) {
        // Host buffers go to the kernel as-is at submit time; nothing to
        // pin or register here.
        return NIXL_SUCCESS;
    }
    if (nixl_mem != BLK_SEG) {
        NIXL_ERROR << "XNVME_NVM: unsupported memory type " << nixl_mem;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    const nixlXnvmeNvmGeo &g = dev_->geo();
    if (!mem.metaInfo.empty() && mem.metaInfo != deviceName_) {
        NIXL_ERROR << "XNVME_NVM: BLK_SEG devId " << mem.devId << " names device '" << mem.metaInfo
                   << "' but this backend opened '" << deviceName_ << "'";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (mem.addr % g.lbaBytes != 0 || mem.len % g.lbaBytes != 0) {
        NIXL_ERROR << "XNVME_NVM: BLK_SEG range offset=" << mem.addr << " len=" << mem.len
                   << " is not a multiple of the " << g.lbaBytes << "-byte logical block";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (mem.addr >= g.capacityBytes || mem.len > g.capacityBytes - mem.addr) {
        NIXL_ERROR << "XNVME_NVM: BLK_SEG range offset=" << mem.addr << " len=" << mem.len
                   << " exceeds the namespace capacity of " << g.capacityBytes << " bytes";
        return NIXL_ERR_INVALID_PARAM;
    }

    {
        std::lock_guard<std::mutex> guard(devIdLock_);
        auto [it, inserted] = devIdPaths_.try_emplace(mem.devId, deviceName_);
        (void)it;
        (void)inserted;
    }

    out = new nixlXnvmeNvmBlkMetadata(mem.devId, mem.addr, mem.len);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlXnvmeNvmEngine::deregisterMem(nixlBackendMD *meta) {
    // Registration metadata only; the bytes stay on the device.
    delete static_cast<nixlXnvmeNvmBlkMetadata *>(meta);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlXnvmeNvmEngine::queryMem(const nixl_reg_dlist_t &descs,
                             std::vector<nixl_query_resp_t> &resp) const {
    resp.clear();
    resp.reserve(descs.descCount());

    const nixlXnvmeNvmGeo &g = dev_->geo();
    for (int i = 0; i < descs.descCount(); i++) {
        const nixlBlobDesc &d = descs[i];
        if (descs.getType() != BLK_SEG) {
            resp.emplace_back(nixl_b_params_t{});
            continue;
        }
        const bool inside = d.addr < g.capacityBytes && d.len <= g.capacityBytes - d.addr &&
            (d.metaInfo.empty() || d.metaInfo == deviceName_);
        if (!inside) {
            resp.emplace_back(std::nullopt);
            continue;
        }
        // No device command: conventional NVM has no existence query; the
        // answer is the namespace identity and geometry.
        resp.emplace_back(nixl_b_params_t{{"serial", g.serial},
                                          {"wwid", g.wwid},
                                          {"capacity_bytes", std::to_string(g.capacityBytes)},
                                          {"lba_bytes", std::to_string(g.lbaBytes)},
                                          {"max_io_bytes", std::to_string(maxIoBytes_)}});
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlXnvmeNvmEngine::prepXfer(const nixl_xfer_op_t &operation,
                             const nixl_meta_dlist_t &local,
                             const nixl_meta_dlist_t &remote,
                             const std::string &remote_agent,
                             nixlBackendReqH *&handle,
                             const nixl_opt_b_args_t *opt_args) const {
    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        NIXL_ERROR << "XNVME_NVM: only NIXL_WRITE and NIXL_READ are supported";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (remote_agent != localAgent) {
        NIXL_WARN << "XNVME_NVM: remote agent (" << remote_agent << ") != local agent ("
                  << localAgent << "); XNVME_NVM is a local storage backend";
    }
    if (local.getType() != DRAM_SEG) {
        NIXL_ERROR << "XNVME_NVM: local descriptor list must be DRAM_SEG, got " << local.getType();
        return NIXL_ERR_INVALID_PARAM;
    }
    if (remote.getType() != BLK_SEG) {
        NIXL_ERROR << "XNVME_NVM: remote descriptor list must be BLK_SEG, got " << remote.getType();
        return NIXL_ERR_INVALID_PARAM;
    }
    const int desc_count = local.descCount();
    if (desc_count == 0) {
        NIXL_ERROR << "XNVME_NVM: empty descriptor list";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (remote.descCount() != desc_count) {
        NIXL_ERROR << "XNVME_NVM: local (" << desc_count << ") and remote (" << remote.descCount()
                   << ") descriptor counts differ";
        return NIXL_ERR_INVALID_PARAM;
    }

    const nixlXnvmeNvmGeo &g = dev_->geo();
    const uint32_t lba = g.lbaBytes;
    auto req_h = std::make_unique<nixlXnvmeNvmBackendReqH>();

    // First pass: validate and count commands, so the task vector is sized
    // once and never grows after its addresses are published.
    size_t total_cmds = 0;
    for (int i = 0; i < desc_count; i++) {
        const nixlMetaDesc &l = local[i];
        const nixlMetaDesc &r = remote[i];
        if (l.len != r.len) {
            NIXL_ERROR << "XNVME_NVM: descriptor " << i << " local len " << l.len
                       << " != remote len " << r.len;
            return NIXL_ERR_INVALID_PARAM;
        }
        if (l.len == 0) {
            NIXL_ERROR << "XNVME_NVM: descriptor " << i << " has zero length";
            return NIXL_ERR_INVALID_PARAM;
        }
        if (l.addr % lba != 0 || r.addr % lba != 0 || l.len % lba != 0) {
            NIXL_ERROR << "XNVME_NVM: descriptor " << i << " (host=0x" << std::hex << l.addr
                       << " offset=0x" << r.addr << " len=0x" << l.len << std::dec
                       << ") is not aligned to the " << lba << "-byte logical block";
            return NIXL_ERR_INVALID_PARAM;
        }
        if (r.addr >= g.capacityBytes || r.len > g.capacityBytes - r.addr) {
            NIXL_ERROR << "XNVME_NVM: descriptor " << i << " offset=" << r.addr << " len=" << r.len
                       << " exceeds the namespace capacity of " << g.capacityBytes << " bytes";
            return NIXL_ERR_INVALID_PARAM;
        }
        const auto *md = static_cast<const nixlXnvmeNvmBlkMetadata *>(r.metadataP);
        if (!md) {
            NIXL_ERROR << "XNVME_NVM: remote descriptor " << i << " has no registration metadata";
            return NIXL_ERR_INVALID_PARAM;
        }
        if (md->len != 0 &&
            (r.len > md->len || r.addr < md->offset || r.addr - md->offset > md->len - r.len)) {
            NIXL_ERROR << "XNVME_NVM: descriptor " << i << " offset=" << r.addr << " len=" << r.len
                       << " lies outside its registration [" << md->offset << ", "
                       << md->offset + md->len << ")";
            return NIXL_ERR_INVALID_PARAM;
        }
        total_cmds += (l.len + maxIoBytes_ - 1) / maxIoBytes_;
    }
    if (total_cmds > UINT32_MAX) {
        NIXL_ERROR << "XNVME_NVM: request needs " << total_cmds << " commands";
        return NIXL_ERR_INVALID_PARAM;
    }

    req_h->tasks_.reserve(total_cmds);
    for (int i = 0; i < desc_count; i++) {
        const nixlMetaDesc &l = local[i];
        const nixlMetaDesc &r = remote[i];
        uint32_t child = 0;
        for (size_t done = 0; done < l.len; done += maxIoBytes_, child++) {
            const size_t bytes = std::min<size_t>(maxIoBytes_, l.len - done);
            nixlXnvmeNvmTask task;
            task.reqH = req_h.get();
            task.descIndex = static_cast<uint32_t>(i);
            task.childIndex = child;
            task.bytes = static_cast<uint32_t>(bytes);
            task.slba = (r.addr + done) / lba;
            task.nlb = static_cast<uint16_t>(bytes / lba - 1);
            task.buf = reinterpret_cast<void *>(l.addr + done);
            req_h->tasks_.push_back(task);
        }
    }

    handle = req_h.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlXnvmeNvmEngine::postXfer(const nixl_xfer_op_t &operation,
                             const nixl_meta_dlist_t &local,
                             const nixl_meta_dlist_t &remote,
                             const std::string &remote_agent,
                             nixlBackendReqH *&handle,
                             const nixl_opt_b_args_t *opt_args) const {
    if (!handle) {
        NIXL_ERROR << "XNVME_NVM: postXfer on null handle";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        NIXL_ERROR << "XNVME_NVM: unsupported operation";
        return NIXL_ERR_INVALID_PARAM;
    }
    return engine_->postXfer(static_cast<nixlXnvmeNvmBackendReqH *>(handle), operation);
}

nixl_status_t
nixlXnvmeNvmEngine::checkXfer(nixlBackendReqH *handle) const {
    return engine_->checkXfer(static_cast<nixlXnvmeNvmBackendReqH *>(handle));
}

nixl_status_t
nixlXnvmeNvmEngine::releaseReqH(nixlBackendReqH *handle) const {
    if (!handle) {
        NIXL_ERROR << "XNVME_NVM: releaseReqH on null handle";
        return NIXL_ERR_INVALID_PARAM;
    }
    engine_->cancelRequest(static_cast<nixlXnvmeNvmBackendReqH *>(handle));
    return NIXL_SUCCESS;
}
