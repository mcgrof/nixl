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
#include "io_uring_backend.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "common/nixl_log.h"
#include "io_uring_progress_engine.h"

namespace {

// A registered payload buffer in fixed or dmabuf mode: the slot the ring
// handed out, and the virtual range the caller addresses it by.
class nixlIoUringBufferMetadata : public nixlBackendMD {
public:
    nixlIoUringBufferMetadata(uint32_t slot, uintptr_t base, size_t len, bool is_dmabuf)
        : nixlBackendMD(true),
          slot(slot),
          base(base),
          len(len),
          isDmabuf(is_dmabuf) {}

    uint32_t slot;
    uintptr_t base;
    size_t len;
    bool isDmabuf;
};

// A BLK_SEG registration: the byte range of the target a devId may address.
// len 0 means the whole target, which is how NIXL itself stores an unbounded
// registration.
class nixlIoUringBlkMetadata : public nixlBackendMD {
public:
    nixlIoUringBlkMetadata(uint64_t dev_id, uint64_t offset, size_t len)
        : nixlBackendMD(true),
          devId(dev_id),
          offset(offset),
          len(len) {}

    uint64_t devId;
    uint64_t offset;
    size_t len;
};

// Reads "dmabuf:<fd>" out of a registration's metaInfo.
bool
parseDmabufMeta(const std::string &meta, int &fd) {
    static const std::string prefix = "dmabuf:";
    if (meta.compare(0, prefix.size(), prefix) != 0) {
        return false;
    }
    try {
        size_t pos = 0;
        const long parsed = std::stol(meta.substr(prefix.size()), &pos, 10);
        if (pos != meta.size() - prefix.size() || parsed < 0 || parsed > INT32_MAX) {
            return false;
        }
        fd = static_cast<int>(parsed);
        return true;
    }
    catch (const std::exception &) {
        return false;
    }
}

bool
parseUnsigned(const std::string &text, unsigned long long max, unsigned long long &out) {
    try {
        size_t pos = 0;
        unsigned long long parsed = std::stoull(text, &pos, 10);
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

const char *
bufferModeName(nixlIoUringBufferMode mode) {
    switch (mode) {
    case nixlIoUringBufferMode::fixed:
        return "fixed";
    case nixlIoUringBufferMode::dmabuf:
        return "dmabuf";
    default:
        return "host";
    }
}

} // namespace

uint32_t
nixlIoUringEngine::lengthAlignment(const nixlIoUringTarget &t) {
    const uint32_t lb = t.logicalBlockBytes;
    return std::max(lb, t.offsetAlignBytes ? t.offsetAlignBytes : lb);
}

uint32_t
nixlIoUringEngine::computeMaxIoBytes(const nixlIoUringTarget &target,
                                     nixlIoUringBufferMode mode,
                                     uint32_t user_max_io_bytes) {
    // A single read or write carries its length in an unsigned, so no request
    // can exceed that however permissive the target is.
    uint64_t limit = UINT32_MAX;
    bool bounded = false;
    auto fold = [&limit, &bounded](uint64_t candidate) {
        if (candidate == 0) {
            return;
        }
        bounded = true;
        if (candidate < limit) {
            limit = candidate;
        }
    };
    // A dma-buf payload is described to the driver by DMA addresses rather
    // than pages, so the descriptor budget a request fits into differs and the
    // queue advertises its own ceiling for it. Where the kernel publishes
    // none, such a request is bounded like any other.
    if (mode == nixlIoUringBufferMode::dmabuf && target.maxHwDmabufBytes) {
        fold(target.maxHwDmabufBytes);
    } else {
        fold(target.maxHwBytes);
    }
    // queue/max_segments is deliberately left out. These are ordinary reads
    // and writes, which the block layer splits when a buffer needs more
    // segments than the queue allows; only a request the driver had to map
    // whole would be refused outright instead.
    fold(user_max_io_bytes);
    if (!bounded) {
        limit = kFallbackMaxIoBytes;
    }

    const uint32_t alignment = lengthAlignment(target);
    if (alignment == 0) {
        return 0;
    }
    limit -= limit % alignment;
    if (limit < alignment) {
        return 0;
    }
    return static_cast<uint32_t>(limit);
}

nixlIoUringEngine::nixlIoUringEngine(const nixlBackendInitParams *init_params,
                                     std::unique_ptr<nixlIoUringQueue> queue)
    : nixlBackendEngine(init_params),
      queue_(std::move(queue)) {
    nixl_status_t status = parseInitParams(init_params->customParams);
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("IO_URING: invalid backend parameters");
    }

    if (deviceName_.empty()) {
        throw std::runtime_error("IO_URING: device_name is required");
    }

    if (!queue_) {
        queue_ = nixlIoUringQueueCreate();
    }

    status = initQueue();
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("IO_URING: queue initialization failed");
    }

    if (init_params->enableProgTh) {
        NIXL_WARN << "IO_URING: progress thread requested; this backend progresses from "
                     "checkXfer() only";
    }

    engine_ = std::make_unique<nixlIoUringProgressEngine>(queue_.get());
    enginePtr_ = engine_.get();

    const nixlIoUringTarget &t = queue_->target();
    NIXL_INFO << "IO_URING: initialized device_name=" << deviceName_
              << " direct=" << (direct_ ? "true" : "false") << " target=" << t.kindName()
              << " serial='" << t.serial << "' model='" << t.model << "' firmware='" << t.firmware
              << "' wwid='" << t.wwid << "' logical_block_bytes=" << t.logicalBlockBytes
              << " mem_align_bytes=" << t.memAlignBytes
              << " offset_align_bytes=" << t.offsetAlignBytes << " size_bytes=" << t.sizeBytes
              << " max_hw_bytes=" << t.maxHwBytes << " max_hw_dmabuf_bytes=" << t.maxHwDmabufBytes
              << " max_io_bytes=" << maxIoBytes_ << " num_tasks=" << numTasks_
              << " buffer_mode=" << bufferModeName(bufferMode_);
}

nixlIoUringEngine::~nixlIoUringEngine() {
    // The progress engine drains in-flight requests before the queue goes
    // away; completions reaped during that drain still route via enginePtr_.
    engine_.reset();
    enginePtr_ = nullptr;
    if (queue_) {
        queue_->close();
    }
}

nixl_status_t
nixlIoUringEngine::parseInitParams(const nixl_b_params_t *custom_params) {
    if (!custom_params) {
        return NIXL_SUCCESS;
    }

    unsigned long long value = 0;

    if (auto it = custom_params->find("device_name"); it != custom_params->end()) {
        deviceName_ = it->second;
    }

    if (auto it = custom_params->find("direct"); it != custom_params->end()) {
        if (!parseBool(it->second, direct_)) {
            NIXL_ERROR << "IO_URING: direct must be 'true' or 'false'";
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    if (auto it = custom_params->find("num_tasks"); it != custom_params->end()) {
        if (!parseUnsigned(it->second, 2048, value) || value == 0 || (value & (value - 1)) != 0) {
            NIXL_ERROR << "IO_URING: num_tasks must be a power of two in [1, 2048], got '"
                       << it->second << "'";
            return NIXL_ERR_INVALID_PARAM;
        }
        numTasks_ = static_cast<uint32_t>(value);
    }

    if (auto it = custom_params->find("max_io_bytes"); it != custom_params->end()) {
        if (!parseUnsigned(it->second, UINT32_MAX, value)) {
            NIXL_ERROR << "IO_URING: invalid max_io_bytes '" << it->second << "'";
            return NIXL_ERR_INVALID_PARAM;
        }
        userMaxIoBytes_ = static_cast<uint32_t>(value);
    }

    if (auto it = custom_params->find("max_io_bytes_force"); it != custom_params->end()) {
        if (!parseUnsigned(it->second, UINT32_MAX, value)) {
            NIXL_ERROR << "IO_URING: invalid max_io_bytes_force '" << it->second << "'";
            return NIXL_ERR_INVALID_PARAM;
        }
        forceMaxIoBytes_ = static_cast<uint32_t>(value);
    }

    if (auto it = custom_params->find("max_dmabuf_bytes"); it != custom_params->end()) {
        if (!parseUnsigned(it->second, UINT64_MAX, value) || value == 0) {
            NIXL_ERROR << "IO_URING: invalid max_dmabuf_bytes '" << it->second << "'";
            return NIXL_ERR_INVALID_PARAM;
        }
        maxDmabufBytes_ = value;
    }

    if (auto it = custom_params->find("expect_serial"); it != custom_params->end()) {
        expectSerial_ = it->second;
    }
    if (auto it = custom_params->find("expect_wwid"); it != custom_params->end()) {
        expectWwid_ = it->second;
    }

    if (auto it = custom_params->find("buffer_mode"); it != custom_params->end()) {
        if (it->second == "host" || it->second.empty()) {
            bufferMode_ = nixlIoUringBufferMode::host;
        } else if (it->second == "fixed") {
            bufferMode_ = nixlIoUringBufferMode::fixed;
        } else if (it->second == "dmabuf") {
            bufferMode_ = nixlIoUringBufferMode::dmabuf;
        } else {
            NIXL_ERROR << "IO_URING: buffer_mode must be host, fixed or dmabuf; got '" << it->second
                       << "'";
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlIoUringEngine::initQueue() {
    nixlIoUringOpenOpts opts;
    opts.direct = direct_;
    opts.numTasks = numTasks_;

    nixl_status_t status = queue_->open(deviceName_, opts, [this](void *cookie, int32_t res) {
        if (enginePtr_) {
            enginePtr_->onCompletion(cookie, res);
        }
    });
    if (status != NIXL_SUCCESS) {
        return status;
    }

    const nixlIoUringTarget &t = queue_->target();
    if (t.sizeBytes == 0 || t.logicalBlockBytes == 0) {
        NIXL_ERROR << "IO_URING: target reports size " << t.sizeBytes << " bytes and logical block "
                   << t.logicalBlockBytes << " bytes; unusable";
        queue_->close();
        return NIXL_ERR_BACKEND;
    }
    // Identity is checked before a single request can be issued: a device name
    // is ephemeral and is not evidence of which drive was opened.
    if (!expectSerial_.empty() && t.serial != expectSerial_) {
        NIXL_ERROR << "IO_URING: serial '" << t.serial << "' does not match expect_serial '"
                   << expectSerial_ << "'";
        queue_->close();
        return NIXL_ERR_INVALID_PARAM;
    }
    if (!expectWwid_.empty() && t.wwid != expectWwid_) {
        NIXL_ERROR << "IO_URING: wwid '" << t.wwid << "' does not match expect_wwid '"
                   << expectWwid_ << "'";
        queue_->close();
        return NIXL_ERR_INVALID_PARAM;
    }
    if (expectSerial_.empty() && expectWwid_.empty()) {
        NIXL_WARN << "IO_URING: no expect_serial/expect_wwid given; target "
                     "identity unvalidated";
    }

    maxIoBytes_ = computeMaxIoBytes(t, bufferMode_, userMaxIoBytes_);
    if (maxIoBytes_ == 0) {
        NIXL_ERROR << "IO_URING: max_io_bytes is smaller than the target's "
                      "direct-I/O alignment";
        queue_->close();
        return NIXL_ERR_INVALID_PARAM;
    }
    if (forceMaxIoBytes_) {
        // Every other term in the ceiling is a limit the kernel or the device
        // enforces at its own end. A caller measuring where that limit bites,
        // or one whose payload is contiguous enough not to need it, may ask
        // for a larger request and see what comes back.
        const uint32_t alignment = lengthAlignment(t);
        const uint32_t forced = forceMaxIoBytes_ - forceMaxIoBytes_ % alignment;
        if (forced == 0) {
            NIXL_ERROR << "IO_URING: max_io_bytes_force=" << forceMaxIoBytes_ << " is below the "
                       << alignment << "-byte direct-I/O alignment";
            queue_->close();
            return NIXL_ERR_INVALID_PARAM;
        }
        if (forced > maxIoBytes_) {
            NIXL_WARN << "IO_URING: max_io_bytes_force=" << forced
                      << " overrides the computed ceiling of " << maxIoBytes_
                      << "; the kernel or device may refuse such requests";
        }
        maxIoBytes_ = forced;
    }
    return NIXL_SUCCESS;
}

const nixlIoUringTarget &
nixlIoUringEngine::target() const {
    return queue_->target();
}

nixlIoUringStats
nixlIoUringEngine::stats() const {
    return engine_->stats();
}

nixl_status_t
nixlIoUringEngine::registerMem(const nixlBlobDesc &mem,
                               const nixl_mem_t &nixl_mem,
                               nixlBackendMD *&out) {
    out = nullptr;

    const bool vram_dmabuf = nixl_mem == VRAM_SEG && bufferMode_ == nixlIoUringBufferMode::dmabuf;
    if (nixl_mem == DRAM_SEG || vram_dmabuf) {
        if (bufferMode_ == nixlIoUringBufferMode::host) {
            // A host buffer reaches the kernel as a pointer on every request;
            // there is nothing to pin or map here.
            return NIXL_SUCCESS;
        }
        if (mem.len == 0) {
            NIXL_ERROR << "IO_URING: a registered buffer needs a non-zero length";
            return NIXL_ERR_INVALID_PARAM;
        }
        const nixlIoUringTarget &t = queue_->target();
        const uint32_t alignment = t.memAlignBytes ? t.memAlignBytes : t.logicalBlockBytes;
        if (mem.addr % alignment != 0 || mem.len % alignment != 0) {
            NIXL_ERROR << "IO_URING: registered buffer addr=" << std::hex << mem.addr
                       << " len=" << mem.len << std::dec << " is not aligned to " << alignment
                       << " bytes";
            return NIXL_ERR_INVALID_PARAM;
        }
        uint32_t slot = IO_URING_NO_SLOT;
        int err;
        bool is_dmabuf = false;
        if (bufferMode_ == nixlIoUringBufferMode::dmabuf) {
            int fd = -1;
            if (!parseDmabufMeta(mem.metaInfo, fd)) {
                NIXL_ERROR << "IO_URING: buffer_mode=dmabuf needs metaInfo 'dmabuf:<fd>' on "
                              "every registration";
                return NIXL_ERR_INVALID_PARAM;
            }
            if (mem.len > maxDmabufBytes_) {
                NIXL_ERROR << "IO_URING: dma-buf of " << mem.len
                           << " bytes exceeds max_dmabuf_bytes " << maxDmabufBytes_
                           << "; register it as several dma-bufs";
                return NIXL_ERR_INVALID_PARAM;
            }
            err = queue_->registerDmabuf(fd, mem.len, slot);
            is_dmabuf = true;
        } else {
            err = queue_->registerHostBuffer(reinterpret_cast<void *>(mem.addr), mem.len, slot);
        }
        if (err) {
            NIXL_ERROR << "IO_URING: buffer registration (" << (is_dmabuf ? "dmabuf" : "fixed")
                       << ", " << mem.len << " bytes) failed: " << strerror(-err);
            return NIXL_ERR_BACKEND;
        }
        out = new nixlIoUringBufferMetadata(slot, mem.addr, mem.len, is_dmabuf);
        return NIXL_SUCCESS;
    }
    if (nixl_mem != BLK_SEG) {
        NIXL_ERROR << "IO_URING: unsupported memory type " << nixl_mem;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    const nixlIoUringTarget &t = queue_->target();
    if (!mem.metaInfo.empty() && mem.metaInfo != deviceName_) {
        NIXL_ERROR << "IO_URING: BLK_SEG devId " << mem.devId << " names target '" << mem.metaInfo
                   << "' but this backend opened '" << deviceName_ << "'";
        return NIXL_ERR_INVALID_PARAM;
    }
    const uint32_t alignment = t.offsetAlignBytes ? t.offsetAlignBytes : t.logicalBlockBytes;
    if (mem.addr % alignment != 0 || mem.len % alignment != 0) {
        NIXL_ERROR << "IO_URING: BLK_SEG range offset=" << mem.addr << " len=" << mem.len
                   << " is not a multiple of the " << alignment
                   << "-byte direct-I/O offset alignment";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (mem.len == 0 && mem.addr != 0) {
        NIXL_ERROR << "IO_URING: an unbounded BLK_SEG registration must start at offset 0";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (mem.addr >= t.sizeBytes || mem.len > t.sizeBytes - mem.addr) {
        NIXL_ERROR << "IO_URING: BLK_SEG range offset=" << mem.addr << " len=" << mem.len
                   << " exceeds the target size of " << t.sizeBytes << " bytes";
        return NIXL_ERR_INVALID_PARAM;
    }

    {
        std::lock_guard<std::mutex> guard(devIdLock_);
        auto [it, inserted] = devIdPaths_.try_emplace(mem.devId, deviceName_);
        if (!inserted && it->second != deviceName_) {
            NIXL_ERROR << "IO_URING: BLK_SEG devId " << mem.devId << " is already bound to '"
                       << it->second << "' and cannot be reused for '" << deviceName_ << "'";
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    out = new nixlIoUringBlkMetadata(mem.devId, mem.addr, mem.len);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlIoUringEngine::deregisterMem(nixlBackendMD *meta) {
    if (!meta) {
        return NIXL_SUCCESS;
    }
    if (auto *buffer = dynamic_cast<nixlIoUringBufferMetadata *>(meta)) {
        // The kernel keeps a registered buffer pinned or mapped until its slot
        // is cleared; requests still referencing it were drained when their
        // own request handles were released.
        const uint32_t slot = buffer->slot;
        const int err = queue_->unregisterBuffer(slot);
        delete buffer;
        if (err) {
            // The kernel still holds the buffer, so the caller must not free
            // the memory behind it. Report the failure rather than let a caller
            // believe the registration is gone.
            NIXL_ERROR << "IO_URING: releasing slot " << slot << " failed: " << strerror(-err)
                       << "; the kernel still holds this buffer and the slot is retired";
            return NIXL_ERR_BACKEND;
        }
        return NIXL_SUCCESS;
    }
    // Registration metadata only; the bytes stay on the target.
    delete static_cast<nixlIoUringBlkMetadata *>(meta);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlIoUringEngine::queryMem(const nixl_reg_dlist_t &descs,
                            std::vector<nixl_query_resp_t> &resp) const {
    resp.clear();
    resp.reserve(descs.descCount());

    const nixlIoUringTarget &t = queue_->target();
    for (int i = 0; i < descs.descCount(); i++) {
        const nixlBlobDesc &d = descs[i];
        if (descs.getType() != BLK_SEG) {
            resp.emplace_back(nixl_b_params_t{});
            continue;
        }
        const bool inside = (d.len != 0 || d.addr == 0) && d.addr < t.sizeBytes &&
            d.len <= t.sizeBytes - d.addr && (d.metaInfo.empty() || d.metaInfo == deviceName_);
        if (!inside) {
            resp.emplace_back(std::nullopt);
            continue;
        }
        // A byte range on a block device or in a file has no existence to
        // query; the useful answer is which target it lands on and how it may
        // be addressed.
        resp.emplace_back(
            nixl_b_params_t{{"serial", t.serial},
                            {"wwid", t.wwid},
                            {"size_bytes", std::to_string(t.sizeBytes)},
                            {"logical_block_bytes", std::to_string(t.logicalBlockBytes)},
                            {"mem_align_bytes", std::to_string(t.memAlignBytes)},
                            {"offset_align_bytes", std::to_string(t.offsetAlignBytes)},
                            {"max_io_bytes", std::to_string(maxIoBytes_)}});
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlIoUringEngine::prepXfer(const nixl_xfer_op_t &operation,
                            const nixl_meta_dlist_t &local,
                            const nixl_meta_dlist_t &remote,
                            const std::string &remote_agent,
                            nixlBackendReqH *&handle,
                            const nixl_opt_b_args_t *opt_args) const {
    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        NIXL_ERROR << "IO_URING: only NIXL_WRITE and NIXL_READ are supported";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (remote_agent != localAgent) {
        NIXL_WARN << "IO_URING: remote agent (" << remote_agent << ") != local agent ("
                  << localAgent << "); IO_URING is a local storage backend";
    }
    const bool local_ok = local.getType() == DRAM_SEG ||
        (local.getType() == VRAM_SEG && bufferMode_ == nixlIoUringBufferMode::dmabuf);
    if (!local_ok) {
        NIXL_ERROR << "IO_URING: local descriptor list must be DRAM_SEG (or VRAM_SEG in "
                      "buffer_mode=dmabuf), got "
                   << local.getType();
        return NIXL_ERR_INVALID_PARAM;
    }
    if (remote.getType() != BLK_SEG) {
        NIXL_ERROR << "IO_URING: remote descriptor list must be BLK_SEG, got " << remote.getType();
        return NIXL_ERR_INVALID_PARAM;
    }
    const int desc_count = local.descCount();
    if (desc_count == 0) {
        NIXL_ERROR << "IO_URING: empty descriptor list";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (remote.descCount() != desc_count) {
        NIXL_ERROR << "IO_URING: local (" << desc_count << ") and remote (" << remote.descCount()
                   << ") descriptor counts differ";
        return NIXL_ERR_INVALID_PARAM;
    }

    const nixlIoUringTarget &t = queue_->target();
    const uint32_t lb = t.logicalBlockBytes;
    const uint32_t mem_align = t.memAlignBytes ? t.memAlignBytes : lb;
    const uint32_t offset_align = t.offsetAlignBytes ? t.offsetAlignBytes : lb;
    const uint32_t len_align = lengthAlignment(t);
    auto req_h = std::make_unique<nixlIoUringBackendReqH>();

    // First pass: validate and count requests, so the task vector is sized
    // once and never grows after its element addresses are published.
    size_t total_reqs = 0;
    for (int i = 0; i < desc_count; i++) {
        const nixlMetaDesc &l = local[i];
        const nixlMetaDesc &r = remote[i];
        if (l.len != r.len) {
            NIXL_ERROR << "IO_URING: descriptor " << i << " local len " << l.len
                       << " != remote len " << r.len;
            return NIXL_ERR_INVALID_PARAM;
        }
        if (l.len == 0) {
            NIXL_ERROR << "IO_URING: descriptor " << i << " has zero length";
            return NIXL_ERR_INVALID_PARAM;
        }
        if (l.addr % mem_align != 0 || r.addr % offset_align != 0 || l.len % len_align != 0) {
            NIXL_ERROR << "IO_URING: descriptor " << i << " (host=" << std::hex << l.addr
                       << " offset=" << r.addr << " len=" << l.len << std::dec
                       << ") violates direct-I/O alignment (memory=" << mem_align
                       << ", offset=" << offset_align << ", length=" << len_align << ")";
            return NIXL_ERR_INVALID_PARAM;
        }
        if (r.addr >= t.sizeBytes || r.len > t.sizeBytes - r.addr) {
            NIXL_ERROR << "IO_URING: descriptor " << i << " offset=" << r.addr << " len=" << r.len
                       << " exceeds the target size of " << t.sizeBytes << " bytes";
            return NIXL_ERR_INVALID_PARAM;
        }
        const auto *md = static_cast<const nixlIoUringBlkMetadata *>(r.metadataP);
        if (!md) {
            NIXL_ERROR << "IO_URING: remote descriptor " << i << " has no registration metadata";
            return NIXL_ERR_INVALID_PARAM;
        }
        if (bufferMode_ != nixlIoUringBufferMode::host) {
            const auto *bmd = static_cast<const nixlIoUringBufferMetadata *>(l.metadataP);
            if (!bmd) {
                NIXL_ERROR << "IO_URING: local descriptor " << i
                           << " is not a registered buffer (buffer_mode requires it)";
                return NIXL_ERR_INVALID_PARAM;
            }
            if (l.addr < bmd->base || l.len > bmd->len || l.addr - bmd->base > bmd->len - l.len) {
                NIXL_ERROR << "IO_URING: local descriptor " << i << " [" << std::hex << l.addr
                           << ", +" << l.len << ") lies outside its registration [" << bmd->base
                           << ", +" << bmd->len << ")" << std::dec;
                return NIXL_ERR_INVALID_PARAM;
            }
        }
        if (md->len != 0 &&
            (r.len > md->len || r.addr < md->offset || r.addr - md->offset > md->len - r.len)) {
            NIXL_ERROR << "IO_URING: descriptor " << i << " offset=" << r.addr << " len=" << r.len
                       << " lies outside its registration [" << md->offset << ", "
                       << md->offset + md->len << ")";
            return NIXL_ERR_INVALID_PARAM;
        }
        total_reqs += (l.len + maxIoBytes_ - 1) / maxIoBytes_;
    }
    if (total_reqs > UINT32_MAX) {
        NIXL_ERROR << "IO_URING: transfer needs " << total_reqs << " requests";
        return NIXL_ERR_INVALID_PARAM;
    }

    req_h->tasks_.reserve(total_reqs);
    for (int i = 0; i < desc_count; i++) {
        const nixlMetaDesc &l = local[i];
        const nixlMetaDesc &r = remote[i];
        uint32_t child = 0;
        for (size_t done = 0; done < l.len; done += maxIoBytes_, child++) {
            const size_t bytes = std::min<size_t>(maxIoBytes_, l.len - done);
            nixlIoUringTask task;
            task.reqH = req_h.get();
            task.descIndex = static_cast<uint32_t>(i);
            task.childIndex = child;
            task.bytes = static_cast<uint32_t>(bytes);
            task.offset = r.addr + done;
            task.buf = reinterpret_cast<void *>(l.addr + done);
            if (bufferMode_ != nixlIoUringBufferMode::host) {
                const auto *bmd = static_cast<const nixlIoUringBufferMetadata *>(l.metadataP);
                task.slot = bmd->slot;
                if (bmd->isDmabuf) {
                    // A dma-buf has no address the caller can pass: the kernel
                    // knows the mapping, and the request names a byte offset
                    // into it.
                    task.buf = reinterpret_cast<void *>(l.addr - bmd->base + done);
                }
            }
            req_h->tasks_.push_back(task);
        }
    }

    handle = req_h.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlIoUringEngine::postXfer(const nixl_xfer_op_t &operation,
                            const nixl_meta_dlist_t &local,
                            const nixl_meta_dlist_t &remote,
                            const std::string &remote_agent,
                            nixlBackendReqH *&handle,
                            const nixl_opt_b_args_t *opt_args) const {
    if (!handle) {
        NIXL_ERROR << "IO_URING: postXfer on null handle";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        NIXL_ERROR << "IO_URING: unsupported operation";
        return NIXL_ERR_INVALID_PARAM;
    }
    return engine_->postXfer(static_cast<nixlIoUringBackendReqH *>(handle), operation);
}

nixl_status_t
nixlIoUringEngine::checkXfer(nixlBackendReqH *handle) const {
    return engine_->checkXfer(static_cast<nixlIoUringBackendReqH *>(handle));
}

nixl_status_t
nixlIoUringEngine::releaseReqH(nixlBackendReqH *handle) const {
    if (!handle) {
        NIXL_ERROR << "IO_URING: releaseReqH on null handle";
        return NIXL_ERR_INVALID_PARAM;
    }
    engine_->cancelRequest(static_cast<nixlIoUringBackendReqH *>(handle));
    return NIXL_SUCCESS;
}
