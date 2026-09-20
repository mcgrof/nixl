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
#ifndef XNVME_NVM_DEV_H
#define XNVME_NVM_DEV_H

#include <cerrno>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "nixl_types.h"

// A command that names no registered buffer.
inline constexpr uint32_t XNVME_NVM_NO_SLOT = UINT32_MAX;

// Narrow seam over one xNVMe device plus its async queue for conventional
// NVM reads and writes. The backend and progress engine only talk to this
// interface; xnvme_nvm_dev.cpp is the sole translation unit that includes
// libxnvme.h, and the unit tests substitute an in-memory namespace.

struct nixlXnvmeNvmOpenOpts {
    // xNVMe asynchronous command interface: io_uring_cmd (NVMe passthrough
    // on /dev/ngXnY), io_uring or libaio (ordinary block I/O on a block
    // device or file), posix or thrpool (emulated, development only).
    std::string async = "io_uring_cmd";
    // O_DIRECT for the block/file paths; ignored by passthrough.
    bool direct = true;
    // 0 = use the namespace the URI resolves to; otherwise an expectation.
    uint32_t nsidOverride = 0;
    // Queue capacity; power of two in [1, 2048].
    uint32_t numTasks = 128;
};

// Everything the backend needs to know about the opened device. Byte
// ceilings that a source does not report are left 0, never guessed.
struct nixlXnvmeNvmGeo {
    std::string devType; // nvme_namespace, block_device, fs_file, nvme_controller, unknown
    uint32_t nsid = 0;
    uint32_t lbaBytes = 0;
    uint64_t capacityBytes = 0;
    uint32_t xnvmeMdtsBytes = 0; // xNVMe's own geo.mdts_nbytes, for the record
    uint32_t sysfsMaxHwBytes = 0; // queue/max_hw_sectors_kb * 1024; 0 = unknown
    uint32_t sysfsMaxSegments = 0; // queue/max_segments; 0 = unknown
    uint32_t sysfsMaxHwDmabufBytes = 0; // queue/max_hw_dmabuf_sectors_kb * 1024; 0 = absent
    uint32_t pageBytes = 0; // host page size, the unit of a scattered user buffer
    uint32_t memAlignBytes = 0; // direct-I/O buffer alignment; 0 = lbaBytes
    uint32_t offsetAlignBytes = 0; // direct-I/O offset alignment; 0 = lbaBytes
    std::string serial;
    std::string model;
    std::string firmware;
    std::string wwid;
};

// xNVMe reports a kernel errno (a negative cqe result) as status code
// type 7 with the errno's low byte in sc.
inline constexpr uint8_t XNVME_NVM_SCT_ERRNO = 0x7;

// Decoded completion of one command. For the ordinary block paths a
// successful command reports the transferred byte count in result; for
// passthrough result is completion dword 0 and success is never short.
struct nixlXnvmeNvmCpl {
    uint8_t sc = 0;
    uint8_t sct = 0;
    uint64_t result = 0;

    bool
    ok() const {
        return sc == 0 && sct == 0;
    }

    // The errno behind a status-code-type-7 completion, else 0.
    int
    kernelErrno() const {
        return sct == XNVME_NVM_SCT_ERRNO ? static_cast<int>(static_cast<int8_t>(sc)) : 0;
    }
};

// Fired from inside poke() for every reaped completion with the cookie
// passed at submission time.
using nixlXnvmeNvmCplCb = std::function<void(void *cookie, const nixlXnvmeNvmCpl &cpl)>;

class nixlXnvmeNvmDevIface {
public:
    virtual ~nixlXnvmeNvmDevIface() = default;

    virtual nixl_status_t
    open(const std::string &uri, const nixlXnvmeNvmOpenOpts &opts, nixlXnvmeNvmCplCb cpl_cb) = 0;

    virtual void
    close() = 0;

    virtual const nixlXnvmeNvmGeo &
    geo() const = 0;

    // True when a successful completion's result is the byte count actually
    // transferred, so the caller can detect a short operation.
    virtual bool
    completionReportsBytes() const = 0;

    // Registered buffers. A device that supports them hands out slot
    // indices; a command then names the slot and, in buf, either the
    // virtual address inside a registered host buffer or the byte offset
    // inside a registered dma-buf. The xNVMe device supports neither.
    virtual bool
    supportsRegistration() const {
        return false;
    }

    virtual int
    registerHostBuffer(void *addr, size_t len, uint32_t &slot) {
        return -ENOTSUP;
    }

    virtual int
    registerDmabuf(int fd, size_t len, uint32_t &slot) {
        return -ENOTSUP;
    }

    virtual int
    unregisterBuffer(uint32_t slot) {
        return -ENOTSUP;
    }

    // Submit one read or write of nlb + 1 logical blocks at slba. slot is
    // XNVME_NVM_NO_SLOT for a plain host pointer. Returns 0 on successful
    // submission, -EAGAIN when the queue is saturated (no command context,
    // SQ full, or in-flight == capacity) so the caller must poke() and
    // retry, or another negative errno on permanent failure.
    virtual int
    submitRead(uint64_t slba, uint16_t nlb, void *buf, uint32_t slot, void *cookie) = 0;

    virtual int
    submitWrite(uint64_t slba, uint16_t nlb, const void *buf, uint32_t slot, void *cookie) = 0;

    // Reap completions, firing cpl_cb inline for each. xNVMe defers ring
    // submission to poke by default, so progress requires poking even when
    // nothing has completed. Returns the number of completions processed or
    // a negative errno.
    virtual int
    poke() = 0;

    virtual uint32_t
    outstanding() const = 0;
};

// sysfs helpers shared by both device implementations (xnvme_nvm_dev.cpp).
namespace nixl_xnvme_nvm {
std::string
readSysfs(const std::string &path);
unsigned long
readSysfsNumber(const std::string &path);
bool
sysfsNamesFor(const std::string &uri, std::string &ctrl, std::string &ns);
} // namespace nixl_xnvme_nvm

// Real xNVMe-backed implementation (xnvme_nvm_dev.cpp).
std::unique_ptr<nixlXnvmeNvmDevIface>
nixlXnvmeNvmDevCreate();

// liburing implementation for registered host and dma-buf buffers on a
// block device or file (xnvme_nvm_uring_dev.cpp).
std::unique_ptr<nixlXnvmeNvmDevIface>
nixlXnvmeNvmUringDevCreate();

#endif // XNVME_NVM_DEV_H
