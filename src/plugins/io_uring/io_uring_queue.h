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
#ifndef IO_URING_QUEUE_H
#define IO_URING_QUEUE_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <sys/types.h>

#include "nixl_types.h"

// A request that names no registered buffer and carries a plain pointer.
inline constexpr uint32_t IO_URING_NO_SLOT = UINT32_MAX;

struct nixlIoUringOpenOpts {
    // O_DIRECT. A registered buffer is only useful with it: a buffered write
    // would copy the payload into the page cache and lose the point of
    // registering it.
    bool direct = true;
    // Ring capacity; a power of two in [1, 2048].
    uint32_t numTasks = 128;
};

// What the backend needs to know about the opened target. A limit that no
// source reports stays 0 and is never guessed at.
struct nixlIoUringTarget {
    enum class Kind { blockDevice, regularFile };

    Kind kind = Kind::blockDevice;
    uint64_t sizeBytes = 0;
    uint32_t logicalBlockBytes = 0;
    // queue/max_hw_sectors_kb in bytes: the largest request the driver
    // accepts without the block layer splitting it.
    uint32_t maxHwBytes = 0;
    // Direct-I/O alignment: what statx reports for a file, else the logical
    // block size.
    uint32_t memAlignBytes = 0;
    uint32_t offsetAlignBytes = 0;
    // Identity as sysfs spells it, for the expect_serial and expect_wwid
    // guards. Empty wherever the kernel publishes none.
    std::string serial;
    std::string model;
    std::string firmware;
    std::string wwid;

    const char *
    kindName() const {
        return kind == Kind::blockDevice ? "block_device" : "regular_file";
    }
};

// Fired from inside poke() for every reaped completion, with the cookie the
// request was submitted with. res is the completion queue entry's result:
// the byte count transferred, or a negative errno.
using nixlIoUringCplCb = std::function<void(void *cookie, int32_t res)>;

// One open target and its ring. The backend and the progress engine reach
// the kernel only through this, so the unit tests can drive both against an
// in-memory queue with no ring and no device.
class nixlIoUringQueue {
public:
    virtual ~nixlIoUringQueue() = default;

    virtual nixl_status_t
    open(const std::string &path, const nixlIoUringOpenOpts &opts, nixlIoUringCplCb cpl_cb) = 0;

    virtual void
    close() = 0;

    virtual const nixlIoUringTarget &
    target() const = 0;

    // Registered buffers. Registration hands back a slot index, and a
    // request then names the slot and the virtual address inside the
    // registered range. A kernel that cannot register the buffer says so
    // here, at registration time, rather than failing the first transfer.
    virtual int
    registerHostBuffer(void *addr, size_t len, uint32_t &slot) = 0;

    virtual int
    unregisterBuffer(uint32_t slot) = 0;

    // Queue one read or write of bytes at offset. slot is IO_URING_NO_SLOT
    // for a plain pointer. Returns 0 once queued, -EAGAIN while the ring is
    // full so the caller must poke() and retry, or another negative errno
    // for a permanent failure.
    virtual int
    submit(bool write, uint64_t offset, uint32_t bytes, void *buf, uint32_t slot, void *cookie) = 0;

    // Hand queued submissions to the kernel and reap completions, firing
    // cpl_cb inline for each. Submission is deferred to here, so a request
    // makes no progress until someone pokes even when nothing has completed.
    // Returns the number of completions processed, or a negative errno when
    // there were none and submission failed.
    virtual int
    poke() = 0;

    virtual uint32_t
    outstanding() const = 0;
};

// Queue limits and identity read out of /sys/dev/block, so a whole disk, one
// of its partitions and a file on a filesystem mounted from it all resolve
// to the same limits. Their own translation unit (io_uring_sysfs.cpp) so the
// CPU-only unit tests get them without liburing.
namespace nixl_io_uring {
std::string
readSysfs(const std::string &path);

unsigned long
readSysfsNumber(const std::string &path);

// The sysfs directory of the whole disk behind dev, which is st_rdev for a
// block device and st_dev for a file. Empty when the kernel has no block
// device for it, as for tmpfs or a network filesystem.
std::string
diskSysfsDir(dev_t dev);

// Fill in every limit and identity field the disk publishes, leaving the
// rest untouched.
void
readDiskLimits(const std::string &disk_dir, nixlIoUringTarget &target);
} // namespace nixl_io_uring

// The liburing-backed queue (io_uring_queue.cpp).
std::unique_ptr<nixlIoUringQueue>
nixlIoUringQueueCreate();

#endif // IO_URING_QUEUE_H
