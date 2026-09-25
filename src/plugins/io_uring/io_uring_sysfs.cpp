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

// Queue limits and device identity, read through /sys/dev/block. Keying the
// lookup on the device number rather than on the path the caller used means a
// stable alias from /dev/disk/by-id, a device-mapper name and a raw kernel
// name all land on the same disk, and it needs no per-subsystem knowledge of
// how device names are spelled.

#include "io_uring_queue.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include <sys/sysmacros.h>
#include <unistd.h>

namespace nixl_io_uring {

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

// A decimal sysfs attribute, or 0 when it is absent or does not parse whole.
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

std::string
diskSysfsDir(dev_t dev) {
    char link[64];
    snprintf(link, sizeof(link), "/sys/dev/block/%u:%u", major(dev), minor(dev));
    char resolved[PATH_MAX];
    if (!::realpath(link, resolved)) {
        return "";
    }

    // A partition holds neither queue limits nor identity of its own: both
    // belong to the disk it was cut from, which is also what bounds a request
    // issued through the partition. sysfs makes that disk the partition's
    // parent directory, and marks a partition by publishing this attribute.
    std::string dir = resolved;
    if (::access((dir + "/partition").c_str(), F_OK) == 0) {
        const size_t slash = dir.find_last_of('/');
        if (slash != std::string::npos && slash != 0) {
            dir.resize(slash);
        }
    }
    return dir;
}

void
readDiskLimits(const std::string &disk_dir, nixlIoUringTarget &target) {
    if (disk_dir.empty()) {
        return;
    }
    const std::string queue = disk_dir + "/queue/";

    // Guard every limit against an implausible value so a malformed attribute
    // cannot widen a request beyond what the kernel would accept.
    const unsigned long lbs = readSysfsNumber(queue + "logical_block_size");
    if (lbs && lbs <= (1ul << 20)) {
        target.logicalBlockBytes = static_cast<uint32_t>(lbs);
    }
    const unsigned long hw_kb = readSysfsNumber(queue + "max_hw_sectors_kb");
    if (hw_kb && hw_kb < (1ul << 22)) {
        target.maxHwBytes = static_cast<uint32_t>(hw_kb) * 1024u;
    }
    const unsigned long dmabuf_kb = readSysfsNumber(queue + "max_hw_dmabuf_sectors_kb");
    if (dmabuf_kb && dmabuf_kb < (1ul << 22)) {
        target.maxHwDmabufBytes = static_cast<uint32_t>(dmabuf_kb) * 1024u;
    }

    // Identity lives partly on the disk and partly on the driver's device
    // below it: NVMe publishes the serial and firmware on the controller,
    // SCSI publishes a model and revision, and neither is guaranteed.
    target.wwid = readSysfs(disk_dir + "/wwid");
    const std::string device = disk_dir + "/device/";
    target.serial = readSysfs(device + "serial");
    target.model = readSysfs(device + "model");
    target.firmware = readSysfs(device + "firmware_rev");
    if (target.firmware.empty()) {
        target.firmware = readSysfs(device + "rev");
    }
}

} // namespace nixl_io_uring
