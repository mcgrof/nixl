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

// sysfs helpers shared by both device implementations. They live apart
// from xnvme_nvm_dev.cpp so the liburing device, which needs no xNVMe
// library, and the CPU-only unit tests can use them without linking one.

#include "xnvme_nvm_dev.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

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
//
// The caller may name the device through any of the stable aliases the
// kernel keeps in /dev/disk/by-id, by-uuid or by-path, and operators are
// usually told to prefer them because the nvmeXnY names are not stable
// across boots. Those basenames spell the model and serial rather than
// the kernel name sysfs is keyed by, so resolve the link before parsing
// it. Skipping this does not fail loudly: the lookup simply misses and
// the device keeps its default geometry, silently losing MDTS, the
// dma-buf ceiling, and the serial used to confirm which drive was opened.
bool
sysfsNamesFor(const std::string &uri, std::string &ctrl, std::string &ns) {
    std::string path = uri;
    char resolved[PATH_MAX];
    if (::realpath(uri.c_str(), resolved)) {
        path = resolved;
    }
    const size_t slash = path.find_last_of('/');
    const std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
    unsigned c = 0, n = 0, part = 0;
    char tail = 0;
    if (sscanf(base.c_str(), "ng%un%u%c", &c, &n, &tail) == 2 ||
        sscanf(base.c_str(), "nvme%un%u%c", &c, &n, &tail) == 2 ||
        /*
         * A partition carries no queue limits and no identity of its own:
         * both live on the namespace it was cut from, and that is what
         * bounds a request made through the partition.  Map nvmeXnYpZ onto
         * nvmeXnY rather than refusing it, or every partitioned device
         * silently loses MDTS and the dma-buf ceiling.
         */
        sscanf(base.c_str(), "nvme%un%up%u%c", &c, &n, &part, &tail) == 3) {
        ctrl = "nvme" + std::to_string(c);
        ns = ctrl + "n" + std::to_string(n);
        return true;
    }
    return false;
}

} // namespace nixl_xnvme_nvm
