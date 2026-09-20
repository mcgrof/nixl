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

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <string>

#include <unistd.h>

#include <gtest/gtest.h>

#include "xnvme_nvm_dev.h"

namespace {

class TempFile {
public:
    TempFile() {
        char path[] = "/tmp/nixl-xnvme-nvm-XXXXXX";
        const int fd = mkstemp(path);
        if (fd >= 0) {
            path_ = path;
            if (ftruncate(fd, kSize) != 0) {
                unlink(path_.c_str());
                path_.clear();
            }
            close(fd);
        }
    }

    ~TempFile() {
        if (!path_.empty()) {
            unlink(path_.c_str());
        }
    }

    const std::string &
    path() const {
        return path_;
    }

    static constexpr size_t kSize = 64 * 1024;

private:
    std::string path_;
};

TEST(XnvmeNvmUringDev, CapacityCountsEachQueuedSqeOnce) {
    TempFile file;
    ASSERT_FALSE(file.path().empty());

    auto dev = nixlXnvmeNvmUringDevCreate();
    nixlXnvmeNvmOpenOpts opts;
    opts.async = "io_uring";
    opts.direct = false;
    opts.numTasks = 4;

    uint32_t completions = 0;
    bool completionError = false;
    const nixl_status_t status =
        dev->open(file.path(), opts, [&](void *, const nixlXnvmeNvmCpl &cpl) {
            completions++;
            completionError |= !cpl.ok() || cpl.result != 4096;
        });
    if (status != NIXL_SUCCESS) {
        GTEST_SKIP() << "io_uring unavailable in this test environment";
    }

    std::array<std::array<uint8_t, 4096>, 5> buffers{};
    for (size_t i = 0; i < buffers.size(); i++) {
        memset(buffers[i].data(), static_cast<int>(i + 1), buffers[i].size());
    }

    for (uint64_t i = 0; i < opts.numTasks; i++) {
        EXPECT_EQ(
            dev->submitWrite(i * 8, 7, buffers[i].data(), XNVME_NVM_NO_SLOT, buffers[i].data()), 0);
    }
    EXPECT_EQ(dev->outstanding(), opts.numTasks);
    EXPECT_EQ(
        dev->submitWrite(32, 7, buffers.back().data(), XNVME_NVM_NO_SLOT, buffers.back().data()),
        -EAGAIN);

    for (int i = 0; dev->outstanding() != 0 && i < 10000; i++) {
        ASSERT_GE(dev->poke(), 0);
    }
    EXPECT_EQ(dev->outstanding(), 0u);
    EXPECT_EQ(completions, opts.numTasks);
    EXPECT_FALSE(completionError);
}

} // namespace
