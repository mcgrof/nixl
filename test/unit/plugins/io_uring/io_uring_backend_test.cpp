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

// CPU-only unit tests for the IO_URING backend. The real backend and
// progress-engine translation units are linked against an in-memory queue, so
// these need no block device, no liburing and no io_uring support in the
// running kernel. The sysfs tests are the exception and say so where they run.

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include "io_uring_backend.h"
#include "io_uring_mock_queue.h"

// These tests always inject a mock queue, so io_uring_queue.cpp is not linked
// and liburing is not needed; satisfy the backend's fallback reference with a
// stub that never returns one.
std::unique_ptr<nixlIoUringQueue>
nixlIoUringQueueCreate() {
    return nullptr;
}

namespace {

constexpr const char *kAgent = "IoUringUnitTester";
constexpr uint32_t kBlock = 512;

nixl_b_params_t
defaultParams() {
    return {{"device_name", "/dev/mock0"}, {"num_tasks", "8"}};
}

class EngineMaker {
public:
    explicit EngineMaker(nixl_b_params_t params = defaultParams(), bool prog_thread = false)
        : params_(std::move(params)) {
        init_.localAgent = kAgent;
        init_.type = "IO_URING";
        init_.customParams = &params_;
        init_.enableProgTh = prog_thread;
        init_.pthrDelay = 50;
        init_.enableTelemetry_ = false;
    }

    // The mock stays reachable through this raw pointer while the engine owns
    // the unique_ptr, and is valid for the engine's lifetime.
    nixlIoUringMockQueue *mock = nullptr;

    std::unique_ptr<nixlIoUringEngine>
    make() {
        auto queue = std::make_unique<nixlIoUringMockQueue>();
        mock = queue.get();
        if (configureMock) {
            configureMock(*mock);
        }
        return std::make_unique<nixlIoUringEngine>(&init_, std::move(queue));
    }

    std::function<void(nixlIoUringMockQueue &)> configureMock;

private:
    nixl_b_params_t params_;
    nixlBackendInitParams init_;
};

// A host buffer aligned well past any alignment these tests configure.
struct AlignedBuf {
    uint8_t *data = nullptr;
    size_t size = 0;

    explicit AlignedBuf(size_t n) : size(n) {
        if (posix_memalign(reinterpret_cast<void **>(&data), 4096, n) != 0) {
            data = nullptr;
        }
    }

    ~AlignedBuf() {
        free(data);
    }

    AlignedBuf(const AlignedBuf &) = delete;
    AlignedBuf &
    operator=(const AlignedBuf &) = delete;
};

// Registered DRAM and BLK descriptors backed by real buffers, for driving
// prepXfer/postXfer/checkXfer at the backend-engine level. Object i lives at
// target offset base + i * stride.
class XferHarness {
public:
    XferHarness(nixlIoUringEngine &engine,
                int count,
                size_t len,
                uint64_t base = 0,
                size_t stride = 0)
        : local_(DRAM_SEG),
          remote_(BLK_SEG),
          engine_(engine) {
        if (stride == 0) {
            stride = len;
        }
        for (int i = 0; i < count; i++) {
            bufs_.push_back(std::make_unique<AlignedBuf>(len));
            EXPECT_NE(bufs_[i]->data, nullptr);
            fill(*bufs_[i], i);

            const uint64_t offset = base + static_cast<uint64_t>(i) * stride;
            nixlBlobDesc blk_desc(offset, len, static_cast<uint64_t>(i), "");
            nixlBackendMD *md = nullptr;
            EXPECT_EQ(engine_.registerMem(blk_desc, BLK_SEG, md), NIXL_SUCCESS);
            EXPECT_NE(md, nullptr);
            mds_.push_back(md);

            nixlMetaDesc local_desc(reinterpret_cast<uintptr_t>(bufs_[i]->data), len, 0);
            local_desc.metadataP = nullptr;
            local_.addDesc(local_desc);

            nixlMetaDesc remote_desc(offset, len, static_cast<uint64_t>(i));
            remote_desc.metadataP = md;
            remote_.addDesc(remote_desc);
        }
    }

    ~XferHarness() {
        for (auto *md : mds_) {
            engine_.deregisterMem(md);
        }
    }

    static uint8_t
    patternByte(int idx, size_t j) {
        return static_cast<uint8_t>(idx * 7 + j * 13 + 1);
    }

    static void
    fill(AlignedBuf &buf, int idx) {
        for (size_t j = 0; j < buf.size; j++) {
            buf.data[j] = patternByte(idx, j);
        }
    }

    void
    poison() {
        for (auto &b : bufs_) {
            memset(b->data, 0xEE, b->size);
        }
    }

    bool
    allMatchPattern() const {
        for (size_t i = 0; i < bufs_.size(); i++) {
            for (size_t j = 0; j < bufs_[i]->size; j++) {
                if (bufs_[i]->data[j] != patternByte(static_cast<int>(i), j)) {
                    return false;
                }
            }
        }
        return true;
    }

    // Drive a whole transfer through the engine; returns the final status.
    nixl_status_t
    run(nixl_xfer_op_t op, int max_iters = 100000) {
        nixlBackendReqH *handle = nullptr;
        nixl_status_t status = engine_.prepXfer(op, local_, remote_, kAgent, handle);
        if (status != NIXL_SUCCESS) {
            return status;
        }
        status = engine_.postXfer(op, local_, remote_, kAgent, handle);
        int iters = 0;
        while (status == NIXL_IN_PROG && iters++ < max_iters) {
            status = engine_.checkXfer(handle);
            std::this_thread::yield();
        }
        engine_.releaseReqH(handle);
        return status;
    }

    std::vector<std::unique_ptr<AlignedBuf>> bufs_;
    nixl_meta_dlist_t local_;
    nixl_meta_dlist_t remote_;

private:
    nixlIoUringEngine &engine_;
    std::vector<nixlBackendMD *> mds_;
};

// Any partition sysfs currently publishes, as (partition dev_t, parent name).
bool
findAnyPartition(dev_t &part, std::string &parent_name) {
    DIR *d = opendir("/sys/class/block");
    if (!d) {
        return false;
    }
    bool found = false;
    while (struct dirent *e = readdir(d)) {
        if (e->d_name[0] == '.') {
            continue;
        }
        const std::string dir = std::string("/sys/class/block/") + e->d_name;
        if (::access((dir + "/partition").c_str(), F_OK) != 0) {
            continue;
        }
        const std::string dev = nixl_io_uring::readSysfs(dir + "/dev");
        unsigned maj = 0, min = 0;
        if (sscanf(dev.c_str(), "%u:%u", &maj, &min) != 2) {
            continue;
        }
        // The parent of /sys/class/block/<part> is reachable through the link
        // target; its basename is the disk's kernel name.
        char resolved[PATH_MAX];
        if (!::realpath(dir.c_str(), resolved)) {
            continue;
        }
        std::string path = resolved;
        const size_t slash = path.find_last_of('/');
        if (slash == std::string::npos || slash == 0) {
            continue;
        }
        path.resize(slash);
        parent_name = path.substr(path.find_last_of('/') + 1);
        part = makedev(maj, min);
        found = true;
        break;
    }
    closedir(d);
    return found;
}

// --- sysfs resolution ------------------------------------------------------

// Keying the lookup on a device number rather than on a device name is what
// lets a file on a mounted filesystem pick up the queue limits of the disk
// underneath it. A name-shaped lookup finds nothing for a file, and silently
// leaves the request-size ceiling unknown.
TEST(IoUringSysfs, AFilesDiskResolvesToItsQueueLimits) {
    struct stat st;
    ASSERT_EQ(stat(".", &st), 0);
    const std::string dir = nixl_io_uring::diskSysfsDir(st.st_dev);
    if (dir.empty()) {
        GTEST_SKIP() << "the working directory has no backing block device";
    }
    EXPECT_EQ(::access((dir + "/queue").c_str(), F_OK), 0) << dir << " has no queue directory";
    EXPECT_NE(::access((dir + "/partition").c_str(), F_OK), 0)
        << dir << " is a partition, not the whole disk";

    nixlIoUringTarget target;
    nixl_io_uring::readDiskLimits(dir, target);
    EXPECT_GT(target.logicalBlockBytes, 0u);
}

// A partition carries neither queue limits nor identity of its own; both
// belong to the disk it was cut from, and that disk is what bounds a request
// issued through the partition.
TEST(IoUringSysfs, APartitionResolvesToItsWholeDisk) {
    dev_t part = 0;
    std::string parent;
    if (!findAnyPartition(part, parent)) {
        GTEST_SKIP() << "no partitioned block device is present";
    }
    const std::string dir = nixl_io_uring::diskSysfsDir(part);
    ASSERT_FALSE(dir.empty());
    EXPECT_EQ(dir.substr(dir.find_last_of('/') + 1), parent);
    EXPECT_EQ(::access((dir + "/queue").c_str(), F_OK), 0);
}

TEST(IoUringSysfs, AnUnknownDeviceNumberYieldsNothing) {
    EXPECT_EQ(nixl_io_uring::diskSysfsDir(makedev(4095, 4095)), "");
    nixlIoUringTarget target;
    nixl_io_uring::readDiskLimits("", target);
    EXPECT_EQ(target.logicalBlockBytes, 0u);
    EXPECT_EQ(target.maxHwBytes, 0u);
}

// --- the request-size ceiling ---------------------------------------------

TEST(IoUringCeiling, SmallestNonZeroSourceWins) {
    nixlIoUringTarget t;
    t.logicalBlockBytes = 512;
    t.maxHwBytes = 128u << 10;
    EXPECT_EQ(nixlIoUringEngine::computeMaxIoBytes(t, nixlIoUringBufferMode::host, 0), 128u << 10);
    EXPECT_EQ(nixlIoUringEngine::computeMaxIoBytes(t, nixlIoUringBufferMode::host, 64u << 10),
              64u << 10);
    t.maxHwBytes = 2u << 20;
    EXPECT_EQ(nixlIoUringEngine::computeMaxIoBytes(t, nixlIoUringBufferMode::host, 0), 2u << 20);
}

// A dma-buf payload is bounded by its own advertised ceiling where the kernel
// publishes one, and by the ordinary one where it does not.
TEST(IoUringCeiling, DmabufUsesItsOwnHardwareCeiling) {
    nixlIoUringTarget t;
    t.logicalBlockBytes = 512;
    t.maxHwBytes = 128u << 10;
    t.maxHwDmabufBytes = 2u << 20;
    EXPECT_EQ(nixlIoUringEngine::computeMaxIoBytes(t, nixlIoUringBufferMode::dmabuf, 0), 2u << 20);
    EXPECT_EQ(nixlIoUringEngine::computeMaxIoBytes(t, nixlIoUringBufferMode::host, 0), 128u << 10);
    EXPECT_EQ(nixlIoUringEngine::computeMaxIoBytes(t, nixlIoUringBufferMode::fixed, 0), 128u << 10);
    // Without the knob a dma-buf request is bounded like any other.
    t.maxHwDmabufBytes = 0;
    EXPECT_EQ(nixlIoUringEngine::computeMaxIoBytes(t, nixlIoUringBufferMode::dmabuf, 0),
              128u << 10);
}

// A target may demand a wider alignment of the payload address than of the
// file offset. Only the offset alignment governs a request's length, so the
// wider address alignment must not shorten what a caller may transfer.
TEST(IoUringCeiling, LengthFollowsTheOffsetAlignmentNotTheAddress) {
    nixlIoUringTarget t;
    t.logicalBlockBytes = 512;
    t.memAlignBytes = 4096;
    t.offsetAlignBytes = 512;
    EXPECT_EQ(nixlIoUringEngine::lengthAlignment(t), 512u);
    EXPECT_EQ(nixlIoUringEngine::computeMaxIoBytes(t, nixlIoUringBufferMode::host, 1536), 1536u);
}

TEST(IoUringCeiling, FallbackAppliesWhenNoSourceNamesALimit) {
    nixlIoUringTarget t;
    t.logicalBlockBytes = 512;
    EXPECT_EQ(nixlIoUringEngine::computeMaxIoBytes(t, nixlIoUringBufferMode::host, 0), 1u << 20);
}

TEST(IoUringCeiling, RoundsDownToTheDirectIoAlignment) {
    nixlIoUringTarget t;
    t.logicalBlockBytes = 4096;
    t.memAlignBytes = 4096;
    t.offsetAlignBytes = 4096;
    EXPECT_EQ(nixlIoUringEngine::computeMaxIoBytes(t, nixlIoUringBufferMode::host, 5000), 4096u);
    // A ceiling below one whole aligned unit can produce no legal request.
    EXPECT_EQ(nixlIoUringEngine::computeMaxIoBytes(t, nixlIoUringBufferMode::host, 100), 0u);
}

// --- parameters -----------------------------------------------------------

TEST(IoUringParams, MissingDeviceNameThrows) {
    EngineMaker maker(nixl_b_params_t{});
    EXPECT_THROW(maker.make(), std::runtime_error);
}

TEST(IoUringParams, InvalidValuesThrow) {
    EXPECT_THROW(EngineMaker({{"device_name", "/dev/mock0"}, {"num_tasks", "3"}}).make(),
                 std::runtime_error);
    EXPECT_THROW(EngineMaker({{"device_name", "/dev/mock0"}, {"direct", "yes"}}).make(),
                 std::runtime_error);
    EXPECT_THROW(EngineMaker({{"device_name", "/dev/mock0"}, {"buffer_mode", "pool"}}).make(),
                 std::runtime_error);
    EXPECT_THROW(EngineMaker({{"device_name", "/dev/mock0"}, {"max_dmabuf_bytes", "0"}}).make(),
                 std::runtime_error);
}

TEST(IoUringParams, ParamsReachTheQueue) {
    EngineMaker maker({{"device_name", "/dev/mock7"}, {"direct", "false"}, {"num_tasks", "16"}});
    auto engine = maker.make();
    EXPECT_EQ(maker.mock->openedPath, "/dev/mock7");
    EXPECT_FALSE(maker.mock->openedOpts.direct);
    EXPECT_EQ(maker.mock->openedOpts.numTasks, 16u);
}

TEST(IoUringParams, QueueOpenFailureThrows) {
    EngineMaker maker;
    maker.configureMock = [](nixlIoUringMockQueue &m) { m.openResult = NIXL_ERR_BACKEND; };
    EXPECT_THROW(maker.make(), std::runtime_error);
}

TEST(IoUringParams, IdentityMismatchThrowsAndMatchSucceeds) {
    EXPECT_THROW(EngineMaker({{"device_name", "/dev/mock0"}, {"expect_serial", "OTHER"}}).make(),
                 std::runtime_error);
    EXPECT_THROW(EngineMaker({{"device_name", "/dev/mock0"}, {"expect_wwid", "x"}}).make(),
                 std::runtime_error);
    EngineMaker ok({{"device_name", "/dev/mock0"},
                    {"expect_serial", "MOCK0"},
                    {"expect_wwid", "mock.wwid.0"}});
    EXPECT_NO_THROW(ok.make());
}

TEST(IoUringParams, CeilingAppliedFromTargetAndUser) {
    EngineMaker maker({{"device_name", "/dev/mock0"}, {"max_io_bytes", "65536"}});
    maker.configureMock = [](nixlIoUringMockQueue &m) { m.maxHwBytes = 1u << 20; };
    auto engine = maker.make();
    EXPECT_EQ(engine->maxIoBytes(), 65536u);
}

TEST(IoUringParams, ForcedCeilingOverridesTheComputedOne) {
    EngineMaker maker({{"device_name", "/dev/mock0"}, {"max_io_bytes_force", "1048576"}});
    maker.configureMock = [](nixlIoUringMockQueue &m) { m.maxHwBytes = 128u << 10; };
    auto engine = maker.make();
    EXPECT_EQ(engine->maxIoBytes(), 1u << 20);
}

TEST(IoUringParams, CeilingCannotUndercutDirectIoAlignment) {
    auto align4k = [](nixlIoUringMockQueue &m) {
        m.memAlignBytes = 4096;
        m.offsetAlignBytes = 4096;
    };
    EngineMaker capped({{"device_name", "/dev/mock0"}, {"max_io_bytes", "512"}});
    capped.configureMock = align4k;
    EXPECT_THROW(capped.make(), std::runtime_error);
    EngineMaker forced({{"device_name", "/dev/mock0"}, {"max_io_bytes_force", "512"}});
    forced.configureMock = align4k;
    EXPECT_THROW(forced.make(), std::runtime_error);
}

// --- registered buffers ---------------------------------------------------

// A kernel that does not carry the interface refuses the registration, and
// that has to surface here rather than as a mysterious transfer failure.
TEST(IoUringRegistered, RegistrationRefusedByTheKernelIsReported) {
    AlignedBuf buf(8192);
    for (const char *mode : {"fixed", "dmabuf"}) {
        EngineMaker maker({{"device_name", "/dev/mock0"}, {"buffer_mode", mode}});
        maker.configureMock = [&buf](nixlIoUringMockQueue &m) {
            m.registerErr = -EINVAL;
            m.dmabufBacking[9] = buf.data;
        };
        auto engine = maker.make();
        nixlBackendMD *md = nullptr;
        EXPECT_EQ(engine->registerMem(
                      nixlBlobDesc(reinterpret_cast<uintptr_t>(buf.data), 8192, 0, "dmabuf:9"),
                      DRAM_SEG,
                      md),
                  NIXL_ERR_BACKEND)
            << "buffer_mode=" << mode;
        EXPECT_EQ(md, nullptr);
    }
}

// A dma-buf the caller names by an fd the kernel does not know fails at
// registration with the kernel's own errno.
TEST(IoUringRegistered, DmabufWithAnUnknownFdIsRefused) {
    EngineMaker maker({{"device_name", "/dev/mock0"}, {"buffer_mode", "dmabuf"}});
    auto engine = maker.make();
    AlignedBuf buf(8192);
    nixlBackendMD *md = nullptr;
    EXPECT_EQ(engine->registerMem(
                  nixlBlobDesc(reinterpret_cast<uintptr_t>(buf.data), 8192, 0, "dmabuf:99"),
                  DRAM_SEG,
                  md),
              NIXL_ERR_BACKEND);
}

TEST(IoUringRegistered, FixedRegistersAndTransfersWithSlots) {
    EngineMaker maker({{"device_name", "/dev/mock0"}, {"buffer_mode", "fixed"}});
    auto engine = maker.make();
    EXPECT_EQ(engine->bufferMode(), nixlIoUringBufferMode::fixed);

    AlignedBuf buf(16384);
    XferHarness::fill(buf, 3);
    nixlBlobDesc dram(reinterpret_cast<uintptr_t>(buf.data), buf.size, 0, "");
    nixlBackendMD *hmd = nullptr;
    ASSERT_EQ(engine->registerMem(dram, DRAM_SEG, hmd), NIXL_SUCCESS);
    ASSERT_NE(hmd, nullptr);
    EXPECT_EQ(maker.mock->registerCalls, 1);
    nixlBackendMD *bmd = nullptr;
    ASSERT_EQ(engine->registerMem(nixlBlobDesc(0, 0, 0, ""), BLK_SEG, bmd), NIXL_SUCCESS);

    // Transfer the second half of the registered buffer to offset 4096.
    nixl_meta_dlist_t local(DRAM_SEG), remote(BLK_SEG);
    nixlMetaDesc l(reinterpret_cast<uintptr_t>(buf.data) + 8192, 8192, 0);
    l.metadataP = hmd;
    local.addDesc(l);
    nixlMetaDesc r(4096, 8192, 0);
    r.metadataP = bmd;
    remote.addDesc(r);
    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_WRITE, local, remote, kAgent, handle), NIXL_SUCCESS);
    nixl_status_t st = engine->postXfer(NIXL_WRITE, local, remote, kAgent, handle);
    for (int i = 0; st == NIXL_IN_PROG && i < 1000; i++) {
        st = engine->checkXfer(handle);
    }
    EXPECT_EQ(st, NIXL_SUCCESS);
    engine->releaseReqH(handle);
    ASSERT_EQ(maker.mock->submittedSlots.size(), 1u);
    EXPECT_EQ(maker.mock->submittedSlots[0], 0u);
    EXPECT_EQ(memcmp(maker.mock->store.data() + 4096, buf.data + 8192, 8192), 0);

    // A local descriptor that is unregistered, or outside its registration,
    // is refused.
    nixl_meta_dlist_t bad(DRAM_SEG);
    nixlMetaDesc unreg(reinterpret_cast<uintptr_t>(buf.data), 8192, 0);
    unreg.metadataP = nullptr;
    bad.addDesc(unreg);
    EXPECT_EQ(engine->prepXfer(NIXL_WRITE, bad, remote, kAgent, handle), NIXL_ERR_INVALID_PARAM);
    nixl_meta_dlist_t outside(DRAM_SEG);
    nixlMetaDesc past(reinterpret_cast<uintptr_t>(buf.data) + 12288, 8192, 0);
    past.metadataP = hmd;
    outside.addDesc(past);
    EXPECT_EQ(engine->prepXfer(NIXL_WRITE, outside, remote, kAgent, handle),
              NIXL_ERR_INVALID_PARAM);

    engine->deregisterMem(hmd);
    EXPECT_EQ(maker.mock->unregisterCalls, 1);
    engine->deregisterMem(bmd);
}

TEST(IoUringRegistered, DmabufUsesOffsetsAndItsOwnCeiling) {
    EngineMaker maker({{"device_name", "/dev/mock0"}, {"buffer_mode", "dmabuf"}});
    AlignedBuf backing(16384);
    maker.configureMock = [&backing](nixlIoUringMockQueue &m) {
        m.dmabufBacking[42] = backing.data;
        m.maxHwBytes = 128u << 10; // what an ordinary request is bounded by
        m.maxHwDmabufBytes = 2u << 20; // what a dma-buf request is bounded by
    };
    auto engine = maker.make();
    EXPECT_EQ(engine->maxIoBytes(), 2u << 20);

    // Registration names the dma-buf by fd. addr is whatever base the caller
    // addresses the buffer by, and never reaches the kernel.
    nixlBackendMD *hmd = nullptr;
    EXPECT_EQ(
        engine->registerMem(
            nixlBlobDesc(reinterpret_cast<uintptr_t>(backing.data), 16384, 0, ""), DRAM_SEG, hmd),
        NIXL_ERR_INVALID_PARAM); // no dmabuf:<fd>
    ASSERT_EQ(engine->registerMem(
                  nixlBlobDesc(reinterpret_cast<uintptr_t>(backing.data), 16384, 0, "dmabuf:42"),
                  DRAM_SEG,
                  hmd),
              NIXL_SUCCESS);
    nixlBackendMD *bmd = nullptr;
    ASSERT_EQ(engine->registerMem(nixlBlobDesc(0, 0, 0, ""), BLK_SEG, bmd), NIXL_SUCCESS);

    XferHarness::fill(backing, 9);
    nixl_meta_dlist_t local(DRAM_SEG), remote(BLK_SEG);
    nixlMetaDesc l(reinterpret_cast<uintptr_t>(backing.data) + 4096, 8192, 0);
    l.metadataP = hmd;
    local.addDesc(l);
    nixlMetaDesc r(65536, 8192, 0);
    r.metadataP = bmd;
    remote.addDesc(r);
    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_WRITE, local, remote, kAgent, handle), NIXL_SUCCESS);
    // The mock fails a dma-buf request carrying an address instead of a byte
    // offset, so a byte-exact store is what proves the offset semantics.
    nixl_status_t st = engine->postXfer(NIXL_WRITE, local, remote, kAgent, handle);
    for (int i = 0; st == NIXL_IN_PROG && i < 1000; i++) {
        st = engine->checkXfer(handle);
    }
    EXPECT_EQ(st, NIXL_SUCCESS);
    engine->releaseReqH(handle);
    ASSERT_EQ(maker.mock->submitted.size(), 1u);
    EXPECT_EQ(maker.mock->submitted[0], std::make_pair(uint64_t{65536}, uint32_t{8192}));
    EXPECT_EQ(memcmp(maker.mock->store.data() + 65536, backing.data + 4096, 8192), 0);
    engine->deregisterMem(hmd);
    engine->deregisterMem(bmd);
}

TEST(IoUringRegistered, DmabufLargerThanTheKernelLimitIsRefused) {
    // The kernel refuses a dma-buf above 1 GiB with a bare EINVAL; refuse it
    // here first, with the reason, and let the limit be a parameter.
    AlignedBuf backing(16384);
    auto make = [&backing](const std::string &limit) {
        nixl_b_params_t p{{"device_name", "/dev/mock0"}, {"buffer_mode", "dmabuf"}};
        if (!limit.empty()) {
            p["max_dmabuf_bytes"] = limit;
        }
        EngineMaker maker(p);
        maker.configureMock = [&backing](nixlIoUringMockQueue &m) {
            m.dmabufBacking[42] = backing.data;
        };
        return maker.make();
    };
    const uintptr_t base = reinterpret_cast<uintptr_t>(backing.data);
    nixlBackendMD *md = nullptr;
    auto engine = make("");
    EXPECT_EQ(engine->registerMem(
                  nixlBlobDesc(base, (size_t(1) << 30) + 4096, 0, "dmabuf:42"), DRAM_SEG, md),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(md, nullptr);
    ASSERT_EQ(
        engine->registerMem(nixlBlobDesc(base, size_t(1) << 30, 0, "dmabuf:42"), DRAM_SEG, md),
        NIXL_SUCCESS);
    engine->deregisterMem(md);

    auto small = make("8192");
    md = nullptr;
    EXPECT_EQ(small->registerMem(nixlBlobDesc(base, 16384, 0, "dmabuf:42"), DRAM_SEG, md),
              NIXL_ERR_INVALID_PARAM);
    ASSERT_EQ(small->registerMem(nixlBlobDesc(base, 8192, 0, "dmabuf:42"), DRAM_SEG, md),
              NIXL_SUCCESS);
    small->deregisterMem(md);
}

TEST(IoUringRegistered, VramIsOnlyReachableAsADmabuf) {
    EngineMaker host;
    auto plain = host.make();
    nixlBackendMD *md = nullptr;
    EXPECT_EQ(plain->registerMem(nixlBlobDesc(0x1000, 4096, 0, "dmabuf:7"), VRAM_SEG, md),
              NIXL_ERR_NOT_SUPPORTED);
    const auto mems = plain->getSupportedMems();
    EXPECT_EQ(std::count(mems.begin(), mems.end(), VRAM_SEG), 0);

    EngineMaker maker({{"device_name", "/dev/mock0"}, {"buffer_mode", "dmabuf"}});
    AlignedBuf backing(8192);
    maker.configureMock = [&backing](nixlIoUringMockQueue &m) {
        m.dmabufBacking[7] = backing.data;
    };
    auto engine = maker.make();
    const auto dmems = engine->getSupportedMems();
    EXPECT_EQ(std::count(dmems.begin(), dmems.end(), VRAM_SEG), 1);
    // The device pointer is any base the caller chooses: only the fd reaches
    // the kernel, and the request carries an offset into the buffer.
    const uintptr_t gpu_base = 0x7f0000000000ull;
    ASSERT_EQ(engine->registerMem(nixlBlobDesc(gpu_base, 8192, 0, "dmabuf:7"), VRAM_SEG, md),
              NIXL_SUCCESS);
    nixlBackendMD *bmd = nullptr;
    ASSERT_EQ(engine->registerMem(nixlBlobDesc(0, 0, 0, ""), BLK_SEG, bmd), NIXL_SUCCESS);
    XferHarness::fill(backing, 5);
    nixl_meta_dlist_t local(VRAM_SEG), remote(BLK_SEG);
    nixlMetaDesc l(gpu_base + 4096, 4096, 0);
    l.metadataP = md;
    local.addDesc(l);
    nixlMetaDesc r(8192, 4096, 0);
    r.metadataP = bmd;
    remote.addDesc(r);
    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_WRITE, local, remote, kAgent, handle), NIXL_SUCCESS);
    nixl_status_t st = engine->postXfer(NIXL_WRITE, local, remote, kAgent, handle);
    for (int i = 0; st == NIXL_IN_PROG && i < 1000; i++) {
        st = engine->checkXfer(handle);
    }
    EXPECT_EQ(st, NIXL_SUCCESS);
    engine->releaseReqH(handle);
    EXPECT_EQ(memcmp(maker.mock->store.data() + 8192, backing.data + 4096, 4096), 0);
    engine->deregisterMem(md);
    engine->deregisterMem(bmd);
}

// --- registration ---------------------------------------------------------

TEST(IoUringRegister, DramNeedsNoMetadataInHostMode) {
    EngineMaker maker;
    auto engine = maker.make();
    nixlBlobDesc desc(0x1000, 4096, 0, "");
    nixlBackendMD *md = reinterpret_cast<nixlBackendMD *>(1);
    EXPECT_EQ(engine->registerMem(desc, DRAM_SEG, md), NIXL_SUCCESS);
    EXPECT_EQ(md, nullptr);
    EXPECT_EQ(maker.mock->registerCalls, 0);
}

TEST(IoUringRegister, UnsupportedMemTypes) {
    EngineMaker maker;
    auto engine = maker.make();
    nixlBlobDesc desc(0, 4096, 0, "");
    nixlBackendMD *md = nullptr;
    EXPECT_EQ(engine->registerMem(desc, VRAM_SEG, md), NIXL_ERR_NOT_SUPPORTED);
    EXPECT_EQ(engine->registerMem(desc, OBJ_SEG, md), NIXL_ERR_NOT_SUPPORTED);
    EXPECT_EQ(engine->registerMem(desc, FILE_SEG, md), NIXL_ERR_NOT_SUPPORTED);
}

TEST(IoUringRegister, BlkRangeChecks) {
    EngineMaker maker;
    auto engine = maker.make();
    const uint64_t size = maker.mock->sizeBytes;
    nixlBackendMD *md = nullptr;

    // The whole target (len 0) and an exact-fit range are both fine.
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(0, 0, 1, ""), BLK_SEG, md), NIXL_SUCCESS);
    engine->deregisterMem(md);
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(4096, 0, 1, ""), BLK_SEG, md),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(size - 4096, 4096, 1, ""), BLK_SEG, md),
              NIXL_SUCCESS);
    engine->deregisterMem(md);
    // metaInfo may repeat this target's path, but may not name another.
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(0, 4096, 1, "/dev/mock0"), BLK_SEG, md),
              NIXL_SUCCESS);
    engine->deregisterMem(md);
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(0, 4096, 1, "/dev/other"), BLK_SEG, md),
              NIXL_ERR_INVALID_PARAM);
    // Misaligned, past the end, and overflowing ranges are refused.
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(100, 4096, 1, ""), BLK_SEG, md),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(0, 1000, 1, ""), BLK_SEG, md),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(size, 4096, 1, ""), BLK_SEG, md),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(size - 512, 4096, 1, ""), BLK_SEG, md),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(0, UINT64_MAX, 1, ""), BLK_SEG, md),
              NIXL_ERR_INVALID_PARAM);
}

TEST(IoUringQuery, InsideRangesAreAccessibleOutsideAreNot) {
    EngineMaker maker;
    auto engine = maker.make();
    const uint64_t size = maker.mock->sizeBytes;
    nixl_reg_dlist_t descs(BLK_SEG);
    descs.addDesc(nixlBlobDesc(0, 4096, 0, ""));
    descs.addDesc(nixlBlobDesc(size, 4096, 1, ""));
    descs.addDesc(nixlBlobDesc(0, 0, 2, "/dev/other"));
    descs.addDesc(nixlBlobDesc(4096, 0, 3, ""));
    std::vector<nixl_query_resp_t> resp;
    ASSERT_EQ(engine->queryMem(descs, resp), NIXL_SUCCESS);
    ASSERT_EQ(resp.size(), 4u);
    ASSERT_TRUE(resp[0].has_value());
    EXPECT_EQ(resp[0]->at("serial"), "MOCK0");
    EXPECT_EQ(resp[0]->at("logical_block_bytes"), "512");
    EXPECT_EQ(resp[0]->at("mem_align_bytes"), "512");
    EXPECT_EQ(resp[0]->at("offset_align_bytes"), "512");
    EXPECT_FALSE(resp[1].has_value());
    EXPECT_FALSE(resp[2].has_value());
    EXPECT_FALSE(resp[3].has_value());
    // Answering a query must cost no I/O.
    EXPECT_EQ(maker.mock->submitAttempts, 0);
}

// --- transfers ------------------------------------------------------------

TEST(IoUringXfer, WriteThenReadRoundTrip) {
    EngineMaker maker;
    auto engine = maker.make();
    XferHarness h(*engine, 4, 8192, 1 << 20);
    EXPECT_EQ(h.run(NIXL_WRITE), NIXL_SUCCESS);
    EXPECT_EQ(memcmp(maker.mock->store.data() + (1 << 20), h.bufs_[0]->data, 8192), 0);
    h.poison();
    EXPECT_EQ(h.run(NIXL_READ), NIXL_SUCCESS);
    EXPECT_TRUE(h.allMatchPattern());

    const auto s = engine->stats();
    EXPECT_EQ(s.reqsSubmitted, 8u);
    EXPECT_EQ(s.reqsCompletedOk, 8u);
    EXPECT_EQ(s.reqsFailed, 0u);
    EXPECT_EQ(s.bytesOkWrite, 4u * 8192u);
    EXPECT_EQ(s.bytesOkRead, 4u * 8192u);
    EXPECT_EQ(s.descsRequested, 8u);
}

TEST(IoUringXfer, DescriptorsSplitAtTheCeiling) {
    EngineMaker maker({{"device_name", "/dev/mock0"}, {"max_io_bytes", "4096"}});
    maker.configureMock = [](nixlIoUringMockQueue &m) { m.capacity = 64; };
    auto engine = maker.make();
    // 10240 bytes is 2.5 ceilings, so three requests with a 2048-byte tail.
    XferHarness h(*engine, 2, 10240, 0, 16384);
    EXPECT_EQ(h.run(NIXL_WRITE), NIXL_SUCCESS);
    ASSERT_EQ(maker.mock->submitted.size(), 6u);
    EXPECT_EQ(maker.mock->submitted[0], std::make_pair(uint64_t{0}, uint32_t{4096}));
    EXPECT_EQ(maker.mock->submitted[1], std::make_pair(uint64_t{4096}, uint32_t{4096}));
    EXPECT_EQ(maker.mock->submitted[2], std::make_pair(uint64_t{8192}, uint32_t{2048}));
    // The second descriptor starts one stride along.
    EXPECT_EQ(maker.mock->submitted[3], std::make_pair(uint64_t{16384}, uint32_t{4096}));
    h.poison();
    EXPECT_EQ(h.run(NIXL_READ), NIXL_SUCCESS);
    EXPECT_TRUE(h.allMatchPattern());
}

TEST(IoUringXfer, QueueDepthExhaustionParksAndResumes) {
    EngineMaker maker; // num_tasks 8
    maker.configureMock = [](nixlIoUringMockQueue &m) { m.capacity = 2; };
    auto engine = maker.make();
    XferHarness h(*engine, 9, 4096);
    EXPECT_EQ(h.run(NIXL_WRITE), NIXL_SUCCESS);
    EXPECT_GT(engine->stats().submitParked, 0u);
    EXPECT_EQ(engine->stats().reqsSubmitted, 9u);
    EXPECT_EQ(engine->stats().maxInflight, 2u);
}

TEST(IoUringXfer, EagainBudgetRetries) {
    EngineMaker maker;
    maker.configureMock = [](nixlIoUringMockQueue &m) { m.eagainBudget = 3; };
    auto engine = maker.make();
    XferHarness h(*engine, 2, 4096);
    EXPECT_EQ(h.run(NIXL_WRITE), NIXL_SUCCESS);
    EXPECT_EQ(maker.mock->submitAccepted, 2);
    EXPECT_GE(maker.mock->submitAttempts, 5);
}

TEST(IoUringXfer, PartialSubmissionThenPermanentFailure) {
    EngineMaker maker;
    maker.configureMock = [](nixlIoUringMockQueue &m) { m.failSubmitAfter = 2; };
    auto engine = maker.make();
    XferHarness h(*engine, 4, 4096);
    EXPECT_EQ(h.run(NIXL_WRITE), NIXL_ERR_BACKEND);
    const auto s = engine->stats();
    EXPECT_EQ(s.reqsSubmitted, 2u);
    EXPECT_EQ(s.submitPermanentErrors, 1u);
    EXPECT_EQ(s.reqsCompletedOk, 2u);
    EXPECT_EQ(s.bytesOkWrite, 2u * 4096u);
}

TEST(IoUringXfer, PermanentQueueErrorFailsInsteadOfStayingInProgress) {
    EngineMaker maker;
    maker.configureMock = [](nixlIoUringMockQueue &m) { m.pokeErrorBudget = 1; };
    auto engine = maker.make();
    XferHarness h(*engine, 2, 4096);
    EXPECT_EQ(h.run(NIXL_WRITE), NIXL_ERR_BACKEND);
    XferHarness later(*engine, 1, 4096);
    EXPECT_EQ(later.run(NIXL_WRITE), NIXL_ERR_BACKEND);
}

TEST(IoUringXfer, TransientQueueErrorRetries) {
    EngineMaker maker;
    maker.configureMock = [](nixlIoUringMockQueue &m) {
        m.pokeErrorBudget = 1;
        m.pokeError = -EINTR;
    };
    auto engine = maker.make();
    XferHarness h(*engine, 2, 4096);
    EXPECT_EQ(h.run(NIXL_WRITE), NIXL_SUCCESS);
}

TEST(IoUringXfer, NegativeCompletionIsBackendError) {
    EngineMaker maker;
    maker.configureMock = [](nixlIoUringMockQueue &m) { m.forceError = -EIO; };
    auto engine = maker.make();
    XferHarness h(*engine, 3, 4096);
    EXPECT_EQ(h.run(NIXL_READ), NIXL_ERR_BACKEND);
    const auto s = engine->stats();
    EXPECT_EQ(s.reqsFailed, 3u);
    EXPECT_EQ(s.bytesOkRead, 0u);
}

// A read or write that moves fewer bytes than asked for leaves the caller's
// range partly untouched, so it is a failure here and not a partial success.
TEST(IoUringXfer, ShortCompletionFails) {
    EngineMaker maker;
    maker.configureMock = [](nixlIoUringMockQueue &m) { m.shortBy = 512; };
    auto engine = maker.make();
    XferHarness h(*engine, 2, 4096);
    EXPECT_EQ(h.run(NIXL_READ), NIXL_ERR_BACKEND);
    const auto s = engine->stats();
    EXPECT_EQ(s.reqsShort, 2u);
    EXPECT_EQ(s.reqsCompletedOk, 0u);
    EXPECT_EQ(s.bytesOkRead, 0u);
}

TEST(IoUringXfer, PrepRejectsBadDescriptors) {
    EngineMaker maker;
    auto engine = maker.make();
    const uint64_t size = maker.mock->sizeBytes;
    std::vector<uint8_t> buf(8192);
    nixlBackendMD *md = nullptr;
    ASSERT_EQ(engine->registerMem(nixlBlobDesc(0, 0, 0, ""), BLK_SEG, md), NIXL_SUCCESS);

    auto attempt = [&](uintptr_t haddr, uint64_t off, size_t llen, size_t rlen) {
        nixl_meta_dlist_t local(DRAM_SEG), remote(BLK_SEG);
        nixlMetaDesc l(haddr, llen, 0);
        l.metadataP = nullptr;
        local.addDesc(l);
        nixlMetaDesc r(off, rlen, 0);
        r.metadataP = md;
        remote.addDesc(r);
        nixlBackendReqH *handle = nullptr;
        nixl_status_t st = engine->prepXfer(NIXL_WRITE, local, remote, kAgent, handle);
        if (st == NIXL_SUCCESS) {
            engine->releaseReqH(handle);
        }
        return st;
    };
    const uintptr_t aligned = reinterpret_cast<uintptr_t>(buf.data()) & ~uintptr_t{kBlock - 1};
    const uintptr_t base = aligned + kBlock; // inside buf, block aligned
    EXPECT_EQ(attempt(base, 0, 4096, 4096), NIXL_SUCCESS);
    EXPECT_EQ(attempt(base + 1, 0, 4096, 4096), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(attempt(base, 100, 4096, 4096), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(attempt(base, 0, 4096, 2048), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(attempt(base, 0, 1000, 1000), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(attempt(base, 0, 0, 0), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(attempt(base, size - 2048, 4096, 4096), NIXL_ERR_INVALID_PARAM);
    engine->deregisterMem(md);

    // Empty lists, mismatched counts and wrong segment types.
    nixl_meta_dlist_t empty_l(DRAM_SEG), empty_r(BLK_SEG);
    nixlBackendReqH *handle = nullptr;
    EXPECT_EQ(engine->prepXfer(NIXL_WRITE, empty_l, empty_r, kAgent, handle),
              NIXL_ERR_INVALID_PARAM);
    nixl_meta_dlist_t obj(OBJ_SEG);
    EXPECT_EQ(engine->prepXfer(NIXL_WRITE, empty_l, obj, kAgent, handle), NIXL_ERR_INVALID_PARAM);
}

TEST(IoUringXfer, DirectIoAlignmentIsStricterThanTheLogicalBlock) {
    EngineMaker maker;
    maker.configureMock = [](nixlIoUringMockQueue &m) {
        m.memAlignBytes = 4096;
        m.offsetAlignBytes = 4096;
    };
    auto engine = maker.make();
    AlignedBuf buf(8192);
    ASSERT_NE(buf.data, nullptr);
    nixlBackendMD *md = nullptr;
    ASSERT_EQ(engine->registerMem(nixlBlobDesc(0, 8192, 0, ""), BLK_SEG, md), NIXL_SUCCESS);

    auto attempt = [&](uintptr_t haddr, uint64_t off, size_t len) {
        nixl_meta_dlist_t local(DRAM_SEG), remote(BLK_SEG);
        nixlMetaDesc l(haddr, len, 0);
        local.addDesc(l);
        nixlMetaDesc r(off, len, 0);
        r.metadataP = md;
        remote.addDesc(r);
        nixlBackendReqH *handle = nullptr;
        nixl_status_t st = engine->prepXfer(NIXL_WRITE, local, remote, kAgent, handle);
        if (st == NIXL_SUCCESS) {
            engine->releaseReqH(handle);
        }
        return st;
    };

    EXPECT_EQ(attempt(reinterpret_cast<uintptr_t>(buf.data), 0, 4096), NIXL_SUCCESS);
    EXPECT_EQ(attempt(reinterpret_cast<uintptr_t>(buf.data) + 512, 0, 4096),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(attempt(reinterpret_cast<uintptr_t>(buf.data), 512, 4096), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(attempt(reinterpret_cast<uintptr_t>(buf.data), 0, 512), NIXL_ERR_INVALID_PARAM);
    engine->deregisterMem(md);
}

TEST(IoUringXfer, TransferOutsideRegistrationRejected) {
    EngineMaker maker;
    auto engine = maker.make();
    std::vector<uint8_t> buf(16384);
    nixlBackendMD *md = nullptr;
    ASSERT_EQ(engine->registerMem(nixlBlobDesc(8192, 8192, 0, ""), BLK_SEG, md), NIXL_SUCCESS);
    const uintptr_t base =
        (reinterpret_cast<uintptr_t>(buf.data()) & ~uintptr_t{kBlock - 1}) + kBlock;
    nixl_meta_dlist_t local(DRAM_SEG), remote(BLK_SEG);
    nixlMetaDesc l(base, 8192, 0);
    l.metadataP = nullptr;
    local.addDesc(l);
    nixlMetaDesc r(4096, 8192, 0); // starts before the registered range
    r.metadataP = md;
    remote.addDesc(r);
    nixlBackendReqH *handle = nullptr;
    EXPECT_EQ(engine->prepXfer(NIXL_WRITE, local, remote, kAgent, handle), NIXL_ERR_INVALID_PARAM);
    engine->deregisterMem(md);
}

TEST(IoUringXfer, HandleRepost) {
    EngineMaker maker;
    auto engine = maker.make();
    XferHarness h(*engine, 2, 4096);
    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_WRITE, h.local_, h.remote_, kAgent, handle), NIXL_SUCCESS);
    for (int round = 0; round < 3; round++) {
        nixl_status_t st = engine->postXfer(NIXL_WRITE, h.local_, h.remote_, kAgent, handle);
        int iters = 0;
        while (st == NIXL_IN_PROG && iters++ < 1000) {
            st = engine->checkXfer(handle);
        }
        EXPECT_EQ(st, NIXL_SUCCESS);
    }
    engine->releaseReqH(handle);
    EXPECT_EQ(engine->stats().reqsSubmitted, 6u);
}

// A released transfer's requests still name the caller's payload buffers, so
// release must not return while the kernel may still be writing into them.
// Without this the caller is free to free memory that is still a DMA target.
TEST(IoUringXfer, ReleaseDrainsInflightBeforeReturning) {
    EngineMaker maker;
    maker.configureMock = [](nixlIoUringMockQueue &m) { m.holdCompletions = true; };
    auto engine = maker.make();
    {
        XferHarness h(*engine, 2, 4096);
        nixlBackendReqH *handle = nullptr;
        ASSERT_EQ(engine->prepXfer(NIXL_WRITE, h.local_, h.remote_, kAgent, handle), NIXL_SUCCESS);
        EXPECT_EQ(engine->postXfer(NIXL_WRITE, h.local_, h.remote_, kAgent, handle), NIXL_IN_PROG);
        EXPECT_EQ(maker.mock->outstanding(), 2u);

        maker.mock->holdCompletions = false;
        EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
        EXPECT_EQ(maker.mock->outstanding(), 0u)
            << "release returned with requests still in flight";
        EXPECT_EQ(engine->stats().reqsCompletedOk, 2u);
    }
    // Destruction is then prompt, with nothing left to drain.
    engine.reset();
}

TEST(IoUringXfer, PrepReleaseWithoutPost) {
    EngineMaker maker;
    auto engine = maker.make();
    XferHarness h(*engine, 2, 4096);
    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_READ, h.local_, h.remote_, kAgent, handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(maker.mock->submitAttempts, 0);
}

TEST(IoUringXfer, ProgressThreadRequestFallsBackToCallerProgress) {
    EngineMaker maker(defaultParams(), true);
    auto engine = maker.make();
    XferHarness h(*engine, 2, 4096);
    EXPECT_EQ(h.run(NIXL_WRITE), NIXL_SUCCESS);
}

} // namespace
