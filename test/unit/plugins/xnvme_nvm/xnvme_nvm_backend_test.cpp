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

// CPU-only unit tests for the XNVME_NVM backend: the real backend and
// progress-engine translation units are linked against an in-memory mock of
// the xNVMe device seam, so no NVMe device, xNVMe library, or io_uring
// support is needed.

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include "xnvme_nvm_backend.h"
#include "xnvme_nvm_mock_dev.h"

// The tests always inject a mock device, so the real xNVMe-backed factory
// (xnvme_nvm_dev.cpp, which needs libxnvme) is never linked; satisfy the
// backend's fallback reference with a stub.
std::unique_ptr<nixlXnvmeNvmDevIface>
nixlXnvmeNvmDevCreate() {
    return nullptr;
}

std::unique_ptr<nixlXnvmeNvmDevIface>
nixlXnvmeNvmUringDevCreate() {
    return nullptr;
}

namespace {

constexpr const char *kAgent = "XnvmeNvmUnitTester";
constexpr uint32_t kLba = 512;

nixl_b_params_t
defaultParams() {
    return {{"device_name", "/dev/mock0"}, {"num_tasks", "8"}};
}

class EngineMaker {
public:
    explicit EngineMaker(nixl_b_params_t params = defaultParams(), bool prog_thread = false)
        : params_(std::move(params)) {
        init_.localAgent = kAgent;
        init_.type = "XNVME_NVM";
        init_.customParams = &params_;
        init_.enableProgTh = prog_thread;
        init_.pthrDelay = 50;
        init_.enableTelemetry_ = false;
    }

    // The mock stays owned by the caller-visible raw pointer; the engine owns
    // the unique_ptr. Valid for the engine's lifetime.
    nixlXnvmeNvmMockDev *mock = nullptr;

    std::unique_ptr<nixlXnvmeNvmEngine>
    make() {
        auto dev = std::make_unique<nixlXnvmeNvmMockDev>();
        mock = dev.get();
        if (configureMock) {
            configureMock(*mock);
        }
        return std::make_unique<nixlXnvmeNvmEngine>(&init_, std::move(dev));
    }

    std::function<void(nixlXnvmeNvmMockDev &)> configureMock;

private:
    nixl_b_params_t params_;
    nixlBackendInitParams init_;
};

// A host buffer aligned to the logical block, as the backend requires.
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

// Registered DRAM + BLK descriptors backed by real buffers, for driving
// prepXfer/postXfer/checkXfer directly at the backend-engine level. Object
// i lives at device offset base + i * stride.
class XferHarness {
public:
    XferHarness(nixlXnvmeNvmEngine &engine,
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

    // Drive a full transfer through the engine; returns the final status.
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
    nixlXnvmeNvmEngine &engine_;
    std::vector<nixlBackendMD *> mds_;
};

// --- ceiling rule --------------------------------------------------------------

TEST(XnvmeNvmCeiling, SmallestNonZeroSourceWins) {
    nixlXnvmeNvmGeo g;
    g.lbaBytes = 512;
    g.idfyMdtsBytes = 2u << 20;
    g.sysfsMaxHwBytes = 128u << 10;
    EXPECT_EQ(nixlXnvmeNvmEngine::computeMaxIoBytes(g, false, nixlXnvmeNvmBufferMode::host, 0),
              128u << 10);
    EXPECT_EQ(
        nixlXnvmeNvmEngine::computeMaxIoBytes(g, false, nixlXnvmeNvmBufferMode::host, 64u << 10),
        64u << 10);
    g.sysfsMaxHwBytes = 0;
    EXPECT_EQ(nixlXnvmeNvmEngine::computeMaxIoBytes(g, false, nixlXnvmeNvmBufferMode::host, 0),
              2u << 20);
}

TEST(XnvmeNvmCeiling, NlbFieldAlwaysCaps) {
    nixlXnvmeNvmGeo g;
    g.lbaBytes = 512;
    g.idfyMdtsBytes = UINT32_MAX; // unbounded MDTS
    EXPECT_EQ(nixlXnvmeNvmEngine::computeMaxIoBytes(g, false, nixlXnvmeNvmBufferMode::host, 0),
              65536u * 512u);
    g.idfyMdtsBytes = 0; // not reported at all: the NLB field still bounds it
    EXPECT_EQ(nixlXnvmeNvmEngine::computeMaxIoBytes(g, false, nixlXnvmeNvmBufferMode::host, 0),
              65536u * 512u);
    // Rounded down to a whole block, never below one block.
    g.lbaBytes = 4096;
    EXPECT_EQ(nixlXnvmeNvmEngine::computeMaxIoBytes(g, false, nixlXnvmeNvmBufferMode::host, 5000),
              4096u);
    EXPECT_EQ(nixlXnvmeNvmEngine::computeMaxIoBytes(g, false, nixlXnvmeNvmBufferMode::host, 100),
              4096u);
}

TEST(XnvmeNvmCeiling, PassthroughIsBoundedBySegmentsTimesPages) {
    nixlXnvmeNvmGeo g;
    g.lbaBytes = 512;
    g.idfyMdtsBytes = 2u << 20;
    g.sysfsMaxHwBytes = 2u << 20;
    g.sysfsMaxSegments = 128;
    g.pageBytes = 4096;
    // The kernel cannot split a passthrough command, so a scattered host
    // buffer is limited to one page per segment; the block paths are not.
    EXPECT_EQ(nixlXnvmeNvmEngine::computeMaxIoBytes(g, true, nixlXnvmeNvmBufferMode::host, 0),
              512u << 10);
    EXPECT_EQ(nixlXnvmeNvmEngine::computeMaxIoBytes(g, false, nixlXnvmeNvmBufferMode::host, 0),
              2u << 20);
    g.sysfsMaxSegments = 0; // unknown: nothing to fold
    EXPECT_EQ(nixlXnvmeNvmEngine::computeMaxIoBytes(g, true, nixlXnvmeNvmBufferMode::host, 0),
              2u << 20);
}

TEST(XnvmeNvmParams, PassthroughCeilingUsesTheSegmentLimit) {
    EngineMaker maker({{"device_name", "/dev/mock0"}, {"async", "io_uring_cmd"}});
    maker.configureMock = [](nixlXnvmeNvmMockDev &m) {
        m.idfyMdtsBytes = 2u << 20;
        m.sysfsMaxSegments = 33;
        m.pageBytes = 4096;
    };
    auto engine = maker.make();
    EXPECT_EQ(engine->maxIoBytes(), 33u * 4096u);
    EngineMaker bio({{"device_name", "/dev/mock0"}, {"async", "io_uring"}});
    bio.configureMock = maker.configureMock;
    auto bio_engine = bio.make();
    EXPECT_EQ(bio_engine->maxIoBytes(), 2u << 20);
}

TEST(XnvmeNvmParams, ForcedCeilingOverridesButNotBeyondNlb) {
    EngineMaker maker({{"device_name", "/dev/mock0"},
                       {"async", "io_uring_cmd"},
                       {"max_io_bytes_force", "1048576"}});
    maker.configureMock = [](nixlXnvmeNvmMockDev &m) {
        m.idfyMdtsBytes = 2u << 20;
        m.sysfsMaxSegments = 128;
        m.pageBytes = 4096; // computed ceiling would be 512 KiB
    };
    auto engine = maker.make();
    EXPECT_EQ(engine->maxIoBytes(), 1u << 20);
    EXPECT_THROW(EngineMaker({{"device_name", "/dev/mock0"},
                              {"max_io_bytes_force", std::to_string(65537ull * 512)}})
                     .make(),
                 std::runtime_error);
    EXPECT_THROW(EngineMaker({{"device_name", "/dev/mock0"}, {"max_io_bytes_force", "100"}}).make(),
                 std::runtime_error);
}

// --- registered buffers --------------------------------------------------------

TEST(XnvmeNvmRegistered, ModeNeedsIoUringAndRegistrationSupport) {
    EXPECT_THROW(EngineMaker({{"device_name", "/dev/mock0"}, {"buffer_mode", "fixed"}}).make(),
                 std::runtime_error); // default async is passthrough
    EXPECT_THROW(EngineMaker({{"device_name", "/dev/mock0"}, {"buffer_mode", "pool"}}).make(),
                 std::runtime_error);
    EngineMaker unsupported(
        {{"device_name", "/dev/mock0"}, {"async", "io_uring"}, {"buffer_mode", "fixed"}});
    EXPECT_THROW(unsupported.make(), std::runtime_error); // mock: registration off
}

TEST(XnvmeNvmRegistered, FixedRegistersAndTransfersWithSlots) {
    EngineMaker maker(
        {{"device_name", "/dev/mock0"}, {"async", "io_uring"}, {"buffer_mode", "fixed"}});
    maker.configureMock = [](nixlXnvmeNvmMockDev &m) { m.registration = true; };
    auto engine = maker.make();
    EXPECT_EQ(engine->bufferMode(), nixlXnvmeNvmBufferMode::fixed);

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

    // A local descriptor that is not registered, or outside its registration, is refused.
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

TEST(XnvmeNvmRegistered, DmabufUsesOffsetsAndItsOwnCeiling) {
    EngineMaker maker(
        {{"device_name", "/dev/mock0"}, {"async", "io_uring"}, {"buffer_mode", "dmabuf"}});
    AlignedBuf backing(16384);
    maker.configureMock = [&backing](nixlXnvmeNvmMockDev &m) {
        m.registration = true;
        m.dmabufBacking[42] = backing.data;
        m.sysfsMaxHwBytes = 128u << 10; // the clamp
        m.sysfsMaxHwDmabufBytes = 2u << 20; // lifted for dma-buf bios
        m.idfyMdtsBytes = 2u << 20;
    };
    auto engine = maker.make();
    EXPECT_EQ(engine->maxIoBytes(), 2u << 20);

    // Registration names the dma-buf by fd; addr is whatever base the caller
    // addresses the buffer by (here its mapping), never sent to the kernel.
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
    // The mock fails a dma-buf command that carries an address instead of
    // the byte offset, so a byte-exact store proves the offset semantics.
    nixl_status_t st = engine->postXfer(NIXL_WRITE, local, remote, kAgent, handle);
    for (int i = 0; st == NIXL_IN_PROG && i < 1000; i++) {
        st = engine->checkXfer(handle);
    }
    EXPECT_EQ(st, NIXL_SUCCESS);
    engine->releaseReqH(handle);
    EXPECT_EQ(memcmp(maker.mock->store.data() + 65536, backing.data + 4096, 8192), 0);
    engine->deregisterMem(hmd);
    engine->deregisterMem(bmd);
}

TEST(XnvmeNvmRegistered, DmabufLargerThanTheKernelLimitIsRefused) {
    // The kernel refuses a dma-buf above 1 GiB with a bare EINVAL; the
    // backend refuses it first with the reason, and the limit is a parameter.
    AlignedBuf backing(16384);
    auto make = [&backing](const std::string &limit) {
        nixl_b_params_t p{
            {"device_name", "/dev/mock0"}, {"async", "io_uring"}, {"buffer_mode", "dmabuf"}};
        if (!limit.empty()) {
            p["max_dmabuf_bytes"] = limit;
        }
        EngineMaker maker(p);
        maker.configureMock = [&backing](nixlXnvmeNvmMockDev &m) {
            m.registration = true;
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
    EXPECT_THROW(make("0"), std::exception);
}

TEST(XnvmeNvmRegistered, VramIsOnlyReachableAsADmabuf) {
    EngineMaker host;
    auto plain = host.make();
    nixlBackendMD *md = nullptr;
    EXPECT_EQ(plain->registerMem(nixlBlobDesc(0x1000, 4096, 0, "dmabuf:7"), VRAM_SEG, md),
              NIXL_ERR_NOT_SUPPORTED);
    const auto mems = plain->getSupportedMems();
    EXPECT_EQ(std::count(mems.begin(), mems.end(), VRAM_SEG), 0);

    EngineMaker maker(
        {{"device_name", "/dev/mock0"}, {"async", "io_uring"}, {"buffer_mode", "dmabuf"}});
    AlignedBuf backing(8192);
    maker.configureMock = [&backing](nixlXnvmeNvmMockDev &m) {
        m.registration = true;
        m.dmabufBacking[7] = backing.data;
    };
    auto engine = maker.make();
    const auto dmems = engine->getSupportedMems();
    EXPECT_EQ(std::count(dmems.begin(), dmems.end(), VRAM_SEG), 1);
    // The "device pointer" is any base the caller chooses; only the fd
    // reaches the kernel and the command carries an offset.
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

TEST(XnvmeNvmCpl, KernelErrnoDecodes) {
    nixlXnvmeNvmCpl cpl;
    cpl.sct = XNVME_NVM_SCT_ERRNO;
    cpl.sc = static_cast<uint8_t>(-EINVAL);
    EXPECT_EQ(cpl.kernelErrno(), -EINVAL);
    cpl.sct = 0;
    EXPECT_EQ(cpl.kernelErrno(), 0);
}

// --- parameters -----------------------------------------------------------------

TEST(XnvmeNvmParams, MissingDeviceNameThrows) {
    EngineMaker maker(nixl_b_params_t{});
    EXPECT_THROW(maker.make(), std::runtime_error);
}

TEST(XnvmeNvmParams, InvalidValuesThrow) {
    EXPECT_THROW(EngineMaker({{"device_name", "/dev/mock0"}, {"num_tasks", "3"}}).make(),
                 std::runtime_error);
    EXPECT_THROW(EngineMaker({{"device_name", "/dev/mock0"}, {"async", "spdk"}}).make(),
                 std::runtime_error);
    EXPECT_THROW(EngineMaker({{"device_name", "/dev/mock0"}, {"direct", "yes"}}).make(),
                 std::runtime_error);
    EXPECT_THROW(EngineMaker({{"device_name", "/dev/mock0"}, {"buffer_mode", "fixed"}}).make(),
                 std::runtime_error);
    EXPECT_THROW(EngineMaker({{"device_name", "/dev/mock0"}, {"buffer_mode", "dmabuf"}}).make(),
                 std::runtime_error);
}

TEST(XnvmeNvmParams, ParamsReachDevice) {
    EngineMaker maker({{"device_name", "/dev/mock7"},
                       {"async", "io_uring"},
                       {"direct", "false"},
                       {"nsid", "3"},
                       {"num_tasks", "16"}});
    auto engine = maker.make();
    EXPECT_EQ(maker.mock->openedUri, "/dev/mock7");
    EXPECT_EQ(maker.mock->openedOpts.async, "io_uring");
    EXPECT_FALSE(maker.mock->openedOpts.direct);
    EXPECT_EQ(maker.mock->openedOpts.nsidOverride, 3u);
    EXPECT_EQ(maker.mock->openedOpts.numTasks, 16u);
}

TEST(XnvmeNvmParams, DeviceOpenFailureThrows) {
    EngineMaker maker;
    maker.configureMock = [](nixlXnvmeNvmMockDev &m) { m.openResult = NIXL_ERR_BACKEND; };
    EXPECT_THROW(maker.make(), std::runtime_error);
}

TEST(XnvmeNvmParams, IdentityMismatchThrowsAndMatchSucceeds) {
    EXPECT_THROW(EngineMaker({{"device_name", "/dev/mock0"}, {"expect_serial", "OTHER"}}).make(),
                 std::runtime_error);
    EXPECT_THROW(EngineMaker({{"device_name", "/dev/mock0"}, {"expect_wwid", "x"}}).make(),
                 std::runtime_error);
    EngineMaker ok({{"device_name", "/dev/mock0"},
                    {"expect_serial", "MOCK0"},
                    {"expect_wwid", "mock.wwid.0"}});
    EXPECT_NO_THROW(ok.make());
}

TEST(XnvmeNvmParams, CeilingAppliedFromDeviceAndUser) {
    EngineMaker maker({{"device_name", "/dev/mock0"}, {"max_io_bytes", "65536"}});
    maker.configureMock = [](nixlXnvmeNvmMockDev &m) { m.idfyMdtsBytes = 1u << 20; };
    auto engine = maker.make();
    EXPECT_EQ(engine->maxIoBytes(), 65536u);
}

// --- registration ---------------------------------------------------------------

TEST(XnvmeNvmRegister, DramNeedsNoMetadata) {
    EngineMaker maker;
    auto engine = maker.make();
    nixlBlobDesc desc(0x1000, 4096, 0, "");
    nixlBackendMD *md = reinterpret_cast<nixlBackendMD *>(1);
    EXPECT_EQ(engine->registerMem(desc, DRAM_SEG, md), NIXL_SUCCESS);
    EXPECT_EQ(md, nullptr);
}

TEST(XnvmeNvmRegister, UnsupportedMemTypes) {
    EngineMaker maker;
    auto engine = maker.make();
    nixlBlobDesc desc(0, 4096, 0, "");
    nixlBackendMD *md = nullptr;
    EXPECT_EQ(engine->registerMem(desc, VRAM_SEG, md), NIXL_ERR_NOT_SUPPORTED);
    EXPECT_EQ(engine->registerMem(desc, OBJ_SEG, md), NIXL_ERR_NOT_SUPPORTED);
    EXPECT_EQ(engine->registerMem(desc, FILE_SEG, md), NIXL_ERR_NOT_SUPPORTED);
}

TEST(XnvmeNvmRegister, BlkRangeChecks) {
    EngineMaker maker;
    auto engine = maker.make();
    const uint64_t cap = maker.mock->capacityBytes;
    nixlBackendMD *md = nullptr;

    // Whole namespace (len 0) and an exact-fit range are fine.
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(0, 0, 1, ""), BLK_SEG, md), NIXL_SUCCESS);
    engine->deregisterMem(md);
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(cap - 4096, 4096, 1, ""), BLK_SEG, md),
              NIXL_SUCCESS);
    engine->deregisterMem(md);
    // The device path may be repeated in metaInfo but not another device.
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
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(cap, 4096, 1, ""), BLK_SEG, md),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(cap - 512, 4096, 1, ""), BLK_SEG, md),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(engine->registerMem(nixlBlobDesc(0, UINT64_MAX, 1, ""), BLK_SEG, md),
              NIXL_ERR_INVALID_PARAM);
}

TEST(XnvmeNvmQuery, InsideRangesAreAccessibleOutsideAreNot) {
    EngineMaker maker;
    auto engine = maker.make();
    const uint64_t cap = maker.mock->capacityBytes;
    nixl_reg_dlist_t descs(BLK_SEG);
    descs.addDesc(nixlBlobDesc(0, 4096, 0, ""));
    descs.addDesc(nixlBlobDesc(cap, 4096, 1, ""));
    descs.addDesc(nixlBlobDesc(0, 0, 2, "/dev/other"));
    std::vector<nixl_query_resp_t> resp;
    ASSERT_EQ(engine->queryMem(descs, resp), NIXL_SUCCESS);
    ASSERT_EQ(resp.size(), 3u);
    ASSERT_TRUE(resp[0].has_value());
    EXPECT_EQ(resp[0]->at("serial"), "MOCK0");
    EXPECT_EQ(resp[0]->at("lba_bytes"), "512");
    EXPECT_FALSE(resp[1].has_value());
    EXPECT_FALSE(resp[2].has_value());
    EXPECT_EQ(maker.mock->submitAttempts, 0);
}

// --- transfers ------------------------------------------------------------------

TEST(XnvmeNvmXfer, WriteThenReadRoundTrip) {
    EngineMaker maker;
    auto engine = maker.make();
    XferHarness h(*engine, 4, 8192, 1 << 20);
    EXPECT_EQ(h.run(NIXL_WRITE), NIXL_SUCCESS);
    EXPECT_EQ(memcmp(maker.mock->store.data() + (1 << 20), h.bufs_[0]->data, 8192), 0);
    h.poison();
    EXPECT_EQ(h.run(NIXL_READ), NIXL_SUCCESS);
    EXPECT_TRUE(h.allMatchPattern());

    const auto s = engine->stats();
    EXPECT_EQ(s.cmdsSubmitted, 8u);
    EXPECT_EQ(s.cmdsCompletedOk, 8u);
    EXPECT_EQ(s.cmdsFailed, 0u);
    EXPECT_EQ(s.bytesOkWrite, 4u * 8192u);
    EXPECT_EQ(s.bytesOkRead, 4u * 8192u);
    EXPECT_EQ(s.descsRequested, 8u);
}

TEST(XnvmeNvmXfer, DescriptorsSplitAtTheCeiling) {
    EngineMaker maker({{"device_name", "/dev/mock0"}, {"max_io_bytes", "4096"}});
    maker.configureMock = [](nixlXnvmeNvmMockDev &m) { m.capacity = 64; };
    auto engine = maker.make();
    // 10000 bytes is not a block multiple; 10240 = 2.5 ceilings -> 3 commands.
    XferHarness h(*engine, 2, 10240, 0, 16384);
    EXPECT_EQ(h.run(NIXL_WRITE), NIXL_SUCCESS);
    ASSERT_EQ(maker.mock->submitted.size(), 6u);
    // First descriptor: slba 0/8/16 with nlb 7, 7, 3 (2048 bytes tail).
    EXPECT_EQ(maker.mock->submitted[0], std::make_pair(uint64_t{0}, uint16_t{7}));
    EXPECT_EQ(maker.mock->submitted[1], std::make_pair(uint64_t{8}, uint16_t{7}));
    EXPECT_EQ(maker.mock->submitted[2], std::make_pair(uint64_t{16}, uint16_t{3}));
    // Second descriptor starts at 16384 / 512 = slba 32.
    EXPECT_EQ(maker.mock->submitted[3], std::make_pair(uint64_t{32}, uint16_t{7}));
    h.poison();
    EXPECT_EQ(h.run(NIXL_READ), NIXL_SUCCESS);
    EXPECT_TRUE(h.allMatchPattern());
}

TEST(XnvmeNvmXfer, QueueDepthExhaustionParksAndResumes) {
    EngineMaker maker; // num_tasks 8
    maker.configureMock = [](nixlXnvmeNvmMockDev &m) { m.capacity = 2; };
    auto engine = maker.make();
    XferHarness h(*engine, 9, 4096);
    EXPECT_EQ(h.run(NIXL_WRITE), NIXL_SUCCESS);
    EXPECT_GT(engine->stats().submitParked, 0u);
    EXPECT_EQ(engine->stats().cmdsSubmitted, 9u);
    EXPECT_LE(engine->stats().maxInflight, 2u);
}

TEST(XnvmeNvmXfer, EagainBudgetRetries) {
    EngineMaker maker;
    maker.configureMock = [](nixlXnvmeNvmMockDev &m) { m.eagainBudget = 3; };
    auto engine = maker.make();
    XferHarness h(*engine, 2, 4096);
    EXPECT_EQ(h.run(NIXL_WRITE), NIXL_SUCCESS);
    EXPECT_EQ(maker.mock->submitAccepted, 2);
    EXPECT_GE(maker.mock->submitAttempts, 5);
}

TEST(XnvmeNvmXfer, PartialSubmissionThenPermanentFailure) {
    EngineMaker maker;
    maker.configureMock = [](nixlXnvmeNvmMockDev &m) { m.failSubmitAfter = 2; };
    auto engine = maker.make();
    XferHarness h(*engine, 4, 4096);
    EXPECT_EQ(h.run(NIXL_WRITE), NIXL_ERR_BACKEND);
    const auto s = engine->stats();
    EXPECT_EQ(s.cmdsSubmitted, 2u);
    EXPECT_EQ(s.submitPermanentErrors, 1u);
    EXPECT_EQ(s.cmdsCompletedOk, 2u);
    EXPECT_EQ(s.bytesOkWrite, 2u * 4096u);
}

TEST(XnvmeNvmXfer, CompletionFailureIsBackendError) {
    EngineMaker maker;
    maker.configureMock = [](nixlXnvmeNvmMockDev &m) { m.forceCplError = true; };
    auto engine = maker.make();
    XferHarness h(*engine, 3, 4096);
    EXPECT_EQ(h.run(NIXL_READ), NIXL_ERR_BACKEND);
    const auto s = engine->stats();
    EXPECT_EQ(s.cmdsFailed, 3u);
    EXPECT_EQ(s.bytesOkRead, 0u);
}

TEST(XnvmeNvmXfer, ShortCompletionOnTheBioPathFails) {
    EngineMaker maker;
    maker.configureMock = [](nixlXnvmeNvmMockDev &m) {
        m.reportsBytes = true;
        m.shortBy = 512;
    };
    auto engine = maker.make();
    XferHarness h(*engine, 2, 4096);
    EXPECT_EQ(h.run(NIXL_READ), NIXL_ERR_BACKEND);
    const auto s = engine->stats();
    EXPECT_EQ(s.cmdsShort, 2u);
    EXPECT_EQ(s.cmdsCompletedOk, 0u);
    EXPECT_EQ(s.bytesOkRead, 0u);
}

TEST(XnvmeNvmXfer, PassthroughIgnoresResultForLength) {
    EngineMaker maker;
    maker.configureMock = [](nixlXnvmeNvmMockDev &m) {
        m.reportsBytes = false;
        m.shortBy = 512; // would be short on the bio path; passthrough never is
    };
    auto engine = maker.make();
    XferHarness h(*engine, 2, 4096);
    EXPECT_EQ(h.run(NIXL_READ), NIXL_SUCCESS);
    EXPECT_EQ(engine->stats().cmdsShort, 0u);
}

TEST(XnvmeNvmXfer, PrepRejectsBadDescriptors) {
    EngineMaker maker;
    auto engine = maker.make();
    const uint64_t cap = maker.mock->capacityBytes;
    std::vector<uint8_t> buf(8192);
    nixlBlobDesc reg(0, 0, 0, "");
    nixlBackendMD *md = nullptr;
    ASSERT_EQ(engine->registerMem(reg, BLK_SEG, md), NIXL_SUCCESS);

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
    const uintptr_t aligned = reinterpret_cast<uintptr_t>(buf.data()) & ~uintptr_t{kLba - 1};
    const uintptr_t base = aligned + kLba; // inside buf, block aligned
    EXPECT_EQ(attempt(base, 0, 4096, 4096), NIXL_SUCCESS);
    EXPECT_EQ(attempt(base + 1, 0, 4096, 4096), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(attempt(base, 100, 4096, 4096), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(attempt(base, 0, 4096, 2048), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(attempt(base, 0, 1000, 1000), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(attempt(base, 0, 0, 0), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(attempt(base, cap - 2048, 4096, 4096), NIXL_ERR_INVALID_PARAM);
    engine->deregisterMem(md);

    // Empty lists, mismatched counts, and wrong segment types.
    nixl_meta_dlist_t empty_l(DRAM_SEG), empty_r(BLK_SEG);
    nixlBackendReqH *handle = nullptr;
    EXPECT_EQ(engine->prepXfer(NIXL_WRITE, empty_l, empty_r, kAgent, handle),
              NIXL_ERR_INVALID_PARAM);
    nixl_meta_dlist_t obj(OBJ_SEG);
    EXPECT_EQ(engine->prepXfer(NIXL_WRITE, empty_l, obj, kAgent, handle), NIXL_ERR_INVALID_PARAM);
}

TEST(XnvmeNvmXfer, TransferOutsideRegistrationRejected) {
    EngineMaker maker;
    auto engine = maker.make();
    std::vector<uint8_t> buf(16384);
    nixlBackendMD *md = nullptr;
    ASSERT_EQ(engine->registerMem(nixlBlobDesc(8192, 8192, 0, ""), BLK_SEG, md), NIXL_SUCCESS);
    const uintptr_t base = (reinterpret_cast<uintptr_t>(buf.data()) & ~uintptr_t{kLba - 1}) + kLba;
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

TEST(XnvmeNvmXfer, HandleRepost) {
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
    EXPECT_EQ(engine->stats().cmdsSubmitted, 6u);
}

TEST(XnvmeNvmXfer, CancelWithInflightCommandsDefersFree) {
    EngineMaker maker;
    maker.configureMock = [](nixlXnvmeNvmMockDev &m) { m.holdCompletions = true; };
    auto engine = maker.make();
    XferHarness h(*engine, 2, 4096);
    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_WRITE, h.local_, h.remote_, kAgent, handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->postXfer(NIXL_WRITE, h.local_, h.remote_, kAgent, handle), NIXL_IN_PROG);
    EXPECT_EQ(maker.mock->outstanding(), 2u);
    // Release while both commands are in flight: the handle must survive
    // until they complete.
    engine->releaseReqH(handle);
    EXPECT_EQ(maker.mock->outstanding(), 2u);
    maker.mock->holdCompletions = false;
    // Any later progress drains and frees it; the destructor also would.
    XferHarness h2(*engine, 1, 4096, 1 << 20);
    EXPECT_EQ(h2.run(NIXL_WRITE), NIXL_SUCCESS);
    EXPECT_EQ(maker.mock->outstanding(), 0u);
}

TEST(XnvmeNvmXfer, PrepReleaseWithoutPost) {
    EngineMaker maker;
    auto engine = maker.make();
    XferHarness h(*engine, 2, 4096);
    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_READ, h.local_, h.remote_, kAgent, handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
    EXPECT_EQ(maker.mock->submitAttempts, 0);
}

TEST(XnvmeNvmXfer, ProgressThreadRequestFallsBackToCallerProgress) {
    EngineMaker maker(defaultParams(), true);
    auto engine = maker.make();
    XferHarness h(*engine, 2, 4096);
    EXPECT_EQ(h.run(NIXL_WRITE), NIXL_SUCCESS);
}

TEST(XnvmeNvmXfer, DestructorDrainsInflight) {
    EngineMaker maker;
    maker.configureMock = [](nixlXnvmeNvmMockDev &m) { m.holdCompletions = true; };
    auto engine = maker.make();
    auto *h = new XferHarness(*engine, 2, 4096);
    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_WRITE, h->local_, h->remote_, kAgent, handle), NIXL_SUCCESS);
    EXPECT_EQ(engine->postXfer(NIXL_WRITE, h->local_, h->remote_, kAgent, handle), NIXL_IN_PROG);
    engine->releaseReqH(handle);
    maker.mock->holdCompletions = false; // the destructor's drain completes them
    delete h;
    engine.reset(); // must return promptly: the mock completes on the drain's poke
}

} // namespace
