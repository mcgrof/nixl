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

// CPU-only unit tests for the XNVME_KV backend: the real backend and
// progress-engine translation units are linked against an in-memory mock of
// the xNVMe device seam, so no KV hardware, xNVMe library, or io_uring
// support is needed.

#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "xnvme_kv_backend.h"
#include "xnvme_kv_mock_dev.h"

// The tests always inject a mock device, so the real xNVMe-backed factory
// (xnvme_kv_dev.cpp, which needs libxnvme) is never linked; satisfy the
// backend's fallback reference with a stub.
std::unique_ptr<nixlXnvmeKvDevIface>
nixlXnvmeKvDevCreate() {
    return nullptr;
}

namespace {

constexpr const char *kAgent = "XnvmeKvUnitTester";

nixl_b_params_t
defaultParams() {
    return {{"device_name", "/dev/mock0"}};
}

class EngineMaker {
public:
    explicit EngineMaker(nixl_b_params_t params = defaultParams(),
                         bool prog_thread = false,
                         uint64_t pthr_delay_us = 50)
        : params_(std::move(params)) {
        init_.localAgent = kAgent;
        init_.type = "XNVME_KV";
        init_.customParams = &params_;
        init_.enableProgTh = prog_thread;
        init_.pthrDelay = pthr_delay_us;
        init_.enableTelemetry_ = false;
    }

    // The mock stays owned by the caller-visible raw pointer; the engine owns
    // the unique_ptr. Valid for the engine's lifetime.
    nixlXnvmeKvMockDev *mock = nullptr;

    std::unique_ptr<nixlXnvmeKvEngine>
    make() {
        auto dev = std::make_unique<nixlXnvmeKvMockDev>();
        mock = dev.get();
        if (configureMock)
            configureMock(*mock);
        return std::make_unique<nixlXnvmeKvEngine>(&init_, std::move(dev));
    }

    std::function<void(nixlXnvmeKvMockDev &)> configureMock;

private:
    nixl_b_params_t params_;
    nixlBackendInitParams init_;
};

// Registered DRAM + OBJ descriptors backed by real buffers, for driving
// prepXfer/postXfer/checkXfer directly at the backend-engine level.
class XferHarness {
public:
    XferHarness(nixlXnvmeKvEngine &engine, int count, size_t len)
        : engine_(engine),
          local_(DRAM_SEG),
          remote_(OBJ_SEG) {
        bufs_.resize(count, std::vector<uint8_t>(len));
        for (int i = 0; i < count; i++) {
            for (size_t j = 0; j < len; j++)
                bufs_[i][j] = static_cast<uint8_t>(i + j + 1);

            nixlBlobDesc obj_desc(0, len, static_cast<uint64_t>(i), keyHex(i));
            nixlBackendMD *md = nullptr;
            EXPECT_EQ(engine_.registerMem(obj_desc, OBJ_SEG, md), NIXL_SUCCESS);
            EXPECT_NE(md, nullptr);
            mds_.push_back(md);

            nixlMetaDesc local_desc(reinterpret_cast<uintptr_t>(bufs_[i].data()), len, 0);
            local_desc.metadataP = nullptr;
            local_.addDesc(local_desc);

            nixlMetaDesc remote_desc(0, len, static_cast<uint64_t>(i));
            remote_desc.metadataP = md;
            remote_.addDesc(remote_desc);
        }
    }

    ~XferHarness() {
        for (auto *md : mds_)
            engine_.deregisterMem(md);
    }

    static std::string
    keyHex(int i) {
        char buf[33];
        snprintf(buf, sizeof(buf), "%032x", 0xb000 + i);
        return buf;
    }

    // Drive a full transfer through the engine; returns the final status.
    nixl_status_t
    run(nixl_xfer_op_t op, int max_iters = 100000) {
        nixlBackendReqH *handle = nullptr;
        nixl_status_t status = engine_.prepXfer(op, local_, remote_, kAgent, handle);
        if (status != NIXL_SUCCESS)
            return status;
        status = engine_.postXfer(op, local_, remote_, kAgent, handle);
        int iters = 0;
        while (status == NIXL_IN_PROG && iters++ < max_iters) {
            status = engine_.checkXfer(handle);
            std::this_thread::yield();
        }
        engine_.releaseReqH(handle);
        return status;
    }

    std::vector<std::vector<uint8_t>> bufs_;
    nixl_meta_dlist_t local_;
    nixl_meta_dlist_t remote_;

private:
    nixlXnvmeKvEngine &engine_;
    std::vector<nixlBackendMD *> mds_;
};

std::string
storedKeyBytes(int i) {
    // keyHex(i) decodes to 16 binary bytes.
    nixlXnvmeKvKey key;
    EXPECT_TRUE(nixlXnvmeKvEngine::convertToKvKey(XferHarness::keyHex(i), key));
    return std::string(reinterpret_cast<const char *>(key.key), key.keyLen);
}

// --- key encoding -----------------------------------------------------------

TEST(XnvmeKvKeyCodec, HexDecodes) {
    nixlXnvmeKvKey key;
    ASSERT_TRUE(nixlXnvmeKvEngine::convertToKvKey("00ff10a5", key));
    EXPECT_EQ(key.keyLen, 4);
    EXPECT_EQ(key.key[0], 0x00);
    EXPECT_EQ(key.key[1], 0xff);
    EXPECT_EQ(key.key[2], 0x10);
    EXPECT_EQ(key.key[3], 0xa5);
}

TEST(XnvmeKvKeyCodec, MaxHexKeyIs32Chars) {
    nixlXnvmeKvKey key;
    ASSERT_TRUE(nixlXnvmeKvEngine::convertToKvKey(std::string(32, 'a'), key));
    EXPECT_EQ(key.keyLen, 16);
    EXPECT_FALSE(nixlXnvmeKvEngine::convertToKvKey(std::string(34, 'a'), key));
}

TEST(XnvmeKvKeyCodec, OddLengthOrNonHexIsNotHex) {
    nixlXnvmeKvKey key;
    EXPECT_FALSE(nixlXnvmeKvEngine::convertToKvKey("abc", key));
    EXPECT_FALSE(nixlXnvmeKvEngine::convertToKvKey("zz", key));
}

TEST(XnvmeKvKeyCodec, ResolvePrefersHexOverRaw) {
    nixlXnvmeKvKey key;
    // "beef" is valid hex, so it must decode to 2 bytes, not 4 raw chars.
    ASSERT_TRUE(nixlXnvmeKvEngine::resolveKvKey(0, "beef", key));
    EXPECT_EQ(key.keyLen, 2);
    EXPECT_EQ(key.key[0], 0xbe);
    EXPECT_EQ(key.key[1], 0xef);
}

TEST(XnvmeKvKeyCodec, ResolveRawUpTo16Bytes) {
    nixlXnvmeKvKey key;
    ASSERT_TRUE(nixlXnvmeKvEngine::resolveKvKey(0, "hello-world", key));
    EXPECT_EQ(key.keyLen, 11);
    EXPECT_EQ(memcmp(key.key, "hello-world", 11), 0);
}

TEST(XnvmeKvKeyCodec, ResolveRejectsOver16NonHex) {
    nixlXnvmeKvKey key;
    EXPECT_FALSE(nixlXnvmeKvEngine::resolveKvKey(0, "0123456789abcdefg", key));
}

TEST(XnvmeKvKeyCodec, EmptyMetaInfoFallsBackToDevId) {
    nixlXnvmeKvKey key;
    ASSERT_TRUE(nixlXnvmeKvEngine::resolveKvKey(0x1122334455667788ULL, "", key));
    EXPECT_EQ(key.keyLen, 8);
    uint64_t dev_id = 0;
    memcpy(&dev_id, key.key, 8);
    EXPECT_EQ(dev_id, 0x1122334455667788ULL);
}

// --- constructor / params ---------------------------------------------------

TEST(XnvmeKvParams, MissingDeviceNameThrows) {
    EngineMaker maker(nixl_b_params_t{});
    EXPECT_THROW(maker.make(), std::runtime_error);
}

TEST(XnvmeKvParams, InvalidNumTasksThrows) {
    for (const char *bad : {"0", "3", "4096", "junk", "-8"}) {
        nixl_b_params_t params = defaultParams();
        params["num_tasks"] = bad;
        EngineMaker maker(params);
        EXPECT_THROW(maker.make(), std::runtime_error) << "num_tasks=" << bad;
    }
}

TEST(XnvmeKvParams, InvalidEnumParamsThrow) {
    for (auto [key, bad] : std::vector<std::pair<const char *, const char *>>{
             {"query_mem_mode", "maybe"}, {"ignore_read_not_found", "yes"}, {"nguid", "abc"}}) {
        nixl_b_params_t params = defaultParams();
        params[key] = bad;
        EngineMaker maker(params);
        EXPECT_THROW(maker.make(), std::runtime_error) << key << "=" << bad;
    }
}

TEST(XnvmeKvParams, DeviceOpenFailureThrows) {
    EngineMaker maker;
    maker.configureMock = [](nixlXnvmeKvMockDev &mock) { mock.openResult = NIXL_ERR_BACKEND; };
    EXPECT_THROW(maker.make(), std::runtime_error);
}

TEST(XnvmeKvParams, ParamsReachDevice) {
    nixl_b_params_t params = defaultParams();
    params["num_tasks"] = "64";
    params["be"] = "io_uring_cmd";
    EngineMaker maker(params);
    auto engine = maker.make();
    EXPECT_EQ(maker.mock->openedUri, "/dev/mock0");
    EXPECT_EQ(maker.mock->openedBe, "io_uring_cmd");
    EXPECT_EQ(maker.mock->openedNumTasks, 64u);
}

TEST(XnvmeKvParams, NguidMismatchThrows) {
    nixl_b_params_t params = defaultParams();
    params["nguid"] = "000102030405060708090a0b0c0d0e0f";
    EngineMaker maker(params);
    maker.configureMock = [](nixlXnvmeKvMockDev &mock) {
        mock.nguidVal = {0xff}; // does not match
    };
    EXPECT_THROW(maker.make(), std::runtime_error);
}

TEST(XnvmeKvParams, NguidMatchSucceeds) {
    nixl_b_params_t params = defaultParams();
    params["nguid"] = "000102030405060708090a0b0c0d0e0f";
    EngineMaker maker(params);
    maker.configureMock = [](nixlXnvmeKvMockDev &mock) {
        for (int i = 0; i < 16; i++)
            mock.nguidVal[i] = static_cast<uint8_t>(i);
    };
    EXPECT_NO_THROW(maker.make());
}

// --- registerMem ------------------------------------------------------------

TEST(XnvmeKvRegister, DramNeedsNoMetadata) {
    EngineMaker maker;
    auto engine = maker.make();
    nixlBlobDesc desc(0x1000, 4096, 0, "");
    nixlBackendMD *md = reinterpret_cast<nixlBackendMD *>(0x1);
    EXPECT_EQ(engine->registerMem(desc, DRAM_SEG, md), NIXL_SUCCESS);
    EXPECT_EQ(md, nullptr);
}

TEST(XnvmeKvRegister, UnsupportedMemType) {
    EngineMaker maker;
    auto engine = maker.make();
    nixlBlobDesc desc(0x1000, 4096, 0, "");
    nixlBackendMD *md = nullptr;
    EXPECT_EQ(engine->registerMem(desc, VRAM_SEG, md), NIXL_ERR_NOT_SUPPORTED);
}

TEST(XnvmeKvRegister, OversizedKeyRejected) {
    EngineMaker maker;
    auto engine = maker.make();
    nixlBlobDesc desc(0, 4096, 0, "this-is-way-longer-than-16-bytes");
    nixlBackendMD *md = nullptr;
    EXPECT_EQ(engine->registerMem(desc, OBJ_SEG, md), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(md, nullptr);
}

// --- data path --------------------------------------------------------------

TEST(XnvmeKvXfer, WriteStoresRetrieveReads) {
    EngineMaker maker;
    auto engine = maker.make();
    XferHarness h(*engine, 4, 512);

    EXPECT_EQ(h.run(NIXL_WRITE), NIXL_SUCCESS);
    EXPECT_EQ(maker.mock->store.size(), 4u);
    EXPECT_EQ(maker.mock->store[storedKeyBytes(0)][0], 1);

    // Clear and read back.
    auto expected = h.bufs_;
    for (auto &buf : h.bufs_)
        std::fill(buf.begin(), buf.end(), 0);
    EXPECT_EQ(h.run(NIXL_READ), NIXL_SUCCESS);
    EXPECT_EQ(h.bufs_, expected);
}

TEST(XnvmeKvXfer, MissingKeyReadFails) {
    EngineMaker maker;
    auto engine = maker.make();
    XferHarness h(*engine, 2, 256);
    EXPECT_EQ(h.run(NIXL_READ), NIXL_ERR_BACKEND);
}

TEST(XnvmeKvXfer, MissingKeyReadIgnored) {
    nixl_b_params_t params = defaultParams();
    params["ignore_read_not_found"] = "true";
    EngineMaker maker(params);
    auto engine = maker.make();
    XferHarness h(*engine, 2, 256);
    EXPECT_EQ(h.run(NIXL_READ), NIXL_SUCCESS);
}

TEST(XnvmeKvXfer, QueueDepthExhaustionRetries) {
    EngineMaker maker;
    maker.configureMock = [](nixlXnvmeKvMockDev &mock) { mock.capacity = 2; };
    auto engine = maker.make();
    XferHarness h(*engine, 16, 128);
    EXPECT_EQ(h.run(NIXL_WRITE), NIXL_SUCCESS);
    EXPECT_EQ(maker.mock->store.size(), 16u);
    // 16 accepted + at least some -EAGAIN rejections along the way.
    EXPECT_GT(maker.mock->submitAttempts, 16);
}

TEST(XnvmeKvXfer, EagainBudgetRetries) {
    EngineMaker maker;
    maker.configureMock = [](nixlXnvmeKvMockDev &mock) { mock.eagainBudget = 5; };
    auto engine = maker.make();
    XferHarness h(*engine, 4, 128);
    EXPECT_EQ(h.run(NIXL_WRITE), NIXL_SUCCESS);
    EXPECT_EQ(maker.mock->store.size(), 4u);
}

TEST(XnvmeKvXfer, PartialSubmissionThenPermanentFailure) {
    EngineMaker maker;
    maker.configureMock = [](nixlXnvmeKvMockDev &mock) { mock.failSubmitAfter = 3; };
    auto engine = maker.make();
    XferHarness h(*engine, 8, 128);
    EXPECT_EQ(h.run(NIXL_WRITE), NIXL_ERR_BACKEND);
    // Submission stops at the first permanent failure.
    EXPECT_EQ(maker.mock->submitAccepted, 3);
}

TEST(XnvmeKvXfer, CompletionFailureIsBackendError) {
    EngineMaker maker;
    maker.configureMock = [](nixlXnvmeKvMockDev &mock) { mock.forceCplError = true; };
    auto engine = maker.make();
    XferHarness h(*engine, 4, 128);
    EXPECT_EQ(h.run(NIXL_WRITE), NIXL_ERR_BACKEND);
}

TEST(XnvmeKvXfer, ValueExceedingDeviceMaxRejected) {
    EngineMaker maker;
    maker.configureMock = [](nixlXnvmeKvMockDev &mock) { mock.vml = 256; };
    auto engine = maker.make();
    XferHarness h(*engine, 2, 512);
    EXPECT_EQ(h.run(NIXL_WRITE), NIXL_ERR_INVALID_PARAM);
}

TEST(XnvmeKvXfer, RetrieveLengthMismatchIsWarnOnly) {
    EngineMaker maker;
    maker.configureMock = [](nixlXnvmeKvMockDev &mock) { mock.forcedRetrieveCdw0 = 9999; };
    auto engine = maker.make();
    XferHarness h(*engine, 2, 128);
    EXPECT_EQ(h.run(NIXL_WRITE), NIXL_SUCCESS);
    EXPECT_EQ(h.run(NIXL_READ), NIXL_SUCCESS); // mismatch only warns
}

TEST(XnvmeKvXfer, HandleRepost) {
    EngineMaker maker;
    auto engine = maker.make();
    XferHarness h(*engine, 2, 128);

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_WRITE, h.local_, h.remote_, kAgent, handle), NIXL_SUCCESS);
    for (int round = 0; round < 2; round++) {
        nixl_status_t status = engine->postXfer(NIXL_WRITE, h.local_, h.remote_, kAgent, handle);
        int iters = 0;
        while (status == NIXL_IN_PROG && iters++ < 1000)
            status = engine->checkXfer(handle);
        EXPECT_EQ(status, NIXL_SUCCESS) << "round " << round;
    }
    engine->releaseReqH(handle);
}

TEST(XnvmeKvXfer, EmptyAndMismatchedDescListsRejected) {
    EngineMaker maker;
    auto engine = maker.make();
    XferHarness h(*engine, 2, 128);

    nixl_meta_dlist_t empty_local(DRAM_SEG), empty_remote(OBJ_SEG);
    nixlBackendReqH *handle = nullptr;
    EXPECT_EQ(engine->prepXfer(NIXL_WRITE, empty_local, empty_remote, kAgent, handle),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(engine->prepXfer(NIXL_WRITE, h.local_, empty_remote, kAgent, handle),
              NIXL_ERR_INVALID_PARAM);
}

TEST(XnvmeKvXfer, CancelWithInflightCommands) {
    EngineMaker maker;
    maker.configureMock = [](nixlXnvmeKvMockDev &mock) { mock.capacity = 64; };
    auto engine = maker.make();
    XferHarness h(*engine, 8, 128);

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_WRITE, h.local_, h.remote_, kAgent, handle), NIXL_SUCCESS);
    nixl_status_t status = engine->postXfer(NIXL_WRITE, h.local_, h.remote_, kAgent, handle);
    // Release immediately; commands may still be in flight. The engine must
    // defer the handle free until they drain (engine destruction drains).
    EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
    (void)status;
}

// Regression (review): releasing a prepped-but-never-posted handle must free
// it immediately — no in-flight commands reference it — instead of parking it
// behind a drain condition that never fires (10s destructor stall / leak).
TEST(XnvmeKvXfer, PrepReleaseWithoutPost) {
    for (bool prog_thread : {false, true}) {
        EngineMaker maker(defaultParams(), prog_thread, 50);
        auto engine = maker.make();
        XferHarness h(*engine, 4, 128);

        nixlBackendReqH *handle = nullptr;
        ASSERT_EQ(engine->prepXfer(NIXL_WRITE, h.local_, h.remote_, kAgent, handle),
                  NIXL_SUCCESS);
        EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
        // Engine destruction must not stall on the never-posted handle; the
        // 120s test timeout catches the old 10s-per-engine drain regression.
    }
}

// --- queryMem ---------------------------------------------------------------

// Regression (review): a parked queryMem EXIST handle whose resumed
// submission fails permanently must complete cleanly — the engine must not
// touch the stack-owned handle after the completion flag is set.
TEST(XnvmeKvQuery, ActualModeParkThenPermanentFail) {
    for (bool prog_thread : {false, true}) {
        nixl_b_params_t params = defaultParams();
        params["query_mem_mode"] = "actual";
        EngineMaker maker(params, prog_thread, 50);
        maker.configureMock = [](nixlXnvmeKvMockDev &mock) {
            mock.eagainBudget = 1; // first EXIST attempt parks the handle
            mock.failSubmitAfter = 0; // the resumed attempt fails permanently
        };
        auto engine = maker.make();

        nixl_reg_dlist_t descs(OBJ_SEG);
        descs.addDesc(nixlBlobDesc(0, 128, 1, "aabb"));
        std::vector<nixl_query_resp_t> resp;
        EXPECT_EQ(engine->queryMem(descs, resp), NIXL_ERR_BACKEND);
        ASSERT_EQ(resp.size(), 1u);
        EXPECT_FALSE(resp[0].has_value());
    }
}

// Regression (review): resp is owned by the engine — a dirty vector passed in
// must not break the index-to-descriptor correspondence.
TEST(XnvmeKvQuery, RespVectorIsReset) {
    EngineMaker maker;
    auto engine = maker.make();

    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(0, 128, 1, "aabb"));
    std::vector<nixl_query_resp_t> resp(7, std::nullopt);
    EXPECT_EQ(engine->queryMem(descs, resp), NIXL_SUCCESS);
    ASSERT_EQ(resp.size(), 1u);
    EXPECT_TRUE(resp[0].has_value());
}

// Regression (review): the plugin's advertised option defaults must
// round-trip through the parameter parser.
TEST(XnvmeKvParams, AdvertisedDefaultsRoundTrip) {
    nixl_b_params_t params = {
        {"device_name", "/dev/mock0"},
        {"be", "io_uring_cmd"},
        {"nsid", "0"},
        {"num_tasks", "128"},
        {"nguid", ""},
        {"query_mem_mode", "assume_success"},
        {"ignore_read_not_found", "false"},
    };
    EngineMaker maker(params);
    EXPECT_NO_THROW(maker.make());
}

// Review: DOCA_MEMOS fails init when the device's max key length exceeds the
// plugin's 16-byte keys; match it.
TEST(XnvmeKvParams, DeviceKeyLengthAbovePluginCapacityThrows) {
    EngineMaker maker;
    maker.configureMock = [](nixlXnvmeKvMockDev &mock) { mock.kml = 32; };
    EXPECT_THROW(maker.make(), std::runtime_error);
}

TEST(XnvmeKvQuery, AssumeSuccessIssuesNoCommands) {
    EngineMaker maker;
    auto engine = maker.make();

    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(0, 128, 1, "aabb"));
    descs.addDesc(nixlBlobDesc(0, 128, 2, "ccdd"));
    std::vector<nixl_query_resp_t> resp;
    EXPECT_EQ(engine->queryMem(descs, resp), NIXL_SUCCESS);
    ASSERT_EQ(resp.size(), 2u);
    EXPECT_TRUE(resp[0].has_value());
    EXPECT_TRUE(resp[1].has_value());
    EXPECT_EQ(maker.mock->submitAttempts, 0);
}

TEST(XnvmeKvQuery, ActualModeMixedBatch) {
    nixl_b_params_t params = defaultParams();
    params["query_mem_mode"] = "actual";
    EngineMaker maker(params);
    auto engine = maker.make();

    // Store one object first.
    XferHarness h(*engine, 1, 128);
    ASSERT_EQ(h.run(NIXL_WRITE), NIXL_SUCCESS);

    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(0, 128, 0, XferHarness::keyHex(0))); // exists
    descs.addDesc(nixlBlobDesc(0, 128, 1, "6e6f70650000")); // absent
    std::vector<nixl_query_resp_t> resp;
    EXPECT_EQ(engine->queryMem(descs, resp), NIXL_SUCCESS);
    ASSERT_EQ(resp.size(), 2u);
    EXPECT_TRUE(resp[0].has_value());
    EXPECT_FALSE(resp[1].has_value());
}

TEST(XnvmeKvQuery, ActualModeBadKeyIsError) {
    nixl_b_params_t params = defaultParams();
    params["query_mem_mode"] = "actual";
    EngineMaker maker(params);
    auto engine = maker.make();

    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(0, 128, 1, "this-is-way-longer-than-16-bytes"));
    std::vector<nixl_query_resp_t> resp;
    EXPECT_EQ(engine->queryMem(descs, resp), NIXL_ERR_BACKEND);
    ASSERT_EQ(resp.size(), 1u);
    EXPECT_FALSE(resp[0].has_value());
}

// --- progress-thread mode ----------------------------------------------------

TEST(XnvmeKvThreaded, WriteReadRoundTrip) {
    EngineMaker maker(defaultParams(), /*prog_thread=*/true, /*pthr_delay_us=*/50);
    auto engine = maker.make();
    XferHarness h(*engine, 8, 256);

    EXPECT_EQ(h.run(NIXL_WRITE), NIXL_SUCCESS);
    auto expected = h.bufs_;
    for (auto &buf : h.bufs_)
        std::fill(buf.begin(), buf.end(), 0);
    EXPECT_EQ(h.run(NIXL_READ), NIXL_SUCCESS);
    EXPECT_EQ(h.bufs_, expected);
}

TEST(XnvmeKvThreaded, QueueExhaustionRetries) {
    EngineMaker maker(defaultParams(), true, 50);
    maker.configureMock = [](nixlXnvmeKvMockDev &mock) { mock.capacity = 2; };
    auto engine = maker.make();
    XferHarness h(*engine, 16, 128);
    EXPECT_EQ(h.run(NIXL_WRITE), NIXL_SUCCESS);
    EXPECT_EQ(maker.mock->store.size(), 16u);
}

TEST(XnvmeKvThreaded, CancelBeforePickupAndInflight) {
    EngineMaker maker(defaultParams(), true, 2000);
    auto engine = maker.make();
    XferHarness h(*engine, 4, 128);

    nixlBackendReqH *handle = nullptr;
    ASSERT_EQ(engine->prepXfer(NIXL_WRITE, h.local_, h.remote_, kAgent, handle), NIXL_SUCCESS);
    ASSERT_EQ(engine->postXfer(NIXL_WRITE, h.local_, h.remote_, kAgent, handle), NIXL_IN_PROG);
    // Cancel quickly — depending on timing this hits the never-picked-up or
    // the in-flight path; both must be safe.
    EXPECT_EQ(engine->releaseReqH(handle), NIXL_SUCCESS);
}

TEST(XnvmeKvThreaded, ActualQuery) {
    nixl_b_params_t params = defaultParams();
    params["query_mem_mode"] = "actual";
    EngineMaker maker(params, true, 50);
    auto engine = maker.make();

    XferHarness h(*engine, 2, 128);
    ASSERT_EQ(h.run(NIXL_WRITE), NIXL_SUCCESS);

    nixl_reg_dlist_t descs(OBJ_SEG);
    descs.addDesc(nixlBlobDesc(0, 128, 0, XferHarness::keyHex(0)));
    descs.addDesc(nixlBlobDesc(0, 128, 9, "6d697373696e67")); // "missing"
    std::vector<nixl_query_resp_t> resp;
    EXPECT_EQ(engine->queryMem(descs, resp), NIXL_SUCCESS);
    ASSERT_EQ(resp.size(), 2u);
    EXPECT_TRUE(resp[0].has_value());
    EXPECT_FALSE(resp[1].has_value());
}

} // namespace
