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
#ifndef XNVME_KV_DEV_H
#define XNVME_KV_DEV_H

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "nixl_types.h"

// Narrow seam over the xNVMe device + async queue. The backend and progress
// engine only ever talk to this interface; the sole translation unit that
// includes libxnvme.h is xnvme_kv_dev.cpp. Unit tests substitute an in-memory
// implementation without linking xNVMe at all.

// NVMe KV command-set status codes (KV spec 1.3). vfio-user-kvssd and xNVMe
// report these with SCT 0x1 (command specific); SPDK and the base spec place
// them in the generic range (SCT 0x0). Match on SC and accept either SCT.
inline constexpr uint8_t XNVME_KV_SC_CAPACITY_EXCEEDED = 0x81;
inline constexpr uint8_t XNVME_KV_SC_INVALID_VALUE_SIZE = 0x85;
inline constexpr uint8_t XNVME_KV_SC_INVALID_KEY_SIZE = 0x86;
inline constexpr uint8_t XNVME_KV_SC_KEY_NOT_EXISTS = 0x87;
inline constexpr uint8_t XNVME_KV_SC_UNRECOVERED_ERR = 0x88;
inline constexpr uint8_t XNVME_KV_SC_KEY_EXISTS = 0x89;

// Decoded NVMe completion for one KV command.
struct nixlXnvmeKvCpl {
    uint8_t sc = 0;
    uint8_t sct = 0;
    // Completion dword 0. Per the KV spec a Retrieve reports the actual value
    // length here; devices that do not implement it (e.g. vfio-user-kvssd)
    // leave it 0.
    uint32_t cdw0 = 0;

    bool
    ok() const {
        return sc == 0 && sct == 0;
    }

    bool
    keyNotExists() const {
        return sc == XNVME_KV_SC_KEY_NOT_EXISTS && (sct == 0 || sct == 1);
    }

    bool
    keyExists() const {
        return sc == XNVME_KV_SC_KEY_EXISTS && (sct == 0 || sct == 1);
    }
};

// Fired from inside poke() for every reaped completion. The cookie is the
// pointer passed at submission time.
using nixlXnvmeKvCplCb = std::function<void(void *cookie, const nixlXnvmeKvCpl &cpl)>;

class nixlXnvmeKvDevIface {
public:
    virtual ~nixlXnvmeKvDevIface() = default;

    // Open the device, create the async queue and discover the KV limits via
    // Identify Namespace (CSI=KV). num_tasks is the queue capacity and must be
    // a power of two in [1, 2048] (xNVMe rejects 4096). nsid_override of 0
    // means "use the namespace the URI resolves to".
    virtual nixl_status_t
    open(const std::string &uri,
         const std::string &be,
         uint32_t nsid_override,
         uint32_t num_tasks,
         nixlXnvmeKvCplCb cpl_cb) = 0;

    virtual void
    close() = 0;

    virtual uint32_t
    nsid() const = 0;

    virtual uint16_t
    maxKeyLen() const = 0;

    virtual uint32_t
    maxValueLen() const = 0;

    virtual std::array<uint8_t, 16>
    nguid() const = 0;

    // Submit one async KV command. Returns 0 on successful submission,
    // -EAGAIN when the queue is saturated (command context pool empty, SQ
    // full, or in-flight == capacity) — the caller must poke() and retry —
    // or another negative errno on permanent failure.
    virtual int
    submitStore(const void *key, uint8_t key_len, void *buf, uint32_t len, void *cookie) = 0;

    virtual int
    submitRetrieve(const void *key, uint8_t key_len, void *buf, uint32_t len, void *cookie) = 0;

    virtual int
    submitExist(const void *key, uint8_t key_len, void *cookie) = 0;

    // Reap completions; fires cpl_cb inline for each. With xNVMe's default
    // command batching this also pushes queued submissions to the kernel, so
    // progress requires poking even when nothing has completed yet. Returns
    // the number of completions processed or a negative errno.
    virtual int
    poke() = 0;

    virtual uint32_t
    outstanding() const = 0;
};

// Real xNVMe-backed implementation (xnvme_kv_dev.cpp).
std::unique_ptr<nixlXnvmeKvDevIface>
nixlXnvmeKvDevCreate();

#endif // XNVME_KV_DEV_H
