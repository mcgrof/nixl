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
#include "xnvme_kv_backend.h"

#include <charconv>
#include <cstring>
#include <iomanip>
#include <stdexcept>

#include "common/nixl_log.h"
#include "xnvme_kv_progress_engine.h"

namespace {

class nixlXnvmeKvMetadata : public nixlBackendMD {
public:
    nixlXnvmeKvMetadata(nixl_mem_t nixl_mem, uint64_t dev_id, const nixlXnvmeKvKey &key)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          devId(dev_id),
          objKey(key) {}

    nixl_mem_t nixlMem;
    uint64_t devId;
    nixlXnvmeKvKey objKey;
};

bool
isValidPrepXferParams(const nixl_xfer_op_t &operation,
                      const nixl_meta_dlist_t &local,
                      const nixl_meta_dlist_t &remote,
                      const std::string &remote_agent,
                      const std::string &local_agent) {
    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        NIXL_ERROR << "XNVME_KV: only NIXL_WRITE and NIXL_READ are supported";
        return false;
    }
    if (remote_agent != local_agent)
        NIXL_WARN << "XNVME_KV: remote agent (" << remote_agent << ") != local agent ("
                  << local_agent << "); XNVME_KV is a local storage backend";
    if (local.getType() != DRAM_SEG) {
        NIXL_ERROR << "XNVME_KV: local descriptor list must be DRAM_SEG, got "
                   << local.getType();
        return false;
    }
    if (remote.getType() != OBJ_SEG) {
        NIXL_ERROR << "XNVME_KV: remote descriptor list must be OBJ_SEG, got "
                   << remote.getType();
        return false;
    }
    return true;
}

} // namespace

std::ostream &
operator<<(std::ostream &os, const nixlXnvmeKvKey &key) {
    auto flags = os.flags();
    for (uint16_t i = 0; i < key.keyLen; i++)
        os << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(key.key[i]);
    os.flags(flags);
    return os;
}

bool
nixlXnvmeKvEngine::convertToKvKey(const std::string &meta_info, nixlXnvmeKvKey &key) {
    if (meta_info.empty())
        return false;
    if (meta_info.size() > XNVME_KV_MAX_OBJECT_KEY_LEN * 2)
        return false;
    if (meta_info.size() & 1)
        return false;

    auto minfo = meta_info.data();
    key.keyLen = meta_info.size() / 2;
    for (uint32_t i = 0; i < key.keyLen; i++) {
        unsigned val = 0;
        auto [ptr, ec] = std::from_chars(minfo + i * 2, minfo + i * 2 + 2, val, 16);
        if (ec != std::errc{} || ptr != minfo + i * 2 + 2)
            return false;
        key.key[i] = static_cast<uint8_t>(val);
    }
    return true;
}

bool
nixlXnvmeKvEngine::resolveKvKey(uint64_t dev_id, const std::string &meta_info, nixlXnvmeKvKey &key) {
    if (meta_info.empty()) {
        key.keyLen = sizeof(dev_id);
        memcpy(key.key, &dev_id, key.keyLen);
        return true;
    }
    if (convertToKvKey(meta_info, key))
        return true;
    if (meta_info.size() <= XNVME_KV_MAX_OBJECT_KEY_LEN) {
        key.keyLen = meta_info.size();
        memcpy(key.key, meta_info.data(), key.keyLen);
        return true;
    }
    return false;
}

nixlXnvmeKvEngine::nixlXnvmeKvEngine(const nixlBackendInitParams *init_params,
                                     std::unique_ptr<nixlXnvmeKvDevIface> dev)
    : nixlBackendEngine(init_params),
      dev_(std::move(dev)) {
    nixl_status_t status = parseInitParams(init_params->customParams);
    if (status != NIXL_SUCCESS)
        throw std::runtime_error("XNVME_KV: invalid backend parameters");

    if (deviceName_.empty())
        throw std::runtime_error("XNVME_KV: device_name is required");

    if (!dev_)
        dev_ = nixlXnvmeKvDevCreate();

    status = initDevice();
    if (status != NIXL_SUCCESS)
        throw std::runtime_error("XNVME_KV: device initialization failed");

    status = createProgressEngine(init_params->enableProgTh, init_params->pthrDelay);
    if (status != NIXL_SUCCESS) {
        dev_->close();
        throw std::runtime_error("XNVME_KV: progress engine initialization failed");
    }

    NIXL_INFO << "XNVME_KV: initialized device_name=" << deviceName_ << " be=" << be_
              << " nsid=" << dev_->nsid() << " num_tasks=" << numTasks_
              << " max_key_len=" << dev_->maxKeyLen() << " max_value_len=" << dev_->maxValueLen()
              << " query_mem_mode=" << (queryMemAssumeSuccess_ ? "assume_success" : "actual")
              << " ignore_read_not_found=" << (ignoreReadNotFound_ ? "true" : "false")
              << " progress_thread=" << (init_params->enableProgTh ? "enabled" : "disabled");
}

nixlXnvmeKvEngine::~nixlXnvmeKvEngine() {
    // The progress engine drains in-flight commands before the device and
    // queue go away; completions reaped during that drain still route
    // through progressEnginePtr_.
    progressEngine_.reset();
    progressEnginePtr_ = nullptr;
    if (dev_)
        dev_->close();
}

nixl_status_t
nixlXnvmeKvEngine::parseInitParams(const nixl_b_params_t *custom_params) {
    if (!custom_params)
        return NIXL_SUCCESS;

    if (auto it = custom_params->find("device_name"); it != custom_params->end())
        deviceName_ = it->second;

    if (auto it = custom_params->find("be"); it != custom_params->end()) {
        if (it->second.empty()) {
            NIXL_ERROR << "XNVME_KV: 'be' must be a non-empty xNVMe backend name";
            return NIXL_ERR_INVALID_PARAM;
        }
        be_ = it->second;
    }

    if (auto it = custom_params->find("nsid"); it != custom_params->end()) {
        try {
            size_t pos = 0;
            unsigned long long parsed = std::stoull(it->second, &pos, 0);
            // 0 = no expectation (use the namespace the URI resolves to).
            if (pos != it->second.size() || parsed > UINT32_MAX) {
                NIXL_ERROR << "XNVME_KV: invalid nsid '" << it->second << "'";
                return NIXL_ERR_INVALID_PARAM;
            }
            nsid_ = static_cast<uint32_t>(parsed);
        }
        catch (const std::exception &) {
            NIXL_ERROR << "XNVME_KV: invalid nsid '" << it->second << "'";
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    if (auto it = custom_params->find("num_tasks"); it != custom_params->end()) {
        try {
            size_t pos = 0;
            unsigned long long parsed = std::stoull(it->second, &pos, 0);
            // xNVMe queue capacity must be a power of two and rejects 4096
            // (xnvme_queue_init requires capacity < 4096).
            if (pos != it->second.size() || parsed == 0 || parsed > 2048 ||
                (parsed & (parsed - 1)) != 0) {
                NIXL_ERROR << "XNVME_KV: num_tasks must be a power of two in [1, 2048], got '"
                           << it->second << "'";
                return NIXL_ERR_INVALID_PARAM;
            }
            numTasks_ = static_cast<uint32_t>(parsed);
        }
        catch (const std::exception &) {
            NIXL_ERROR << "XNVME_KV: invalid num_tasks '" << it->second << "'";
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    if (auto it = custom_params->find("nguid");
        it != custom_params->end() && !it->second.empty()) {
        const std::string &val = it->second;
        if (val.size() != 32) {
            NIXL_ERROR << "XNVME_KV: nguid must be exactly 32 hex characters";
            return NIXL_ERR_INVALID_PARAM;
        }
        for (size_t i = 0; i < val.size(); i += 2) {
            unsigned parsed = 0;
            auto [ptr, ec] = std::from_chars(val.data() + i, val.data() + i + 2, parsed, 16);
            if (ec != std::errc{} || ptr != val.data() + i + 2) {
                NIXL_ERROR << "XNVME_KV: nguid is not valid hex";
                return NIXL_ERR_INVALID_PARAM;
            }
        }
        nguid_ = val;
    }
    if (nguid_.empty())
        NIXL_WARN << "XNVME_KV: no expected 'nguid' provided; namespace identity unvalidated";

    if (auto it = custom_params->find("ignore_read_not_found"); it != custom_params->end()) {
        if (it->second == "true") {
            ignoreReadNotFound_ = true;
        } else if (it->second == "false") {
            ignoreReadNotFound_ = false;
        } else {
            NIXL_ERROR << "XNVME_KV: ignore_read_not_found must be 'true' or 'false'";
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    if (auto it = custom_params->find("query_mem_mode"); it != custom_params->end()) {
        if (it->second == "assume_success") {
            queryMemAssumeSuccess_ = true;
        } else if (it->second == "actual") {
            queryMemAssumeSuccess_ = false;
        } else {
            NIXL_ERROR << "XNVME_KV: query_mem_mode must be 'assume_success' or 'actual'";
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlXnvmeKvEngine::initDevice() {
    nixl_status_t status = dev_->open(
        deviceName_, be_, nsid_, numTasks_, [this](void *cookie, const nixlXnvmeKvCpl &cpl) {
            // progressEnginePtr_, not the unique_ptr: during the engine
            // destructor's drain the unique_ptr already reads null while the
            // progress-engine object is still alive inside its destructor.
            if (progressEnginePtr_)
                progressEnginePtr_->onCompletion(cookie, cpl);
        });
    if (status != NIXL_SUCCESS)
        return status;

    if (dev_->maxKeyLen() > XNVME_KV_MAX_OBJECT_KEY_LEN) {
        NIXL_ERROR << "XNVME_KV: device max key length " << dev_->maxKeyLen()
                   << " exceeds plugin capacity " << XNVME_KV_MAX_OBJECT_KEY_LEN;
        dev_->close();
        return NIXL_ERR_BACKEND;
    }

    if (!nguid_.empty()) {
        nixlXnvmeKvKey expected;
        if (!convertToKvKey(nguid_, expected) || expected.keyLen != 16) {
            NIXL_ERROR << "XNVME_KV: failed to decode expected nguid";
            dev_->close();
            return NIXL_ERR_INVALID_PARAM;
        }
        auto actual = dev_->nguid();
        if (memcmp(expected.key, actual.data(), actual.size()) != 0) {
            NIXL_ERROR << "XNVME_KV: NGUID mismatch: namespace does not match expected 'nguid'";
            dev_->close();
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlXnvmeKvEngine::createProgressEngine(bool enable_prog_thread, uint64_t pthr_delay_us) {
    if (enable_prog_thread) {
        if (pthr_delay_us == 0)
            NIXL_WARN << "XNVME_KV: progress-thread delay is 0us; thread will busy-spin";
        progressEngine_ = std::make_unique<nixlXnvmeKvThreadedEngine>(
            dev_.get(), ignoreReadNotFound_, std::chrono::microseconds(pthr_delay_us));
    } else {
        progressEngine_ =
            std::make_unique<nixlXnvmeKvNoThreadEngine>(dev_.get(), ignoreReadNotFound_);
    }
    progressEnginePtr_ = progressEngine_.get();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlXnvmeKvEngine::registerMem(const nixlBlobDesc &mem,
                               const nixl_mem_t &nixl_mem,
                               nixlBackendMD *&out) {
    out = nullptr;

    if (nixl_mem == DRAM_SEG) {
        // DRAM buffers are passed straight through to io_uring_cmd at submit
        // time; no pinning or registration is needed.
        return NIXL_SUCCESS;
    }
    if (nixl_mem != OBJ_SEG) {
        NIXL_ERROR << "XNVME_KV: unsupported memory type " << nixl_mem;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixlXnvmeKvKey key;
    if (!resolveKvKey(mem.devId, mem.metaInfo, key)) {
        NIXL_ERROR << "XNVME_KV: cannot derive a KV key from metaInfo of length "
                   << mem.metaInfo.size() << " (max " << XNVME_KV_MAX_OBJECT_KEY_LEN
                   << " raw bytes or " << XNVME_KV_MAX_OBJECT_KEY_LEN * 2 << " hex chars)";
        return NIXL_ERR_INVALID_PARAM;
    }

    auto kv_md = std::make_unique<nixlXnvmeKvMetadata>(nixl_mem, mem.devId, key);
    out = kv_md.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlXnvmeKvEngine::deregisterMem(nixlBackendMD *meta) {
    // Releases registration metadata only; the object stays on the device.
    delete static_cast<nixlXnvmeKvMetadata *>(meta);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlXnvmeKvEngine::queryMem(const nixl_reg_dlist_t &descs,
                            std::vector<nixl_query_resp_t> &resp) const {
    resp.clear();
    if (queryMemAssumeSuccess_) {
        // No device command is issued in assume_success mode.
        resp.reserve(descs.descCount());
        for (int i = 0; i < descs.descCount(); i++)
            resp.emplace_back(nixl_b_params_t{});
        return NIXL_SUCCESS;
    }
    return progressEngine_->queryMem(descs, resp);
}

nixl_status_t
nixlXnvmeKvEngine::prepXfer(const nixl_xfer_op_t &operation,
                            const nixl_meta_dlist_t &local,
                            const nixl_meta_dlist_t &remote,
                            const std::string &remote_agent,
                            nixlBackendReqH *&handle,
                            const nixl_opt_b_args_t *opt_args) const {
    if (!isValidPrepXferParams(operation, local, remote, remote_agent, localAgent))
        return NIXL_ERR_INVALID_PARAM;

    const int desc_count = local.descCount();
    if (desc_count == 0) {
        NIXL_ERROR << "XNVME_KV: empty descriptor list";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (remote.descCount() != desc_count) {
        NIXL_ERROR << "XNVME_KV: local (" << desc_count << ") and remote (" << remote.descCount()
                   << ") descriptor counts differ";
        return NIXL_ERR_INVALID_PARAM;
    }

    auto req_h = std::make_unique<nixlXnvmeKvBackendReqH>(desc_count);
    req_h->storedOperation_ = operation;
    if (operation == NIXL_READ)
        req_h->ignoreNotFound_ = ignoreReadNotFound_;

    req_h->valueIovecs_.resize(desc_count);
    req_h->taskContexts_.resize(desc_count);
    req_h->objectKeys_.reserve(desc_count);

    for (int i = 0; i < desc_count; i++) {
        req_h->valueIovecs_[i].iov_base = reinterpret_cast<void *>(local[i].addr);
        req_h->valueIovecs_[i].iov_len = local[i].len;

        const auto *kv_md = static_cast<const nixlXnvmeKvMetadata *>(remote[i].metadataP);
        if (!kv_md) {
            NIXL_ERROR << "XNVME_KV: remote descriptor " << i << " has no registration metadata";
            return NIXL_ERR_INVALID_PARAM;
        }
        req_h->objectKeys_.push_back(kv_md->objKey);
    }

    handle = req_h.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlXnvmeKvEngine::postXfer(const nixl_xfer_op_t &operation,
                            const nixl_meta_dlist_t &local,
                            const nixl_meta_dlist_t &remote,
                            const std::string &remote_agent,
                            nixlBackendReqH *&handle,
                            const nixl_opt_b_args_t *opt_args) const {
    if (!handle) {
        NIXL_ERROR << "XNVME_KV: postXfer on null handle";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        NIXL_ERROR << "XNVME_KV: unsupported operation";
        return NIXL_ERR_INVALID_PARAM;
    }
    return progressEngine_->postXfer(static_cast<nixlXnvmeKvBackendReqH *>(handle), operation);
}

nixl_status_t
nixlXnvmeKvEngine::checkXfer(nixlBackendReqH *handle) const {
    return progressEngine_->checkXfer(static_cast<nixlXnvmeKvBackendReqH *>(handle));
}

nixl_status_t
nixlXnvmeKvEngine::releaseReqH(nixlBackendReqH *handle) const {
    if (!handle) {
        NIXL_ERROR << "XNVME_KV: releaseReqH on null handle";
        return NIXL_ERR_INVALID_PARAM;
    }
    progressEngine_->cancelRequest(static_cast<nixlXnvmeKvBackendReqH *>(handle));
    return NIXL_SUCCESS;
}
