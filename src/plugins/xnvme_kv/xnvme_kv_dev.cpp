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
#include "xnvme_kv_dev.h"

#include <cerrno>
#include <cstring>

#include <libxnvme.h>

#include "common/nixl_log.h"

namespace {

class nixlXnvmeKvRealDev : public nixlXnvmeKvDevIface {
public:
    nixlXnvmeKvRealDev() = default;

    ~nixlXnvmeKvRealDev() override {
        close();
    }

    nixl_status_t
    open(const std::string &uri,
         const std::string &be,
         uint32_t nsid_override,
         uint32_t num_tasks,
         nixlXnvmeKvCplCb cpl_cb) override {
        struct xnvme_opts opts = xnvme_opts_default();

        opts.be = be.c_str();
        opts.admin = "nvme";

        dev_ = xnvme_dev_open(uri.c_str(), &opts);
        if (!dev_) {
            NIXL_ERROR << "XNVME_KV: xnvme_dev_open(" << uri << ", be=" << be
                       << ") failed: " << strerror(errno);
            return NIXL_ERR_BACKEND;
        }

        // On char-device passthrough the URI *is* the namespace; the kernel
        // rejects commands whose nsid differs from the bound one, so an
        // override can only ever act as an expectation check.
        nsid_ = xnvme_dev_get_nsid(dev_);
        if (nsid_override && nsid_override != nsid_) {
            NIXL_ERROR << "XNVME_KV: " << uri << " is namespace " << nsid_
                       << ", not the requested nsid " << nsid_override;
            close();
            return NIXL_ERR_INVALID_PARAM;
        }

        nixl_status_t status = identifyKv();
        if (status != NIXL_SUCCESS) {
            close();
            return status;
        }

        int err = xnvme_queue_init(dev_, num_tasks, 0, &queue_);
        if (err) {
            NIXL_ERROR << "XNVME_KV: xnvme_queue_init(capacity=" << num_tasks
                       << ") failed: " << strerror(-err);
            close();
            return NIXL_ERR_BACKEND;
        }

        cplCb_ = std::move(cpl_cb);
        return NIXL_SUCCESS;
    }

    void
    close() override {
        if (queue_) {
            NIXL_INFO << "XNVME_KV: device stats: submitted=" << statSubmitted_
                      << " saturation_retries=" << statSaturated_;
            int err = xnvme_queue_term(queue_);
            if (err)
                NIXL_WARN << "XNVME_KV: xnvme_queue_term failed: " << strerror(-err);
            queue_ = nullptr;
        }
        if (dev_) {
            xnvme_dev_close(dev_);
            dev_ = nullptr;
        }
    }

    uint32_t
    nsid() const override {
        return nsid_;
    }

    uint16_t
    maxKeyLen() const override {
        return maxKeyLen_;
    }

    uint32_t
    maxValueLen() const override {
        return maxValueLen_;
    }

    std::array<uint8_t, 16>
    nguid() const override {
        return nguid_;
    }

    int
    submitStore(const void *key, uint8_t key_len, void *buf, uint32_t len, void *cookie) override {
        return submit(XNVME_KV_OP_STORE, key, key_len, buf, len, cookie);
    }

    int
    submitRetrieve(const void *key, uint8_t key_len, void *buf, uint32_t len, void *cookie)
        override {
        return submit(XNVME_KV_OP_RETRIEVE, key, key_len, buf, len, cookie);
    }

    int
    submitExist(const void *key, uint8_t key_len, void *cookie) override {
        return submit(XNVME_KV_OP_EXIST, key, key_len, nullptr, 0, cookie);
    }

    int
    poke() override {
        tlsPokingDev_ = this;
        int ret = xnvme_queue_poke(queue_, 0);
        tlsPokingDev_ = nullptr;
        return ret;
    }

    uint32_t
    outstanding() const override {
        return queue_ ? xnvme_queue_get_outstanding(queue_) : 0;
    }

private:
    enum xnvmeKvOp {
        XNVME_KV_OP_STORE,
        XNVME_KV_OP_RETRIEVE,
        XNVME_KV_OP_EXIST,
    };

    nixl_status_t
    identifyKv() {
        struct xnvme_spec_kvs_idfy *idfy;

        idfy = static_cast<struct xnvme_spec_kvs_idfy *>(xnvme_buf_alloc(dev_, sizeof(*idfy)));
        if (!idfy) {
            NIXL_ERROR << "XNVME_KV: failed to allocate identify buffer";
            return NIXL_ERR_BACKEND;
        }
        memset(idfy, 0, sizeof(*idfy));

        struct xnvme_cmd_ctx ctx = xnvme_cmd_ctx_from_dev(dev_);
        int err = xnvme_adm_idfy_ns_csi(&ctx, nsid_, XNVME_SPEC_CSI_KV, &idfy->base);
        if (err || xnvme_cmd_ctx_cpl_status(&ctx)) {
            NIXL_ERROR << "XNVME_KV: Identify Namespace (CSI=KV) failed for nsid " << nsid_
                       << ": err=" << err << " sc=" << std::hex
                       << static_cast<unsigned>(ctx.cpl.status.sc) << " sct="
                       << static_cast<unsigned>(ctx.cpl.status.sct) << std::dec
                       << " — not a KV namespace?";
            xnvme_buf_free(dev_, idfy);
            return NIXL_ERR_INVALID_PARAM;
        }

        // KV format 0 carries the limits (KV spec 1.3, Figures 39/40).
        maxKeyLen_ = idfy->ns.kvf[0].kml;
        maxValueLen_ = idfy->ns.kvf[0].vml;
        memcpy(nguid_.data(), idfy->ns.nguid, sizeof(nguid_));

        xnvme_buf_free(dev_, idfy);

        if (maxKeyLen_ == 0 || maxValueLen_ == 0) {
            NIXL_ERROR << "XNVME_KV: KV identify reports kml=" << maxKeyLen_
                       << " vml=" << maxValueLen_ << "; unusable namespace";
            return NIXL_ERR_BACKEND;
        }
        return NIXL_SUCCESS;
    }

    int
    submit(xnvmeKvOp op, const void *key, uint8_t key_len, void *buf, uint32_t len, void *cookie) {
        struct xnvme_cmd_ctx *ctx = xnvme_queue_get_cmd_ctx(queue_);

        if (!ctx) {
            statSaturated_++;
            return -EAGAIN;
        }

        // Pooled contexts come back with the previous command intact; the
        // xnvme_kvs_* setters do not cover every dword (e.g. an Exist after
        // a Store would inherit cdw10 and stale key/key_hi tail bytes, which
        // are reserved fields a strict controller may reject).
        memset(&ctx->cmd, 0, sizeof(ctx->cmd));
        xnvme_cmd_ctx_set_cb(ctx, cplTrampoline, cookie);

        int err;
        switch (op) {
        case XNVME_KV_OP_STORE:
            err = xnvme_kvs_store(ctx, nsid_, key, key_len, buf, len, 0);
            break;
        case XNVME_KV_OP_RETRIEVE:
            err = xnvme_kvs_retrieve(ctx, nsid_, key, key_len, buf, len, 0);
            break;
        default:
            err = xnvme_kvs_exist(ctx, nsid_, key, key_len);
            break;
        }

        if (err) {
            xnvme_queue_put_cmd_ctx(queue_, ctx);
            // Normalize both saturation signals (-EBUSY: in-flight == capacity,
            // -EAGAIN: no SQE available) to the single retry-later code.
            if (err == -EBUSY || err == -EAGAIN) {
                statSaturated_++;
                return -EAGAIN;
            }
            return err;
        }
        statSubmitted_++;
        return 0;
    }

    static void
    cplTrampoline(struct xnvme_cmd_ctx *ctx, void *cb_arg) {
        // xNVMe only fires callbacks from within xnvme_queue_poke(), which we
        // always enter through poke() below, so the thread-local is valid.
        nixlXnvmeKvRealDev *dev = tlsPokingDev_;
        nixlXnvmeKvCpl cpl;

        cpl.sc = ctx->cpl.status.sc;
        cpl.sct = ctx->cpl.status.sct;
        cpl.cdw0 = ctx->cpl.cdw0;

        xnvme_queue_put_cmd_ctx(ctx->async.queue, ctx);

        if (dev && dev->cplCb_)
            dev->cplCb_(cb_arg, cpl);
    }

    static thread_local nixlXnvmeKvRealDev *tlsPokingDev_;

    struct xnvme_dev *dev_ = nullptr;
    struct xnvme_queue *queue_ = nullptr;
    nixlXnvmeKvCplCb cplCb_;
    uint64_t statSubmitted_ = 0;
    uint64_t statSaturated_ = 0;
    uint32_t nsid_ = 0;
    uint16_t maxKeyLen_ = 0;
    uint32_t maxValueLen_ = 0;
    std::array<uint8_t, 16> nguid_ = {};
};

thread_local nixlXnvmeKvRealDev *nixlXnvmeKvRealDev::tlsPokingDev_ = nullptr;

} // namespace

std::unique_ptr<nixlXnvmeKvDevIface>
nixlXnvmeKvDevCreate() {
    return std::make_unique<nixlXnvmeKvRealDev>();
}
