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

// An xNVMe-independent storage backend over io_uring, for registered host
// buffers and GPU dma-bufs.
//
// This is a packaging of the engine XNVME_NVM already uses, not a second
// implementation of it: the backend, progress engine, sysfs helpers and the
// liburing device are the same translation units, built here without the
// xNVMe dependency.  NIXL already ships ordinary io_uring in its POSIX
// plugin; what this adds is dma-buf registration, target-specific mappings
// and raw-block endpoint semantics.
//
// The engine asks for a device implementation twice: the liburing one for
// registered buffers, and a second for plain host buffers, which upstream
// satisfies with xNVMe.  The liburing device already serves both — it
// prepares unregistered read/write as well as the fixed variants — so the
// factory below returns it for either request, and no source is duplicated
// to achieve that.

#include "backend/backend_plugin.h"
#include "xnvme_nvm_backend.h"
#include "xnvme_nvm_dev.h"

std::unique_ptr<nixlXnvmeNvmDevIface>
nixlXnvmeNvmDevCreate() {
    return nixlXnvmeNvmUringDevCreate();
}

namespace {
const nixl_b_params_t &
get_io_uring_backend_options() {
    static const nixl_b_params_t options = {
        {"device_name", ""},
        // io_uring only: this backend has no passthrough transport, so the
        // parameter that selects one is absent rather than ignored.
        {"direct", "true"},
        {"num_tasks", "128"},
        {"max_io_bytes", "0"},
        {"max_io_bytes_force", "0"},
        {"expect_serial", ""},
        {"expect_wwid", ""},
        {"buffer_mode", "host"},
        {"max_dmabuf_bytes", "1073741824"},
    };
    return options;
}
} // namespace

using io_uring_plugin_t = nixlBackendPluginCreator<nixlXnvmeNvmEngine>;

#ifdef STATIC_PLUGIN_IO_URING

nixlBackendPlugin *
createStaticIO_URINGPlugin() {
    return io_uring_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                     "IO_URING",
                                     "0.1.0",
                                     get_io_uring_backend_options(),
                                     {DRAM_SEG, VRAM_SEG, BLK_SEG});
}

#else

extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return io_uring_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                     "IO_URING",
                                     "0.1.0",
                                     get_io_uring_backend_options(),
                                     {DRAM_SEG, VRAM_SEG, BLK_SEG});
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}

#endif
