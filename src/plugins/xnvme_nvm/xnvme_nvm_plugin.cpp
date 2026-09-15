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
#include "backend/backend_plugin.h"
#include "xnvme_nvm_backend.h"

namespace {
const nixl_b_params_t &
get_xnvme_nvm_backend_options() {
    static const nixl_b_params_t options = {
        {"device_name", ""},
        {"async", "io_uring_cmd"},
        {"direct", "true"},
        {"nsid", "0"},
        {"num_tasks", "128"},
        {"max_io_bytes", "0"},
        {"max_io_bytes_force", "0"},
        {"expect_serial", ""},
        {"expect_wwid", ""},
        {"buffer_mode", "host"},
    };
    return options;
}
} // namespace

using xnvme_nvm_plugin_t = nixlBackendPluginCreator<nixlXnvmeNvmEngine>;

#ifdef STATIC_PLUGIN_XNVME_NVM

nixlBackendPlugin *
createStaticXNVME_NVMPlugin() {
    return xnvme_nvm_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                      "XNVME_NVM",
                                      "0.1.0",
                                      get_xnvme_nvm_backend_options(),
                                      {DRAM_SEG, BLK_SEG});
}

#else

extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return xnvme_nvm_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                      "XNVME_NVM",
                                      "0.1.0",
                                      get_xnvme_nvm_backend_options(),
                                      {DRAM_SEG, BLK_SEG});
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}

#endif
