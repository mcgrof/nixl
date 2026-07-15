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
#include "xnvme_kv_backend.h"

namespace {
const nixl_b_params_t &
get_xnvme_kv_backend_options() {
    static const nixl_b_params_t options = {
        {"device_name", ""},
        {"be", "io_uring_cmd"},
        {"nsid", "0"},
        {"num_tasks", "128"},
        {"nguid", ""},
        {"query_mem_mode", "assume_success"},
        {"ignore_read_not_found", "false"},
    };
    return options;
}
} // namespace

using xnvme_kv_plugin_t = nixlBackendPluginCreator<nixlXnvmeKvEngine>;

#ifdef STATIC_PLUGIN_XNVME_KV

nixlBackendPlugin *
createStaticXNVME_KVPlugin() {
    return xnvme_kv_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                     "XNVME_KV",
                                     "0.1.0",
                                     get_xnvme_kv_backend_options(),
                                     {DRAM_SEG, OBJ_SEG});
}

#else

extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return xnvme_kv_plugin_t::create(NIXL_PLUGIN_API_VERSION,
                                     "XNVME_KV",
                                     "0.1.0",
                                     get_xnvme_kv_backend_options(),
                                     {DRAM_SEG, OBJ_SEG});
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}

#endif
