# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Backend sources (no xNVMe dependency) exported for the unit tests, which
# link them against a mock device instead of xnvme_nvm_dev.cpp.
xnvme_nvm_backend_sources = files(
    'xnvme_nvm_backend.cpp',
    'xnvme_nvm_backend.h',
    'xnvme_nvm_dev.h',
    'xnvme_nvm_progress_engine.cpp',
    'xnvme_nvm_progress_engine.h',
    'xnvme_nvm_sysfs.cpp',
)

xnvme_nvm_sources = [
    xnvme_nvm_backend_sources,
    files('xnvme_nvm_dev.cpp', 'xnvme_nvm_uring_dev.cpp', 'xnvme_nvm_plugin.cpp'),
]

plugin_deps = [nixl_infra, nixl_common_dep, xnvme_dep, io_uring_dep]

if 'XNVME_NVM' in static_plugins
    xnvme_nvm_backend_lib = static_library('XNVME_NVM',
        xnvme_nvm_sources,
        dependencies: plugin_deps,
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: false,
        name_prefix: 'libplugin_')
else
    xnvme_nvm_backend_lib = shared_library('XNVME_NVM',
        xnvme_nvm_sources,
        dependencies: plugin_deps,
        cpp_args: ['-fPIC'],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: true,
        name_prefix: 'libplugin_',
        install_dir: plugin_install_dir,
        install_rpath: '$ORIGIN/..')
    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "XNVME_NVM=' + xnvme_nvm_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true
        )
    endif
endif

xnvme_nvm_backend_interface = declare_dependency(link_with: xnvme_nvm_backend_lib)
