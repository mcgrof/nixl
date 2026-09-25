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

# Everything that does not itself talk to liburing is listed separately so the
# CPU-only unit tests can link it against an in-memory queue.
io_uring_backend_sources = files(
    'io_uring_backend.cpp',
    'io_uring_backend.h',
    'io_uring_progress_engine.cpp',
    'io_uring_progress_engine.h',
    'io_uring_queue.h',
    'io_uring_sysfs.cpp',
)

io_uring_sources = [
    io_uring_backend_sources,
    files('io_uring_queue.cpp', 'io_uring_plugin.cpp'),
]

io_uring_plugin_deps = [nixl_infra, nixl_common_dep, io_uring_dep]

if 'IO_URING' in static_plugins
    io_uring_backend_lib = static_library('IO_URING',
        io_uring_sources,
        dependencies: io_uring_plugin_deps,
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: false,
        name_prefix: 'libplugin_')
else
    io_uring_backend_lib = shared_library('IO_URING',
        io_uring_sources,
        dependencies: io_uring_plugin_deps,
        cpp_args: ['-fPIC'],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: true,
        name_prefix: 'libplugin_',
        install_dir: plugin_install_dir,
        install_rpath: '$ORIGIN/..')
    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "IO_URING=' + io_uring_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true
        )
    endif
endif

io_uring_backend_interface = declare_dependency(link_with: io_uring_backend_lib)
