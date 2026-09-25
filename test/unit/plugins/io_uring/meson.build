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

# CPU-only unit tests: the backend and progress engine link against an
# in-memory queue, so neither liburing nor a block device is needed.
gtest_dep = dependency('gtest', main: true, version: '>=1.11.0', required: false)
if gtest_dep.found()
    io_uring_unit_test = executable('io_uring_backend_test',
        ['io_uring_backend_test.cpp', 'io_uring_mock_queue.h'],
        io_uring_backend_sources,
        dependencies: [nixl_dep, nixl_infra, nixl_common_dep, gtest_dep],
        include_directories: [nixl_inc_dirs, utils_inc_dirs,
                              include_directories('../../../../src/plugins/io_uring')],
        cpp_args: ['-DNIXL_UNIT_TEST'],
        install: false)
    test('io_uring_backend_unit_test', io_uring_unit_test,
         timeout: 120, suite: ['unit', 'plugins', 'io_uring'])
else
    message('gtest not found, skipping io_uring unit tests')
endif
