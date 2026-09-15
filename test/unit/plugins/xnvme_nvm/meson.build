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

# Integration test against a real KV namespace; skips (exit 77) unless
# NIXL_XNVME_NVM_DEVICE is set in the environment.
nixl_xnvme_nvm_app = executable('nixl_xnvme_nvm_test', 'nixl_xnvme_nvm_test.cpp',
                               dependencies: [nixl_dep, nixl_infra, absl_log_dep],
                               include_directories: [nixl_inc_dirs, utils_inc_dirs],
                               install: true)
test('xnvme_nvm_plugin_test', nixl_xnvme_nvm_app, suite: ['unit', 'plugins', 'xnvme_nvm'])


# Focused benchmark (STORE/RETRIEVE/EXIST sweeps); same env gating.
# CPU-only unit tests (mock device, no xNVMe library or KV hardware needed).
gtest_dep = dependency('gtest', main: true, version: '>=1.11.0', required: false)
if gtest_dep.found()
    xnvme_nvm_unit_test = executable('xnvme_nvm_backend_test',
        ['xnvme_nvm_backend_test.cpp', 'xnvme_nvm_mock_dev.h'],
        xnvme_nvm_backend_sources,
        dependencies: [nixl_dep, nixl_infra, nixl_common_dep, gtest_dep],
        include_directories: [nixl_inc_dirs, utils_inc_dirs,
                              include_directories('../../../../src/plugins/xnvme_nvm')],
        cpp_args: ['-DNIXL_UNIT_TEST'],
        install: false)
    test('xnvme_nvm_backend_unit_test', xnvme_nvm_unit_test,
         timeout: 120, suite: ['unit', 'plugins', 'xnvme_nvm'])
else
    message('gtest not found, skipping xnvme_nvm unit tests')
endif
