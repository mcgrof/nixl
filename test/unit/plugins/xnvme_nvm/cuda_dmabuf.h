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

// Export a CUDA device allocation as a dma-buf, so the harness can drive
// the plugin's VRAM path: the payload stays in GPU memory and the NVMe
// controller DMAs straight into and out of it.
//
// The CUDA driver API is loaded with dlopen rather than linked, so a build
// on a host with no CUDA toolkit still produces this binary and the arm
// simply reports that it is unavailable at run time.

#ifndef NIXL_TEST_CUDA_DMABUF_H
#define NIXL_TEST_CUDA_DMABUF_H

#include <cstdint>
#include <cstdio>
#include <string>

#include <dlfcn.h>

namespace cuda_dmabuf {

// From cuda.h; spelled out so no CUDA headers are needed to build.
constexpr unsigned kHandleTypeDmabufFd = 0x1;
// CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE. This is NOT the default
// for a good reason: it makes the NVIDIA driver take its FORCE_PCIE path,
// which sets skip_iommu and puts raw physical BAR addresses in the
// scatterlist instead of calling nv_dma_map_peer (nv-dmabuf.c:1033, 965).
// With the IOMMU translating, the importing NVMe device then DMAs to
// addresses nothing ever mapped for it, and every command dies in a
// DMAR fault. It is only right when the IOMMU is in passthrough. The
// default mapping asks the driver to map for the peer device properly.
constexpr unsigned kFlagMappingTypePcie = 0x1;

struct Api {
    void *lib = nullptr;
    int (*cuInit)(unsigned) = nullptr;
    int (*cuDeviceGet)(int *, int) = nullptr;
    int (*cuDevicePrimaryCtxRetain)(void **, int) = nullptr;
    int (*cuCtxSetCurrent)(void *) = nullptr;
    int (*cuMemAlloc)(unsigned long long *, size_t) = nullptr;
    int (*cuMemFree)(unsigned long long) = nullptr;
    int (*cuMemsetD8)(unsigned long long, unsigned char, size_t) = nullptr;
    int (*cuMemcpyHtoD)(unsigned long long, const void *, size_t) = nullptr;
    int (*cuMemcpyDtoH)(void *, unsigned long long, size_t) = nullptr;
    int (*cuMemGetHandleForAddressRange)(void *, unsigned long long, size_t, unsigned, unsigned long long) = nullptr;
    int (*cuGetErrorName)(int, const char **) = nullptr;
    int (*cuDeviceGetName)(char *, int, int) = nullptr;
};

inline Api &
api() {
    static Api a;
    return a;
}

inline std::string
errName(int rc) {
    const char *n = nullptr;
    if (api().cuGetErrorName && api().cuGetErrorName(rc, &n) == 0 && n) {
        return n;
    }
    return "CUDA error " + std::to_string(rc);
}

// The PCIe mapping flag that actually took, for the record; -1 until a
// successful export.
inline long &
mappingFlagUsed() {
    static long f = -1;
    return f;
}

inline bool
init(int ordinal, std::string &err) {
    Api &a = api();
    if (a.lib) {
        return true;
    }
    a.lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!a.lib) {
        err = "dlopen(libcuda.so.1): " + std::string(dlerror() ? dlerror() : "failed");
        return false;
    }
#define SYM(n)                                                     \
    do {                                                           \
        *(void **)(&a.n) = dlsym(a.lib, #n);                       \
        if (!a.n) {                                                \
            err = "libcuda.so.1 has no " #n;                       \
            return false;                                          \
        }                                                          \
    } while (0)
    SYM(cuInit);
    SYM(cuDeviceGet);
    SYM(cuDevicePrimaryCtxRetain);
    SYM(cuCtxSetCurrent);
    SYM(cuMemFree);
    SYM(cuMemsetD8);
    SYM(cuMemGetHandleForAddressRange);
    SYM(cuDeviceGetName);
#undef SYM
    // These carry a _v2 suffix in the driver ABI.
    *(void **)(&a.cuMemAlloc) = dlsym(a.lib, "cuMemAlloc_v2");
    *(void **)(&a.cuMemcpyHtoD) = dlsym(a.lib, "cuMemcpyHtoD_v2");
    *(void **)(&a.cuMemcpyDtoH) = dlsym(a.lib, "cuMemcpyDtoH_v2");
    *(void **)(&a.cuGetErrorName) = dlsym(a.lib, "cuGetErrorName");
    if (!a.cuMemAlloc || !a.cuMemcpyHtoD || !a.cuMemcpyDtoH) {
        err = "libcuda.so.1 is missing the _v2 memory entry points";
        return false;
    }

    int rc = a.cuInit(0);
    if (rc != 0) {
        err = "cuInit: " + errName(rc);
        return false;
    }
    int dev = 0;
    rc = a.cuDeviceGet(&dev, ordinal);
    if (rc != 0) {
        err = "cuDeviceGet(" + std::to_string(ordinal) + "): " + errName(rc);
        return false;
    }
    void *ctx = nullptr;
    rc = a.cuDevicePrimaryCtxRetain(&ctx, dev);
    if (rc != 0) {
        err = "cuDevicePrimaryCtxRetain: " + errName(rc);
        return false;
    }
    rc = a.cuCtxSetCurrent(ctx);
    if (rc != 0) {
        err = "cuCtxSetCurrent: " + errName(rc);
        return false;
    }
    char name[256] = {0};
    if (a.cuDeviceGetName(name, sizeof(name), dev) == 0) {
        printf("  cuda: device %d is %s\n", ordinal, name);
    }
    return true;
}

// One device allocation exported as a dma-buf. `ptr` is a device address
// and must never be dereferenced by the CPU.
struct Buf {
    unsigned long long raw = 0; // what cuMemFree needs
    unsigned long long ptr = 0; // the aligned address handed to NIXL
    int fd = -1;
    size_t len = 0;
};

inline bool
alloc(size_t len, size_t align, Buf &out, std::string &err) {
    Api &a = api();
    // Over-allocate so the exported range can start on `align`, which the
    // plugin requires of any registered buffer and O_DIRECT requires of
    // the transfer.
    const size_t pad = align ? align : 4096;
    int rc = a.cuMemAlloc(&out.raw, len + pad);
    if (rc != 0) {
        err = "cuMemAlloc(" + std::to_string(len + pad) + "): " + errName(rc);
        return false;
    }
    out.ptr = (out.raw + pad - 1) & ~(unsigned long long)(pad - 1);
    out.len = len;

    // Default mapping first: it is the one that gets an IOMMU mapping made
    // for the importing device. PCIe-forced is the fallback for hosts that
    // refuse the default (an IOMMU-passthrough setup, typically).
    unsigned long long flags[2] = {0, kFlagMappingTypePcie};
    for (int i = 0; i < 2; i++) {
        int fd = -1;
        rc = a.cuMemGetHandleForAddressRange(&fd, out.ptr, len, kHandleTypeDmabufFd, flags[i]);
        if (rc == 0 && fd >= 0) {
            out.fd = fd;
            if (mappingFlagUsed() < 0) {
                mappingFlagUsed() = (long)flags[i];
                printf("  cuda: dma-buf export used mapping flags=0x%llx (%s)\n",
                       flags[i],
                       flags[i] == 0 ? "default, driver maps for the peer device"
                                     : "FORCE_PCIE, driver skips the IOMMU");
                fflush(stdout);
            }
            return true;
        }
        if (i == 1) {
            err = "cuMemGetHandleForAddressRange: " + errName(rc);
        }
    }
    a.cuMemFree(out.raw);
    out.raw = out.ptr = 0;
    return false;
}

inline void
free(Buf &b) {
    if (b.raw) {
        api().cuMemFree(b.raw);
        b.raw = b.ptr = 0;
    }
}

inline bool
toDevice(unsigned long long dst, const void *src, size_t n) {
    return api().cuMemcpyHtoD(dst, src, n) == 0;
}

inline bool
fromDevice(void *dst, unsigned long long src, size_t n) {
    return api().cuMemcpyDtoH(dst, src, n) == 0;
}

inline bool
memsetDevice(unsigned long long dst, unsigned char v, size_t n) {
    return api().cuMemsetD8(dst, v, n) == 0;
}

} // namespace cuda_dmabuf

#endif // NIXL_TEST_CUDA_DMABUF_H
