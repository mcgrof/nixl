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

// Device harness for the XNVME_NVM plugin through the NIXL agent API.
//
// Writes deterministic patterns to a caller-owned byte range of one
// namespace, reads them back into pre-poisoned buffers and verifies every
// byte; exercises batches larger than the queue, object reuse across
// generations, and interleaved reads and writes. Emits one JSON line per
// request and per descriptor (with the command split the backend applies)
// so a kernel-side NVMe trace can be joined on (opcode, slba, nlb).
//
// Gated on --device or NIXL_XNVME_NVM_DEVICE; exits with the meson skip
// code (77) when neither is given so CI without a device passes cleanly.

#include <fcntl.h>
#include <getopt.h>
#include <linux/dma-heap.h>
#include <linux/udmabuf.h>

// A hugetlb memfd needs these; older glibc headers do not carry them.
#ifndef MFD_HUGETLB
#define MFD_HUGETLB 0x0004U
#endif
#ifndef MFD_HUGE_2MB
#define MFD_HUGE_2MB (21U << 26)
#endif
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "nixl.h"

#include "cuda_dmabuf.h"
#include "nixl_descriptors.h"
#include "nixl_params.h"

namespace {

struct Options {
    std::string backend = "XNVME_NVM"; // or POSIX: the upstream file reference
    std::string device;
    std::string async = "io_uring_cmd";
    std::string expectSerial;
    std::string expectWwid;
    std::string traceOut;
    uint64_t rangeOffset = 0;
    uint64_t rangeLen = 64ull << 20;
    uint32_t objects = 32;
    uint64_t objectSize = 1ull << 20;
    uint32_t batch = 0; // 0 = all objects in one request
    uint32_t numTasks = 32;
    uint64_t maxIoBytes = 0;
    uint64_t maxIoBytesForce = 0;
    bool hugepage = false; // 2 MiB-aligned buffers advised MADV_HUGEPAGE
    std::string bufferMode = "host"; // host | fixed | dmabuf
    // dmabuf mode only: sysheap (/dev/dma_heap/system), heap:<name> for any
    // other dma-buf heap (a CMA heap, say), udmabuf, udmabuf-huge (a
    // hugetlb memfd, so the exporter hands out 2 MiB folios), or cuda.
    //
    // Every source but cuda is host memory: the dma-buf describes ordinary
    // CPU pages and the transfer is still host DRAM to storage. cuda is the
    // GPU-direct arm -- the payload is a CUDA device allocation exported as
    // a dma-buf, registered as VRAM_SEG, and the drive DMAs into and out of
    // VRAM with no host copy on the data path.
    std::string dmabufSource = "sysheap";
    int cudaDevice = 0; // --dmabuf-source cuda: which GPU to allocate on
    bool expectRefusal = false; // pass when the kernel or device refuses every command
    uint64_t generation = 0;
    uint64_t seed = 0x5eed;
    uint32_t alignment = 4096;
};

int failures = 0;
FILE *trace = nullptr;

void
check(bool ok, const std::string &what) {
    printf("  %s: %s\n", ok ? "OK" : "FAIL", what.c_str());
    if (!ok) {
        failures++;
    }
}

uint64_t
nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

uint64_t
splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Every 8-byte word of object `id` in generation `gen` at object byte
// offset `off` is a function of (seed, id, gen, off) and nothing else.
void
fillPattern(uint8_t *buf, size_t len, const Options &o, uint32_t id, uint64_t gen) {
    for (size_t off = 0; off < len; off += 8) {
        uint64_t w = splitmix64(o.seed ^ (static_cast<uint64_t>(id) << 40) ^ (gen << 56) ^ off);
        memcpy(buf + off, &w, std::min<size_t>(8, len - off));
    }
}

size_t
firstMismatch(const uint8_t *buf, size_t len, const Options &o, uint32_t id, uint64_t gen) {
    for (size_t off = 0; off < len; off += 8) {
        uint64_t w = splitmix64(o.seed ^ (static_cast<uint64_t>(id) << 40) ^ (gen << 56) ^ off);
        const size_t n = std::min<size_t>(8, len - off);
        if (memcmp(buf + off, &w, n) != 0) {
            return off;
        }
    }
    return len;
}

void
poison(uint8_t *buf, size_t len) {
    static const uint8_t p[4] = {0xDE, 0xAD, 0xBE, 0xEF};
    for (size_t i = 0; i < len; i++) {
        buf[i] = p[i & 3];
    }
}

nixl_status_t
pollXfer(nixlAgent &agent, nixlXferReqH *req, uint64_t &done_ns) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    for (;;) {
        nixl_status_t status = agent.getXferStatus(req);
        if (status != NIXL_IN_PROG) {
            done_ns = nowNs();
            return status;
        }
        if (std::chrono::steady_clock::now() > deadline) {
            done_ns = nowNs();
            return NIXL_ERR_UNKNOWN;
        }
        std::this_thread::yield();
    }
}

struct Device {
    uint64_t capacity = 0;
    uint64_t lba = 0;
    uint64_t maxIo = 0;
    std::string serial, wwid;
};

class Harness {
public:
    Harness(const Options &o, nixlAgent &agent, nixlBackendH *backend)
        : o_(o),
          agent_(agent),
          posix_(o.backend == "POSIX"),
          remoteType_(posix_ ? FILE_SEG : BLK_SEG) {
        extra_.backends = {backend};
        if (posix_) {
            // The POSIX plugin takes an fd in devId; O_DIRECT keeps the
            // reference on the same caching terms as the block arms.
            fd_ = open(o_.device.c_str(), O_RDWR | O_CREAT | O_DIRECT, 0644);
            if (fd_ < 0) {
                perror("open");
                exit(1);
            }
        }
        gpu_ = o_.bufferMode == "dmabuf" && o_.dmabufSource == "cuda";
        if (gpu_) {
            std::string err;
            if (!cuda_dmabuf::init(o_.cudaDevice, err)) {
                fprintf(stderr, "cuda dma-buf unavailable: %s\n", err.c_str());
                exit(1);
            }
            gpuBufs_.resize(o_.objects);
            staging_.resize(o_.objectSize);
        }
        bufs_.resize(o_.objects);
        dmabufFds_.assign(o_.objects, -1);
        mapLen_.assign(o_.objects, 0);
        const size_t huge = 2u << 20;
        for (uint32_t i = 0; i < o_.objects; i++) {
            size_t align = o_.alignment, size = o_.objectSize;
            if (o_.hugepage) {
                align = huge;
                size = (size + huge - 1) / huge * huge;
            }
            if (gpu_) {
                // GPU-direct: the payload is device memory. There is no CPU
                // mapping at all, so patterns are staged through a host
                // buffer and copied with the CUDA driver; the I/O path only
                // ever sees the exported dma-buf.
                const size_t map = (size + 4095) / 4096 * 4096;
                std::string err;
                if (!cuda_dmabuf::alloc(map, o_.alignment, gpuBufs_[i], err)) {
                    fprintf(stderr, "cuda dma-buf allocation failed: %s\n", err.c_str());
                    exit(1);
                }
                dmabufFds_[i] = gpuBufs_[i].fd;
                bufs_[i] = reinterpret_cast<uint8_t *>(gpuBufs_[i].ptr);
                mapLen_[i] = map;
            } else if (o_.bufferMode == "dmabuf") {
                // The payload lives in a dma-buf; the harness maps it only
                // to fill and verify patterns, the I/O path never sees the
                // mapping.
                const size_t map = (size + 4095) / 4096 * 4096;
                dmabufFds_[i] = allocDmabuf(o_.dmabufSource, map);
                if (dmabufFds_[i] < 0) {
                    fprintf(stderr, "dma-buf allocation (%s) failed\n", o_.dmabufSource.c_str());
                    exit(1);
                }
                void *p = mmap(nullptr, map, PROT_READ | PROT_WRITE, MAP_SHARED, dmabufFds_[i], 0);
                if (p == MAP_FAILED) {
                    perror("mmap(dma-buf)");
                    exit(1);
                }
                bufs_[i] = static_cast<uint8_t *>(p);
                mapLen_[i] = map;
            } else {
                bufs_[i] = static_cast<uint8_t *>(aligned_alloc(align, size));
                if (!bufs_[i]) {
                    fprintf(stderr, "aligned_alloc failed\n");
                    exit(1);
                }
                if (o_.hugepage && madvise(bufs_[i], size, MADV_HUGEPAGE) != 0) {
                    perror("madvise(MADV_HUGEPAGE)");
                }
            }
            // Touch every page so the backing is decided before the first
            // I/O and a huge-page buffer is one physical extent.
            if (gpu_) {
                cuda_dmabuf::memsetDevice(gpuBufs_[i].ptr, 0, size);
            } else {
                memset(bufs_[i], 0, size);
            }
        }
        nixl_reg_dlist_t dram(gpu_ ? VRAM_SEG : DRAM_SEG);
        for (uint32_t i = 0; i < o_.objects; i++) {
            const std::string meta =
                o_.bufferMode == "dmabuf" ? "dmabuf:" + std::to_string(dmabufFds_[i]) : "";
            const size_t reg_len = o_.bufferMode == "dmabuf" ? mapLen_[i] : o_.objectSize;
            dram.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(bufs_[i]), reg_len, 0, meta));
        }
        nixl_reg_dlist_t blk(remoteType_);
        blk.addDesc(nixlBlobDesc(o_.rangeOffset, posix_ ? 0 : o_.rangeLen, devId(), ""));
        check(agent_.registerMem(dram, &extra_) == NIXL_SUCCESS,
              gpu_ ? "registerMem(VRAM_SEG)" : "registerMem(DRAM_SEG)");
        check(agent_.registerMem(blk, &extra_) == NIXL_SUCCESS,
              posix_ ? "registerMem(FILE_SEG fd)" : "registerMem(BLK_SEG range)");
        dram_ = dram;
        blk_ = blk;
    }

    ~Harness() {
        agent_.deregisterMem(dram_, &extra_);
        agent_.deregisterMem(blk_, &extra_);
        for (uint32_t i = 0; i < bufs_.size(); i++) {
            if (gpu_) {
                // The fd belongs to the exported device allocation.
                if (dmabufFds_[i] >= 0) {
                    close(dmabufFds_[i]);
                }
                cuda_dmabuf::free(gpuBufs_[i]);
            } else if (dmabufFds_[i] >= 0) {
                munmap(bufs_[i], mapLen_[i]);
                close(dmabufFds_[i]);
            } else {
                free(bufs_[i]);
            }
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    static constexpr size_t kHuge2M = 2u << 20;

    static int
    allocDmabuf(const std::string &source, size_t len) {
        if (source == "udmabuf") {
            return udmabufAlloc(len, false);
        }
        if (source == "udmabuf-huge") {
            return udmabufAlloc(len, true);
        }
        if (source == "sysheap") {
            return heapAlloc("system", len);
        }
        if (source.rfind("heap:", 0) == 0) {
            return heapAlloc(source.substr(5), len);
        }
        if (source == "cuda") {
            fprintf(stderr, "cuda dma-bufs are allocated on the device, not here\n");
            return -1;
        }
        fprintf(stderr, "unknown --dmabuf-source '%s'\n", source.c_str());
        return -1;
    }

    static int
    heapAlloc(const std::string &name, size_t len) {
        const std::string path = "/dev/dma_heap/" + name;
        int heap = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (heap < 0) {
            perror(path.c_str());
            return -1;
        }
        struct dma_heap_allocation_data a;
        memset(&a, 0, sizeof(a));
        a.len = len;
        a.fd_flags = O_RDWR | O_CLOEXEC;
        int rc = ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &a);
        close(heap);
        if (rc < 0) {
            perror("DMA_HEAP_IOCTL_ALLOC");
            return -1;
        }
        return static_cast<int>(a.fd);
    }

    static int
    udmabufAlloc(size_t len, bool hugetlb) {
        // A hugetlb memfd must be a multiple of the huge page; the dma-buf is
        // created over the requested length either way.
        unsigned flags = MFD_ALLOW_SEALING;
        size_t memlen = len;
        if (hugetlb) {
            flags |= MFD_HUGETLB | MFD_HUGE_2MB;
            memlen = (len + kHuge2M - 1) / kHuge2M * kHuge2M;
        }
        int mem = memfd_create("nixl-xnvme-nvm", flags);
        if (mem < 0 || ftruncate(mem, memlen) != 0 || fcntl(mem, F_ADD_SEALS, F_SEAL_SHRINK) != 0) {
            perror(hugetlb ? "memfd(hugetlb; is nr_hugepages large enough?)" : "memfd");
            return -1;
        }
        struct udmabuf_create c;
        memset(&c, 0, sizeof(c));
        c.memfd = mem;
        c.flags = UDMABUF_FLAGS_CLOEXEC;
        c.offset = 0;
        c.size = len;
        int dev = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
        if (dev < 0) {
            perror("open(/dev/udmabuf)");
            return -1;
        }
        int fd = ioctl(dev, UDMABUF_CREATE, &c);
        close(dev);
        close(mem);
        if (fd < 0) {
            perror("UDMABUF_CREATE");
        }
        return fd;
    }

    uint64_t
    devId() const {
        return posix_ ? static_cast<uint64_t>(fd_) : 0;
    }

    Device
    queryDevice() {
        Device d;
        if (posix_) {
            // No namespace geometry behind a file: the POSIX plugin submits
            // one read or write per descriptor and the kernel does the rest,
            // so the planned split is one command per object and the trace
            // join key is a file offset, not a device LBA.
            d.maxIo = o_.objectSize;
            d.lba = 512;
            d.serial = "file";
            d.wwid = o_.device;
            dev_ = d;
            return d;
        }
        std::vector<nixl_query_resp_t> resp;
        nixl_status_t st = agent_.queryMem(blk_, resp, &extra_);
        check(st == NIXL_SUCCESS && resp.size() == 1 && resp[0].has_value(),
              "queryMem(BLK_SEG range) reports the namespace");
        if (st == NIXL_SUCCESS && !resp.empty() && resp[0].has_value()) {
            d.capacity = std::stoull(resp[0]->at("capacity_bytes"));
            d.lba = std::stoull(resp[0]->at("lba_bytes"));
            d.maxIo = std::stoull(resp[0]->at("max_io_bytes"));
            d.serial = resp[0]->at("serial");
            d.wwid = resp[0]->at("wwid");
        }
        dev_ = d;
        return d;
    }

    uint64_t
    objectOffset(uint32_t id) const {
        return o_.rangeOffset + static_cast<uint64_t>(id) * o_.objectSize;
    }

    // One request over objects [first, first + count) in the given
    // generation. Returns the request status; writes trace lines.
    nixl_status_t
    transfer(nixl_xfer_op_t op, uint32_t first, uint32_t count, uint64_t gen, const char *tag) {
        nixl_xfer_dlist_t local(gpu_ ? VRAM_SEG : DRAM_SEG);
        nixl_xfer_dlist_t remote(remoteType_);
        for (uint32_t i = first; i < first + count; i++) {
            if (op == NIXL_WRITE) {
                fillPattern(scratch(i), o_.objectSize, o_, i, gen);
            } else {
                poison(scratch(i), o_.objectSize);
            }
            commitScratch(i);
            local.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(bufs_[i]), o_.objectSize, 0));
            remote.addDesc(nixlBasicDesc(objectOffset(i), o_.objectSize, devId()));
        }

        nixlXferReqH *req = nullptr;
        const std::string agent_name = "XnvmeNvmHarness";
        nixl_status_t st = agent_.createXferReq(op, local, remote, agent_name, req, &extra_);
        if (st != NIXL_SUCCESS) {
            check(false, std::string(tag) + ": createXferReq");
            return st;
        }
        const uint64_t t0 = nowNs();
        st = agent_.postXferReq(req);
        uint64_t t1 = t0;
        if (st == NIXL_IN_PROG || st == NIXL_SUCCESS) {
            st = pollXfer(agent_, req, t1);
        } else {
            t1 = nowNs();
        }
        agent_.releaseXferReq(req);
        requestId_++;

        if (trace) {
            fprintf(trace,
                    "{\"event\":\"request\",\"id\":%u,\"tag\":\"%s\",\"op\":\"%s\",\"gen\":%llu,"
                    "\"descs\":%u,\"bytes\":%llu,\"t_post_ns\":%llu,\"t_done_ns\":%llu,"
                    "\"status\":%d}\n",
                    requestId_,
                    tag,
                    op == NIXL_WRITE ? "write" : "read",
                    (unsigned long long)gen,
                    count,
                    (unsigned long long)(count * o_.objectSize),
                    (unsigned long long)t0,
                    (unsigned long long)t1,
                    (int)st);
            for (uint32_t i = first; i < first + count; i++) {
                const uint64_t off = objectOffset(i);
                fprintf(trace,
                        "{\"event\":\"desc\",\"request\":%u,\"object\":%u,\"offset\":%llu,"
                        "\"len\":%llu,\"cmds\":[",
                        requestId_,
                        i,
                        (unsigned long long)off,
                        (unsigned long long)o_.objectSize);
                // The split the backend applies (DESIGN.md section 4).
                bool firstCmd = true;
                for (uint64_t done = 0; done < o_.objectSize && dev_.maxIo && dev_.lba;
                     done += dev_.maxIo) {
                    const uint64_t bytes = std::min<uint64_t>(dev_.maxIo, o_.objectSize - done);
                    fprintf(trace,
                            "%s[%llu,%llu]",
                            firstCmd ? "" : ",",
                            (unsigned long long)((off + done) / dev_.lba),
                            (unsigned long long)(bytes / dev_.lba - 1));
                    firstCmd = false;
                }
                fprintf(trace, "]}\n");
            }
            fflush(trace);
        }
        lastElapsedNs_ = t1 - t0;
        return st;
    }

    bool
    verify(uint32_t first, uint32_t count, uint64_t gen, const char *tag) {
        for (uint32_t i = first; i < first + count; i++) {
            const size_t bad = firstMismatch(readBack(i), o_.objectSize, o_, i, gen);
            if (bad != o_.objectSize) {
                printf("  FAIL: %s: object %u gen %llu first mismatch at byte %zu\n",
                       tag,
                       i,
                       (unsigned long long)gen,
                       bad);
                failures++;
                return false;
            }
        }
        return true;
    }

    uint64_t lastElapsedNs_ = 0;

private:
    // On the GPU arm the payload has no CPU mapping, so patterns are built
    // in a host staging buffer and copied across; everywhere else these
    // resolve to the payload itself and cost nothing.
    uint8_t *
    scratch(uint32_t i) {
        return gpu_ ? staging_.data() : bufs_[i];
    }

    void
    commitScratch(uint32_t i) {
        if (gpu_ && !cuda_dmabuf::toDevice(gpuBufs_[i].ptr, staging_.data(), o_.objectSize)) {
            fprintf(stderr, "cuMemcpyHtoD failed for object %u\n", i);
            exit(1);
        }
    }

    const uint8_t *
    readBack(uint32_t i) {
        if (!gpu_) {
            return bufs_[i];
        }
        if (!cuda_dmabuf::fromDevice(staging_.data(), gpuBufs_[i].ptr, o_.objectSize)) {
            fprintf(stderr, "cuMemcpyDtoH failed for object %u\n", i);
            exit(1);
        }
        return staging_.data();
    }

    const Options &o_;
    nixlAgent &agent_;
    nixl_opt_args_t extra_;
    bool posix_ = false;
    nixl_mem_t remoteType_ = BLK_SEG;
    int fd_ = -1;
    nixl_reg_dlist_t dram_{DRAM_SEG};
    nixl_reg_dlist_t blk_{BLK_SEG};
    std::vector<uint8_t *> bufs_;
    std::vector<int> dmabufFds_;
    std::vector<size_t> mapLen_;
    bool gpu_ = false;
    std::vector<cuda_dmabuf::Buf> gpuBufs_;
    std::vector<uint8_t> staging_;
    Device dev_;
    uint32_t requestId_ = 0;
};

void
report(const char *what, uint64_t bytes, uint64_t ns) {
    printf("  %s: %llu bytes in %.3f ms (%.1f MiB/s, harness clock, includes submission)\n",
           what,
           (unsigned long long)bytes,
           ns / 1e6,
           ns ? bytes / 1048576.0 / (ns / 1e9) : 0.0);
}

int
run(const Options &o) {
    nixlAgentConfig cfg;
    cfg.useProgThread = false;
    nixlAgent agent("XnvmeNvmHarness", cfg);

    nixl_b_params_t params;
    if (o.backend == "POSIX") {
        params = {{"use_uring", "true"}};
    } else {
        params = {
            {"device_name", o.device},
            {"async", o.async},
            {"num_tasks", std::to_string(o.numTasks)},
        };
        if (o.maxIoBytes) {
            params["max_io_bytes"] = std::to_string(o.maxIoBytes);
        }
        if (o.maxIoBytesForce) {
            params["max_io_bytes_force"] = std::to_string(o.maxIoBytesForce);
        }
        params["buffer_mode"] = o.bufferMode;
        if (!o.expectSerial.empty()) {
            params["expect_serial"] = o.expectSerial;
        }
        if (!o.expectWwid.empty()) {
            params["expect_wwid"] = o.expectWwid;
        }
    }

    nixlBackendH *backend = nullptr;
    nixl_status_t st = agent.createBackend(o.backend, params, backend);
    check(st == NIXL_SUCCESS && backend, "createBackend(" + o.backend + ")");
    if (st != NIXL_SUCCESS) {
        return 1;
    }

    Harness h(o, agent, backend);
    const Device d = h.queryDevice();
    printf("  device: serial='%s' wwid='%s' capacity=%llu lba=%llu max_io_bytes=%llu\n",
           d.serial.c_str(),
           d.wwid.c_str(),
           (unsigned long long)d.capacity,
           (unsigned long long)d.lba,
           (unsigned long long)d.maxIo);
    if (trace) {
        fprintf(trace,
                "{\"event\":\"device\",\"backend\":\"%s\",\"buffer_mode\":\"%s\","
                "\"join_key\":\"%s\","
                "\"device\":\"%s\",\"async\":\"%s\",\"serial\":\"%s\","
                "\"wwid\":\"%s\",\"capacity_bytes\":%llu,\"lba_bytes\":%llu,"
                "\"max_io_bytes\":%llu,\"range_offset\":%llu,\"range_len\":%llu,"
                "\"objects\":%u,\"object_size\":%llu,\"num_tasks\":%u,\"seed\":%llu}\n",
                o.backend.c_str(),
                o.bufferMode.c_str(),
                o.backend == "POSIX" ? "file_offset" : "device_lba",
                o.device.c_str(),
                o.async.c_str(),
                d.serial.c_str(),
                d.wwid.c_str(),
                (unsigned long long)d.capacity,
                (unsigned long long)d.lba,
                (unsigned long long)d.maxIo,
                (unsigned long long)o.rangeOffset,
                (unsigned long long)o.rangeLen,
                o.objects,
                (unsigned long long)o.objectSize,
                o.numTasks,
                (unsigned long long)o.seed);
    }

    const uint32_t batch = o.batch ? std::min(o.batch, o.objects) : o.objects;
    const uint64_t gen = o.generation;

    if (o.expectRefusal) {
        // Failure injection: every command is expected to be refused by
        // the kernel or the device. The request must fail with the
        // backend error, the handle must be releasable, and the backend
        // must still take and fail a second request afterwards.
        nixl_status_t first = h.transfer(NIXL_WRITE, 0, batch, gen, "refused-write");
        check(first == NIXL_ERR_BACKEND, "refused write reported as NIXL_ERR_BACKEND");
        nixl_status_t second = h.transfer(NIXL_READ, 0, batch, gen, "refused-read");
        check(second == NIXL_ERR_BACKEND,
              "refused read after a refused write reported the same way");
        return failures ? 1 : 0;
    }

    // 1. Write every object in batches, read every object back, verify.
    uint64_t wr_ns = 0, rd_ns = 0;
    bool ok = true;
    for (uint32_t f = 0; f < o.objects && ok; f += batch) {
        const uint32_t n = std::min(batch, o.objects - f);
        ok = h.transfer(NIXL_WRITE, f, n, gen, "write-all") == NIXL_SUCCESS;
        wr_ns += h.lastElapsedNs_;
    }
    check(ok, "write all objects");
    report("write", static_cast<uint64_t>(o.objects) * o.objectSize, wr_ns);
    ok = true;
    for (uint32_t f = 0; f < o.objects && ok; f += batch) {
        const uint32_t n = std::min(batch, o.objects - f);
        ok = h.transfer(NIXL_READ, f, n, gen, "read-all") == NIXL_SUCCESS;
        rd_ns += h.lastElapsedNs_;
        ok = ok && h.verify(f, n, gen, "read-all");
    }
    check(ok, "read all objects back byte-exact into poisoned buffers");
    report("read", static_cast<uint64_t>(o.objects) * o.objectSize, rd_ns);

    // 2. Reuse: rewrite the odd objects in the next generation, then read
    //    even and odd objects and expect each in its own generation.
    ok = true;
    for (uint32_t i = 1; i < o.objects && ok; i += 2) {
        ok = h.transfer(NIXL_WRITE, i, 1, gen + 1, "rewrite-odd") == NIXL_SUCCESS;
    }
    check(ok, "rewrite odd objects in the next generation");
    ok = true;
    for (uint32_t i = 0; i < o.objects && ok; i++) {
        const uint64_t g = (i & 1) ? gen + 1 : gen;
        ok = h.transfer(NIXL_READ, i, 1, g, "read-mixed") == NIXL_SUCCESS &&
            h.verify(i, 1, g, "read-mixed");
    }
    check(ok, "even objects keep the old generation, odd objects carry the new one");

    // 3. Interleave: a write of the first half and a read of the second
    //    half posted back to back, both must complete correctly.
    if (o.objects >= 2) {
        const uint32_t half = o.objects / 2;
        ok = h.transfer(NIXL_WRITE, 0, half, gen + 2, "interleave-write") == NIXL_SUCCESS;
        ok = ok &&
            h.transfer(NIXL_READ, half, o.objects - half, 0, "interleave-read") == NIXL_SUCCESS;
        for (uint32_t i = half; i < o.objects && ok; i++) {
            ok = h.verify(i, 1, (i & 1) ? gen + 1 : gen, "interleave-read");
        }
        ok = ok && h.transfer(NIXL_READ, 0, half, 0, "interleave-verify") == NIXL_SUCCESS &&
            h.verify(0, half, gen + 2, "interleave-verify");
        check(ok, "interleaved write and read complete byte-exact");
    }

    // 4. A batch above the queue capacity in one request parks and resumes.
    if (o.objects > o.numTasks) {
        ok = h.transfer(NIXL_READ, 0, o.objects, 0, "read-over-capacity") == NIXL_SUCCESS;
        for (uint32_t i = 0; i < o.objects && ok; i++) {
            ok = h.verify(i,
                          1,
                          i < o.objects / 2 ? gen + 2 : ((i & 1) ? gen + 1 : gen),
                          "read-over-capacity");
        }
        check(ok, "one request larger than num_tasks completes byte-exact");
    }

    return failures ? 1 : 0;
}

void
usage(const char *prog) {
    fprintf(stderr,
            "usage: %s --device PATH [--backend XNVME_NVM|POSIX]\n"
            "  [--async io_uring_cmd|io_uring|libaio|posix|thrpool]\n"
            "  [--range-offset BYTES] [--range-len BYTES] [--objects N] [--object-size BYTES]\n"
            "  [--batch N] [--num-tasks N] [--max-io-bytes BYTES] [--max-io-bytes-force BYTES]\n"
            "  [--hugepage] [--buffer-mode host|fixed|dmabuf]\n"
            "  [--dmabuf-source sysheap|heap:<name>|udmabuf|udmabuf-huge|cuda]\n"
            "  [--cuda-device N]   GPU ordinal for --dmabuf-source cuda\n"
            "  [--expect-refusal] [--generation N] [--seed N]\n"
            "  [--expect-serial S] [--expect-wwid W] [--trace-out FILE]\n",
            prog);
}

} // namespace

int
main(int argc, char **argv) {
    Options o;
    if (const char *env = getenv("NIXL_XNVME_NVM_DEVICE")) {
        o.device = env;
    }

    static const struct option longopts[] = {
        {"backend", required_argument, nullptr, 'B'},
        {"device", required_argument, nullptr, 'd'},
        {"async", required_argument, nullptr, 'a'},
        {"range-offset", required_argument, nullptr, 'o'},
        {"range-len", required_argument, nullptr, 'l'},
        {"objects", required_argument, nullptr, 'n'},
        {"object-size", required_argument, nullptr, 's'},
        {"batch", required_argument, nullptr, 'b'},
        {"num-tasks", required_argument, nullptr, 'q'},
        {"max-io-bytes", required_argument, nullptr, 'm'},
        {"max-io-bytes-force", required_argument, nullptr, 'F'},
        {"hugepage", no_argument, nullptr, 'H'},
        {"buffer-mode", required_argument, nullptr, 'M'},
        {"dmabuf-source", required_argument, nullptr, 'D'},
        {"cuda-device", required_argument, nullptr, 'G'},
        {"expect-refusal", no_argument, nullptr, 'R'},
        {"generation", required_argument, nullptr, 'g'},
        {"seed", required_argument, nullptr, 'S'},
        {"expect-serial", required_argument, nullptr, 'E'},
        {"expect-wwid", required_argument, nullptr, 'W'},
        {"trace-out", required_argument, nullptr, 't'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };
    int c;
    while ((c = getopt_long(
                argc, argv, "B:d:a:o:l:n:s:b:q:m:F:HM:D:G:Rg:S:E:W:t:h", longopts, nullptr)) != -1) {
        switch (c) {
        case 'B':
            o.backend = optarg;
            break;
        case 'd':
            o.device = optarg;
            break;
        case 'a':
            o.async = optarg;
            break;
        case 'o':
            o.rangeOffset = strtoull(optarg, nullptr, 0);
            break;
        case 'l':
            o.rangeLen = strtoull(optarg, nullptr, 0);
            break;
        case 'n':
            o.objects = strtoul(optarg, nullptr, 0);
            break;
        case 's':
            o.objectSize = strtoull(optarg, nullptr, 0);
            break;
        case 'b':
            o.batch = strtoul(optarg, nullptr, 0);
            break;
        case 'q':
            o.numTasks = strtoul(optarg, nullptr, 0);
            break;
        case 'm':
            o.maxIoBytes = strtoull(optarg, nullptr, 0);
            break;
        case 'F':
            o.maxIoBytesForce = strtoull(optarg, nullptr, 0);
            break;
        case 'H':
            o.hugepage = true;
            break;
        case 'M':
            o.bufferMode = optarg;
            break;
        case 'D':
            o.dmabufSource = optarg;
            break;
        case 'G':
            o.cudaDevice = atoi(optarg);
            break;
        case 'R':
            o.expectRefusal = true;
            break;
        case 'g':
            o.generation = strtoull(optarg, nullptr, 0);
            break;
        case 'S':
            o.seed = strtoull(optarg, nullptr, 0);
            break;
        case 'E':
            o.expectSerial = optarg;
            break;
        case 'W':
            o.expectWwid = optarg;
            break;
        case 't':
            o.traceOut = optarg;
            break;
        default:
            usage(argv[0]);
            return 2;
        }
    }
    if (o.device.empty()) {
        printf("no --device and NIXL_XNVME_NVM_DEVICE unset; skipping XNVME_NVM device test\n");
        return 77; // meson skip
    }
    if (o.bufferMode != "host" && o.bufferMode != "fixed" && o.bufferMode != "dmabuf") {
        fprintf(stderr, "buffer-mode must be host, fixed or dmabuf\n");
        return 2;
    }
    if (o.backend != "XNVME_NVM" && o.backend != "POSIX") {
        fprintf(stderr, "backend must be XNVME_NVM or POSIX\n");
        return 2;
    }
    if (o.objects == 0 || o.objectSize == 0 || o.objectSize % 8 != 0 ||
        static_cast<uint64_t>(o.objects) * o.objectSize > o.rangeLen) {
        fprintf(stderr, "objects * object_size must fit in range_len and be a multiple of 8\n");
        return 2;
    }
    if (!o.traceOut.empty()) {
        trace = fopen(o.traceOut.c_str(), "w");
        if (!trace) {
            perror("trace-out");
            return 2;
        }
    }

    printf("=== %s device harness: %s async=%s objects=%u size=%llu range=[%llu,+%llu) ===\n",
           o.backend.c_str(),
           o.device.c_str(),
           o.async.c_str(),
           o.objects,
           (unsigned long long)o.objectSize,
           (unsigned long long)o.rangeOffset,
           (unsigned long long)o.rangeLen);
    const int rc = run(o);
    if (trace) {
        fclose(trace);
    }
    printf(failures ? "%s device harness: FAIL (%d)\n" : "%s device harness: PASS\n",
           o.backend.c_str(),
           failures);
    return rc;
}
