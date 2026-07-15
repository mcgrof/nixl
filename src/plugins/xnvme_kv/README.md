# XNVME_KV Plugin

The XNVME_KV plugin is a local storage backend that maps NIXL transfers onto
the NVMe Key-Value command set (KV spec 1.3) using the open
[xNVMe](https://github.com/xnvme/xnvme) library and Linux `io_uring` command
passthrough. It reproduces the observable behavior of the DOCA_MEMOS backend
without requiring CUDA, DOCA, BlueField, ConnectX, or any GPU — the entire
data path is CPU-side and runs against any NVMe namespace that implements the
KV command set, including the software
[vfio-user-kvssd](https://github.com/safl/vfio-user-kvssd) device under
stock QEMU (>= 10.1).

## Architecture

```text
NIXL agent API (DRAM_SEG <-> OBJ_SEG)
  |
  v
nixlXnvmeKvEngine            backend entry points, parameter parsing,
  |                          OBJ_SEG key resolution, registration
  v
nixlXnvmeKv*ProgressEngine   request queueing, submission/retry, completion
  |                          bookkeeping, cancellation; caller-progress or
  |                          progress-thread flavor
  v
nixlXnvmeKvDevIface          narrow seam over xNVMe (xnvme_kv_dev.cpp is the
  |                          ONLY translation unit including libxnvme.h)
  v
xnvme_kvs_store/retrieve/exist -> xnvme async queue -> IORING_OP_URING_CMD
  |
  v
NVMe KV namespace (/dev/ngXnY char device)
```

- `NIXL_WRITE` maps to KV Store, `NIXL_READ` to KV Retrieve, and
  `queryMem` in `actual` mode to KV Exist. One NIXL descriptor produces
  exactly one KV command; there is no coalescing.
- Only `DRAM_SEG` (local) and `OBJ_SEG` (remote) descriptor lists are
  supported. The backend is local: pass your own agent name as the remote
  agent.
- DRAM buffers are handed to `io_uring_cmd` passthrough as-is — no copies,
  no pinning, no registration. A buffer must stay alive until its command
  completes; the engine guarantees it never releases a request with commands
  in flight.
- `deregisterMem` releases registration metadata only; it never deletes
  objects from the device.
- NIXL cannot report per-descriptor partial success: a request's status is
  the sticky first error observed, and independent requests have no ordering
  guarantees.

## Why /dev/ngXnY and io_uring_cmd

KV namespaces are not block devices: Linux exposes namespaces whose command
set the block layer cannot drive as *generic char devices* (`/dev/ngXnY`)
only. Vendor-defined commands (and the KV command set) are issued through
NVMe passthrough. The `io_uring_cmd` path (`IORING_OP_URING_CMD`, kernel >=
5.19, big-SQE/big-CQE rings) is the asynchronous passthrough interface —
the synchronous `NVME_IOCTL_IO64_CMD` ioctl would serialize every command.
The plugin requires the xNVMe `io_uring_cmd` backend by default and does not
fall back to the sync ioctl path for data transfers.

## Build

xNVMe (>= 0.7.5) built with liburing enabled must be discoverable via
pkg-config:

```sh
# xNVMe, CPU-only
git clone https://github.com/xnvme/xnvme.git && cd xnvme
meson setup build --prefix=$PREFIX \
    -Dwith-liburing=enabled -Dwith-spdk=disabled \
    -Dwith-cuda=disabled -Dwith-hip=disabled
meson compile -C build && meson install -C build
export PKG_CONFIG_PATH=$PREFIX/lib/x86_64-linux-gnu/pkgconfig

# NIXL with the plugin (auto-detected; explicit enable errors when missing)
meson setup build
# or: meson setup build -Denable_plugins=XNVME_KV
```

No UCX, CUDA, ROCm, or DOCA is needed for this plugin.

## Test environment: QEMU + vfio-user-kvssd

Real NVMe KV hardware is rare; the supported software target is
`vfu_kvssd`, the vfio-user NVMe-KV device used by xNVMe CI (key <= 16
bytes, value <= 4096 bytes, in-memory backing):

```sh
# 1. Serve a KV SSD on a unix socket (static release binaries available).
vfu_kvssd --socket /tmp/vfu_kvssd.sock --capacity 2G

# 2. Attach to a QEMU >= 10.1 guest. Guest RAM must be SHARED for
#    vfio-user DMA (memory-backend-memfd + share=on; with libvirt use
#    <memoryBacking><source type="memfd"/><access mode="shared"/>).
qemu-system-x86_64 \
  -machine q35,accel=kvm,memory-backend=mem \
  -object memory-backend-memfd,id=mem,size=16G,share=on \
  ... \
  -device '{"driver":"vfio-user-pci","socket":{"type":"unix","path":"/tmp/vfu_kvssd.sock"}}'

# 3. In the guest the namespace appears as a generic char device
#    (e.g. /dev/ng4n1). Verify with xNVMe before involving NIXL:
kvs idfy-ns /dev/ng4n1 --nsid 1 --be io_uring_cmd --admin nvme
kvs store   /dev/ng4n1 --be io_uring_cmd --admin nvme --key hello --value world
kvs exist   /dev/ng4n1 --be io_uring_cmd --admin nvme --key hello
kvs retrieve /dev/ng4n1 --be io_uring_cmd --admin nvme --key hello
kvs delete  /dev/ng4n1 --be io_uring_cmd --admin nvme --key hello
```

Set `VFU_KVSSD_TRACE=1` on the server to trace every NVMe command (opcodes
STORE 0x01, RETRIEVE 0x02, LIST 0x06, DELETE 0x10, EXIST 0x14) and verify
the plugin's command patterns independently of NIXL.

## Backend parameters

| Parameter | Default | Description |
|---|---|---|
| `device_name` | (required) | xNVMe device URI, normally `/dev/ngXnY` |
| `be` | `io_uring_cmd` | xNVMe backend name |
| `nsid` | `0` (unset) | Expected namespace ID, validated against the namespace the URI resolves to (on char-device passthrough the URI *is* the namespace, so this is an expectation check, not an override) |
| `num_tasks` | `128` | Async queue capacity; power of two in [1, 2048] (xNVMe rejects 4096) |
| `nguid` | (unset) | Expected NGUID, 32 hex chars; validated against Identify Namespace (CSI=KV) at init. Not a substitute for opening the right namespace |
| `query_mem_mode` | `assume_success` | `assume_success`: queryMem engages every descriptor without device commands. `actual`: one KV Exist per descriptor |
| `ignore_read_not_found` | `false` | Treat Retrieve `KEY_NOT_EXISTS` as success |

## Key encoding

An NVMe KV key is 1–16 bytes. For each registered `OBJ_SEG` descriptor the
key is derived from `metaInfo`/`devId`:

1. Empty `metaInfo`: the 8 raw native-endian bytes of `devId`.
2. `metaInfo` of even length <= 32 that parses as hex: decoded to binary
   (hex always wins — an all-hex string is never treated as raw bytes).
3. Otherwise, `metaInfo` of <= 16 bytes: used verbatim.
4. Anything longer: registration fails with `NIXL_ERR_INVALID_PARAM`.

## Queueing and progress

Submission backpressure (xNVMe command-context exhaustion, SQ-full
`-EAGAIN`, in-flight == capacity `-EBUSY`) parks the request on a FIFO
pending queue with a resume index; later progress resumes it. New requests
never jump ahead of parked ones. The first *permanent* submission failure
stops the batch: unsubmitted descriptors are abandoned, already-submitted
commands drain, and the request completes with the sticky first-error
status.

An xNVMe queue is not thread-safe, so exactly one of two disciplines is in
force:

- **Caller progress** (default): one mutex serializes all queue operations;
  `checkXfer()` reaps completions, resumes parked requests, and returns the
  request state.
- **Progress thread** (`enableProgTh`): a dedicated thread is the sole owner
  of the queue; `postXfer`/`releaseReqH`/`queryMem` only enqueue lightweight
  records under a short-held mutex, and `checkXfer()` is lock-free.

Note that xNVMe defers ring submission to `xnvme_queue_poke()` by default,
so the engine pokes on every progress step, including right after
`postXfer`.

`releaseReqH` on an incomplete request cancels it: parked work is dropped,
in-flight commands drain, and the handle is freed only after the drain (the
destructors bound the drain at 10 s and deliberately leak rather than free
memory that in-flight commands still reference).

## Differences from DOCA_MEMOS

- The KV engine is the open xNVMe library over `io_uring_cmd`, not the DOCA
  MEMOS SDK over BlueField/CMX; nothing links against DOCA.
- Device limits (max key/value length) come from Identify Namespace
  (CSI=KV) `kvf[0]` rather than DOCA capability queries; note that the
  Linux xNVMe path never marks `csi=KV` on the device handle, so the plugin
  issues the CSI-specific identify itself.
- EXIST results arrive as NVMe completion status (`0x00` = exists, `0x87` =
  not found) rather than DOCA's error-callback statuses. The plugin accepts
  both SCT placements for KV status codes: command-specific (0x1, what
  vfio-user-kvssd and xNVMe's enum use) and generic (0x0, where the base
  spec and SPDK put them).
- `num_tasks` defaults to 128 (not 8192) and is bounded by xNVMe's queue
  capacity limit of 2048.
- The Retrieve actual-value-length check reads completion dword 0 per the
  KV spec; devices that leave it 0 (vfio-user-kvssd does) are exempted from
  the length-mismatch warning instead of warning on every transfer.

## Limitations / unvalidated areas

- Validated end-to-end only against QEMU + vfio-user-kvssd (value <= 4 KiB,
  key <= 16 B, in-memory). Real KV SSD hardware and CMX/BlueField behavior
  (in particular Retrieve length reporting in cdw0, large-value transfers
  above 4 KiB, and device-side error paths like capacity-exceeded) are
  unvalidated.
- QEMU/vfio-user throughput/latency numbers validate semantics and command
  patterns only; they are not representative of real device performance.
- One command per descriptor; no vectored values, no KV List, no Delete
  through the transfer API (deregisterMem intentionally never deletes).
- `queryMem(actual)` waits for its EXIST batch inline (no timeout by
  design; the wait is bounded by device completion).
- GPU (VRAM_SEG) support is out of scope for this backend revision.
