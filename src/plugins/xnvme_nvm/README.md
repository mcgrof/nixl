# XNVME_NVM Plugin

The XNVME_NVM plugin is a local storage backend that maps NIXL transfers
onto NVMe Read and Write commands against a conventional (block) namespace
through the open [xNVMe](https://github.com/xnvme/xnvme) library. The
remote side of a transfer is a byte range of the namespace, the local side
is host memory. No GPU, CUDA, DOCA or network transport is involved.

It is the conventional-NVM sibling of the `XNVME_KV` plugin: the same
device-seam and progress-engine design, but one NIXL descriptor becomes
one or more Read/Write commands instead of exactly one KV command, and a
descriptor addresses an offset, not a key.

## Descriptor contract

| Side | Type | `addr` | `len` | `devId` | `metaInfo` at registration |
| --- | --- | --- | --- | --- | --- |
| local | `DRAM_SEG` | host virtual address | bytes | ignored | ignored |
| remote | `BLK_SEG` | byte offset from the start of the namespace | bytes | caller's device slot id | empty, or the same path as `device_name` |

- One backend instance opens one namespace (`device_name`). A `BLK_SEG`
  registration with `len == 0` covers the whole namespace; otherwise the
  range must lie inside its capacity. Transfers must lie inside their
  registration and inside the capacity.
- `addr` and `len` on both sides must be multiples of the logical block
  size; the check happens before anything is submitted.
- Descriptor lists pair by index and paired lengths must be equal.
- `NIXL_WRITE` is host to device, `NIXL_READ` is device to host. A request's
  status is the sticky first error; NIXL cannot report per-descriptor
  partial success.
- `queryMem` on a `BLK_SEG` list issues no command: a range inside the
  namespace is reported accessible with `serial`, `wwid`, `capacity_bytes`,
  `lba_bytes` and `max_io_bytes`; a range outside it is reported absent.

## Command formation

Each descriptor is split into commands of at most `max_io_bytes`, the
smallest of: the controller's Identify MDTS, the 16-bit NLB field
(`65536 * lba_bytes`), the kernel's `max_hw_sectors_kb` for the namespace
(read from sysfs when the path is `/dev/ngXnY` or `/dev/nvmeXnY`), for
passthrough the kernel's `max_segments` times the host page size, and the
`max_io_bytes` parameter. The segment term exists because a passthrough
command maps the host buffer as it is and the kernel rejects one that
needs more segments than the queue allows (`EINVAL`) instead of splitting
it; an ordinary host buffer can need one segment per page, so a larger
command only succeeds when the pages happen to be physically contiguous.
The block paths split and merge in the kernel, so there the device may see
commands both smaller and larger than `max_io_bytes`. xNVMe's own
`geo.mdts_nbytes` is logged but not used: for the `nvme` sync interface it
is capped to `(max_segments - 1) * physical_block_size`, a segment model
rather than a device or kernel limit.

A request completes when every command of every descriptor has completed.
A failed command fails the request but the remaining in-flight commands
still drain before the handle can be released. On the ordinary block paths
a command that transfers fewer bytes than requested is a failed command,
never a partial success. Counters (`descs_requested`, `cmds_submitted`,
`cmds_completed_ok`, `cmds_failed`, `cmds_short`, `submit_parked`,
`submit_permanent_errors`, `bytes_ok_read`, `bytes_ok_write`,
`max_inflight`) are logged at teardown and readable from the engine.

## Transports

| `async` | Handle | Kernel path |
| --- | --- | --- |
| `io_uring_cmd` (default) | `/dev/ngXnY` | NVMe passthrough over `IORING_OP_URING_CMD`; commands above `max_hw_sectors_kb` are rejected by the kernel, not split |
| `io_uring` | `/dev/nvmeXnY` or a file | `IORING_OP_READV/WRITEV` with `O_DIRECT` (`direct=true`); the block layer splits at `max_sectors_kb` |
| `libaio` | block device or file | Linux AIO, reference only |
| `posix`, `thrpool` | anything | emulated by xNVMe, development only |

The pairing is checked at init: passthrough needs a namespace character
device, the block paths need a block device or file.

Host buffers are handed to the kernel as-is: no copy, no pinning, no
registration. `buffer_mode` accepts only `host`; `fixed` is refused because
the pinned xNVMe never consumes its `register_buffers` option, and `dmabuf`
is refused until a kernel path for it exists.

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `device_name` | (required) | xNVMe URI of the namespace, block device or file |
| `async` | `io_uring_cmd` | asynchronous command interface, see above |
| `direct` | `true` | `O_DIRECT` for the block and file paths |
| `nsid` | `0` (unset) | expected namespace id, validated against the opened handle |
| `num_tasks` | `128` | queue capacity; power of two in [1, 2048] |
| `max_io_bytes` | `0` (auto) | per-command ceiling; folded into the minimum above |
| `max_io_bytes_force` | `0` (off) | replaces the computed ceiling, bounded only by the NLB field; the kernel or device may refuse the commands. For physically contiguous (huge-page) buffers that can legitimately exceed the segment term, and for failure injection |
| `expect_serial` | (unset) | controller serial that must match sysfs before any command |
| `expect_wwid` | (unset) | namespace WWID that must match sysfs before any command |
| `buffer_mode` | `host` | only `host` is supported |

At init the plugin logs the device type, identity, geometry, every ceiling
source and the resulting `max_io_bytes`. A command that the kernel refuses
is reported with its errno (xNVMe carries it as status code type 7).

## Queueing and progress

One xNVMe queue with one owner. The caller's thread drives submission,
completion reaping and retries from `checkXfer()`; one mutex serializes
every queue operation. Saturation (`-EAGAIN`, `-EBUSY`, no command context)
parks the request FIFO with a resume index; later progress resumes it and
new requests never pass parked ones. xNVMe defers ring submission to
`xnvme_queue_poke()`, so the engine pokes after every submission batch.

A NIXL progress thread is acknowledged with a warning; the backend still
progresses from `checkXfer()` only.

`releaseReqH` on an incomplete request cancels it: parked work is dropped,
in-flight commands drain, and the handle is freed only after the drain.
Destruction bounds the drain at 10 s and then deliberately leaks rather
than free memory the kernel may still DMA into.

## Build and test

xNVMe (>= 0.7.5) with liburing must be discoverable through pkg-config;
see the `XNVME_KV` README for the build recipe. The plugin is auto-detected
or selected with `-Denable_plugins=XNVME_NVM`. Test targets are only
generated for non-release build types.

- `xnvme_nvm_backend_test`: gtest suite over an in-memory mock of the
  device seam; needs no device or xNVMe library.
- `nixl_xnvme_nvm_test`: device harness through the agent API. Skips
  unless `--device` or `NIXL_XNVME_NVM_DEVICE` names a device. Writes
  deterministic patterns to a caller-owned range, reads them back into
  poisoned buffers and verifies every byte; reuses objects across
  generations, interleaves reads and writes, and posts one request larger
  than `num_tasks`. `--trace-out` writes one JSON line per request and per
  descriptor, including the `[slba, nlb]` split the backend applies, so a
  kernel-side trace of `nvme_setup_cmd` can be joined on it.

```sh
nixl_xnvme_nvm_test --device /dev/ng0n1 --async io_uring_cmd \
    --range-offset 0 --range-len $((64<<20)) --objects 32 \
    --object-size $((1<<20)) --num-tasks 8 --expect-serial nixlpd0 \
    --trace-out harness.jsonl
```

## Limitations

- One namespace per backend instance; `devId` is recorded, not a device
  selector.
- No fixed or dma-buf buffers, no GPU memory, no metadata/PI, no Write
  Zeroes, Copy or Dataset Management.
- Caller progress only.
- Write completion is not a durability guarantee: no Flush or FUA is
  issued.
