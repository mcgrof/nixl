# NIXL IO_URING Plugin

This plugin moves bytes between an agent's host memory (`DRAM_SEG`) and a byte
range on local storage, using io_uring. The storage side is a block device or a
preallocated regular file, addressed as a flat byte range through `BLK_SEG`.

## Why this exists alongside POSIX and GDS

The POSIX plugin addresses files as `FILE_SEG` and already submits through
io_uring as one of its queue implementations, but it hands the kernel a plain
pointer on every request. GDS drives NVIDIA's cuFile, also over `FILE_SEG`.
GUSLI does address a block device as `BLK_SEG`, but it is a client for an
external GUSLI server and links `libgusli_clnt` from outside this tree.

What is missing between them is a backend that addresses a device or a file as
a byte range using nothing but the kernel and liburing, and that can use
io_uring's registered-buffer table, where the kernel pins a payload buffer once
and later requests name it by slot instead of by address.

## Buffer modes

`buffer_mode` selects how a payload reaches the kernel.

- `host` (default) passes a pointer with every request. Nothing is registered.
- `fixed` registers host memory once and submits the fixed read and write
  variants against its slot.

## A worked example

To write 2 MiB of host memory to the start of an NVMe namespace, a caller
creates the backend with `device_name=/dev/nvme0n1`, `buffer_mode=fixed` and
the drive's `expect_serial`, registers its aligned buffer as `DRAM_SEG`,
registers `(offset 0, length 2 MiB)` as `BLK_SEG`, and issues a `NIXL_WRITE`
between the two descriptor lists.

## Request size

A transfer larger than one request is split rather than refused. The per-request
ceiling is the smallest limit any source names: the queue's
`max_hw_sectors_kb` and the caller's `max_io_bytes`. It is rounded down to the
direct-I/O length granularity. Where no source names a limit, requests are
capped at 1 MiB, which is this backend's own choice and not a claim about the
device.

Limits are read from `/sys/dev/block` keyed on the target's device number, so a
whole disk, one of its partitions, and a file on a filesystem mounted from it
all resolve to the same disk.

## Parameters

| parameter | default | meaning |
|---|---|---|
| `device_name` | required | block device or preallocated regular file |
| `direct` | `true` | open with `O_DIRECT` |
| `num_tasks` | `128` | ring capacity; a power of two in [1, 2048] |
| `buffer_mode` | `host` | `host` or `fixed` |
| `max_io_bytes` | `0` | caller's ceiling; 0 leaves it to the target |
| `max_io_bytes_force` | `0` | override every derived ceiling; warns rather than refuses |
| `expect_serial` | empty | refuse to open unless the target reports this serial |
| `expect_wwid` | empty | refuse to open unless the target reports this wwid |

`expect_serial` and `expect_wwid` are checked before any request is issued. A
device name is ephemeral and is not evidence of which drive was opened; if a
caller is about to write to a drive, naming the drive it expects is the only
thing that makes that safe. Omitting both warns.

## Limits worth knowing before using it

- The target is opened `O_RDWR`, so a read-only device or file cannot be used.
- A regular file is never extended: the addressable range is whatever the file
  already holds, so preallocate it. Preallocate *and* initialize it if you
  intend to read, or a hole will return zeros without reaching the device.
- Alignment is enforced from the target's reported direct-I/O alignment even
  when `direct=false`.
- Progress is driven from `checkXfer()` on the caller's thread. A requested
  NIXL progress thread is acknowledged with a warning and caller progress is
  used.
- `releaseReqH()` drains the transfer's outstanding requests before returning,
  because those requests name the caller's payload buffers.

## Testing

The unit tests link the backend and progress engine against an in-memory queue,
so they need neither liburing nor a block device:

```
meson setup build -Denable_plugins=IO_URING
meson test -C build --suite io_uring
```

Two of them read the host's real `/sys` to check device-number resolution and
skip where the host has no backing block device or no partition. The liburing
layer itself is exercised by running against real storage, not by these tests.
