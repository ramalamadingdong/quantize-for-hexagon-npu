# Pitfalls: symptom → cause → fix

Real error strings, so you can grep for the one you have.

## Quantization-time (host)

**`ValueError: autodetected range of [0.0, inf] is not finite` / `[nan, nan]` /
`supplied range of [-inf, inf] is not finite`** from
`calibrate.py::collect_absolute_value` → `CalibrationMethod.Percentile` and
`.Entropy` build `np.histogram`s and cannot handle `±inf` attention masks; both
break identically (ORT 1.22). → Use `CalibrationMethod.MinMax` **with**
`op_types_to_quantize=["MatMul"]`. MinMax doesn't raise, but if the inf-bearing
tensors are themselves quantized it silently writes **inf/nan scale and
zero_point**. If you need a histogram method, replace `-inf` with -1e4 first.

**`[ONNXRuntimeError] : 7 : INVALID_PROTOBUF : Load model from
/tmp/ort.quant.*/augmented_model.onnx failed:Protobuf parsing failed.`** → the
**calibrator** (not the quantizer) writes one monolithic augmented model over the
2 GB protobuf limit. → `quantize_static(..., use_external_data_format=True)`;
ensure the fp32 input was also saved with external data; ensure `TMPDIR` has
several GB free.

**`RuntimeError: assertNonNull ... onnx version converter is unable to parse input
model`** (`quantize.py:742 → quant_utils.py:1224 → version_converter.py:39`) →
16-bit QDQ triggered an automatic opset bump to 21, which serializes the whole
>2 GB model to one protobuf string. → Pre-bump `opset_import` to 21 in place and
re-save with `save_as_external_data=True`. `UseQDQContribOps` does **not** dodge
this (ORT skips contrib ops for 16-bit precisely because it plans to bump).
Caveat: an in-place `opset_import` edit skips real op conversion, so it is valid
only when no op has a breaking spec change up to 21 (Clip, Pad, Resize, ReduceSum,
Squeeze moved axes attribute→input) — run `onnx.checker` and compare fp32 outputs
before/after.

**External-data file stuck at ~2× the needed size** → `onnx`'s writer opens the
`.data` file `r+b`, seeks, and **never truncates**, so re-saving over an existing
`.data` leaves stale trailing bytes (observed 3.14 GB where 1.57 GB was needed). →
Delete the old `.data` first (or use a new `location=`), then re-save. Assert the
on-disk size before transferring multi-GB artifacts.

**`Invalid model for HBA quantization algorithm.`** → CLE's preconditions aren't
met. CLE matches only single-branch patterns
(`Conv(r)→BN(r)→act(o)→Conv(r)→BN(r)→act(o)` and the DepthwiseConv variant) with
**no matched tensor consumed by more than one op**, only **Relu** activations
(Relu6 is auto-rewritten; anything else makes CLE skip the preceding conv), and it
needs detectable batchnorm beta/gamma in the pre-conversion model. The message
means CLE only **partially** ran and accuracy loss is likely. → Don't reach for
CLE on a transformer or a BN-stripped export.

## QAIRT tooling

**Exit 0 but the DLC isn't quantized** → you omitted `--input_list`; every other
quantization option is silently ignored (`Skipping quantization, no input_list
provided`). The docs call one of `--input_list` / `--enable_float_fallback`
"mandatory" but it is **not enforced**. → Pass `--input_list` and assert the
output is quantized.

**`Skipping calibration` on a DLC you wanted re-calibrated** → the input DLC is
already fully quantized. → Add `--ignore_quantization_overrides` alongside
`--input_list`, or convert with `--export_format=DLC_STRIP_QUANT` first.

**`Invalid combination: --input_list and --enable_float_fallback cannot be
provided at the same time.`** (legacy tool says `--float_fallback`) → mutually
exclusive. `--enable_float_fallback` also conflicts with
`--ignore_quantization_overrides` (`Cannot determine quantization encodings for any
tensor`).

**`Invalid combination: --bias_bw and --bias_bitwidth cannot be provided at the
same time.`** → legacy alias + modern flag on the legacy converters. On
`qairt-quantizer` the `_bw` names don't exist at all. The guard is a literal
`in sys.argv` test, so it won't fire for args passed via a config/response file.

**`Invalid combination: legacy quantizer options: --act_quantizer or
--param_quantizer cannot be combined with --act_quantizer_calibration or
--param_quantizer_calibration`** → pick the modern pair. Same `sys.argv` caveat.

**A wrong-target context binary with no error at all** → you passed the HTP
options JSON directly to `qnn-context-binary-generator --config_file`. No backend
extensions load, so `dsp_arch` / `soc_model` are ignored. → Pass the
`{"backend_extensions": {...}}` wrapper.

**A config key that does nothing** → HTP backend-extension configs do **not**
reject unknown keys; `libQnnHtpNetRunExtensions.so` only logs `Unknown Key: ...
provided in HTP extensions config`. Also check section placement
(`extended_udma` / `weight_sharing_enabled` are **context**-level) and that
`graph_names` isn't still the sample's placeholder `graph_name_1`, which never
binds. → Grep the embedded schema:
`strings lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so | grep -A40 'schema validation'`.

**`ImportError: libscipy_openblas64_-<hash>.so: cannot open shared object file`**
from `qairt-quantizer`/`qairt-converter` → broken user-site numpy on the system
python3. `PYTHONNOUSERSITE=1` alone does **not** work (no system numpy fallback).
→ Clean python3.10 venv with working numpy.

**`qnn-context-binary-generator: error while loading shared libraries:
libc++.so.1`** → the SDK ships **no** `libc++.so.1`; pointing `LD_LIBRARY_PATH` at
the SDK cannot fix it. → `sudo apt install clang libc++-dev libc++abi-dev`.

**`ModuleNotFoundError: No module named 'yaml'`** from a converter →
`pip install pyyaml`. Note `check-python-dependency` covers the SDK's own pins but
**not** `onnx`/`onnxruntime`/`onnxsim` — install those separately.

**`qairt-net-run: command not found`** → the Dragonwing docs' `qairt-net-run
--version` is a typo; no such binary exists in any ABI dir of 2.46.0.260424. →
`qnn-net-run --version`.

## Device: creation and loading

**`Failed to create device: 14001` / `Device Creation failure` /
`QNN_DEVICE_ERROR_INVALID_CONFIG`** (14001 = `QNN_MIN_ERROR_DEVICE` 14000 + 1) →
on Ubuntu-on-Dragonwing this is almost always a **missing FastRPC runtime**, not a
bad library path and not a wrong `soc_model`/`htp_arch`. Hand-extracting the QAIRT
debs (`dpkg-deb -x` + `cp` + `ldconfig`) is **not enough**: the backend loads and
reports its version, then device creation fails. →

```bash
sudo apt-get install -y qcom-fastrpc1 qcom-fastrpc-dev qcom-libdmabufheap-dev
#   (or: libqnn1 qairt-libs qairt-dsp-binaries — these pull qcom-fastrpc1)
sudo ldconfig && sudo systemd-sysusers
getent group fastrpc || sudo groupadd --system fastrpc
sudo usermod -aG fastrpc "$USER" && sudo reboot
# verify:
ls /usr/lib/aarch64-linux-gnu/libcdsprpc.so /usr/lib/dsp/cdsp/fastrpc_shell_3 /dev/fastrpc-cdsp
systemctl status cdsprpcd; id | grep fastrpc
```

Does **not** apply to VFIO/UMD DSP platforms (no kernel fastrpc node). If 14001
persists after that, check `ADSP_LIBRARY_PATH`/`CDSP_LIBRARY_PATH` and SoC-arch
mismatch next. On QCS8275, ORT also needs `htp_arch:"75"` or device create fails
with `QNN_DEVICE_ERROR_INVALID_CONFIG` regardless.

**`DspTransport.openSession qnn_open failed, 0x80000406` → `IDspTransport: Unable
to load lib 0x80000406` → `DspTransport.getHandle failed, error 0x00000008` →
`Failed to load skel, error: 1002` → `Transport layer setup failed: 14001`** →
`0x80000406` = `AEE_EUNABLETOLOAD` (`DSP_AEE_EOFFSET 0x80000400 + 6`): the **skel
failed to load after the PD was already created**, mis-surfaced as a transport
failure. **Do not chase the transport or blame the BSP/VFIO.** → Check in order:
(1) is `libQnnHtpV<NN>Skel.so` in a searched dir — on CDSP targets that is
`CDSP_LIBRARY_PATH`, `;`-separated (the fix on SA8797P was
`export CDSP_LIBRARY_PATH=/usr/lib/rfsa/adsp/hexagon-v81`); (2) does the filename
match what the stub requests; (3) is the skel loadable at all — wrong arch or
unresolved symbols give the same code.

**`QNN_COMMON_ERROR_INCOMPATIBLE_BINARIES` / `QNN_CONTEXT_ERROR_BINARY_VERSION` /
`QNN binary blob version check failed`** at load → host prepare SDK newer than the
device runtime, or wrong arch. Silent at compile time. → Recompile with a prepare
SDK **≤** the device runtime and the correct `dsp_arch`. Older-binary-on-newer-
runtime is fine (init acceleration auto-disables). Also
`Loading a context binary generated for a different HTP arch may give
indeterminate result.`

**`QNN_CONTEXT_ERROR_NOT_FINALIZED` (0x138b = 5003)** from `QnnContext_getBinary`
→ the graphs in that context were **not finalized**. This is **not** proof that
on-device context serialization is impossible — the SDK ships an aarch64
`qnn-context-binary-generator` (`bin/aarch64-ubuntu-gcc9.4`,
`bin/aarch64-oe-linux-gcc11.2`), and ORT QNN EP generates context binaries
on-device by design. → Finalize the graphs. Decode QNN error ints against
`QnnCommon.h` before inferring a capability limit.

## Device: execution and memory

**`QNN graph execute error. Error code: 6001`** (ORT surfaces it as
`FAIL : Non-zero status code returned while running QNN_<id> node`) → a **generic**
HTP graph-execute/resource failure, not a dedicated OOM code. Three observed
causes, in the order worth checking:

1. **VTCM exhaustion from co-resident contexts.** Measured on QCS8275 (v75, 8 MB
   accepted ceiling): a 3-context action-head loop 6001'd under
   `htp_performance_mode:"burst"` and ran fine at default (2054 ms), with and
   without `htp_graph_finalization_optimization_mode:"3"` — so burst, not O3, was
   the trigger. The interaction is **unexplained and n=1** (VTCM is sized by the
   separate `vtcm_mb` knob; perf mode is nominally DCVS/power), so treat "burst
   reserves more VTCM" as an empirical observation, not a mechanism. → Use burst
   only for load→run→free stages; default mode for co-resident sessions; or
   pre-compile EPContext binaries so sessions run one at a time.
2. **A single graph too heavy to execute.** A monolithic 2B decoder reported
   `read_total_bytes=17791440896` (~17.8 GB) at graph prep and 6001'd; 4-layer
   chunks at ~0.5 GB each ran. Use `read_total_bytes` as the size diagnostic. Do
   not confuse this with host DDR: on a 10 GB board, multi-GB INT16 graphs plus
   ORT sessions get **SIGKILLed by the OOM killer** — that is a DDR limit, and
   tuning `vtcm_mb` won't help.
3. **Stale/dirty DSP state.** A run killed mid-HTP-execute leaves DSP contexts
   allocated and produces a transient 6001 on the next large graph (observed once,
   on a ViT). → Kill lingering processes, smoke-test a small known-good model to
   confirm DSP health, re-run clean. Avoid SIGKILL mid-execute; on some boards you
   cannot reboot to reset.

**`Failed to initialize qnn_model_wrapper`** at session creation → requested
`vtcm_mb` exceeds what the backend accepts for the detected SoC (rejected before
graph prep). → Sweep `vtcm_mb` to find the ceiling (8 MB on QCS8275/v75 with
QAIRT 2.46 + ORT-QNN 1.23).

**`register memHandle failed, calculated size N bytes, supported size is less than
2GB`** → one buffer too large. `rpcmem_alloc(int heapid, uint32 flags, int size)`
takes a **signed int**, so no single buffer reaches 2 GiB. → Split so no
graph/weight buffer exceeds 2 GB.

**`failed to allocate iova for: <n>.cdsp0_cb, ret: -12` (ENOMEM)** → the cDSP is a
32-bit processor with a 4 GB VA space of which only ~3.75 GB is usable; you
exceeded the near-memory IOVA window. → Split into parts ≤ ~1.5 GB. AI Hub ships
Qwen3-8B W4A16 as **5 parts, largest 1.55 GB, with no `extended_udma`** —
splitting, not far mapping, is the primary remedy.

**`Extended UDMA not enabled or not supported by platform for graph <id>, skipping
far mapping allocation`** → far mapping off. `extended_udma` is **v81+**,
**prepare-time only** (`Ignoring context config to use extended UDMA which is
applicable to prepare only` means you set it too late), belongs in the `context`
block as JSON `true` (not `1`), and its log lines only appear at `--log_level
verbose`. The same "skipping far mapping" tail can also come from `Core spill-fill
buffer size X MB is below far mapping minimum Y MB`.

**A benchmark you can't trust** → the NPU/CDSP is effectively an **exclusive**
resource, and the limit is DSP address space, not host RAM (mapping fails with
tens of GB free, as root, while each binary loads fine alone). Measured on one
device: ~1523 MiB concurrently-mapped weights clean, ~1941 MiB `Failed to map
buffer`, 2875 MiB hard fail with `err 1002` = `QNN_COMMON_ERROR_MEM_ALLOC`. It is
not a pure volume limit — repeated map/unmap **fragments** the space, so identical
totals succeed or fail depending on history. Two processes each mapping a ~2.9 GB
bundle contend for the same budget: the second hits 1002 and **both** see inflated
latency. Worst part: QNN reports the mapping failure but still logs `Initialize
Qnn graph from binary file successfully` and hands back a handle producing a
garbage-length output vector. → Enforce a conservative per-device soft budget in
your runner, fail loudly, cross-check output arity against the graph header, and
never benchmark without confirmed exclusive NPU access.

**Latency that includes compilation** → with ORT QNN EP, QNN graph finalize happens
inside `InferenceSession()` construction (measured 147 s ViT / ~150 s DiT vs
~1.2 s inference on QCS8275/ORT 1.23). → Time session creation separately from
`run()`, discard one warmup run (RPC/shared-memory registration + DSP clock ramp),
and report mean **and** max/p95.

**The fastest solo graph is the one that breaks under concurrency** → a
de-partitioned 1-partition DiT was faster alone but 6001'd when held resident
alongside the action encoder and decoder, **even in default perf mode**; the
17-partition build, whose CPU boundaries give a smaller peak working set, fit
3-concurrent. → Optimize partition count for solo latency, then **re-test under
your real concurrency**, and keep both variants. The durable fix is EPContext
caching so sessions load→run→free one at a time. Partition count is not a knob you
set — steer it via quantization config and
`nodes_to_exclude`/`op_types_to_quantize`, and confirm from the QNN EP partition
log.

## ORT QNN EP specifics

**Session "succeeds" but nothing ran on the NPU** → see "Prove it ran on the NPU"
in [ort-qnn-ep.md](ort-qnn-ep.md). `get_providers()` proves nothing.

**`SyntaxError` pasting the ORT QNN EP docs' profiling example** →
`provider_options = [ "k": v, ... ]` is not valid Python, and the same examples
misspell `vtcm_mb` as `"vtcm_mv"`. Fix the brackets, keep the typo, and the option
is **silently ignored** — QNN EP does no unknown-key validation.

**`ERROR: <file> is not a valid wheel filename` / `Invalid wheel filename (wrong
number of parts): 'ortqnn'`** → pip parses distribution/version/ABI from the
PEP-427 filename; you renamed the wheel. → `wget` without `-O`.

**`ensurepip is not available` / `error: externally-managed-environment`** →
`sudo apt install -y python3.12-venv python3-full`, then make a venv. Don't reach
for `--break-system-packages`.

**`pip install onnxruntime-qnn` gives you a Windows wheel, or nothing** → the 1.x
line on PyPI is Windows-only; aarch64 Linux exists only in 2.x
(`manylinux_2_34`, so glibc ≥ 2.34, Python ≥ 3.11) and is a plugin-EP build with a
different registration flow. → Pin the line you're coding against.

## Portability confusion

**A "portable" artifact that isn't** → don't judge portability by extension.

| Artifact | Portable across SoC? |
|---|---|
| `.dlc`, uncached | **Yes** (graph prep happens on device) |
| `.dlc` through `snpe-dlc-graph-prepare --htp_socs` | No — carries a target cache |
| `qnn-model-lib-generator` `.so` | SoC-portable, OS/arch-locked |
| context binary `.bin` | **No** — locked to HTP arch + soc_model + QAIRT version |
| EPContext `*_ctx.onnx` | **No** — an `.onnx` that wraps a compiled context |
| plain QDQ `.onnx`, `.tflite` | Yes |
