# Path C — ONNX Runtime QNN EP + EPContext

## Install (aarch64 Linux)

Pin the line you are documenting — the two generations have different APIs:

- **2.x is the maintained line** (Qualcomm-owned, `github.com/onnxruntime/onnxruntime-qnn`;
  2.8.0 on 2026-11-12, built against **QAIRT 2.52** and ORT 1.29, compatible
  with ORT ≥ 1.24.1). It is a **plugin EP**: a separate package that loads
  into a stock `pip install onnxruntime`. Wheels: `manylinux_2_34_aarch64`
  (inference), `manylinux_2_35_x86_64` (AOT compile only, cannot execute),
  `win_arm64`, `win_amd64`; `Requires-Python >=3.11`. `manylinux_2_34` means
  **glibc ≥ 2.34** — fine on Ubuntu 24.04 Dragonwing images, rules out older
  QLI/Yocto.
- **1.x on PyPI is Windows-only** (every wheel 1.17.3–1.24.3 is `win_amd64` /
  `win_arm64`). On aarch64 Linux the 1.x route is the Edge Impulse wheel below.

### 2.x (default where glibc ≥ 2.34)

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install onnxruntime onnxruntime-qnn          # pin both in anything you ship
```

The 2.x aarch64 wheel **bundles the QNN runtime**: `libQnnHtp.so`,
`libQnnSystem.so`, `libQnnHtpPrepare.so`, `libQnnHtpV{68,69,73,75,79,81}Stub.so`
+ `…Skel.so`, `libGenie.so`, all from the QAIRT build it was made with. The EP
warns whenever `ADSP_LIBRARY_PATH` is already set (`Using existing
ADSP_LIBRARY_PATH setting of …, which may cause the HTP backend to fail`) — seen
even when it pointed at the wheel directory, where it was harmless. If you set
it, point it at the wheel dir so host stub and DSP skel come from the same build;
mixing in system skels from a different QAIRT fails the build-id check.

```python
import onnxruntime as ort, onnxruntime_qnn as qnn_ep
ort.register_execution_provider_library("QNNExecutionProvider", qnn_ep.get_library_path())
devs = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
so = ort.SessionOptions()
so.add_provider_for_devices(devs, {"backend_type": "htp"})   # same option keys as 1.x
sess = ort.InferenceSession("model.onnx", sess_options=so)
...
del sess
ort.unregister_execution_provider_library("QNNExecutionProvider")
```

Measured on an IQ-9075 (ORT 1.30 + onnxruntime-qnn 2.6.0, Docker on Ubuntu
24.04): this loads AI Hub `precompiled_qnn_onnx` artifacts on HTP, and the skel
build-id check passes against the wheel's skel.

**Under 2.x, only `add_provider_for_devices` attaches the EP.** Verified on x86_64
(ORT 1.32 + onnxruntime-qnn 2.8.0): even *after* registration, the 1.x
`providers=[("QNNExecutionProvider", opts)]` form silently gives a CPU-only
session (`get_providers()` → `['CPUExecutionProvider']`), and
`so.add_provider("QNNExecutionProvider", …)` raises `Provider configuration is
not supported`. The examples below are written 1.x-style; under 2.x, pass the
same `opts` dict to `add_provider_for_devices`. The maintained option
reference is `docs/execution_providers/QNN-ExecutionProvider.md` in the 2.x
repo, not the onnxruntime.ai page.

### 1.x (glibc < 2.34, or code already written against 1.x)

```bash
sudo apt install -y python3.12-venv python3-full     # else: ensurepip is not available
python3.12 -m venv .venv && . .venv/bin/activate     # else: externally-managed-environment
wget https://cdn.edgeimpulse.com/qc-ai-docs/wheels/onnxruntime_qnn-1.23.0-cp312-cp312-linux_aarch64.whl
pip install onnxruntime_qnn-1.23.0-cp312-cp312-linux_aarch64.whl numpy==1.26.4
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

**Do not rename the wheel** — pip parses name/version/ABI from the PEP-427
filename and rejects `ortqnn.whl` with `Invalid wheel filename (wrong number of
parts)`. Use `wget` without `-O`. Other ORT/Python combos: build from
`edgeimpulse/onnxruntime-qnn-linux-aarch64`.

This 1.x wheel does **not** bundle the QNN backend libs:

```bash
sudo apt-get install -s libqnn1 qairt-libs qairt-dsp-binaries   # -s first: expect "0 upgraded"
sudo apt-get install -y --no-install-recommends libqnn1 qairt-libs qairt-dsp-binaries
```

*(Qualcomm Ubuntu images only. These pull `qcom-fastrpc1`, which is what actually
fixes device-create 14001 — see [pitfalls.md](pitfalls.md).)*

## Device-side libraries and env

For 1.x or for system-installed QNN libs (the 2.x wheel supplies its own, above).
On `LD_LIBRARY_PATH`: `libQnnHtp.so`, `libQnnSystem.so`, `libQnnHtpV<NN>Stub.so`,
`libcdsprpc.so`. On `ADSP_LIBRARY_PATH`: the directory that actually contains
`libQnnHtpV<NN>Skel.so` and its `.cat`.

```bash
NN=75                                       # your dsp_arch, without the 'v'
export ADSP_LIBRARY_PATH="/usr/lib/rfsa/adsp/hexagon-v$NN;/usr/lib/rfsa/adsp;/dsp"
export CDSP_LIBRARY_PATH="/usr/lib/rfsa/adsp/hexagon-v$NN;/usr/share/fastrpc"
export LD_LIBRARY_PATH=/usr/lib:/lib:$LD_LIBRARY_PATH
```

- **Separator is `;`, not `:`** — a wrong separator fails silently.
- Skel dir is image-specific: `/usr/lib/dsp/cdsp` on Qualcomm Ubuntu once
  `qairt-dsp-binaries` is installed, `/usr/lib/rfsa/adsp` on QLI/Yocto,
  `/usr/share/qcom/<soc>/Qualcomm/<board>/dsp/cdsp` in a dpkg-extracted tree.
- On **compute-DSP (CDSP)** automotive / UMD-VFIO targets the loader reads
  `CDSP_LIBRARY_PATH` and **ignores** `ADSP_LIBRARY_PATH` — set both. This was the
  single fix that got Genie working on SA8797P. Documented only in the SDK's
  `docs/QAIRT-Docs/QNN/general/htp/htp_auto_tutorial_3.html`.
- Conditional: `libQnnHtpPrepare.so` only if you finalize graphs on-device (not
  needed when loading a prebuilt EPContext/.bin); `libQnnHtpNetRunExtensions.so`
  only for `qnn-net-run` with a backend-extensions JSON, never for ORT inference.

## Session setup

```python
import onnxruntime as ort
so = ort.SessionOptions()
so.log_severity_level = 3
# Bring-up gate ONLY. Mutually exclusive with listing CPUExecutionProvider.
so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

qnn = {"backend_type": "htp",            # or backend_path: "/usr/lib/libQnnHtp.so" — at most ONE
       "htp_arch": "75",                 # string
       "htp_performance_mode": "burst",
       "vtcm_mb": "8"}
sess = ort.InferenceSession("model_w8a8.onnx", sess_options=so,
                            providers=[("QNNExecutionProvider", qnn)])
```

- **All option values must be strings** — an int silently fails the dict type check.
- `backend_type` (`cpu|gpu|htp|saver`, plus `ir` on current main) is the newer
  alias for `backend_path`. Supplying **both** throws `Only one of backend_type
  and backend_path should be set.` Neither → falls back to HTP with a WARNING.
- Listing `"CPUExecutionProvider"` **and** `disable_cpu_ep_fallback` aborts
  session creation: `INVALID_ARGUMENT : Conflicting session configuration`.
- `htp_performance_mode`: `burst|balanced|default|high_performance|
  high_power_saver|low_balanced|low_power_saver|power_saver|
  sustained_high_performance|extreme_power_saver` (default `default`). 2.x also
  takes it per run: `run_options.add_run_config_entry("qnn.perf_mode", "burst")`.
  An unrecognized value only logs `Htp performance mode
  not valid.` at WARNING and keeps the default — verify your spelling.
- `htp_graph_finalization_optimization_mode`: `""|"0"` (default), `"1"` faster
  prep/less optimal, `"2"`, `"3"` longest prep/most optimal. Applied at graph
  *finalize*, so set it in the session that **generates** the EPContext binary;
  it is a no-op when loading a prebuilt one.
- `vtcm_mb` is parsed with unguarded `std::stoi` and applied only if > 0 (`<= 0`
  logs `Skip invalid vtcm_mb`; non-numeric throws). It is also a graph config, so
  it gets **baked into the binary**.
- `enable_htp_fp16_precision` no longer controls anything that matters: since
  QAIRT 2.35, HTP float ops use fp16 math on float-capable SoCs **regardless** of
  it (1.x defaulted it to `'1'`, 2.x to `'0'`), and `'0'` does **not** give you
  fp32 on HTP. To debug fp32-vs-NPU numerics, compare against CPU EP instead.
  `htp_bf16_enable` is the separate BF16 switch.
- `offload_graph_io_quantization` defaults to **true** (graph-input Q / output DQ
  left to CPU). Setting `disable_cpu_ep_fallback` **forcibly resets it to false**,
  logged only at INFO — so the full-HTP gate silently moves I/O quant onto QNN and
  can change latency. Benchmark with your shipping flag combination.
- `soc_model` is a string (default `"0"` = unknown): a numeric ID, or in 2.x a
  chip-family **name** (`"SM8750"`, case-insensitive) for a non-exhaustive list of
  well-known chips — anything unrecognized needs the number. Set it together with
  `htp_arch` when compiling off-device. A comma-separated list (`"60,88"`) turns
  on multi-SoC FCB compilation (below).
- `htp_bf16_enable` is gated in `GetCapability`: with `soc_model` unknown or < 88,
  QNN EP returns **zero** assigned nodes and only WARNs (`BF16 mode is enabled but
  ...`) — the whole model silently runs on CPU.
- Now documented in the 2.x option reference: `enable_htp_spill_fill_buffer`,
  `htp_bf16_enable`, `enable_vtcm_backup_buffer_sharing` (prefer its alias
  `htp_share_resource_optimization`), `disable_file_mapped_weights` (Windows
  ARM64 only), `qnn_ir_backend_path`, `op_affinity` (JSON pinning op types to a
  backend), `htp_num_cores`, `framework_op_trace` (writes `qnn_op_trace.json`,
  an ONNX-op → QNN-op map including unsupported ops; schema moved
  `compilation_target` under `soc_traces[]`). Still worth verifying against your
  build: `dump_json_qnn_graph` + `json_qnn_graph_dir`, `dump_qnn_ir_dlc`,
  `op_packages`, `extended_udma`, `skip_qnn_version_check`,
  `enable_htp_shared_memory_allocator`.
- **Large-context levers (2.x):**
  - `htp_reused_io_limit_mb` (QAIRT ≥ 2.45): tells HTP how much I/O is really
    live at once. By default QNN assumes every graph's I/O is resident, inflates
    the PD-size estimate, and can fail PD placement with error **1002**.
  - `enable_htp_prepare_and_load=1`: compile and reload the context in one
    session, "bypass[ing] the single process-domain (PD) memory limit imposed by
    the JIT flow". Add `ep.context_enable=1` to also persist `_ctx.onnx` + `.bin`.
    Mutually exclusive with `enable_htp_prepare_only`, and ignored with a warning
    on an already-compiled `_ctx.onnx`.
  - `enable_htp_prepare_only=1` (needs `ep.context_enable=1`): compile and
    write the context, skip inference.
  - `enable_htp_graph_splitting=1` (runtime QAIRT ≥ 2.49, or context creation
    *fails*): the backend splits the graph into separately-prepared subgraphs to
    cut prepare time (`GPE_KWAY_PARTITIONS`, default 8). No effect on an
    already-compiled binary.
- `op_packages` format is
  `<OpType>:<PackagePath>:<InterfaceSymbolName>[:<Target>]`, comma-separated.
  Parsing is lenient — a wrong field count is skipped with only a WARNING
  (`Skip registration`), so a typo silently leaves the op unregistered and the
  node falls off the NPU.
- HTP subsystem restart (SSR) surfaces as `StatusCode::ENGINE_ERROR` from `Run`.
  Recovery is **recreating the InferenceSession**, not retrying `Run`.

**The legacy onnxruntime.ai QNN EP page's examples were broken** (as of 2026-07,
`QNN-ExecutionProvider.md` L491-501, L539-549 in the microsoft/onnxruntime docs;
the maintained 2.x copy no longer has the typo): `provider_options = [ "k": v, ... ]` is not valid Python, and
they misspell VTCM as `"vtcm_mv"`. QNN EP does **no** unknown-key validation, so
fix the brackets, keep the typo, and VTCM stays unset with no warning. Cross-check
names against the `QNN_*` string constants in `qnn_execution_provider.cc`.

## Quantize for QNN EP (on x86_64 — ORT quantization is x86_64-only)

Prefer the QNN-specific helpers over hand-rolled settings:

```python
from onnxruntime.quantization import QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import (
    get_qnn_qdq_config, qnn_preprocess_model)

model_changed = qnn_preprocess_model("model.onnx", "model.preproc.onnx",
                                     fuse_layernorm=True)      # needs opset>=17
src = "model.preproc.onnx" if model_changed else "model.onnx"

qnn_config = get_qnn_qdq_config(src, MyDataReader(),
                                activation_type=QuantType.QUInt16,   # A16W8
                                weight_type=QuantType.QUInt8)
quantize(src, "model.qdq.onnx", qnn_config)
```

- `qnn_preprocess_model` is **recommended, not required** (without it you get a
  `Please consider to run pre-processing` warning). It runs shape inference + ORT
  graph optimization, then fuses `Erf→Gelu`, `ReduceL2→LpNormalization`,
  `Reshape/Transpose→SpaceToDepth`, optionally `ReduceMean→LayerNormalization`
  (skipped with a warning below opset 17), and names every unnamed non-Constant
  node `qnn_preproc_node_N`. It returns True **only if it changed the model**, and
  writes the output only then — hence the ternary. Gotcha: shape inference and
  optimization do **not** set the changed flag, so when no fusion/rename applies
  you fall back to the original and lose that cleanup.
- Node naming matters because the downstream knobs are name-keyed:
  `nodes_to_exclude`, `TensorQuantOverrides` `recv_nodes`, readable QNN
  graph/profiling output.
- `get_qnn_qdq_config` sets two QNN-safe values you'd otherwise find by trial:
  `MinimumRealRange=0.0001` ("necessary for EPs like QNN that require a minimum
  floating-point range") and `DedicatedQDQPair=False`. It does **not** accept
  `extra_options` — it builds them; mutate `qnn_config.extra_options` on the
  returned `StaticQuantConfig` for anything else.
- It also auto-injects QNN requirements you would otherwise get wrong: per-tensor
  overrides for MatMul initializers (**QNN has no per-channel MatMul**);
  `_make_static_inputs_use_default_weight_type` for the both-16-bit
  MatMul/LayerNorm case; and fixed 16-bit output scales — Sigmoid uint16 →
  scale 1/65536 zp 0, Sigmoid int16 → 1/32768 zp 0, Tanh uint16 → 1/32768
  zp 32768, Tanh int16 → 1/32768 zp 0. Hand-rolled 16-bit QDQ without these
  fails or loses accuracy on those ops.
- **`per_channel=True` interacts badly with two of those fixes**: the MatMul
  per-tensor override only applies to initializers you haven't already overridden,
  so an `init_overrides` entry carrying an `axis` key still wins and can emit an
  unsupported per-channel MatMul; and `_process_layernorm` bails under
  per-channel, silently dropping the 16-bit LayerNorm correction. If you set
  `per_channel=True` alongside `op_types_to_quantize=["MatMul"]`, know that you
  are relying on the *converter* not honouring per-channel for MatMul.
- 16-bit and 4-bit QDQ need **opset ≥ 21** or ORT's `com.microsoft` contrib Q/DQ.
  `get_qnn_qdq_config` sets `extra_options['UseQDQContribOps']=True` automatically
  when the ai.onnx opset is < 21 and a 16-/4-bit type appears.
- `ForceQuantizeNoInputCheck=True` is set by the **generic** `get_qdq_config`,
  **not** by `get_qnn_qdq_config`. Without it, latent ops like MaxPool/Transpose
  stay float when their input isn't already quantized. Add it yourself if wanted.
- QNN EP consumes **QDQ only** — it partitions via QDQ node units and has no
  QOperator/QLinear* builders. Never set `QuantFormat.QOperator`.
- `MatMulConstBOnly` is a **no-op in QDQ format**. Use
  `nodes_to_exclude` / `op_types_to_quantize` instead.
- Exclusion sets differ, and only apply when `op_types_to_quantize` is None:
  `get_qnn_qdq_config` excludes `{"Cast"}`; generic `get_qdq_config` excludes
  `{"Cast","DequantizeLinear","QuantizeLinear"}`. Both set
  `use_external_data_format` automatically when the model has external data or
  `ByteSize() >= 2147483648`.
- Mixed precision: pass
  `init_overrides={"Op4_out": [{"quant_type": QuantType.QUInt16}]}` with base
  A8W8. `add_qtype_converts` defaults True and — only when `init_overrides` is
  non-empty — runs `MixedPrecisionTensorQuantOverridesFixer` to insert
  `convert`/`recv_nodes` so u8↔u16 boundaries become legal DQ→Q (**activations
  only**; weights are not auto-converted). Inspect the result via
  `qnn_config.extra_options['TensorQuantOverrides']`.
- Recent ORT also exposes `dynamic_input_shapes=[('input','1,3,256,256')]` and
  `inputs_to_make_channel_last` / `outputs_to_make_channel_last`. Channel-last
  **rewrites the graph I/O to NHWC**, so your feed/fetch code must change; rank
  must be ≥3. (verify for your version)

`CalibrationDataReader.get_next()` must return a dict keyed by **exact graph input
names**, shapes matching the declared (usually batch-1) inputs, and `None` to
terminate. The reader is **single-pass** — construct a fresh one per call.

## AOT-compile the context locally (no cloud, no token)

```python
so = ort.SessionOptions()
so.add_session_config_entry("ep.context_enable", "1")
so.add_session_config_entry("ep.context_file_path", "policy_ctx.onnx")  # else <input>_ctx.onnx
so.add_session_config_entry("ep.context_embed_mode", "0")               # 0 = sidecar .bin
sess = ort.InferenceSession("policy_w8a8.onnx", sess_options=so,
                            providers=[("QNNExecutionProvider", {"backend_type": "htp"})])
# Session CREATION emits the artifacts. No Run needed.
```

These are **session** config entries, not provider options — putting them in
`provider_options` silently does nothing.

Then **verify the compiled artifact**, which is the step everyone skips:

```python
ref  = ort.InferenceSession("policy_w8a8.onnx", providers=[("QNNExecutionProvider", qnn)])
comp = ort.InferenceSession("policy_ctx.onnx",  providers=[("QNNExecutionProvider", qnn)])
a, b = ref.run(None, feed)[0], comp.run(None, feed)[0]
assert np.allclose(a, b, atol=1e-5), np.abs(a - b).max()
```

- Sidecar name in current ORT is `<ctx_model_basename>_qnn.bin` (e.g.
  `policy_ctx_qnn.bin`); the `[input]_QNN_[hash].bin` form in the docs is
  **stale**. Discover the real name from the `ep_cache_context` attribute of the
  EPContext node. It is recorded as a path **relative to the ONNX file** — ship
  both in the same folder and never rename or edit the `.bin`.
- Loading either model from a **memory buffer** makes `ep.context_file_path`
  **mandatory** — ORT cannot otherwise resolve the sidecar's folder.
- `ep.context_embed_mode=1` conflicts with `ep.share_ep_contexts` and
  `enable_vtcm_backup_buffer_sharing`: ORT only logs at ERROR ("... conflict with
  EP context embed mode. Inference will not work as expected!") and lets
  compilation succeed. Leave embed mode at 0 for any sharing scenario.
  `enable_file_mapped_weights` is also incompatible and is force-disabled.
- EPContext generation **and** loading only work on QPU backends (HTP/DSP/GPU).
  With the QNN **CPU** backend, `GetCapability` logs `Qnn context cache only works
  for HTP/DSP/GPU backend.` at ERROR and assigns zero nodes — the graph runs
  entirely on the ORT CPU EP, looking successful with no acceleration and no
  context binary.
- Generate on the aarch64 device with HTP, or on an x86_64 host with the x86_64
  QNN libs (which cannot execute) — the 2.x `manylinux_2_35_x86_64` wheel ships
  them, so an x86 AOT host no longer needs a separate QAIRT install. When compiling off-device, **pin `soc_model`
  and `htp_arch` explicitly** — defaults are unknown/none and the binary may not
  load.
- ORT ≥ 1.23 has a first-class API:
  `sess_options.add_provider("QNNExecutionProvider", {...})` (1.x) or
  `sess_options.add_provider_for_devices(devs, {...})` (2.x — `add_provider`
  raises there), then `onnxruntime.ModelCompiler(sess_options, "in.onnx",
  embed_compiled_data_into_model=False).compile_to_file("out_ctx.onnx")`. The
  C/C++ `OrtCompileApi` landed in 1.22; the Python wrapper did not exist before
  1.23. Verified with 2.8.0 on a bare x86_64 host (no QAIRT installed,
  `htp_arch: "75"`): writes `out_ctx.onnx` + `out_ctx_qnn.bin`.
- Weight sharing across models is an offline flow requiring an **x86_64** host:
  `ep.share_ep_contexts=1` on every session in the group plus
  `ep.stop_share_ep_contexts=1` on the last (which writes
  `[model1_name]_[ep].bin`). Or wrap a `qnn-context-binary-generator` output with
  `onnxruntime/python/tools/qnn/gen_qnn_ctx_onnx_model.py`.
- **Flexible Context Binary (FCB) / multi-SoC** (2.x, QAIRT ≥ 2.48, x86_64 host
  only): give `soc_model` and/or `htp_arch` a comma-separated list
  (`"htp_arch": "73,81", "soc_model": "60,88"`, same length if both are given),
  with `ep.context_enable=1`. One EPContext model then carries one compiled
  context per listed SoC, packed into a DLC. It is still a context per target
  rather than a portable binary. Not combinable with `ep.share_ep_contexts`.
  Prune targets for shipping with
  `python -m onnxruntime_qnn.deploy_multi_soc_ep_context --input_bin …_qnn.bin --list`
  (Python 3.12 wheel only — the helper extension ships only there).

**Why bother:** first-load HTP graph prep dominates startup. Measured on QCS8275
/ v75 with ORT-QNN 1.23 on specific INT8/INT16 graphs: **ViT 147 s, DiT ~150 s,
~10–15 min for a full 7-graph pipeline** — versus ~1.2 s inference. Reload of a
cached context drops session creation to ~0.01 s for a tiny MLP.

## Prove it ran on the NPU

`sess.get_providers()` proves **nothing** — it lists *registered* providers, not
node placement, and returns `QNNExecutionProvider` first even if every node fell
to CPU. Sessions were observed to "succeed" while QNN device creation had failed
and everything ran on CPU.

Positive checks, strongest first:

1. `session.disable_cpu_ep_fallback=1` → ORT raises at **InferenceSession
   construction** if any node cannot be taken by QNN. This proves QNN EP
   assignment, **not HTP specifically** (QNN EP with `backend_type:"cpu"` also
   passes) — pair it with the HTP backend and the log evidence below. Debug-only:
   remove it for production, where partial partitioning may be fine.
2. HTP graph-prep stages at VERBOSE (`log_severity_level=0`), in order:
   `Graph Preparation Initializing → Graph Optimizations → Post Graph
   Optimization → Graph Sequencing for Target → VTCM Allocation →
   Parallelization Optimization → Finalizing Graph Sequence → Completion`.
   `VTCM Allocation` and `Graph Sequencing for Target` have no CPU-EP analogue.
3. `====== DDR bandwidth summary ======` with `spill_bytes` / `fill_bytes` /
   `write_total_bytes` / `read_total_bytes`, plus fastrpc `rpcmem_init`. QAIRT
   2.46's HTP backend prints this on each successful device bringup — a useful if
   undocumented tell.
4. Never grep the backend string for `"htp"` — that's only the *value* of
   `backend_type`. ORT names the provider `QNNExecutionProvider` in Python
   (`QNN` for the C++ `AppendExecutionProvider` short name).

`VerifyEachNodeIsAssignedToAnEp` ("Some nodes were not assigned to the preferred
execution providers ...") fires on essentially every QNN EP run and proves
**neither** success nor failure — ORT keeps shape ops on CPU by design, and
`offload_graph_io_quantization` defaults to `'1'`, deliberately putting graph-I/O
Q/DQ on the CPU EP. A healthy QDQ partition looks like 34/36 or 3004/3005 nodes on
QNN. **"Zero CPU nodes" is not the criterion.** If the only CPU nodes are graph-I/O
Q/DQ (`QNN EP is configured to not take Q nodes that consume a graph input`), set
`offload_graph_io_quantization` to `'0'` rather than assuming the model is
unsupported.

## Probe VTCM on a new SoC

```python
# Sweep on a SMALL already-quantized model — each accepted value triggers a real
# graph finalize, so a multi-GB model takes minutes per step.
def mk(v):
    so = ort.SessionOptions(); so.log_severity_level = 3
    q = {"backend_type": "htp", "htp_arch": "75"}
    if v: q["vtcm_mb"] = str(v)                      # STRINGS only
    return ort.InferenceSession(M, sess_options=so,
                                providers=[("QNNExecutionProvider", q), "CPUExecutionProvider"])

for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16]:
    try: mk(v); print(f"vtcm_mb={v} OK")
    except Exception as e: print(f"vtcm_mb={v} FAIL {e}")     # print the FULL exception
```

The largest value that constructs is an upper bound on what the QNN HTP backend
**accepts for that device/arch** — not a direct read of physical silicon.
Rejection surfaces as `FAIL : Failed to initialize qnn_model_wrapper` at session
creation. Read the full exception to confirm it is a VTCM/finalize error and not
an unrelated op-support failure, and cross-check against the Hexagon arch spec or
`qnn-platform-validator`.
