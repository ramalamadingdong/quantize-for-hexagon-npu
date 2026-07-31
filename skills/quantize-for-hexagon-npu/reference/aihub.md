# Path A — Qualcomm AI Hub (cloud compile + quantize)

## Setup

```bash
pip install qai-hub                                # pure-Python client; runs on aarch64 too
qai-hub configure --api_token "$QAI_HUB_TOKEN"     # Account > Settings > API Token
qai-hub list-devices                               # never hardcode a device string
qai-hub list-frameworks                            # which QAIRT versions the cloud accepts
```

Device names are SoC/marketing names, not EVK names — enumerate them:

```bash
python -c "import qai_hub as hub; print([d.name for d in hub.get_devices()])"
# e.g. 'QCS8275 (Proxy)', 'Dragonwing IQ-8275 EVK', 'Dragonwing IQ-9075 EVK'
```

`IQ8` / `IQ-8275` match nothing under the Proxy naming, and there is no
`QCS9075 (Proxy)`. `(Proxy)` means latency was measured on a stand-in device that
matches only on NPU/Hexagon version.

## How AI Hub quantizes (the part worth copying)

- Quantize jobs run **AIMET** server-side — not `onnxruntime.quantization`.
- Output is AI Hub's **"fake quantization" ONNX**: QuantizeLinear/DequantizeLinear
  pairs where **weights are still stored as float** followed by a QuantizeLinear.
  This differs from standard static ONNX QDQ (folded int8 initializers).
- **Default range scheme is `mse_minimizer`**, not min-max: it builds a
  per-tensor histogram and picks a range that clips outliers. This is why AI Hub
  gets away with quantizing whole transformers where a naive
  `CalibrationMethod.MinMax` over all ops collapses. `min_max` is offered for
  when outliers are semantically important.
- Docs recommend **500–1000 calibration samples**; the official tutorial uses 100.
- `qai-hub-models` itself emits `--range_scheme min_max` for `w8a16` and for any
  mixed-precision config, and nothing (i.e. the default) for `w8a8`.

## Already-quantized QDQ ONNX → context binary

```python
import qai_hub as hub
dev = hub.Device("QCS8275 (Proxy)")               # exact string from get_devices()
job = hub.submit_compile_job(
    model="policy_w8a8.onnx",                     # static-shape QDQ ONNX
    device=dev,
    options="--target_runtime qnn_context_binary",  # tflite|qnn_dlc|onnx|precompiled_qnn_onnx
)
assert job.wait().success, job.get_status()       # else get_target_model() AttributeErrors
job.get_target_model().download("policy.bin")
```

This works in practice — a locally produced `onnxruntime.quantization` QDQ model
compiled fine and yielded a `.bin` byte-identical in size to the EPContext one.
But it is **not the documented contract**: the quantize docs state AI Hub's own
fake-quant ONNX is "the only ONNX quantization format" officially accepted as
compile input, so a folded-int8 QDQ model may be rejected. If it is, run it
through a quantize job instead.

`options` is the singular string param on `submit_compile_job`; `compile_options`
is the plural list on `submit_compile_and_link_jobs`. `input_specs` is only needed
for torch.export/TorchScript or dynamic-shape ONNX. Prefer
`model.download(<dir>)` and read the returned name — ONNX-family targets with
external weights arrive as a zip/directory, not a single file.

## Float model → quantize job → compile

```python
import numpy as np, qai_hub as hub
client = hub.Client()

# 1. compile to ONNX FIRST — the quantize docs require this even for ONNX input,
#    so AI Hub's optimizer can apply hardware-compatibility passes beforehand.
onnx_job = client.submit_compile_job(model="model.onnx", device=dev,
                                     options="--target_runtime onnx")
assert onnx_job.wait().success
src = onnx_job.get_target_model()

# 2. quantize. dtypes are ENUM members, NOT the strings "int8".
calib = {"image_tensor": [cal[i:i+1].astype(np.float32) for i in range(len(cal))]}
qjob = client.submit_quantize_job(
    model=src,
    calibration_data=calib,                       # keys == exact ONNX input names
    weights_dtype=hub.QuantizeDtype.INT8,
    activations_dtype=hub.QuantizeDtype.INT16,    # A16W8
    options="--range_scheme min_max",              # omit for the mse_minimizer default
)
assert qjob.wait().success
q = qjob.get_target_model()

# 3. compile for the target
cjob = client.submit_compile_job(model=q, device=dev,
                                 options="--target_runtime qnn_context_binary")
assert cjob.wait().success
cjob.get_target_model().download("model.bin")
```

Signature: `submit_quantize_job(model, calibration_data,
weights_dtype=QuantizeDtype.INT8, activations_dtype=QuantizeDtype.INT8,
name=None, options='', project=None)`. There is **no `device=`** on a quantize job.

The client-side dtype guard is a **conjunction**, so it is looser than it reads
(`qai_hub/client.py`):

```python
if weights_dtype != QuantizeDtype.INT8 and activations_dtype not in [
        QuantizeDtype.INT8, QuantizeDtype.INT16]:
    raise ValueError("Weights must be INT8 and activations must be INT8 or INT16.")
```

So `weights_dtype=INT4, activations_dtype=INT16` **passes** client validation
(server-side acceptance untested). Passing the string `"int8"` raises
`AttributeError` — the client does `.value`.

## Quantize-job options

Only two are documented:

- `--range_scheme {auto|mse_minimizer|min_max}` — default `auto`, which currently
  resolves to `mse_minimizer`.
- `--lite_mp percentage=<N>;override_qtype=<int16|fp16>` — Lite Mixed Precision:
  promote the most quantization-sensitive layers. Bare `--lite_mp` defaults to
  `percentage=10;override_qtype=int16`. Beta; tested only on non-generative models.

`--quantize_full_type` / `--quantize_weight_type` / `--quantize_io` are
**compile**-job options — passing them to a quantize job fails.

The precision matrix on `quantize_examples.html` is **stale** (claims mixed
precision unsupported and int4 "long-term") — `--lite_mp` ships and w4a16 works.
Validate empirically.

## Compile-job options that matter

- `--target_runtime`:
  - `qnn_context_binary` — `.bin`, pre-optimized for **one SoC**, NPU-only.
  - `qnn_dlc` — **hardware-agnostic** and forward-compatible: a DLC built on one
    SDK version is guaranteed to run on newer ones. The portable choice.
  - `precompiled_qnn_onnx` — an `.onnx` directory wrapping a context binary for
    ORT's QNN EP. Works across Android/Linux/Windows, performance equal to a raw
    context binary, and the intended vehicle for **>1 GB** models (LLMs, SD).
    The `.bin` is referenced by *relative* path — moving or renaming it breaks it.
  - `tflite` (default on Android), `onnx` (default on Windows).
- `--quantize_full_type {int8|int16|w8a16|w4a8|w4a16}` — quantizes during
  translation; **`w8a16` is recommended over `int16`**. With no calibration data
  a **random** dataset is substituted, making the result "a proxy for achievable
  performance only". Rejected for AIMET inputs and for the ONNX target.
- `--quantize_io [true|false]` (default off) — by default a quantized model keeps
  **float32 graph I/O**, costing a conversion on-device. Turning it on moves that
  conversion into your integration code. `--quantize_io_type {int8|uint8}` is
  TFLite-only.
- `--quantize_weight_type float16` — weights-only, **TFLite-target only**. For
  fp16 on HTP use `--qnn_options default_graph_htp_precision=FLOAT16`.
- `--truncate_64bit_io [true|false]` — **required** for TFLite and QNN targets
  when the source has int64 I/O. `--truncate_64bit_tensors` covers intermediates.
- `--force_channel_last_input <name>[,name]` / `--force_channel_last_output` —
  accept/emit NHWC instead of inserting transposes; also converts a supplied
  representative dataset's layout. Not usable with the ONNX target.
- `--output_names "a,b"` — replaces the default `output_0`, `output_1`, …. Must
  name every output.
- `--qnn_options <opt>[;<opt>]` — e.g.
  `default_graph_htp_vtcm_size=<MB>` (0 = device max),
  `default_graph_htp_optimizations=FINALIZE_OPTIMIZATION_FLAG=<1-3>` (shorthand
  `O=<n>`; 1 fastest prepare, 3 most optimized, default 3),
  `context_enable_graphs=<name>` (must be a valid C++ identifier),
  `default_graph_htp_enable_weights_packing=true`.
  These replace the deprecated `--qnn_graph_name`,
  `--qnn_context_binary_vtcm`, `--qnn_context_binary_optimization_level`.
- **Pin the SDK on both jobs**: `--compile-options="--qairt_version 2.45"
  --profile-options="--qairt_version 2.45"`. Only versions from
  `qai-hub list-frameworks` are accepted. Read the device's version with
  `qnn-net-run --version` (the Dragonwing docs' `qairt-net-run` is a typo — no
  such binary ships).

Profile with `hub.submit_profile_job(model=<compiled target model>, device=...)`
on a **real** device, not `(Proxy)` — proxies match on NPU version only, so any
CPU/GPU fallback makes proxy latency unrepresentative. Options:
`--dequantize_outputs` (default true), `--max_profiler_iterations` (default 100).

`.aimet` input is a **directory** naming convention, not a format: any dir with
`.aimet` in the name holding one `.onnx`, a matching `.encodings`, and optionally
a `.data` (only when the ONNX uses external weights).

## qai-hub-models CLI (catalog models)

```bash
pip install qai-hub-models
qai-hub-models export <model_id> --precision w8a8 \
  --target-runtime=qnn_context_binary \
  --device="QCS8275 (Proxy)" \
  --compile-options="--qairt_version 2.46" --profile-options "--qairt_version 2.46"
# artifacts: export_assets/<model>-qnn_context_binary-w8a8-<chipset>/<model>.bin
```

- Use the **`qai-hub-models export` CLI**;
  `python -m qai_hub_models.models.<id>.export` emits a DeprecationWarning.
- Flag-spelling split: Python API `options="--target_runtime ..."` (underscore),
  CLI `--target-runtime=...` (hyphen).
- Use `--precision`. A `--quantize` flag is only added when the model supports
  more than one precision, and its help says it "always supercedes the
  `--precision` argument" — `--precision` is the stable one.
- `--precision` is parsed order-insensitively (`w4a16 == a16w4`); bit widths map
  to `INT<bits>` and must exist in `QuantizeDtype` (INT4/INT8/INT16 only), so
  `w6a8` raises. `_mixed_int16` / `_mixed_fp16` suffixes pick the Lite-MP type.
- The CLI swallows tracebacks — debug with `QAIHM_CLI_VERBOSE_EXCEPTIONS=1`.
- Set `QAIHM_STORE_ROOT` off any quota'd home **before** the first fetch
  (defaults to `$HOME/.qaihm`; a prequantized 8B W4A16 checkpoint is ~33 GB), and
  redirect `HF_HOME` too.
- Prebuilt asset download:
  `qai-hub-models fetch <model_id> -r qnn_context_binary -p w8a8 -c qualcomm-qcs8275`.
  The positional arg is the lowercase **model id** (`pi05`, not `Pi0.5` — it is
  `str.lower()`-forced and a display name 404s). If no chipset-specific asset
  exists it **silently falls back** to the generic one.
- `TargetRuntime.supports_precision()` is a client-side gate: TFLITE accepts only
  `w8a8` among quantized precisions; QNN/Genie paths accept
  w8a8/w8a16/w4a16/w4/w16a16 + mixed; `geniex_llamacpp` only mxfp4/q8_0/q4_0.

## LLMs on AI Hub

LLM/VLM recipes set `separate_quantize_script: true` — **quantization runs
locally, not as a cloud job**:

```bash
python -m qai_hub_models.models.<llm>.quantize --checkpoint <hf_repo_or_dir> \
  -o ./quantized_model [--precision w4a16] [--num-samples 20] \
  [--use-seq-mse] [--use-ada-scale] [--use-spin-quant r1,r3]
```

Defaults: `--num-samples 20`, `--calibration-sequence-length 2048`,
`--context-length 4096`. AdaScale falls back to 128 samples / 512 iterations.

**A GPU is required** (AIMET-ONNX, Linux only, cp310 wheels): ~40 GB VRAM for 3B,
~80 GB for 8B. Budget ~1 h plain, 5+ h with `--use-seq-mse`. Only *quantization*
is GPU-gated — export for deployment is not. Per-model `MIN_MEMORY_RECOMMENDED`
is checked against RAM+swap but is **advisory only**: it warns and proceeds, so
you get OOM-killed mid-run. Measured: CPU-only 4B W4A16 was OOM-killed at ~94 GB
peak on a 62 GB + 32 GB swap host — calibration is cheap, the **final ONNX
export** is the peak. The pipeline itself was correct; it is purely a ceiling.

Export:

```bash
qai-hub-models export <llm> --checkpoint DEFAULT_W4A16 \
  --target-runtime geniex_qairt --chipset qualcomm-snapdragon-8-elite-gen5 \
  --compile-options "--qairt_version 2.46" --skip-profiling --output-dir <dir>
```

- `--sequence-length 128,1` builds prompt-processor and token-generator in one run.
- `--model-cache-mode disable` is **required** with a custom `--checkpoint`.
- Prefer `--target-runtime geniex_qairt` (`genie` is deprecated). Both emit a
  classic Genie bundle runnable by the device's stock `libGenie.so` /
  `genie-t2t-run` — `geniex_qairt` is an *export-target label*, not a different
  on-device runtime. Do not confuse it with `geniex_llamacpp` (GGUF).
- An AIMET-ONNX quantized LLM checkpoint is a **directory**: `model.encodings` +
  a large shared external blob (`model.data`, or `model.onnx.data` in newer
  versions — ~32 GB for an 8B, i.e. roughly fp32-sized, because W4 packing
  happens at AI Hub compile time) + ONNX graph(s) + `config.json` /
  `tokenizer.json` / `args.json`. Always `ls` it.
- Copying only the `.bin` files from a Genie bundle fails: `genie-t2t-run` needs
  `tokenizer.json`, `tokenizer_config.json`, `config.json`, `genie_config.json`,
  `htp_backend_ext_config.json`, `tool-versions.yaml` (newer also
  `text-generator.json`, `sample_prompt.txt`). rsync the whole directory, and
  read the required runtime from `tool-versions.yaml` rather than hardcoding a
  QAIRT version.

**No Genie/QNN path exists for MoE or linear/recurrent-attention models.** Check
before planning:

```bash
grep -rln 'genie_compatible: *true' <site-packages>/qai_hub_models/models/*/info.yaml
```

`gpt_oss_20b` is `genie_compatible: false` and ships as MXFP4 GGUF with no
model.py/export.py/quantize.py at all; the `qwen3_5` family (Gated DeltaNet) is
`geniex_llamacpp`-only. For MoE, plan on llama.cpp / ggml-hexagon. AI Hub LLM
recipes cap at **8B dense**.

### LLM-quantization environment traps

- **Build on Python 3.10 with the model's exact pinned AIMET.** qai-hub-models
  pins `aimet-onnx==2.26.0` behind a `python_version == "3.10"` marker, so a
  3.11/3.12 env installs **no** pinned AIMET and you silently get an unvalidated
  version. Observed consequence: "Quantization completed successfully", a correct
  first token from the calibration graph, and **garbage** from the separately
  exported seq-128 / seq-1 deployment graphs. Always validate across the
  seq-128 → seq-1 switch *before* paying for a compile:
  ```bash
  python -m qai_hub_models.models.<model>.demo --checkpoint <ckpt> \
    --context-length N --sequence-length 128 --max-output-tokens 16 \
    --seed 42 --prompt '...'
  ```
- AIMET version floors are per-feature, enforced at literal call sites: **2.8.0**
  plain calibration and Seq-MSE, **2.26.0** AdaScale, **2.33.0** SpinQuant.
  AIMET-ONNX is **Linux only** (use WSL on Windows).
- `libpymo import failed ... libpython3.10.so.1.0: cannot open shared object
  file` — `aimet_onnx` installs **lazy stubs** on load failure, so this surfaces
  mid-run. Common with uv / python-build-standalone CPython and conda:
  ```bash
  export LD_LIBRARY_PATH="$(python -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))'):$LD_LIBRARY_PATH"
  python -c "import aimet_onnx.common.py_libpymo"   # verify up front
  ```
- `NameError: name 'init_empty_weights' is not defined` inside transformers →
  missing optional `accelerate` (imported behind `is_accelerate_available()`, so
  NameError not ImportError). `pip install accelerate`.
- `"No active RMSNorms found"` from SpinQuant → you applied it *after* building
  `QuantizationSimModel`; the inserted `QcQuantizeOp` nodes break the
  RMSNorm→linear connectivity walk. Apply `apply_spinquant_to_onnx` to the
  **float** graph, and run `fix_node_names_in_dynamo_exported_onnx` first (dynamo
  export names ops `node_Conv_*`, losing the `/self_attn/v_proj/` scope R2 needs).
- aimet-onnx 2.33.0+ rejects an initializer shared across consumers of different
  op types (Qwen3's tied `lm_head.weight` feeds both a Gather and a MatMul) →
  `aimet_onnx.utils.duplicate_shared_initializers(onnx_model.graph)` before
  `QuantizationSimModel` **and** before `apply_spinquant` (idempotent).
- On Blackwell / RTX 5090 (sm_120), the Llama extra pins torch 2.4.1+cu121 which
  has no sm_120 kernels. Install
  `torch==2.7.1 torchvision==0.22.1 --index-url .../whl/cu128` and verify with a
  real CUDA matmul, not `is_available()`. Don't `pip install
  qai-hub-models[llama...]` directly (the extra downgrades torch) — use a
  constraints file and install `onnxruntime-gpu` **last** with `--no-deps`.

## AI Hub error decoder

- **3110** → a node's input/output types aren't supported by the QNN backend.
  Usually a node that wasn't properly quantized, or an unquantized tensor passing
  through a quantized graph. Also unsupported rank (most HTP ops support <5 dims)
  and unsupported dtypes, especially `[u]int32`. Re-quantize that region and
  check the QNN HTP Backend Op Definition Supplement.
- `AI Hub device "IQ-8275" not found` → enumerate with `hub.get_devices()`.
- `ValidationError: Model cannot be published: no release assets available` →
  version-specific: qai-hub-models 0.58.0 shipped `models/qwen3_8b/` without
  `release-assets.yaml` while `can_promote_to_published()` only does
  `os.path.exists`. Copy the file from upstream into the installed dir.

The repo moved: `github.com/quic/ai-hub-models` → `qualcomm/ai-hub-models`, and
the package to `src/qai_hub_models/`. Git/web/API follow the rename;
`raw.githubusercontent.com` does **not**, so raw URLs need both the new org and
the `src/` prefix.
