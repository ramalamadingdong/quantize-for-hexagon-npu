---
name: quantize-for-hexagon-npu
description: >-
  Quantize and compile a neural network to run on a Qualcomm Hexagon NPU / HTP —
  via Qualcomm AI Hub cloud jobs, the local QAIRT/QNN toolchain
  (qairt-converter, qairt-quantizer, qnn-context-binary-generator, snpe-dlc-*),
  or ONNX Runtime's QNN Execution Provider with EPContext AOT caching.
  Use for: INT8/INT16/W4A16 PTQ of ONNX or PyTorch models, QDQ calibration,
  building .dlc / context-binary .bin / *_ctx.onnx artifacts, picking
  soc_model + dsp_arch/htp_arch, VTCM sizing, Genie LLM bundles, and debugging
  "runs on CPU instead of NPU", QNN 6001, 14001, INVALID_PROTOBUF, or accuracy
  collapse after quantization. The skill must plan and close an accuracy check:
  export/preprocess/layout parity locally, then final deployed HTP accuracy on a
  board or AI Hub when required. Targets Dragonwing IQ-8275 / IQ-9075 / RB3 Gen
  2, QCS/SA/SM chipsets, and Snapdragon HTP v68-v81.
---

# Quantize & compile models for Qualcomm Hexagon NPU (HTP)

Distilled from real bring-ups on QCS8275 (v75), QCS9075 (v73) and SA8797P (v81),
plus the QAIRT 2.46 SDK, AI Hub docs, and ONNX Runtime source. Numbers labelled
*measured* come from one board and one SDK version — re-measure on yours.

## Non-negotiable: accuracy closure is part of the job

When this skill is active, do **not** stop at "conversion succeeded" or "profiled
on NPU". Treat accuracy as a deliverable with an explicit closure status.

At the **start** of the work, ask or infer:

- What reference matters: PyTorch/source model, fp32 ONNX, a golden output set, or
  task-level metric?
- What threshold matters: cosine, max-abs/relative error, top-k, mAP/IoU/RMSE,
  simulator success rate, or domain-specific acceptance?
- Whether a real target board is reachable over SSH/ADB, whether AI Hub inference
  is allowed, or whether this is a **local-only** check.
- The app IO contract for every artifact: input/output names, shape, dtype,
  layout, normalization, and raw-file byte order.

During the work, preserve enough artifacts to rerun accuracy: fixed-shape fp32
ONNX, quantized/compiled artifact, calibration set, held-out parity inputs,
preprocessing script, exact commands, SDK/ORT/AI Hub versions, and logs.

At the **end**, always run the strongest check available and report one of:

- **PASS** — thresholds met for the stated reference and runtime.
- **FAIL** — thresholds missed; include the first failing metric and next action
  (layout fix, bit-width escalation, per-op override, calibration change,
  layerwise diff, etc.).
- **UNPROVEN** — local parity was checked, but final deployed HTP numerics were
  not proven because no board/AI Hub inference path was available. Say this
  plainly; do not imply NPU accuracy from successful compile/profile alone.

Final responses should include an `ACCURACY STATUS` block with: status, reference,
runtime tested, metrics, IO contract, commands/log paths, and the remaining ask
from the user if final HTP accuracy is still unproven.

See [reference/accuracy-debug.md](reference/accuracy-debug.md) for the detailed
closure protocol and local-vs-device decision tree.

## Pick a path

**Default to Path B (QAIRT native) whenever the host can run it.** It is the
vendor toolchain the other two are built on: fully offline, no account or upload,
version-pinnable, and the only route that gives per-op encoding overrides, `.dlc`
output, and Genie/LLM bundles. Fall back only for the specific reasons below.

Check host support first — this decides the default:

```bash
uname -srm            # QAIRT converters/quantizers ship for x86_64 Linux,
                      # x86_64 Windows, and arm64x Windows — NOT aarch64 Linux
```

| Host | Default |
|---|---|
| **x86_64 Linux** | **Path B.** The best-supported combination. |
| **x86_64 / arm64 Windows** | **Path B**, using `bin/x86_64-windows-msvc` or `bin/arm64x-windows-msvc`. |
| **aarch64 Linux** (the board itself, an Arm Mac VM, Graviton) | Path C — no converter/quantizer binary exists for this ABI. Or cross-compile from an x86_64 host and copy the artifact over. |
| **macOS** (any arch) | Path A or C — QAIRT ships no macOS host tools at all. |

Then override the default only for these reasons:

| Go with | When |
|---|---|
| **A. AI Hub** (cloud) | You want AIMET-grade PTQ without doing the work (its default `mse_minimizer` range scheme beats naive MinMax); a **W4A16 LLM**, which needs a 40–80 GB GPU locally; or a quick catalog-model artifact. Requires uploading weights and an account. |
| **C. ORT QNN EP** (local) | You already have QDQ ONNX and just want it on the NPU; you want to iterate **on the board** with no x86 host; or you want EPContext caching to kill graph-prep time. |

Avoid Path B when: the host isn't in the support table above; you need AIMET
algorithms it doesn't expose locally (AdaRound/AMP/AutoQuant need
`--use_aimet_quantizer` plus extra installs); or you want a W4A16 LLM and have no
GPU.

Paths compose — quantize on AI Hub, then compile locally with QAIRT for a SoC AI
Hub doesn't list.

Details: [reference/qairt.md](reference/qairt.md) ·
[reference/aihub.md](reference/aihub.md) ·
[reference/ort-qnn-ep.md](reference/ort-qnn-ep.md)

## Get the SDK (Path B, first step)

The download URL embeds a 4-part build id whose last field is the build **date**,
and there is no listing API — so never hardcode a version. Resolve it:

```bash
scripts/fetch-latest-qairt.sh                      # print latest version + URL
scripts/fetch-latest-qairt.sh --download ~/qairt   # fetch (~1.8 GB) and unzip
```

It walks minor versions upward with a parallel ranged-GET sweep (no auth),
seeded from AI Hub's `get_frameworks()` when a token is configured — AI Hub lags
the download site, so its answer is a floor, not the latest.
Latest verified 2026-11-16: **2.52.0.261104**. Then:

```bash
export QAIRT_ROOT=~/qairt/qairt/2.52.0.261104   # archive extracts to qairt/<ver>/
source "$QAIRT_ROOT/bin/envsetup.sh"
```

## Rules that hold on every path

1. **HTP runs quantized graphs with fully static shapes.** Fix shapes *before*
   calibrating. Dynamic shapes and unsupported ops go to CPU.
   ```bash
   python3 -m onnxruntime.tools.make_dynamic_shape_fixed in.onnx out.onnx \
     --input_name pixel_values --input_shape 1,3,224,224      # or --dim_param seq_len --dim_value 156
   ```
2. **Float32 does not necessarily fall back to CPU.** Since QAIRT 2.35, HTP
   float ops run as fp16 on SoCs with float support, whatever ORT's
   `enable_htp_fp16_precision` says (it no longer changes HTP precision). So an
   fp32 model runs *on HTP* as fp16 — it looks accelerated while being far slower
   than int8. Quantization buys throughput, not file size (see rule 6).
3. **Compiled context binaries are not portable.** A `.bin` / `*_ctx.onnx` is
   locked to HTP arch + `soc_model` + QAIRT version. Portable: `.dlc` (no
   graph-prep cache), plain QDQ `.onnx`, `.tflite`. Runtime SDK must be **≥**
   the SDK that prepared the binary — that direction only. (A Flexible Context
   Binary packs several per-SoC contexts into one artifact — still one compiled
   context per target, see [reference/ort-qnn-ep.md](reference/ort-qnn-ep.md).)
4. **Confirm the SoC on the board itself**, because vendor scaffolds and sample
   configs routinely target a different part than the one in front of you.
   ```bash
   tr -d '\0' < /proc/device-tree/model; echo
   tr '\0' '\n' < /proc/device-tree/compatible          # a fallback LIST, not one name
   cat /sys/devices/soc0/machine /sys/devices/soc0/soc_id
   ```
   `compatible` lists the family chain — a QCS8275 reports both `qcom,qcs8275`
   and `qcs8300`, which is expected, not a contradiction. The Linux `soc_id`
   (e.g. 675) is **not** the QNN `soc_model`. Target by SoC *name string* where a
   tool accepts one; the `soc_model` integer enum is frozen and has no entry for
   QCS8275 or QCS9075 — see [reference/soc-targets.md](reference/soc-targets.md).
5. **Calibrate on representative data.** Static PTQ fits activation ranges from
   this set. Random tensors smoke-test the pipeline and produce a silently
   inaccurate model. Keep a *disjoint* held-out set for parity.
6. **Quantization may not shrink the file.** INT16 weights are 2 bytes, same as
   bf16 (measured: 1.611 GB → 1.611 GB). `op_types_to_quantize=["MatMul"]`
   leaves other weights fp32, so output can be *larger* than the source.
7. **Verify placement, don't assume it.** `sess.get_providers()` proves nothing.
   See "Prove it ran on the NPU" in
   [reference/ort-qnn-ep.md](reference/ort-qnn-ep.md).

## Quantization decisions

**Activation ladder.** HTP supports A8, A16 and FP16; accuracy *and* power rise
in that order. Start A8 → A16 → try **A16W16** before FP16 (docs expect it to
beat A16W8 on PTQ accuracy at better power than FP16). Weight width is chosen
independently. A16W16 is enabled only on selected SoCs and only for
Conv2d/DepthConv2d/TransposeConv2D/FullyConnected/MatMul/Batchnorm/LayerNorm.

**Unsigned, not signed.** QNN ops typically support `uint8`/`uint16`;
`quantize_static`'s `QInt8`/`QInt8` defaults are CPU-oriented. On HTP,
`MatMul` accepts only `(uint8,uint8)`, `(uint8,uint16)`, `(uint16,uint8)` —
**both-16-bit MatMul is unsupported**. Prefer A16W8
(`activation_type=QUInt16, weight_type=QUInt8`) or A8W8.

**Symmetric weights, asymmetric activations.** QAIRT defaults *both* schemas to
asymmetric — pass `--param_quantizer_schema symmetric` explicitly.

**4-bit is symmetric weight-only:** QNN supports INT4 only with zero-point 0.

**Bit width is per component, not per model.** Try INT8, gate each component on
a host-CPU cosine *and* max-abs vs a golden, escalate only what fails. Measured
on GR00T N1.7 (QCS8275, QAIRT 2.46, ORT-QNN 1.23) — an existence proof, not a
rule: a 2B decoder collapsed at INT8 (**cos 0.041, maxErr 6460**) and needed
INT16 (0.99994), while a *shallow MLP* also needed INT16 (INT8 gave cos 0.9918
but **maxErr 21**), and the 16-layer DiT shipped fine at INT8 (0.9975 on HTP).
Neither "INT8 for MLPs" nor "INT16 for transformers" is a law.

**For transformers, the lever is quantizing MatMuls only** — not the bit width:
```python
quantize_static(fp32, out, calibration_data_reader=Reader(feeds),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt16, weight_type=QuantType.QInt16,
    use_external_data_format=True,
    op_types_to_quantize=["MatMul"],          # leave Softmax/LayerNorm/mask fp32
    calibrate_method=CalibrationMethod.MinMax)
```
Measured: MinMax over **all** ops → cos **0.85** (unusable); MatMul-only →
**0.999921** vs the fp32 ONNX reference's own 0.999922. Root cause of the
collapse, reproduced on a minimal mask+MatMul graph (ORT 1.22): tensors carrying
`-inf` attention-mask values get **inf/nan scale and zero_point** and the
quantizer emits a corrupted model *with no error*. `Percentile` and `Entropy`
instead raise `ValueError: ... range of [-inf, inf] is not finite`. Note this is
naive MinMax PTQ; the vendor path (AIMET / AI Hub, per-op encoding overrides)
reaches good accuracy at W4A16, so low bit width per se is not the problem.

Leaving non-MatMul ops in fp32 does **not** push them to CPU — HTP executes
float ops; the measured MatMul-only DiT put 3084/3101 nodes on QNN.
Fragmentation comes from unsupported **op types**, not from float precision.

More: [reference/quantization-recipes.md](reference/quantization-recipes.md)

## Validate in this order

Each step catches failures the next one hides. Run this ladder automatically at
handoff, stopping only where the environment makes the next rung impossible.

0. **IO-contract diff for every artifact** — record input/output name, shape,
   dtype, layout, and normalization. Treat NHWC↔NCHW and fp32↔fp16 app IO changes
   as accuracy risks even when byte counts match; raw files can be silently
   scrambled.
1. **Source/export parity** — PyTorch/source model vs fixed-shape fp32 ONNX. If
   this fails, fix export or preprocessing before quantizing/compiling.
2. **Split/graph-surgery parity, if any** — fp32 chain vs whole fp32 model must
   be cos ≈ **1.0** (measured 1.000000). Anything less is a wiring bug.
3. **Host-runnable model parity** — fp32 ONNX vs QDQ ONNX, CPU-runnable DLC, or
   any host executable representation. Gate on cosine **and** max-abs-error.
   Cosine is scale-invariant and masks localized blow-ups; that is exactly how
   the `maxErr 21` component passed at cos 0.9918. Report which reference you
   used: vs PyTorch/bf16 golden folds in export error (what deployment cares
   about); vs the fp32 **ONNX** isolates pure quantization/conversion error.
   Never compare the two against one threshold.
4. **Deployed HTP runtime** — run the actual `.dlc`, context binary, or
   `*_ctx.onnx` on a board, or submit an AI Hub inference job. This is required
   for final NPU accuracy because HTP fuses ops and uses its own accumulator
   widths. An x86 host cannot honestly prove deployed HTP numerics, and a
   prepare/emulation path that produces no real outputs is not an accuracy run.
5. **Task-level** — gate on the statistic your consumer actually uses. A learned
   CartPole policy had **0.25** max INT8 error on a ±1 action scale yet balanced
   for the full 500-step cap, because the controller is bang-bang and only the
   *sign* matters. Where magnitude is load-bearing, 0.25 is not free.

If only rungs 0-3 are possible, report **UNPROVEN for final HTP accuracy** and ask
for a reachable board or permission to use AI Hub inference. Do not treat
successful conversion, profiling, parameter/MAC equality, or "all ops placed on
NPU" as an accuracy result.

Full closure protocol, local-only checks, layerwise diffing, and
`qairt-accuracy-debugger`: [reference/accuracy-debug.md](reference/accuracy-debug.md)

## When things break

Symptom → cause → fix, with the real error strings:
[reference/pitfalls.md](reference/pitfalls.md)

The three you will hit first:

- **Everything ran on CPU, session "succeeded"** → set
  `session.disable_cpu_ep_fallback=1` to make ORT raise at construction. Then
  confirm HTP specifically (that flag also passes with the QNN *CPU* backend).
  With `onnxruntime-qnn` **2.x**, the 1.x `providers=[("QNNExecutionProvider",
  …)]` form is silently CPU-only even after registering the plugin — use
  `add_provider_for_devices`.
- **`QNN graph execute error. Error code: 6001`** → generic HTP execute/resource
  failure, not a dedicated OOM code. Work it in order: VTCM exhaustion from
  co-resident contexts → a single graph too heavy (check `read_total_bytes`) →
  stale DSP state.
- **`Failed to create device: 14001` / `QNN_DEVICE_ERROR_INVALID_CONFIG`** → on
  Ubuntu-on-Dragonwing this is almost always a **missing FastRPC runtime**, not
  a bad path and not a wrong `soc_model`. `sudo apt-get install -y libqnn1
  qairt-libs qairt-dsp-binaries` (pulls `qcom-fastrpc1`), then `ldconfig`,
  add yourself to group `fastrpc`, reboot.

## Models too large for one graph

`>2 GB` ONNX quantization, VTCM/concurrency budgeting, and splitting a decoder
at residual boundaries: [reference/large-models.md](reference/large-models.md)

Two headlines: ORT calibration (not quantization) is what fails past 2 GB — fix
with `use_external_data_format=True`; and a monolithic 2B decoder reporting
~17.8 GB `read_total_bytes` will 6001, while 4-layer chunks at ~0.5 GB each run.
