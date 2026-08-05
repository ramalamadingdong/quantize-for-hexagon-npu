# Accuracy debugging

## Accuracy closure protocol for agents

Accuracy is not an optional follow-up. Whenever the skill quantizes, converts,
compiles, changes preprocessing, changes app IO, or profiles a model, finish with
an explicit accuracy status.

### Decide what can be proven in this environment

Use the strongest available rung:

| Environment available | What to run | What you may claim |
|---|---|---|
| Local host only, fp32/source model runnable | Source/PyTorch vs fp32 ONNX; fp32 ONNX vs QDQ/host-runnable converted model; IO-contract diff | Export, preprocessing, layout, and host conversion parity only |
| Local x86 host with QAIRT/QNN but no HTP board | Same as above; inspect DLC/context IO; optionally run CPU-runnable artifacts if the CPU backend accepts them | Still **not** final deployed HTP accuracy |
| Physical target board reachable by SSH/ADB | Run actual deployed artifact on HTP and compare outputs against the held-out reference | Deployed HTP tensor accuracy for those samples |
| AI Hub inference allowed | Submit inference on the compiled artifact/model and compare outputs | Deployed/cloud HTP accuracy for that target |

Do **not** claim final NPU accuracy from any of these alone: successful compile,
AI Hub profiling, op placement, equal parameter/MAC counts, x86 HTP prepare logs,
or an x86 run path that produced no real outputs. If the HTP-targeted DLC/context
will not run under the local CPU backend, say so and continue with ONNX/export/IO
parity rather than forcing an invalid comparison.

### Always diff the app IO contract

Before comparing numbers, print a table for each artifact:

```text
artifact | input name/shape/dtype/layout | output name/shape/dtype/layout | notes
```

Flag these as high-risk until proven by a harness:

- NHWC app IO vs NCHW app IO. The byte count can match while the image is
  scrambled.
- Float32 app IO vs Float16/native app IO. Decide whether the application casts
  or the runtime casts.
- Changed output layout, channel order, normalization, resize/crop, padding,
  token dtype, mask dtype, or raw file order.

For image models, create at least one deterministic test pattern in addition to
real held-out samples (for example channel ramps or a checkerboard) so NCHW/NHWC
or RGB/BGR swaps are visually/numerically obvious.

### End every run with this block

```text
ACCURACY STATUS: PASS | FAIL | UNPROVEN
Reference: <PyTorch/source | fp32 ONNX | golden files | task metric>
Runtime tested: <CPU ORT | QNN CPU | board HTP | AI Hub HTP | not available>
Artifacts compared: <paths / job ids>
IO contract: <input/output names, shapes, dtypes, layouts for each artifact>
Metrics: <cosine, max_abs, rtol/atol, task score, thresholds>
Commands/logs: <repro commands and output paths>
Conclusion: <what is proven>
Next action/user ask: <only if FAIL or UNPROVEN>
```

If local checks pass but board/AI Hub is unavailable, the correct conclusion is:
"local export/preprocess/layout parity passed; final deployed HTP accuracy is
UNPROVEN until we run on a board or AI Hub inference."

## Gate on host CPU before touching the device

Run the fp32 ONNX on `CPUExecutionProvider` as reference against the quantized
artifact over **held-out** samples, and gate on **both** cosine and max-abs:

```
RESULT <name>: fp32_cos=0.999922 quant_cos=0.990267 quant_maxErr=0.9412
```

**Cosine alone is scale-invariant and dominated by the largest components — it
masks localized blow-ups.** The canonical case: a shallow MLP at INT8 scored
cos **0.9918** with **maxErr 21** on an un-normalized output; promoting just that
component to INT16 gave 0.99999. Also report the argmax output dimension
(`abs_err.max(axis=0).argmax()`) so you can localize which channel was hurt.

Rule of thumb: for high-dimensional outputs (hidden states, action chunks) use
cosine as primary with max-abs as a tripwire; for low-dimensional control outputs
do the reverse.

**Say which reference you used.** Against a PyTorch/bf16 golden folds in export
error and is what deployment cares about; against the fp32 **ONNX** isolates pure
quantization error. When no golden was captured, fall back to quant-vs-fp32-ONNX
but report it under a **different result key** — the two are not comparable
against one threshold. Report fp32-ONNX-vs-golden too when a golden exists, so
export error and quantization error can be attributed separately.

Host-CPU QDQ is a proxy: HTP fuses ops and uses its own accumulator widths.
Measured, they matched closely (a toy MLP's 0.041 max error was identical host and
HTP; a DiT went 0.990 host → 0.9975 HTP) — but re-measure on device.

Two named INT8 failure signatures:

- **cos ≈ 0.04 on an LLM** → INT8 activation **compounding** through deep
  attention, not a calibration bug. Escalate to INT16.
- **high cos with huge maxErr** → outlier blowup, not adequacy.

## Layerwise diff: same model good on CPU, wrong on NPU

```bash
qnn-net-run --debug --backend libQnnCpu.so --model libQnnModelDlc.so --dlc_path m_q.dlc \
    --input_list calib.txt --output_dir out_cpu
qnn-net-run --debug --backend libQnnHtp.so --model libQnnModelDlc.so --dlc_path m_q.dlc \
    --input_list calib.txt --output_dir out_htp
# diff per-layer raw outputs; find the FIRST diverging op
```

- `--debug` makes every intermediate tensor app-readable, which **suppresses some
  HTP fusion** — numerics triage only, never timing.
- `qnn-net-run` defaults to parsing every `--input_list` file as **float32** and
  casting to the graph dtype. Pass **`--use_native_input_files`** whenever any
  input is int32/bool, or int token ids silently become zeros (all-padding) and
  your *reference* is wrong. Pair with `--output_data_type native_only` for
  non-float outputs.

## qairt-accuracy-debugger / qnn-accuracy-debugger

Six independent modes. The default (no mode flag) runs, in order:
`--framework_runner` (golden intermediates via onnx/tf/tflite) →
`--inference_engine` (target intermediates; emits `tensor_mapping.json` and
`qnn_model_graph_struct.json`) → `--verification`.

Standalone modes:
`--compare_encodings --input <qnn_model_net.json> --aimet_encodings_json <aimet.json>`
(Excel of encoding mismatches; `--params_only` / `--activations_only` /
`--specific_node`), `--tensor_inspection`, `--quant_checker`.

Verifiers: RtolAtol, AdjustedRtolAtol, TopK, L1Error, CosineSimilarity, MSE, MAE,
SQNR, ScaledDiff.

**`--quant_checker` exists only on `qnn-accuracy-debugger`** as of QAIRT
2.46.0.260424 (which is Beta and marked for deprecation); `qairt-accuracy-debugger`
has no quant_checker component. Its `--config_file` JSON keys:
`WEIGHT_COMPARISON_ALGORITHMS`, `BIAS_COMPARISON_ALGORITHMS`,
`ACT_COMPARISON_ALGORITHMS`, `INPUT_DATA_ANALYSIS_ALGORITHMS` (lists of
`{"algo_name":..., "threshold":"..."}` drawn from minmax / maxdiff / sqnr / stats /
data_range_analyzer / data_distribution_analyzer), plus `QUANTIZATION_ALGORITHMS`
(`"None"`, `"cle"`, and `"pcq"`/`"cle_pcq"` on QNN) and `QUANTIZATION_VARIATIONS`.
**Valid variations are only `tf`, `enhanced`, `adjusted`, `symmetric`** — the
`"asymmetric"` in the tools.html sample is **silently ignored**, and that sample
uses typographic quotes that aren't valid JSON. Outputs (HTML/CSV/histograms) land
in `<working-dir>/results`.

**`--binary_snooping` is QNN/SNPE-engine only** (added only when
`self.engine in [Engine.QNN, Engine.SNPE]`); for `Engine.QAIRT` the equivalent is
`--snooping <str>`. ONNX-only. It binary-searches the graph, quantizing one half
while the rest runs fp16/32. Keys in `graph_result.json` / `subgraph_result.json`
are `<start>_<end>` (`318_473` = subgraph 318→473 quantized, rest fp16/32); needs
`-qo/--quantization_overrides` and `--min_graph_size`. Culprits are the **largest
error score**, so use an error-magnitude verifier (`--verifier mse`) — with
CosineSimilarity a high score means agreement, not a culprit.

## Debug PLACEMENT, not just numerics

**Do not do graph surgery to avoid a *predicted* CPU fallback until you have
measured it.** A per-embodiment `Gather` of a (32,·,·) table with an int64 index
was expected to fragment (planned fix: freeze the index to a const-0 initializer
and constant-fold) — it actually ran 34/36 nodes on QNN at cos 0.9992. Read ORT's
node-assignment log first (`qnn_execution_provider.cc ... GetCapability`,
`qnn_node_group.cc IsSupported Operators of type X are not supported by QNN EP`).

**NaN guards fragment graphs.** PyTorch exports often leave inert
`Where(IsNaN(Softmax), const, Softmax)` guards. `IsNaN` is not QNN-supported, so 16
guards produced **17 HTP partitions** (3084/3101 nodes on QNN, 499 ms). Rewiring
each `Where` consumer straight to `Softmax` collapsed it to **1 partition**
(3004/3005 nodes) at unchanged accuracy (cos 0.990267) — but bought only **~6%**
(499 → ~466 ms). Partition hops were not the bottleneck; INT8 compute was. Before
dropping a NaN guard, verify it is genuinely inert for your inputs: a row where
every key is `-inf`-masked really does produce NaN softmax, which is why the guard
exists. And re-check concurrency — the de-partitioned build had a *larger* peak
working set and 6001'd when co-resident (see [pitfalls.md](pitfalls.md)).

**Order of attack** (surgery last): (1) provider options — `burst`, `vtcm_mb`,
graph finalization mode; (2) EPContext caching; (3) model-level cost (does this
graph run N times per inference?); (4) graph surgery.

## Reducing HTP spill/fill without editing the model

At VERBOSE QNN logging, graph prep prints:

```
====== DDR bandwidth summary ======
spill_bytes=0
fill_bytes=0
write_total_bytes=22784
read_total_bytes=171264
```

`spill_bytes=0` / `fill_bytes=0` means the prepared graph fits VTCM with no DDR
spilling; nonzero means VTCM-starved. This is the **compiler's estimate for the
prepared graph**, not a runtime counter. For measured VTCM usage and HVX/HMX
utilization use `--profiling_level=detailed --profiling_option optrace` (note:
`--profiling_option`, **singular**).

Levers:

- Raise `graphs[].vtcm_mb` (0 = device max, which requires `soc_model`/`dsp_arch`
  in `devices[]`; a `vtcm_size` in bytes **overrides** it).
- Drop A16 → A8 (16-bit activations cost 2× TCM). Weights also occupy TCM.
- For **CNNs specifically**: apply a large padding early in a long conv chain then
  VALID/zero padding downstream (2 input tiles instead of 3), and shrink **width
  and depth** — the strongest structural levers on TCM. Large activations in the
  *body* of the network, even at low input resolution, can make a graph **fail to
  prepare**, not merely run slowly.

`read_total_bytes` doubles as the size diagnostic when a graph 6001s on execute
(~17.8 GB for a monolithic 2B decoder vs ~0.5 GB for a 4-layer chunk).

## Profiling levels for `qnn-net-run`

- `basic` — total execution time (µs).
- `detailed` — + per-op cycles.
- lint — needs **both** `--profiling_level=backend` on the CLI **and**
  `"profiling_level": "linting"` in the HTP config's `devices[]`.
- `optrace` — `--profiling_level=detailed --profiling_option optrace`, the only
  level giving real VTCM usage and HVX/HMX utilization.

Read the log with
`qnn-profile-viewer --reader libQnnHtpProfilingReader.so --input_log
<dir>/qnn-profiling-data_0.log --output profile.csv`, or
`libQnnChrometraceProfilingReader.so` for an averaged chrometrace JSON.

## Gate on task behaviour, not tensors alone

Run the simulator/task benchmark against the quantized graphs at matched seeds and
gate on **the statistic your consumer actually uses** — sign, argmax, saturation.

Two measured data points:

- A learned CartPole policy showed **~0.25** max INT8-vs-FP32 output error on a
  ±1 action scale (vs 0.041 for a toy MLP) yet held the pole for the full 500-step
  cap over 20 episodes on CPU **and** HTP (~0.58 ms) — because the controller is
  bang-bang and only the *sign* matters. Where magnitude is load-bearing
  (continuous torque, velocity targets), 0.25 is not free.
- A GR00T N1.7 quantized action head matched fp32 at 10/10 on two LIBERO tasks.
  Caveats worth preserving whenever quoting a number like that: the run used
  ONNX-on-CPU for the quantized submodules with the backbone still fp32 (not the
  on-HTP build), and n=10 over 2 of 10 tasks is too small to compare against the
  published 189/200 = 94% baseline. Read it as "no observed regression", not as a
  superior score.

**Cheapest harness:** instantiate the **real fp32 policy** — so preprocessing,
glue, and un-normalization stay byte-identical to the reference — then
monkey-patch only the quantized submodules to onnxruntime CPU sessions running the
very same QDQ `.onnx` files you ship. That isolates quantization error from
HTP-backend error. It does not replace on-device validation.
