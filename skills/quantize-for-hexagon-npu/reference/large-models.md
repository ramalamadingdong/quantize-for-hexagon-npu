# Models too large for one graph

## >2 GB ONNX: two distinct failures

```python
quantize_static(fp32, out, calibration_data_reader=Reader(feeds),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt16, weight_type=QuantType.QInt16,
                use_external_data_format=True,        # (1)
                op_types_to_quantize=["MatMul"],
                calibrate_method=CalibrationMethod.MinMax)
```

**1. Calibration fails, not quantization.** The calibrator writes one monolithic
`/tmp/ort.quant.*/augmented_model.onnx` and cannot reload it past the 2 GB
protobuf limit → `INVALID_PROTOBUF : ... Protobuf parsing failed`.
`use_external_data_format=True` makes both the augmented calibration model and the
output use external data. It **defaults to False** in `quantize_static` — only
`get_qdq_config` auto-sets it. Also ensure `TMPDIR` has several GB free, and that
the fp32 input was itself saved with external data (a bf16→fp32 conversion roughly
doubles size, which is what pushes a ~2 GB model over).

**2. The 16-bit opset bump fails.** ORT calls `update_opset_version` →
`onnx.version_converter.convert_version(model, 21)`, which serializes the whole
model to one protobuf string → `RuntimeError: ... onnx version converter is unable
to parse input model`. Pre-bump yourself:

```python
def bump_opset(path, target=21):
    m = onnx.load(path, load_external_data=True)
    changed = False
    for oi in m.opset_import:
        if oi.domain in ("", "ai.onnx") and oi.version < target:
            oi.version = target; changed = True
    if changed:
        onnx.save(m, path, save_as_external_data=True,
                  all_tensors_to_one_file=True, location=Path(path).name + ".data")
```

`UseQDQContribOps` does **not** dodge this — ORT skips contrib ops for 16-bit
precisely because it plans to bump. Caveat: an in-place `opset_import` edit skips
real op conversion, so it is valid only when no op in your graph has a breaking
spec change up to 21 (Clip, Pad, Resize, ReduceSum, Squeeze moved axes
attribute→input). Run `onnx.checker` and compare fp32 outputs before/after.

**Compact external data after any graph surgery.** `onnx`'s writer opens the
`.data` file `r+b`, seeks, and **never truncates**, so re-saving over an existing
`.data` leaves stale trailing bytes (observed 3.14 GB where 1.57 GB was needed):

```python
m = onnx.load(p, load_external_data=True)
os.remove(p + ".data")
onnx.save(m, p, save_as_external_data=True, all_tensors_to_one_file=True,
          location="<name>.onnx.data", size_threshold=1024)
```

Assert the on-disk `.data` size before spending time transferring multi-GB
artifacts.

## Pin static shapes for HTP

```python
def fix_input_shapes(path, feed):
    """Pin symbolic input dims to the concrete calibration shapes."""
    m = onnx.load(path, load_external_data=True)
    for vi in m.graph.input:
        if vi.name not in feed: continue
        arr = feed[vi.name]
        for i, d in enumerate(vi.type.tensor_type.shape.dim):
            if i < arr.ndim and (d.HasField("dim_param") or d.dim_value == 0):
                d.ClearField("dim_param"); d.dim_value = int(arr.shape[i])
    onnx.save(m, path, save_as_external_data=True, all_tensors_to_one_file=True,
              location=Path(path).name + ".data")
```

Or use `python3 -m onnxruntime.tools.make_dynamic_shape_fixed`.

## Splitting a large decoder

For **text generation**, use the vendor path: AI Hub/Genie exports prompt-processor
+ token-generator components as multi-part W4A16 context binaries in a
`genie_bundle`. That path is text-in/text-out only — it **cannot return embeddings
or hidden states**. (QAIRT also ships a text-embedding path, `GenieEmbedding.h` /
`genie-t2e-run`, but it is text-query-in / one-embedding-out and cannot take
image/state tensors, tap a chosen decoder layer, or run a non-LLM head.)

For **intermediate tensors** (feeding a downstream action head), split the ONNX
yourself at residual boundaries and run the chunks through ORT-QNN:

```python
from onnx.utils import extract_model
# Cut at post-block residual adds. Chunk0 must ALSO export the per-forward shared
# tensors later chunks cannot recompute: RoPE cos/sin and the additive causal mask.
COS  = "/layers.0/self_attn/Unsqueeze_6_output_0"
SIN  = "/layers.0/self_attn/Unsqueeze_7_output_0"
MASK = "/Add_output_0"
ENTER = {0: "inputs_embeds", 4: "/layers.3/Add_1_output_0",
         8: "/layers.7/Add_1_output_0", 12: "/layers.11/Add_1_output_0"}
# chunk0:       wrapper inputs        -> [hidden, COS, SIN, MASK]
# chunk1..N-1:  [hidden,COS,SIN,MASK] -> [hidden]
extract_model(SRC, dst, ins, outs)
```

**Why the aux tensors:** `extract_model` does pure reachability from the declared
inputs, and the cos/sin/mask producers depend on `position_ids` / `attention_mask`,
which are **not** inputs to an interior chunk — so extraction fails or leaves
dangling names. The alternative (feeding `position_ids`/`attention_mask` into every
chunk) duplicates the RoPE/mask subgraph in each.

**These tensor names are artifacts of one `torch.onnx.export` run** at one opset
with one exporter version. `/Add_output_0` in particular could resolve to an
unrelated node. Rediscover them by inspecting your graph.

**Gate the split before quantizing.** A correct split is numerically exact — chain
the fp32 chunks against the whole (bf16→fp32) model and expect cos ≈ **1.0**
(measured 1.000000, maxErr 0). Anything materially below that is a wiring bug
(wrong boundary tensor, missing cos/sin/mask), not acceptable loss — do **not**
accept a loose `cos > 0.999` here. Only then quantize each chunk and re-gate the
quantized chain against the same fp32 reference at a looser threshold (measured
**0.999937** for INT16 MatMul-only).

Calibration feeds for interior chunks must be **produced**, not captured: run the
fp32 chain over the captured whole-model inputs and hand the resulting dicts to the
reader.

`extract_model` loads and re-serializes the whole model, so a >2 GB source needs
external-data handling.

**Measured on QCS8275:** 4 chunks of 4 layers, ~403 MB each, ~113 ms/chunk on HTP
in default mode (~493 ms for the chain vs **5583 ms** on the board CPU; ~208 ms
total under burst). Chunk count is a bisection starting point, not a constant. The
monolithic 2B failed to execute at all (6001, ~17.8 GB `read_total_bytes`).

## Borrow Qualcomm's own LLM split pattern

`qai_hub_models/models/_shared/{llm,qwen3_vl}`:

- AIMET-ONNX at W4A16 (`default_precision = Precision.w4a16`).
- **Two instantiations of the same decoder** — prompt processor
  (`sequence_length > 1`) and token generator (`sequence_length == 1`,
  `is_token_generator=True`).
- Each split into N parts of `num_layers_per_split` layers; graph names
  `{prompt|token}_ar{seq}_cl{ctx}_{i}_of_{N}`.
- LM head optionally its own part, and converted from Linear to **Conv2d** (as are
  the MLP/QKV projections).

Splitting there is about per-graph ONNX/context-binary size and memory, **not VTCM
specifically** — borrow the pattern, not a byte limit.

## Expect a thin host/CPU glue layer

For a VLA/VLM pipeline, these stay on the host by choice, not because HTP can't:

- `embed_tokens` lookup (the vocab table is huge, not because HTP lacks Gather)
- `masked_scatter` of image embeds into `inputs_embeds` at `visual_pos_masks`
- data-dependent position-index computation
- a LayerNorm sitting exactly on a graph boundary
- the denoise / decode loop

Everything compute-heavy stays on the NPU. Both NVIDIA's ONNX export and
Qualcomm's Qwen-VL HTP recipe push the same ops host-side.

To run a board-side pipeline without installing torch/transformers on a
memory-tight device, seed a NumPy runner from **one** host CPU capture: obs-derived
seeds, the **exact init noise the loop used** (capture the actions tensor at the
FIRST encoder call, not a re-drawn random), the golden output, and the glue
weights/scalars. One capture pins one static sequence length, so closed-loop use
needs the padding rule in
[quantization-recipes.md](quantization-recipes.md#fixed-shapes-vs-variable-length-inputs).

## Multi-graph input ordering is a silent-corruption trap

When a runtime takes one flat input buffer per graph, it slices using the
**per-graph input-tensor order recorded in the context binary** — not your
declaration order and not the producing graph's output order. A mismatch is a
silent reinterpretation: plausible garbage, no error.

Tensor names sort **lexicographically**, so the real order is
`key_cache_l0, l1, l10, l11 ... l17, l2 ... l9` — feeding `l0..l17` numerically
scrambles most layers. Names and cos/sin positions can also differ between
producer and consumer.

Generate the order by introspecting the binaries into a header and `static_assert`
it against the shapes your code assumes. To prove packing (not just numerics):
dump each stage's exact flat input buffer, re-split it into per-tensor `.raw` files
using the graph's **recorded** input order, replay through `qnn-net-run` on the
same backend/context binary, and diff. Byte-identical output proves the packing.
`--use_native_input_files` is mandatory for int32 inputs.

## Concurrency budget

VTCM is a fixed on-chip budget (8 MB accepted on QCS8275/v75 with QAIRT 2.46) and
each resident HTP context consumes some. Consequences measured on that board:

- The backbone ran **one context at a time** (load → run → free), which let it use
  `htp_performance_mode:"burst"` safely — a ~2.6× per-op win (3752 ms → 1458 ms).
- The action-head loop held **3 concurrent** contexts and 6001'd under burst, so it
  stayed at default mode (2054 ms). Net E2E 3512 ms/chunk, all the gain from the
  backbone.
- Loading all 7 components, or even 3 with 2 big ones, got **SIGKILLed** — that one
  is host DDR (10 GB board), not VTCM.

The durable fix for both is **EPContext caching**: pre-compile each session's
context so sessions load→run→free quickly and never more than one is resident,
which also kills the ~10–15 min recompile per run.
