# quantize-for-hexagon-npu

A portable skill for getting a neural network quantized and running on the **Qualcomm Hexagon NPU
(HTP)**. It covers all three compile routes — Qualcomm AI Hub cloud jobs, the local QAIRT/QNN
toolchain, and ONNX Runtime's QNN Execution Provider with EPContext AOT caching — plus SoC targeting,
calibration recipes, a validation ladder, and a pitfalls section keyed by the actual error strings you
will see. One `SKILL.md` (Agent Skills standard) serves two harnesses:

- **Claude Code** as a plugin (this repo is also a plugin marketplace).
- **qGenie / Codex CLI** as an installable skill (`$skill-installer`).

## Layout

```
quantize-for-hexagon-npu/
  .claude-plugin/
    marketplace.json                  # marketplace catalog; the plugin source is the repo root ("./")
    plugin.json                       # plugin manifest (skills auto-discovered from skills/)
  skills/
    quantize-for-hexagon-npu/
      SKILL.md                        # decision guide + universal rules (auto-loads when relevant)
      agents/openai.yaml              # qGenie/Codex UI metadata (display name, prompt)
      scripts/
        fetch-latest-qairt.sh         # resolve + download the latest QAIRT SDK (no hardcoded version)
      reference/                      # loaded on demand
        aihub.md                      # Path A: AI Hub compile/quantize jobs, LLM + AIMET flow
        qairt.md                      # Path B: qairt-converter/quantizer, context binaries, HTP config
        ort-qnn-ep.md                 # Path C: QNN EP options, EPContext, proving NPU placement
        quantization-recipes.md       # calibration, bit-width choice, transformer MatMul-only recipe
        soc-targets.md                # name strings vs dsp_arch vs soc_model; VTCM
        accuracy-debug.md             # validation ladder, layerwise diffing, spill/fill
        large-models.md               # >2 GB ONNX, graph splitting, concurrency budget
        pitfalls.md                   # symptom -> cause -> fix, by error string
  README.md
```

## Install in Claude Code (team)

This repo is a marketplace. Add it once, then install the plugin:

```text
/plugin marketplace add ramalamadingdong/quantize-for-hexagon-npu
/plugin install quantize-for-hexagon-npu@hexagon-npu
```

The skill auto-loads when you work on NPU quantization/compilation, or invoke it with
`/quantize-for-hexagon-npu:quantize-for-hexagon-npu`.

## Install in qGenie / Codex CLI

qGenie's agent runs the Codex/Agent Skills format. Use its built-in installer from inside the agent:

```text
$skill-installer install --repo ramalamadingdong/quantize-for-hexagon-npu --path skills/quantize-for-hexagon-npu
```

This copies the skill into `~/.config/qgenie-cli/agent/skills/quantize-for-hexagon-npu/` (restart the
agent to pick it up), then trigger it with `$quantize-for-hexagon-npu`. Manual alternative: copy
`skills/quantize-for-hexagon-npu/` into that skills directory.

## What it is opinionated about

- **QAIRT native is the default**, whenever the host can run it: it is the vendor toolchain the other
  two routes are built on, fully offline, version-pinnable, and the only route giving per-op encoding
  overrides, `.dlc` output, and Genie/LLM bundles. `SKILL.md` opens with a host-support table
  (x86_64 Linux, x86_64/arm64 Windows — *not* aarch64 Linux, *not* macOS) and the specific reasons to
  fall back to AI Hub (AIMET-grade PTQ, W4A16 LLMs needing a 40-80 GB GPU) or ORT QNN EP (already have
  QDQ ONNX, want to iterate on the board, want EPContext caching).
- **Never hardcode an SDK version.** `scripts/fetch-latest-qairt.sh` resolves the newest QAIRT build:
  a parallel ranged-GET sweep that needs no auth walks minor versions upward (seeded from AI Hub's
  `get_frameworks()` when a token is configured — AI Hub lags the download site, so it is only a floor). The download URL embeds a build *date* and the endpoint has no
  listing API, so a pinned version in a doc is guaranteed to go stale.
- **Read SoC targeting out of the SDK, not out of a table.** The `Qnn_SocModel_t` integer enum is
  marked `@deprecated ... will no longer be updated` and has no entry for QCS8275 or QCS9075 — those
  parts are targeted by SoC *name string*. `soc-targets.md` gives the commands to enumerate both from
  your own install, and explains why 67/82/77 keep appearing for the Dragonwing EVKs (family aliasing,
  not a bug).
- **For transformers, the lever is quantizing MatMuls only** — not the bit width. Naive MinMax over
  all op types collapsed a measured DiT to cos 0.85; MatMul-only reached 0.999921 against the fp32
  ONNX reference's own 0.999922. The `±inf` attention mask is the root cause, and it corrupts silently.
- **Verify placement; don't assume it.** `sess.get_providers()` proves nothing. Sessions were observed
  to "succeed" with every node on CPU after QNN device creation had already failed.
- **Accuracy closure is mandatory.** The skill tells the agent to plan accuracy at the start, preserve
  held-out parity inputs and commands, diff every artifact's app IO contract, run the strongest
  available check at handoff, and end with `ACCURACY STATUS: PASS | FAIL | UNPROVEN`. Local-only
  export/preprocess/layout parity is useful, but final deployed HTP accuracy requires a board or AI Hub
  inference.
- **Gate on cosine *and* max-abs-error.** Cosine is scale-invariant and masks localized blow-ups — a
  measured component passed at cos 0.9918 while carrying maxErr 21.

## Provenance and caveats

Distilled from bring-ups on QCS8275 (v75), QCS9075 (v73) and SA8797P (v81), then reconciled against
QAIRT 2.46.0.260424, `qai-hub` 0.52.0 / `qai-hub-models` 0.58.0, the AI Hub documentation, and
ONNX Runtime source. Refreshed 2026-11-16 against QAIRT 2.52.0.261104, `onnxruntime-qnn` 2.8.0
(plugin EP; registration + AOT compile checked on an x86_64 host), `qai-hub` 0.58.0 /
`qai-hub-models` 0.66.0 and the AI Hub release notes (compile → link replaces the removed
`qnn_context_binary` target; `submit_profile_job` → `submit_inference_job(profile=True)`).

- Numbers labelled **measured** come from one board and one SDK/ORT version. They are existence
  proofs, not specifications — re-measure on your part. Several are explicitly marked `n=1`.
- Flag names, defaults and schema keys were verified against a QAIRT 2.46 install. Anything
  version-sensitive is marked `(verify for your SDK version)` rather than dropped.
- Where a claim is unresolved or a vendor doc contradicts the shipped tool, the skill says so instead
  of picking a side silently.
