# AIMET Workflow

This note describes the practical AIMET workflow in this repo, with emphasis on the lowest-risk path when the source model code has been modified and rebuilding the full PyTorch/AIMET graph is unreliable.

## Goal

Use `build_sim_quantized_pdl.py` to:

1. Load the FP32 Panoptic-DeepLab checkpoint.
2. Build an AIMET `QuantizationSimModel`.
3. Calibrate it with representative Cityscapes images.
4. Optionally recover accuracy with PTQ improvements such as CLE, BN fold, Bias Correction, AdaRound, SeqMSE, and BN re-estimation.
5. Export deployable artifacts.
6. Evaluate the quantized result, preferably from exported ONNX when local model code is unstable.

## Recommended Mental Model

There are two phases:

1. `Quantize + export`
   Run AIMET once in the environment where the model still builds correctly.
2. `Evaluate exported artifact`
   Prefer evaluating the exported ONNX afterward, because it is less coupled to the Python model source.

That second phase is the safest fallback for your current situation.

## Prerequisites

Run:

```bash
bash setup.sh
```

This installs Python dependencies, Cityscapes helpers, `neural-compressor`, and downloads the default FP32 checkpoint to `weights/model_final_bd324a.pkl`.

For evaluation and meaningful calibration, prepare Cityscapes with this structure:

```text
<cityscapes_root>/
  leftImg8bit/
    train/
    val/
  gtFine/
    val/
```

The repo includes `quantization/downloader.py` for authenticated Cityscapes downloads.

## Main AIMET Entry Point

The canonical script is:

```bash
python build_sim_quantized_pdl.py ...
```

Important inputs:

- `--calib_images`: calibration image file or directory
- `--weights_path`: FP32 checkpoint, default `weights/model_final_bd324a.pkl`
- `--model_category`: `PANOPTIC_DEEPLAB` or `DEEPLAB_V3_PLUS`
- `--image_height`, `--image_width`
- `--num_calib`, `--calib_size`
- `--batch_size`, `--num_workers`
- `--config_file`: AIMET quantizer config JSON
- `--export_path`, `--export_prefix`

## Actual Workflow Inside `build_sim_quantized_pdl.py`

The script currently does this:

1. Load the FP32 model with `build_model(...)`.
2. Optionally fold custom `Conv2d(norm=...)` wrappers with `--enable_custom_conv_bn_fold`.
3. Collect, deduplicate, and sample calibration images.
4. Build the calibration dataloader.
5. Optionally apply Cross-Layer Equalization with `--enable_cle`.
6. Wrap the model with `AimetTraceWrapper` so AIMET sees tensor-only outputs.
7. Optionally fold BatchNorm with `--enable_bn_fold`.
8. Optionally apply Bias Correction with `--enable_bias_correction`.
9. Optionally run AdaRound with `--enable_adaround`.
10. Build the AIMET `QuantizationSimModel`.
11. Optionally run SeqMSE with `--enable_seq_mse`.
12. Compute activation encodings from calibration data.
13. Optionally re-estimate BN stats with `--enable_bn_reestimation`.
14. Optionally save the AIMET sim checkpoint with `--save_quant_checkpoint`.
15. Export the quantized model to ONNX QDQ and also export the AIMET `sim_export` bundle.

## Artifact Outputs

If export is enabled, the script writes:

- `<export_path>/<export_prefix>.onnx`
  This is the exported QDQ ONNX model. This is the preferred artifact for later evaluation when Python model code is unstable.
- `<export_path>/sim_export/...`
  AIMET export bundle from `sim.export(...)`.
- `--save_quant_checkpoint <path>`
  Optional pickled AIMET sim checkpoint. Useful if you want to resume AIMET-side inspection, but it still depends more on the Python/AIMET environment than ONNX evaluation does.

## Recommended Baseline Command

This is a sensible starting point for your repo:

```bash
python build_sim_quantized_pdl.py \
  --calib_images /path/to/cityscapes/leftImg8bit/train \
  --num_calib 100 \
  --model_category PANOPTIC_DEEPLAB \
  --image_height 512 \
  --image_width 1024 \
  --export_path ./quantized_export \
  --export_prefix panoptic_deeplab_int8 \
  --config_file ./config/fully_symmetric.json \
  --enable_custom_conv_bn_fold \
  --enable_bn_fold \
  --enable_cle \
  --enable_bn_reestimation
```

## Accuracy-Recovery Knobs

Use these only as needed:

- `--enable_custom_conv_bn_fold`
  Good first step for this repo because some convolutions carry internal BN wrappers.
- `--enable_bn_fold`
  Standard PTQ optimization.
- `--enable_cle`
  Helpful for difficult layers before calibration.
- `--enable_bias_correction`
  Useful when bias shift causes visible accuracy drop.
- `--enable_bn_reestimation`
  Often helpful after quantization.
- `--enable_adaround`
  Stronger weight PTQ recovery, but slower.
- `--enable_seq_mse`
  Useful activation/range tuning, but do not combine with AdaRound in this script.

Important constraint:

- `--enable_seq_mse` and `--enable_adaround` are mutually exclusive in the current implementation.

## Recommended Evaluation Strategy

When local model source has drifted, prefer:

1. Export ONNX once from the working environment.
2. Evaluate the exported ONNX later with `run_eval.py`.
3. Skip `--fp32_weights` unless you specifically want FP32 comparison.

Why this is safer:

- ONNX evaluation does not need to rebuild the quantized PyTorch graph.
- It avoids dependence on an AIMET checkpoint for normal benchmarking.
- It better matches deployment-time behavior.

## Exported-Only Eval

The safest command is:

```bash
python run_eval.py \
  --cityscapes_root /path/to/cityscapes \
  --quant_weights ./quantized_export/panoptic_deeplab_int8.onnx \
  --model_category PANOPTIC_DEEPLAB \
  --image_height 512 \
  --image_width 1024 \
  --split val \
  --onnx_provider CPUExecutionProvider
```

Notes:

- You can switch to `--onnx_provider CUDAExecutionProvider` if ONNX Runtime GPU is installed correctly.
- If you omit `--fp32_weights`, the script evaluates only the exported model.
- This repo’s `run_eval.py` has been adjusted so `.onnx` evaluation does not require importing the AIMET quantized loader first.

## FP32 vs INT8 Comparison

If the local FP32 model path still works, you can compare both:

```bash
python run_eval.py \
  --cityscapes_root /path/to/cityscapes \
  --fp32_weights ./weights/model_final_bd324a.pkl \
  --quant_weights ./quantized_export/panoptic_deeplab_int8.onnx \
  --model_category PANOPTIC_DEEPLAB \
  --image_height 512 \
  --image_width 1024 \
  --split val
```

This prints:

- `mIoU`
- `FPS`
- `Avg_Inference_Time_ms`
- FP32 vs INT8 delta

PCC is only computed when the quantized backend is PyTorch. It is skipped for ONNX in the current script.

## Suggested Operating Modes

Use one of these modes depending on repo stability:

### Mode 1: Full AIMET tuning

Use when the model code still builds and you want best PTQ recovery.

- Run `build_sim_quantized_pdl.py`
- Tune CLE / BN fold / BN re-estimation / Bias Correction / AdaRound / SeqMSE
- Export ONNX
- Evaluate ONNX

### Mode 2: Export-first, eval-later

Use when source code sometimes breaks after local library/model edits.

- Quantize and export once in the last known-good environment
- Preserve the `.onnx` artifact
- Use `run_eval.py --quant_weights <exported.onnx>` for later benchmarking

### Mode 3: AIMET checkpoint debugging

Use only when you specifically need AIMET internals again.

- Save `--save_quant_checkpoint`
- Reload it later through `run_eval.py` or AIMET tooling
- Expect stronger dependency on Python package and model compatibility

## Practical Recommendations For This Repo

- Prefer `config/fully_symmetric.json` as the documented baseline if that is your current working config.
- Keep calibration and evaluation image sizes aligned with the export size.
- Start with `100` to `800` calibration images and increase only if needed.
- Treat exported ONNX as the handoff artifact for downstream validation.
- Use AIMET checkpoint export only as a debugging artifact, not the main deployment artifact.

## Current Local Reality

In this workspace, exported ONNX artifacts already exist under `quantized_model/`, but Cityscapes evaluation data is not present locally, so full evaluation cannot be run here until `leftImg8bit` and `gtFine` are available.

Once the dataset is present, the exported-only eval path is the lowest-risk command to run first.
