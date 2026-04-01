# QuantizedPDL v2 — Quantization Workflows

The repository implements quantization for the PDL model using PyTorch, AIMET and further libraries like ONNX, INC ONNX. Model detail can be found in the paper: https://arxiv.org/pdf/1911.10194

This document explains the three main quantization/export entrypoints in `QuantizedPDL_v2`:

* **AIMET**: `build_sim_quantized_pdl.py`
* **ONNX Runtime PTQ**: `build_quantized_pdl.py`
* **INC for ONNX**: `build_neural_compressed_pdl.py`

The goal of all three flows is the same: start from the FP32 PDL checkpoint, prepare a representative calibration set, and export an INT8-ready model for deployment or further evaluation.

---
## Enviroment set up:
Please run: 
```
bash setup.sh
```
This one will be included:
* Libraries installation.
* Original fp32 weights downloading.

After finished install libraries and download weight, please download data with below scripts:
```
!python "/kaggle/working/QuantizedPDL_v2/quantization/downloader.py" -d "/kaggle/working/" leftImg8bit_trainvaltest.zip
!mkdir -p /kaggle/working/cityscapes
!unzip -q -o /kaggle/working/leftImg8bit_trainvaltest.zip "leftImg8bit/train/*" -d /kaggle/working/cityscapes
!unzip -q -o /kaggle/working/leftImg8bit_trainvaltest.zip "leftImg8bit/val/*" -d /kaggle/working/cityscapes
!rm /kaggle/working/leftImg8bit_trainvaltest.zip
!python "/kaggle/working/QuantizedPDL_v2/quantization/downloader.py" -d "/kaggle/working/" gtFine_trainvaltest.zip
!unzip -q -o /kaggle/working/gtFine_trainvaltest.zip "gtFine/val/*" -d /kaggle/working/cityscapes
!rm /kaggle/working/gtFine_trainvaltest.zip
```

Please note that downloader.py requires user name and password to download datas from citiscapes cite
If you don't got an account yet, please create one here https://www.cityscapes-dataset.com/register/
---

## 1. Repository purpose

`QuantizedPDL_v2` implements post-training quantization flows for the PDL model family. The repo currently supports:

* **PyTorch + AIMET simulation/export** for model-side quantization analysis and tuning
* **ONNX Runtime static quantization** for an ONNX-native PTQ path
* **Intel Neural Compressor (onnx-neural-compressor)** for an alternative ONNX quantization path with configurable node exclusion

The repo also includes helper utilities for:

* calibration dataset creation and sampling
* evaluation on Cityscapes-style data
* custom Conv+BN folding for wrapped convolution blocks
* optional Cross-Layer Equalization, SeqMSE, AdaRound, Bias Correction, and QuantAnalyzer analysis

---

## 2. Common workflow across all three scripts

All three scripts follow roughly the same high-level pipeline:

1. **Load the FP32 PDL model** from the training checkpoint.
2. **Build a dummy input** using the configured image size.
3. **Collect calibration images** from a file or folder.
4. **Sample a representative calibration subset** using `--num_calib` and `--seed`.
5. **Create a calibration loader** with the configured batch size and worker count.
6. **Optionally apply model preprocessing** such as custom Conv+BN folding, batch norm folding, cross-layer equalization, or SeqMSE.
7. **Export FP32 ONNX or build a quantization simulation** depending on the selected flow.
8. **Run calibration / compute encodings / quantize**.
9. **Export artifacts** and optionally run diagnostics such as QuantAnalyzer.

### Shared required inputs

At minimum, the flows expect:

* a trained PDL checkpoint (`--weights_path` or the default repo checkpoint)
* calibration images (`--calib_images`)
* image size (`--image_height`, `--image_width`)
* model type (`DEEPLAB_V3_PLUS` or `PANOPTIC_DEEPLAB`)

---

## 3. AIMET flow — `build_sim_quantized_pdl.py`

### What it is for

Use the AIMET flow when you want the richest quantization workflow on the PyTorch model itself:

* build a QuantSim model
* compute encodings from calibration data
* apply advanced PTQ improvements
* inspect sensitivity with QuantAnalyzer
* export quantized artifacts from AIMET

This is the most feature-rich pipeline in the repo.

### Main capabilities

`build_sim_quantized_pdl.py` supports:

* calibration image loading and sampling
* optional **custom Conv+BN folding** before AIMET steps
* optional **Cross-Layer Equalization (CLE)**
* optional **batch norm folding**
* optional **Bias Correction**
* optional **AdaRound**
* optional **Sequential MSE (SeqMSE)**
* optional **BN re-estimation**
* optional **QuantAnalyzer** evaluation against Cityscapes
* optional export of the quantized model and checkpoint

### Key inputs

Important arguments include:

* `--calib_images`: file or directory used for calibration
* `--weights_path`: FP32 checkpoint path
* `--model_category`: `DEEPLAB_V3_PLUS` or `PANOPTIC_DEEPLAB`
* `--image_height`, `--image_width`
* `--num_calib`, `--batch_size`, `--num_workers`, `--seed`
* `--quant_scheme`
* `--default_output_bw`, `--default_param_bw`
* `--export_path`, `--export_prefix`
* `--config_file`

Optional optimization flags:

* `--enable_custom_conv_bn_fold`
* `--enable_cle`
* `--enable_bn_fold`
* `--enable_bias_correction`
* `--enable_adaround`
* `--enable_seq_mse`
* `--enable_bn_reestimation`
* `--run_quant_analyzer`

### Flow details

1. **Load the FP32 model** with `build_model(...)`.
2. **Optionally fold custom Conv+BN wrappers** before AIMET processing.
3. **Collect and sample calibration images**.
4. **Create the calibration dataloader**.
5. **Optionally apply CLE** before building QuantSim.
6. **Wrap the model for AIMET tracing**.
7. **Optionally fold batch norm** into the traced model.
8. **Optionally run Bias Correction** using a temporary quantized copy.
9. **Optionally run AdaRound** to generate weight encodings.
10. **Optionally run SeqMSE** to improve quantization ranges before encoding computation.
11. **Create QuantSim and compute encodings** using calibration data.
12. **Optionally re-estimate BN statistics** and fold BN effects into quantization scales.
13. **Optionally run QuantAnalyzer** using Cityscapes evaluation.
14. **Export the quantized model** unless `--no_export` is used.

### When to use AIMET

Choose this flow when you need:

* the strongest PTQ tuning options
* PyTorch-side debugging and sensitivity analysis
* QuantAnalyzer reports
* experimentation with AdaRound / Bias Correction / SeqMSE

### Minimal example

```bash
!mkdir /kaggle/working/quantized_model
!mkdir /kaggle/working/checkpoint
!cd /oath/to/QuantizedPDL_v2 && python build_sim_quantized_pdl.py \
    --calib_images /kaggle/working/cityscapes/leftImg8bit/train --num_calib 100 \
    --export_path /kaggle/working/quantized_model \
    --save_quant_checkpoint /kaggle/working/checkpoint/panoptic_deeplab_int8_state_dict.pkl \
    --enable_bn_fold \
    --enable_cle \
    --enable_custom_conv_bn_fold \
    --enable_bn_reestimation \
    --config_file /kaggle/working/QuantizedPDL_v2/config/fully_symmetric.json
    # --enable_seq_mse
    # --enable_bn_reestimation \
    # --enable_adaround \
    # --enable_cle \
```

### Notes

* `--enable_seq_mse` and `--enable_adaround` are mutually exclusive in the script logic.
* `--run_quant_analyzer` requires `--cityscapes_root`.
* This is the best entrypoint when model quality recovery matters more than export simplicity.

---

## 4. ONNX Runtime PTQ flow — `build_quantized_pdl.py`

### What it is for

Use this flow when you want an **ONNX-native static quantization** path built around **ONNX Runtime**. It exports FP32 ONNX first, optionally preprocesses the graph, and then runs static PTQ using a calibration data reader.

This path is useful when your deployment target is ONNX Runtime or when you want quantization decisions to happen directly in the ONNX graph.

### Main capabilities

`build_quantized_pdl.py` includes:

* FP32 model export to ONNX
* calibration loader to ONNX `CalibrationDataReader` bridge
* ONNX Runtime `quant_pre_process(...)`
* ONNX Runtime `quantize_static(...)`
* optional **SmoothQuant** preprocessing
* optional **SeqMSE** on the PyTorch model before ONNX export
* optional **custom Conv+BN folding**
* optional **Cross-Layer Equalization**
* optional **QuantAnalyzer** support
* tensor quantization overrides through JSON
* configurable symmetric range behavior and minimum real range

### Key inputs

Core arguments include:

* `--weights_path`
* `--calib_images`
* `--model_category`
* `--image_height`, `--image_width`
* `--num_calib`, `--calib_max_samples`
* `--batch_size`, `--num_workers`, `--seed`
* `--export_path`
* `--fp32_name`
* `--preprocessed_name`
* `--quant_name`

Quantization options include:

* `--quant_format`
* `--activation_type`
* `--weight_type`
* `--tensor_quant_overrides_json`
* `--calib_tensor_range_symmetric`
* `--min_real_range`

### Flow details

1. **Load the PyTorch PDL model**.
2. **Optionally apply custom Conv+BN folding / CLE / SeqMSE** before export.
3. **Export the model to FP32 ONNX**.
4. **Create the calibration loader** from sampled calibration images.
5. **Convert calibration batches into NumPy** through `LoaderCalibrationDataReader`.
6. **Run ONNX preprocessing** with `quant_pre_process(...)`.
7. **Apply static quantization** with `quantize_static(...)`.
8. **Save the final INT8 ONNX model** and print ONNX op statistics.

### Why this flow exists

Compared with the AIMET path, this script is closer to deployment reality for ONNX Runtime:

* quantization is performed on the exported ONNX graph
* calibration happens through ONNX Runtime APIs
* graph-level options such as SmoothQuant and tensor override JSON can be applied before final export

### Minimal example

```
!cd /path/to/QuantizedPDL_v2 && python build_quantized_pdl.py \
    --weights_path /kaggle/working/QuantizedPDL_v2/weights/model_final_bd324a.pkl \
    --calib_images /kaggle/input/datasets/ippapi/cityscrapes/kaggle/working/cityscapes/leftImg8bit/train \
    --cityscapes_root /kaggle/working/cityscapes \
    --num_calib 50 \
    --export_path /kaggle/working/quantized_model \
    --enable_custom_conv_bn_fold \
    --activation_symmetric \
    --weight_symmetric \
    --per_channel \
    --calibration_method minmax \
    --force_qoperator
```

### Notes

* This script bridges your dataloader to ONNX Runtime using a custom `LoaderCalibrationDataReader`.
* It exposes more ONNX-graph-specific controls than the AIMET script.
* Use this path when the final consumer is ONNX Runtime and you want a direct PTQ export path.

---

## 5. INC ONNX flow — `build_neural_compressed_pdl.py`

### What it is for

Use this flow when you want ONNX quantization driven by **Intel Neural Compressor for ONNX** (`onnx-neural-compressor`).

This script keeps the same general PDL export/calibration flow, but swaps the final quantization stage to INC. It is especially useful when you want finer control over which nodes are quantized or excluded.

### Main capabilities

`build_neural_compressed_pdl.py` includes:

* FP32 ONNX export
* calibration loader to ONNX/NumPy conversion
* ONNX preprocessing with `quant_pre_process(...)`
* static quantization through `onnx_neural_compressor.quantization.quantize`
* custom quant config via `INCExcludeOutputQuantConfig`
* node exclusion support (`nodes_to_exclude`)
* selectable op types to quantize
* optional custom Conv+BN folding
* optional CLE
* optional SeqMSE
* optional QuantAnalyzer support

### Key implementation detail

The script defines a custom config class:

* `INCExcludeOutputQuantConfig(config.StaticQuantConfig)`

This class configures:

* `quant_format=QuantFormat.QOperator`
* `activation_type=QuantType.QInt8`
* `weight_type=QuantType.QInt8`
* `per_channel=True` by default
* `reduce_range=True` by default
* default quantized op types: `Conv`, `MatMul`, `Add`, `Mul`
* explicit `nodes_to_exclude` support

That makes this path convenient when some graph outputs or fragile nodes should remain in floating point.

### Flow details

1. **Load the FP32 PDL model**.
2. **Optionally preprocess the model** with custom Conv+BN folding, CLE, and/or SeqMSE.
3. **Export FP32 ONNX**.
4. **Preprocess the ONNX graph**.
5. **Create a calibration reader** for the ONNX graph inputs.
6. **Build an INC static quantization config**.
7. **Quantize the ONNX model using INC**.
8. **Save the quantized model** and report ONNX operator statistics.

### Example

```
!cd /path/to/QuantizedPDL_v2 && python build_neural_compressed_pdl.py \
    --weights_path /kaggle/working/QuantizedPDL_v2/weights/model_final_bd324a.pkl \
    --calib_images /kaggle/input/datasets/ippapi/cityscrapes/kaggle/working/cityscapes/leftImg8bit/train \
    --cityscapes_root /kaggle/working/cityscapes \
    --num_calib 50 \
    --export_path /kaggle/working/quantized_model \
    --enable_custom_conv_bn_fold \
    --activation_symmetric \
    --weight_symmetric \
    --per_channel \
    --calibration_method minmax \
    --force_qoperator
```

### When to use INC

Choose this flow when you need:

* an ONNX quantization path outside plain ONNX Runtime PTQ
* more explicit node exclusion behavior
* a cleaner route for selective quantization of graph regions

---

## 6. Choosing between the three flows

### Use **AIMET** when

* you want the most mature PTQ recovery features
* you need AdaRound, Bias Correction, BN re-estimation, or QuantAnalyzer-driven analysis
* your primary debugging surface is still the PyTorch model

### Use **ONNX Runtime PTQ** when

* the final deployment target is ONNX Runtime
* you want a straightforward ONNX graph PTQ pipeline
* you want SmoothQuant or ONNX Runtime preprocessing controls

### Use **INC ONNX** when

* you need selective ONNX quantization behavior
* you want to exclude fragile nodes from quantization more explicitly
* you want to compare INC against native ONNX Runtime PTQ on the same exported graph

---

## 7. Calibration data guidance

For all three flows, calibration quality matters a lot.

Recommended practices:

* use representative production-like images
* keep resolution matched to deployment resolution
* avoid calibration sets that are too small or too homogeneous
* start with the script defaults, then adjust `--num_calib` and `--calib_max_samples`
* fix `--seed` to make comparisons reproducible

---