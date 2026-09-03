# Project Context

This file consolidates the markdown and runbook context that used to be spread across multiple files.

Important:

- This file contains both current active workflow notes and older historical TensorFlow-era deployment notes.
- Historical sections are preserved because they still explain why certain problems appeared and how they were diagnosed.
- Some file paths mentioned in the historical sections may no longer exist after cleanup.

## Current V4 Active Status (2026-08-23)

The active deployment workflow is V4 2.5D. The V2 and TensorFlow sections below are retained as historical debugging records and must not be used as current run instructions.

### Active files

- `vxm_2p5d_v4.py`: fixed-grid model and geometry contract
- `Voxelmorph/2.5D/train_2p5d_pytorchV4.py` and notebook: training/resume
- `Voxelmorph/2.5D/prepare_calibration_data_v4.py`: representative fixed-grid PTQ inputs
- `vitis_run_v4.py`: Vitis AI 2.5 quantization and compilation
- root `vxm_2p5d_export.py`: canonical export/board CPU reference helper
- `Voxelmorph/compare_2p5d_v4_fusion_3d_v2_test.ipynb`: local quality comparison
- `Voxelmorph/fpga_inference_v4.ipynb`: board quality and measurement notebook
- `Voxelmorph/fpga_deployment_timing.py`: metric-free board timing
- `Voxelmorph/fpga_deployment_power.py`: matched board inference-only power
- `Voxelmorph/measure_local_inference_power_latency.py`: matched local CPU/GPU measurement
- `windows_metrics_package/`: LibreHardwareMonitor and NVIDIA telemetry runner

### Fixed geometry contract

- canonical volume: `[D,H,W]=[96,112,96]`
- external DPU tensor: NHWC `[1,112,96,16]`
- channels: eight moving slices followed by eight fixed slices
- axial: native `112x96`
- coronal: `96x96` plus centered 8/8 vertical padding
- sagittal: transpose `96x112` to `112x96`
- images: trilinear resampling
- segmentations: nearest-neighbor resampling
- padding: image value `-1.0`, label value `0`

The non-square input was chosen because it matches the canonical axial anatomy and avoids either extra square padding (`112x112`) or field-of-view reduction (`96x96`). Letterboxing was chosen over direct stretching to preserve aspect ratio and permit explicit inverse geometry.

### Current Vitis configuration

- Vitis AI: 2.5
- image: `xilinx/vitis-ai-cpu:2.5`
- conda environment: `vitis-ai-pytorch`
- compiler: `vai_c_xir`
- target: `DPUCZDX8G_ISA1_B4096`
- architecture: `/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json`
- network: `vxm_2p5d_pt_v4_canonical`

### Current board package

Place these beside the board notebook:

- `fpga_inference_v4.ipynb`
- `fpga_deployment_timing.py`
- `fpga_deployment_power.py`
- root `vxm_2p5d_export.py`
- V4 checkpoint `2p5d_dense_pt_v4_canonical_best.pth`
- compiled V4 xmodel and matching bitstream
- all 18 MR/CT volumes and segmentation arrays used by the nine deterministic pairs

The first V4 board attempt omitted the `.npy` files. Every pair was skipped and aggregation later called `np.max` on an empty list. The complete test-data upload fixed the missing-file failure; aggregation must still guard empty results and report the missing-data cause directly.

### Measurement scopes

Do not combine these windows:

1. **Quality evaluation:** includes segmentation warp and Dice/TRE/MI/SSIM.
2. **Inference-to-warp latency:** normalized in-memory volumes through warped intensity output; measured without the power monitor.
3. **Inference-only power:** separate execution of the same inference-to-warp path.

The completed local inference-only run covers 2.5D and 3D on CPU and GPU. The metric-free board timing run covers 2.5D DPU and ARM CPU. The direct board inference-only power run is implemented but has not yet produced its result file.

### Required next board run

Run all prerequisite cells in `Voxelmorph/fpga_inference_v4.ipynb`, then execute the cell titled **Inference-only deployment latency, power, and energy**. It writes:

`inference_pipeline_power_latency_v4.json`

Download it to:

`Voxelmorph/artifacts/results/inference_pipeline_breakdown/inference_pipeline_power_latency_v4.json`

Then update `PROJECT_EXPERIMENT_REPORT.md`, the detailed V4 analysis, this status section, and the power/energy figures. Until that file exists, direct FPGA inference-only energy is pending. Earlier board energy values cover the full quality-evaluation pipeline.

## Superseded V2 Active Repo Surface (historical)

### 2.5D active files

- `train_2p5d_pytorchV2.ipynb`
- `fusion_2p5d_v2.ipynb`
- `gpu_inference_pytorch.ipynb`
- `fpga_inference.ipynb`
- `fpga_metrics.ipynb`
- `prepare_calibration_data.py`
- `prepare_test_data.py`
- `metrics_config.json`

### 2.5D legacy reference kept on purpose

- `train_2p5d_pytorch.ipynb`
  last known training notebook behind `2p5d_dense_pt_best.pth`

### 2.5D local layout

- `data/test_data/`
  local MR/CT sample volumes used by inference and board bundles
- `data/calibration_data/`
  generated calibration dataset for Vitis export
- `docs/runbooks/`
  board-side operational notes used by pack scripts

### Former active Vitis path

- `vitis_run.py`
- `vxm_2p5d_export.py`
- `vitis_toolkit/`
- `fpga_inference.ipynb`

Guidance:

- Historical guidance (superseded): use V2 notebooks for research. Current work uses the V4 notebook and pipeline listed above.
- Use `train_2p5d_pytorch.ipynb` only as the historical FPGA-working reference.
- The 2.5D directory is no longer carrying parity tooling.

## Superseded V2 Board Workflows (historical)

### FPGA inference bundle

Build the standalone board bundle on the host manually:

```bash
ROOT_DIR=/home/j/FPGA_Inference
OUT_DIR="$ROOT_DIR/Voxelmorph/2.5D/artifacts/bundles/fpga_inference"
TARBALL="$ROOT_DIR/Voxelmorph/2.5D/artifacts/bundles/board_fpga_inference_bundle.tar.gz"

mkdir -p "$OUT_DIR"

cp "$ROOT_DIR/Voxelmorph/2.5D/fpga_inference.ipynb" "$OUT_DIR/"
cp "$ROOT_DIR/out/v2.5/vxm_2p5d_pt_v3/compiled/vxm_2p5d_pt_v3.xmodel" "$OUT_DIR/"
cp "$ROOT_DIR/Voxelmorph/2.5D/data/test_data/1BA001_mr.npy" "$OUT_DIR/"
cp "$ROOT_DIR/Voxelmorph/2.5D/data/test_data/1BA005_ct.npy" "$OUT_DIR/"

if [[ -f "$ROOT_DIR/Voxelmorph/2.5D/data/test_data/1BA001_mr_seg.npy" ]]; then
  cp "$ROOT_DIR/Voxelmorph/2.5D/data/test_data/1BA001_mr_seg.npy" "$OUT_DIR/"
fi

if [[ -f "$ROOT_DIR/Voxelmorph/2.5D/data/test_data/1BA005_ct_seg.npy" ]]; then
  cp "$ROOT_DIR/Voxelmorph/2.5D/data/test_data/1BA005_ct_seg.npy" "$OUT_DIR/"
fi

tar -C "$OUT_DIR" -czf "$TARBALL" \
  fpga_inference.ipynb \
  vxm_2p5d_pt_v3.xmodel \
  1BA001_mr.npy \
  1BA005_ct.npy \
  $(test -f "$OUT_DIR/1BA001_mr_seg.npy" && printf '1BA001_mr_seg.npy ') \
  $(test -f "$OUT_DIR/1BA005_ct_seg.npy" && printf '1BA005_ct_seg.npy ')
```

Outputs:

- bundle dir: `Voxelmorph/2.5D/artifacts/bundles/fpga_inference`
- tarball: `Voxelmorph/2.5D/artifacts/bundles/board_fpga_inference_bundle.tar.gz`

Upload the tarball to the board Jupyter area:

```bash
scp Voxelmorph/2.5D/artifacts/bundles/board_fpga_inference_bundle.tar.gz xilinx@<BOARD_IP>:~/jupyter_notebooks/
```

On the board:

```bash
cd /home/xilinx/jupyter_notebooks
rm -rf fpga_run_clean
mkdir -p fpga_run_clean
tar -xzf board_fpga_inference_bundle.tar.gz -C fpga_run_clean
cd fpga_run_clean
md5sum vxm_2p5d_pt_v3.xmodel
```

Expected xmodel md5:

- `5ca95fe3bbadaef3defc45ad498ba9e7`

Open `fpga_inference.ipynb` from `fpga_run_clean` and run all cells.

Expected outputs:

- `./fpga_results/results.json`
- `./fpga_results/single_slice.png`
- `./fpga_results/volume_results.png`
- `./fpga_results/performance.png`
- `./fpga_results/warped_volume.npy`
- `./fpga_results/flow_volume.npy`

### FPGA metrics bundle

Files in the upload bundle:

- `fpga_metrics.ipynb`
- `metrics_config.json`
- `vxm_2p5d_pt_v3.xmodel`
- `1BA001_mr.npy`
- `1BA005_ct.npy`
- `1BA001_mr_seg.npy`
- `1BA005_ct_seg.npy`

Build the metrics bundle on the host manually:

```bash
ROOT_DIR=/home/j/FPGA_Inference
OUT_DIR="$ROOT_DIR/Voxelmorph/2.5D/artifacts/bundles/metrics"
TARBALL="$ROOT_DIR/Voxelmorph/2.5D/artifacts/bundles/board_metrics_bundle.tar.gz"

mkdir -p "$OUT_DIR"

cp "$ROOT_DIR/Voxelmorph/2.5D/fpga_metrics.ipynb" "$OUT_DIR/"
cp "$ROOT_DIR/Voxelmorph/2.5D/metrics_config.json" "$OUT_DIR/"
cp "$ROOT_DIR/out/v2.5/vxm_2p5d_pt_v3/compiled/vxm_2p5d_pt_v3.xmodel" "$OUT_DIR/"
cp "$ROOT_DIR/Voxelmorph/2.5D/data/test_data/1BA001_mr.npy" "$OUT_DIR/"
cp "$ROOT_DIR/Voxelmorph/2.5D/data/test_data/1BA005_ct.npy" "$OUT_DIR/"

if [[ -f "$ROOT_DIR/Voxelmorph/2.5D/data/test_data/1BA001_mr_seg.npy" ]]; then
  cp "$ROOT_DIR/Voxelmorph/2.5D/data/test_data/1BA001_mr_seg.npy" "$OUT_DIR/"
fi

if [[ -f "$ROOT_DIR/Voxelmorph/2.5D/data/test_data/1BA005_ct_seg.npy" ]]; then
  cp "$ROOT_DIR/Voxelmorph/2.5D/data/test_data/1BA005_ct_seg.npy" "$OUT_DIR/"
fi

tar -C "$OUT_DIR" -czf "$TARBALL" \
  fpga_metrics.ipynb \
  metrics_config.json \
  vxm_2p5d_pt_v3.xmodel \
  1BA001_mr.npy \
  1BA005_ct.npy \
  $(test -f "$OUT_DIR/1BA001_mr_seg.npy" && printf '1BA001_mr_seg.npy ') \
  $(test -f "$OUT_DIR/1BA005_ct_seg.npy" && printf '1BA005_ct_seg.npy ')
```

Recommended board flow:

```bash
cd /home/xilinx/jupyter_notebooks
mkdir -p fpga_metrics_run
cd fpga_metrics_run
tar -xzf ../board_metrics_bundle.tar.gz
```

Host bundle path:

```bash
Voxelmorph/2.5D/artifacts/bundles/board_metrics_bundle.tar.gz
```

Open `fpga_metrics.ipynb` in Jupyter and run all cells.

The notebook writes simple files into `./fpga_metrics`:

- `metrics.json`
- `slice_times_ms.csv`
- `power.csv` if power rails are available
- `preview.png`
- `timing_power.png`

## Historical V2 Benchmark: 2.5D Fusion vs 3D V2

Generated:

- `2026-04-09T16:56:30.098541Z`

Devices:

- `cpu`
- `cuda`

### Model Size

| Model | Params | Checkpoint MB | Estimated FP Tensor MB |
| --- | ---: | ---: | ---: |
| 2.5D V2 | 137,621 | 0.56 | 0.53 |
| 3D V2 | 313,507 | 1.21 | 1.20 |

### CPU results

Source:

- cpu timings regenerated in this session from the updated notebook logic using the `gen` conda environment

| Method | Dice After | Dice Gain | Dice Error | MI After | MI Gain | Model ms | Post ms | Total ms | Peak Mem MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.5D axial | 0.7252 +/- 0.0399 | 0.1540 | 0.2748 | 0.6092 +/- 0.0096 | 0.0937 | 505.4 | 418.3 | 923.7 | n/a |
| 2.5D mean fused | 0.7687 +/- 0.0226 | 0.1975 | 0.2313 | 0.5992 +/- 0.0092 | 0.0837 | 1558.3 | 464.5 | 2022.8 | n/a |
| 2.5D smoothed 0.75 | 0.7685 +/- 0.0227 | 0.1973 | 0.2315 | 0.6014 +/- 0.0096 | 0.0859 | 1558.3 | 773.5 | 2331.8 | n/a |
| 3D V2 | 0.7609 +/- 0.0189 | 0.1897 | 0.2391 | 0.5772 +/- 0.0084 | 0.0618 | 528.4 | 510.3 | 1038.7 | n/a |

CPU win counts:

| Category | 2.5D axial | 2.5D mean fused | 2.5D smoothed 0.75 | 3D V2 |
| --- | ---: | ---: | ---: | ---: |
| Best Dice | 0 | 3 | 3 | 3 |
| Best MI | 9 | 0 | 0 | 0 |
| Fastest | 7 | 0 | 0 | 2 |

Head-to-head:

- `2.5D mean fused` vs `3D V2`
  - Dice delta: `+0.0078`
  - MI delta: `+0.0219`
  - total runtime delta: `+984.1 ms`

### CUDA results

Source:

- gpu timings loaded from the earlier notebook execution artifacts in the root `compare_2p5d_fusion_3d_v2_test` output folder

| Method | Dice After | Dice Gain | Dice Error | MI After | MI Gain | Model ms | Post ms | Total ms | Peak Mem MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.5D axial | 0.7252 +/- 0.0399 | 0.1540 | 0.2748 | 0.6092 +/- 0.0096 | 0.0937 | 101.9 | 67.2 | 169.1 | n/a |
| 2.5D mean fused | 0.7687 +/- 0.0226 | 0.1975 | 0.2313 | 0.5992 +/- 0.0092 | 0.0837 | 255.9 | 109.7 | 365.6 | n/a |
| 2.5D smoothed 0.75 | 0.7685 +/- 0.0227 | 0.1973 | 0.2315 | 0.6014 +/- 0.0096 | 0.0859 | 255.9 | 512.4 | 768.3 | n/a |
| 3D V2 | 0.7609 +/- 0.0189 | 0.1897 | 0.2391 | 0.5772 +/- 0.0084 | 0.0617 | 254.4 | 129.7 | 384.1 | n/a |

CUDA win counts:

| Category | 2.5D axial | 2.5D mean fused | 2.5D smoothed 0.75 | 3D V2 |
| --- | ---: | ---: | ---: | ---: |
| Best Dice | 0 | 3 | 3 | 3 |
| Best MI | 9 | 0 | 0 | 0 |
| Fastest | 9 | 0 | 0 | 0 |

Head-to-head:

- `2.5D mean fused` vs `3D V2`
  - Dice delta: `+0.0078`
  - MI delta: `+0.0220`
  - total runtime delta: `-18.5 ms`

### CPU vs GPU speedup and drift

| Method | Model Speedup x | Post Speedup x | Total Speedup x | Dice Drift | MI Drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2.5D axial | 4.96 | 6.22 | 5.46 | -0.000011 | +0.000001 |
| 2.5D mean fused | 6.09 | 4.24 | 5.53 | +0.000011 | +0.000003 |
| 2.5D smoothed 0.75 | 6.09 | 1.51 | 3.03 | -0.000001 | +0.000003 |
| 3D V2 | 2.08 | 3.93 | 2.70 | -0.000002 | -0.000012 |

Interpretation:

- On CPU, best Dice is `2.5D mean fused` and best MI is `2.5D axial`.
- On GPU, best Dice is `2.5D mean fused` and best MI is `2.5D axial`.
- Fastest method on both CPU and GPU is `2.5D axial`.
- `2.5D mean fusion` remains the strongest overlap method overall.
- `3D V2` remains competitive while using a single volumetric pass.
- `2.5D axial` is the simplest and fastest single-view baseline.

Benchmark outputs:

- Per-device outputs: `artifacts/results/compare_2p5d_fusion_3d_v2_test/cpu`
- Per-device outputs: `artifacts/results/compare_2p5d_fusion_3d_v2_test/cuda`
- Combined benchmark JSON: `artifacts/results/compare_2p5d_fusion_3d_v2_test/benchmark_summary.json`
- Combined CSV: `artifacts/results/compare_2p5d_fusion_3d_v2_test/per_pair_metrics.csv`

## Historical TensorFlow / Early FPGA Deployment Notes

This section preserves the older deployment history. It is useful context, but it is not the current active PyTorch V2 path.

### Project overview

Goal:

- Deploy a VoxelMorph 2.5D deformable image registration model on FPGA (`Xilinx ZCU104` with DPU) for accelerated brain MR-CT registration.

Original challenge:

- FPGA requires INT8 quantization.
- The original VoxelMorph model produced flow fields with `±20` pixel magnitudes.
- That did not fit well into INT8 representation.
- Result: severe quantization clipping and degraded performance.

### Timeline of the original issues and solutions

#### Issue 1: Initial FPGA deployment showed weak flow because calibration looked like zero

Symptoms:

- quantization calibration showed `Flow range: [0.0000, 0.0000]`
- model produced near-zero flow fields on FPGA
- registration completely failed

Root causes:

1. model loading bug in the quantization pipeline
   - `brain_reg_models.py` was building a new architecture from scratch
   - then loading weights by name
   - `flow_scale` ended up with uninitialized or random weights instead of the trained `0.1` factor
2. architecture mismatch
   - model builder created a `42`-layer architecture
   - weights came from a `43`-layer model that included `flow_scale`

Expected evidence:

```python
[MODEL] ✅ flow_scale layer found (scale=0.1)
Calibration flow range: [-10.1119, 10.0651]
```

Actual evidence at the time:

```python
[MODEL] Built 2.5D Dense architecture: 112×96
[MODEL] ✅ Loaded 32 weight variables
Calibration flow range: [0.0000, 0.0000]
```

Historical fix:

```python
# BEFORE
model = build_vxm_2p5d_dense_core(...)
model.load_weights(weights_path, by_name=True)

# AFTER
model = tf.keras.models.load_model(weights_path, compile=False)
```

Why that mattered:

- it loaded the exact saved model
- it preserved the `flow_scale` layer weights
- it preserved layer config like `trainable=False`

#### Issue 2: Dynamic shape rejection by the Vitis compiler

Symptom:

```text
AssertionError: [ERROR] Invalid shape of input layer:
shape: [1, None, None, 7] (N,H,W,C), name: moving_stack
```

Root cause:

- the trained model `2p5d_dense_tf_best.h5` was saved with dynamic spatial dimensions `[None, None, None, 7]`
- the Vitis AI compiler requires fixed input shapes

Historical solution:

- create `fix_model_shape.py`
- build a new model with fixed shapes `[None, 112, 96, 7]`
- transfer all weights
- verify outputs are identical
- save `2p5d_dense_tf_best_fixed.h5`

Verification:

```python
flow_trained = trained_model.predict([dummy_moving, dummy_fixed])
flow_fixed = fixed_model.predict([dummy_moving, dummy_fixed])
diff = np.abs(flow_trained - flow_fixed).max()
# Result: 0.000000
```

#### Issue 3: Understanding the original flow scaling architecture

Historical investigation:

```python
flow_unscaled = [-46.49, 243.25]
flow_scaled = [-4.65, 24.33]
```

Key finding at that time:

- `flow_scale` applied a `0.1x` multiplier during training
- the network learned to predict large internal flow
- the scaled output became the actual optimized output

Why this mattered:

- the team initially thought a `10x` multiply was needed during inference
- that turned out to be wrong for that integrated-scaling model

#### Issue 4: Incorrect 10x scaling in inference

Symptoms on FPGA:

- flow range `[-48.7500, 79.3750]`
- quiver arrows all pointed roughly one direction
- Dice got worse after registration, for example `0.65 -> 0.31`
- warped images looked severely distorted

Historical wrong logic:

```python
flow_scaled = int8_output / OUTPUT_SCALE
flow = flow_scaled * 10.0
```

Why that was wrong in that historical model:

- the model had already learned with `flow_scale`
- dequantized output was already the intended flow
- multiplying by `10x` over-amplified it

Historical correction:

```python
flow = output_data.astype(np.float32) / OUTPUT_SCALE
```

#### Issue 5: Input quantization scale mismatch

Symptoms:

- even after output scaling was fixed, FPGA still produced uniform flow
- DPU clearly was not seeing properly scaled inputs

Historical root cause:

- input quantization scales were assigned by tensor index instead of tensor name
- moving and fixed stacks got the wrong scales

Example of the wrong mapping:

```python
INPUT_SCALE_0 = 64
INPUT_SCALE_1 = 128
moving_input_q = moving_input * INPUT_SCALE_0
fixed_input_q = fixed_input * INPUT_SCALE_1
```

Correct semantic mapping:

```python
FIXED_SCALE = None
MOVING_SCALE = None

for tensor in input_tensors:
    fix_point = tensor.get_attr('fix_point')
    scale = 2**fix_point
    if 'fixed' in tensor.name.lower():
        FIXED_SCALE = scale
    elif 'moving' in tensor.name.lower():
        MOVING_SCALE = scale
```

Then:

```python
moving_input_q = moving_input * MOVING_SCALE
fixed_input_q = fixed_input * FIXED_SCALE
```

And DPU input order had to match tensor names:

```python
if 'fixed' in input_tensors[0].name.lower():
    inputs = [fixed_input_q, moving_input_q]
else:
    inputs = [moving_input_q, fixed_input_q]
```

### Historical complete TensorFlow solution architecture

#### Training phase

1. standard VoxelMorph 2.5D encoder-decoder
2. add a DPU-compatible `flow_scale` layer

```python
flow_unscaled = Conv2D(2, 3, padding='same', name='flow_unscaled')(x)

flow_scaled = Conv2D(2, 1, padding='same', use_bias=False,
                     trainable=False, name='flow_scale')(flow_unscaled)

scale_weights = np.zeros((1, 1, 2, 2), dtype=np.float32)
scale_weights[0, 0, 0, 0] = 0.1
scale_weights[0, 0, 1, 1] = 0.1
scale_layer.set_weights([scale_weights])
scale_layer.trainable = False
```

Training outputs mentioned in the historical docs:

- `Voxelmorph/trained_weights/2p5d_dense_tf_best.h5`
- validation Dice around `72.31%`
- historical claim: close to original `~73%`

#### Quantization phase

1. fix dynamic shapes:
   - input: `2p5d_dense_tf_best.h5`
   - output: `2p5d_dense_tf_best_fixed.h5`
2. load the complete model with `tf.keras.models.load_model`
3. quantize with Vitis AI
   - calibration data: `16` samples from the training set
   - quantized output: `out/v2.5/vxm_2p5d_tf/quant/q_model.h5`
   - calibration flow range: `[-10.1119, 10.0651]`

4. compile for DPU:
   - input: quantized model
   - output: `vxm_2p5d_tf.xmodel`
   - target: `DPUCZDX8G_ISA1_B4096`

Historical quantization scales extracted from xmodel:

```text
Inputs:
  quant_fixed_stack:  fix_point=6, scale=64
  quant_moving_stack: fix_point=7, scale=128
Output:
  quant_flow_scale_fix: fix_point=4, scale=16
```

#### Historical FPGA inference phase

1. load xmodel and extract scales

```python
graph = xir.Graph.deserialize(XMODEL_PATH)
# Extract FIXED_SCALE=64, MOVING_SCALE=128, OUTPUT_SCALE=16
```

2. quantize inputs:

```python
moving_input_q = (moving_input * MOVING_SCALE).astype(np.int8)
fixed_input_q = (fixed_input * FIXED_SCALE).astype(np.int8)
```

3. pass to DPU in correct order

```python
if 'fixed' in input_tensors[0].name.lower():
    inputs = [fixed_input_q, moving_input_q]
else:
    inputs = [moving_input_q, fixed_input_q]

job_id = dpu_runner.execute_async(inputs, [output_data])
```

4. dequantize without extra scaling

```python
flow = output_data.astype(np.float32) / OUTPUT_SCALE
```

Historical expectation:

- final flow should land roughly in `±5-25` pixels for brain registration

### Historical design-decision rationale

#### Why use a `flow_scale` layer

Alternatives that were considered and rejected:

- reduce learning rate or model capacity
  - rejected because it would require extensive tuning and could hurt accuracy
- custom post-training quantization scales
  - rejected because Vitis AI gave limited per-layer control
- Lambda layer for scaling
  - rejected because Lambda layers are not DPU-compatible

Chosen historical option:

- `1x1 Conv2D` with fixed `0.1` weights
- DPU-compatible
- trainable end-to-end in the sense that the model adapts to the scaled output
- clean integration into the model graph

#### Why `load_model()` instead of build + load weights

Problem with build + load:

```python
model = build_vxm_2p5d_dense_core(...)
model.load_weights(weights_path, by_name=True)
```

Why `load_model()` worked:

- it loaded architecture
- it loaded exact layer connections
- it loaded all weights
- it loaded `flow_scale` configuration and `trainable=False`

#### Why remove the `10x` scaling factor

Historical lesson:

- `flow_scale` was part of the trained model
- it was not an external pre/post process
- no inverse scaling should be applied afterward

Analogy recorded in the old notes:

- if you train a model to output Celsius using a built-in Fahrenheit→Celsius layer, you do not convert it back afterward

#### Why match inputs by tensor name

Reasoning:

- tensor order can change across framework/compiler/serialization boundaries
- semantic matching is more robust than positional matching

### Historical model structure summary

```text
Input (Moving): [1, 112, 96, 7]
Input (Fixed):  [1, 112, 96, 7]
    ↓
Encoder (4 blocks): [16→32→32→32] with strided conv downsampling
    ↓
Bottleneck: 32 filters
    ↓
Decoder (4 blocks): [32→32→32→16] with bilinear upsampling + skip connections
    ↓
Final Convs: [32→16]
    ↓
flow_unscaled: Conv2D(2, 3x3)
    ↓
flow_scale: Conv2D(2, 1x1, weights=0.1, trainable=False)
    ↓
Output: [1, 112, 96, 2]
```

Historical quantization info:

```text
Inputs:
  fixed_stack:  INT8, scale=64  (fix_point=6)
  moving_stack: INT8, scale=128 (fix_point=7)

Output:
  flow_scale_fix: INT8, scale=16 (fix_point=4)
```

Historical note about capacity:

- `INT8` capacity at scale `16` is `[-8, 7.9375]`
- the notes claimed the actual flow range fit that comfortably, while also discussing percentile clipping near `10`

### Historical inference pipeline summary

```text
1. Load image volumes (MR, CT)
2. Extract 2.5D slice stacks (7 slices, window_radius=3)
3. Downsample to 112×96
4. Quantize:
   - moving: ×128 → INT8
   - fixed: ×64 → INT8
5. Pass to DPU in order: [fixed, moving]
6. DPU computes flow (INT8)
7. Dequantize: flow = int8_output / 16
8. Apply flow to warp moving image
9. Compute metrics (Dice score, etc.)
```

### Historical status checklist recorded in the old summary

Completed in that earlier phase:

1. implemented `flow_scale`
2. trained model with flow scaling
3. fixed model loading bug in quantization pipeline
4. created fixed-shape model
5. quantized successfully with flow range `[-10.11, 10.07]`
6. compiled to `.xmodel`
7. fixed FPGA inference code
   - removed incorrect `10x`
   - fixed input scale assignment
   - fixed input order
8. updated GPU inference to match FPGA logic

Pending at that time:

1. run updated FPGA inference on PYNQ
2. verify flow range `±5-25`
3. verify non-uniform flow patterns
4. verify Dice improves after registration
5. compare FPGA vs GPU performance and accuracy

Historical expected results after those fixes:

```text
FPGA Inference:
  Flow range: [-5, +25] pixels
  Flow pattern: Diverse, spatially-varying
  Dice before: ~0.65
  Dice after: ~0.72-0.75
  Inference time: ~2-5ms per slice

GPU Inference:
  Flow range: [-5, +25] pixels
  Dice after: ~0.72-0.75
  Inference time: ~35-40ms per slice
```

### Historical lessons learned

1. model architecture for FPGA
   - always use DPU-compatible layers
   - integrate scaling into the model
   - train end-to-end with FPGA constraints in mind
2. quantization pipeline
   - load complete models when possible
   - use semantic matching by name instead of by index
   - verify calibration data produces reasonable ranges
3. debugging quantized models
   - inspect intermediate outputs
   - compare against floating-point baseline
   - visualize flow fields
   - verify input quantization, DPU execution, and output dequantization separately
4. INT8 best practices
   - understand what range the model naturally produces
   - do not assume you need to undo training-time transformations
   - use actual data, not only dummy data

### Historical technical insights

#### Why flow scaling worked in the old TensorFlow model

Without `flow_scale`:

```text
Loss = MSE(warped_image, fixed_image) + λ * smoothness(flow)
Model learns flow magnitude ±20 pixels
```

With `flow_scale`:

```text
Loss = MSE(warped_image, fixed_image) + λ * smoothness(flow_scaled)
Model learns:
  flow_unscaled should be larger
  flow_scaled should be the correct final flow
```

The old notes explicitly answered:

- yes, the model learns to compensate internally
- therefore the scaled output is the correct final output

#### Historical quantization scale selection note

Calibration flow range:

- `[-10.1119, 10.0651]`

But output scale:

- `16`
- representable range roughly `[-8, 7.94]`

Old explanation:

- the quantizer used percentile-based thresholds rather than absolute max
- clipping the top `1-5%` of outliers could still improve majority quantization quality

#### Why a uniform flow field indicated input corruption

Reasoning recorded in the old notes:

- brain registration should produce spatially varying deformation
- different structures move differently
- uniform arrows meant the DPU was effectively seeing bad inputs or garbage

That clue led to the input quantization scale mismatch discovery.

### Historical conclusion

The old TensorFlow deployment effort concluded that successful FPGA deployment required understanding:

1. model architecture
2. training dynamics
3. quantization mechanics
4. inference pipeline
5. debugging strategies

Its key stated insight:

- `flow_scale` was not a preprocessing step
- it was an integral part of that trained model
- no extra post-scaling was needed in that design

Historical performance claim at the end of that document:

- FPGA should produce accurate registration with roughly `20x` speedup over GPU (`2ms` vs `40ms` per slice)

## Historical Deployment Success Note

This was a shorter success-oriented summary of the same TensorFlow-era flow-scaling path.

### Historical root cause

- original model produced flow around `±20` pixels
- INT8 quantization at scale `64` could represent only around `±2` pixels
- roughly `72%` of values were clipped
- FPGA produced near-zero flow

### Historical solution as recorded there

#### 1. Training-time flow scaling (`0.1x`)

Code pattern:

```python
flow_unscaled = layers.Conv2D(2, 3, padding='same', name='flow_unscaled',
                              kernel_initializer='zeros',
                              bias_initializer='zeros')(x)

flow_scaled = layers.Conv2D(2, 1, padding='same', use_bias=False,
                           trainable=False, name='flow_scale')(flow_unscaled)

scale_weights = np.zeros((1, 1, 2, 2), dtype=np.float32)
scale_weights[0, 0, 0, 0] = 0.1
scale_weights[0, 0, 1, 1] = 0.1
scale_layer.set_weights([scale_weights])
scale_layer.trainable = False
```

Files the historical note said were modified:

- `Voxelmorph/2.5D/train_2p5d_tensorflow.ipynb`
- `brain_reg_models.py`

Historical training results:

- retrained model: `Voxelmorph/trained_weights/2p5d_dense_tf_best.h5`
- validation Dice: `72.31%`
- claimed to match original `~73%`
- flow output range after scaling: `±2 pixels`

#### 2. Model shape fix for the compiler

File:

- `fix_model_shape.py`

Output:

- `Voxelmorph/trained_weights/2p5d_dense_tf_best_fixed.h5`

Verification:

- models produced identical outputs with max diff `0.000000`

#### 3. Quantization and compilation

Quantization output in that note:

```text
[TF-PIPELINE] Calibration flow range: [-10.1119, 10.0651]
[TF-PIPELINE] Calibration flow stats: mean=0.0634, std=1.4177
```

DPU quantization info:

```text
DPU Subgraph: subgraph_quant_bottleneck_conv

Inputs:
  quant_moving_stack: shape=[1, 112, 96, 7], fix_point=7, scale=128
  quant_fixed_stack: shape=[1, 112, 96, 7], fix_point=6, scale=64

Outputs:
  quant_flow_scale_fix: shape=[1, 112, 96, 2], fix_point=4, scale=16
```

Compilation output:

- xmodel: `out/v2.5/vxm_2p5d_tf/compile/vxm_2p5d_tf.xmodel`
- md5: `cf5d8da3a569b32eb97b2a76bf4735a9`
- DPU subgraphs: `1`
- total operations: `137`

#### 4. Historical FPGA inference dequantization using `10x`

This success note recorded a different inference assumption than the later complete summary:

```python
FLOW_SCALE_FACTOR = 10.0

flow_scaled = output_data.astype(np.float32) / OUTPUT_SCALE
flow = flow_scaled * FLOW_SCALE_FACTOR
```

Combined operation noted there:

```text
flow_real = int8_output / 16.0 * 10.0 = int8_output * 0.625
```

This is historically important because it shows the evolution of the debugging, but it is not consistent with the later summary’s “remove 10x” conclusion.

### Historical deployment checklist from the success note

Completed:

1. implemented DPU-compatible `flow_scale`
2. retrained model
3. created fixed-shape model
4. quantized successfully
5. compiled for ZCU104 DPU
6. updated FPGA inference with `10x` dequantization
7. updated GPU inference for consistency

User next steps written in that note:

1. copy `vxm_2p5d_tf.xmodel` to the PYNQ board
2. update FPGA inference with `OUTPUT_SCALE=16.0`
3. run FPGA inference and verify flow magnitude
4. compare FPGA vs GPU metrics

Historical files listed as updated:

Training:

- `Voxelmorph/2.5D/train_2p5d_tensorflow.ipynb`
- `Voxelmorph/trained_weights/2p5d_dense_tf_best.h5`
- `Voxelmorph/trained_weights/2p5d_dense_tf_best_fixed.h5`

Model builder:

- `brain_reg_models.py` in root and in `Voxelmorph/2.5D/`
- `fix_model_shape.py`

Inference:

- `Voxelmorph/2.5D/fpga_inference.ipynb`
- `Voxelmorph/2.5D/gpu_inference.ipynb`

Pipeline:

- `vitis_run.py`

Historical flow range summary from that note:

| Stage | Range | Representation |
| --- | --- | --- |
| Model Output (unscaled) | ±20 pixels | Would clip in INT8 |
| Model Output (0.1x scaled) | ±2 pixels | Fits INT8 quantization |
| DPU INT8 Output | ±32 | INT8 [-128, 127] |
| DPU Float (÷16) | ±2 pixels | Dequantized scaled flow |
| Final Flow (×10) | ±20 pixels | Real deformation field |

Historical expected results from that note:

Before fix:

- FPGA flow range around `±0.2 pixels`
- registration quality poor

After fix:

- FPGA flow range around `±20 pixels`
- registration quality should match GPU around `~73% Dice`

Historical verification commands from that note:

Check that the fixed TensorFlow model had the expected `flow_scale` weights:

```bash
docker run --rm -v $(pwd):/workspace -w /workspace xilinx/vitis-ai-cpu:2.5 \
  bash -c "conda activate vitis-ai-tensorflow2 && python -c '
import tensorflow as tf
model = tf.keras.models.load_model(\"Voxelmorph/trained_weights/2p5d_dense_tf_best_fixed.h5\", compile=False)
scale_layer = model.get_layer(\"flow_scale\")
print(f\"Flow scale weights: {scale_layer.get_weights()[0][0,0,:,:]}\")'
"
```

Expected:

```text
Flow scale weights: [[0.1 0. ]
                     [0.  0.1]]
```

Check xmodel output scale:

```bash
docker run --rm -v $(pwd):/workspace -w /workspace xilinx/vitis-ai-cpu:2.5 \
  bash -c "conda activate vitis-ai-tensorflow2 && python -c '
import xir
graph = xir.Graph.deserialize(\"/workspace/out/v2.5/vxm_2p5d_tf/compile/vxm_2p5d_tf.xmodel\")
root = graph.get_root_subgraph()
for sg in root.toposort_child_subgraph():
    if sg.get_attr(\"device\") == \"DPU\":
        for t in sg.get_output_tensors():
            print(f\"DPU output scale: {2**t.get_attr(\\\"fix_point\\\")}\")'
"
```

Expected:

```text
DPU output scale: 16
```

Historical documentation references named in that note:

- an issue diagnosis note
- a quantization proposal note
- a retraining guide
- an implementation summary

Historical support note:

- if FPGA inference still showed weak flow, verify:
  1. `OUTPUT_SCALE = 16.0`
  2. `FLOW_SCALE_FACTOR = 10.0`
  3. correct `.xmodel` with md5 `cf5d8da3a569b32eb97b2a76bf4735a9`

## Historical Quantization Fix Proposal

This was the proposal document written before or during the TensorFlow-era deployment work.

### Problem identified

- VoxelMorph 2.5D produced flow around `±20-26 pixels`
- INT8 quantization range at scale `64` was around `±2 pixels`
- result: `72%` of flow values clipped

Stated root cause:

- cross-subject brain registration requires large displacements
- at `112x96`, that naturally leads to large flow

### Options considered there

#### Option 1: Scale flow output layer

Idea:

```python
flow = flow_layer(x)
flow_scaled = flow / 10.0
```

And in FPGA inference:

```python
flow_fpga = (flow_int8 / OUTPUT_SCALE) * 10.0
```

Pros:

- simple
- no accuracy loss
- works with existing hardware

Cons:

- requires retraining

#### Option 2: Quantize at lower resolution

Idea:

- quantize on `56x48` instead of `112x96`
- hope the flow magnitude scales down proportionally

Pros:

- no model changes

Cons:

- does not fully solve the problem
- reduces accuracy

#### Option 3: Output clipping + renormalization

Idea:

1. clip flow to around `±4 pixels` during calibration
2. use scale `32` (`fix_point=5`)
3. accept that large displacements clip

Pros:

- works with existing model and hardware

Cons:

- accuracy loss for large displacements

#### Option 4: Use INT16 quantization if supported

Pros:

- no accuracy loss

Cons:

- hardware may not support it
- doubles memory and bandwidth

### Recommended action in that proposal

- implement Option 1
- add a scaling layer
- retrain
- scale back up in FPGA inference

Historical proposed implementation steps:

```python
flow = conv_flow(x)
flow_scaled = tf.keras.layers.Lambda(lambda x: x / 10.0, name='flow_scale')(flow)
model = tf.keras.Model(inputs=[moving, fixed], outputs=flow_scaled)
```

Then:

1. retrain with scaled output
2. update FPGA inference:

```python
flow = (output_data.astype(np.float32) / OUTPUT_SCALE) * 10.0
```

3. re-quantize

### Quick workaround in that proposal

If retraining was not possible, the proposal suggested keeping clipping explicit:

```python
flow = output_data.astype(np.float32) / OUTPUT_SCALE

if OUTPUT_SCALE == 64:
    print(\"WARNING: Flow clipped to ±2 pixels during quantization\")
    print(\"Large displacements (>2px) will be underestimated\")
    print(\"Consider retraining with scaled output for full accuracy\")
```

What that workaround accepted:

- small flows would be accurate
- large flows would clip
- Dice would likely be lower than GPU

### Verification targets from the proposal

After implementing the fix, the proposal wanted:

1. GPU flow around `±2 pixels` in the scaled representation
2. quantization clipping below `5%`
3. FPGA flow output to match GPU within quantization error
4. FPGA Dice within `5%` of GPU

## Auxiliary Hidden Repo: `.MambaMorph`

This directory is not part of the active V2 / Vitis path, but it does have its own README and should be captured here because it is still present in the repo.

Title:

- `MambaMorph: a Mamba-based Framework for Medical MR-CT Deformable Registration`

Tutorial note:

- install Mamba from `https://github.com/state-spaces/mamba`

Framework and result figures are referenced in that README through GitHub-hosted images.

Train command shown there:

```bash
python ./scripts/torch/train_cross.py --gpu 1 --epochs 1 --batch-size 1 --model-dir output/train_debug --model mm-feat
```

Test command shown there:

```bash
python ./scripts/torch/test_cross.py --gpu 0 --model mm-feat --load-model "/home/guotao/code/voxelmorph-dev/output/train_s46/min_train.pt"
```

Dataset note:

- MambaMorph was implemented on a brain MR-CT dataset named `SR-Reg`
- the README says the dataset is developed from `SynthRAD 2023`
- the README points to `https://github.com/mileswyn/MambaMorph` for dataset access

Paper:

- `https://arxiv.org/abs/2401.13934`

Citation recorded there:

```bibtex
@article{wang2025mamba,
  title={Mamba-based deformable medical image registration with an annotated brain MR-CT dataset},
  author={Wang, Yinuo and Guo, Tao and Yuan, Weimin and Shu, Shihao and Meng, Cai and Bai, Xiangzhi},
  journal={Computerized Medical Imaging and Graphics},
  pages={102566},
  year={2025},
  publisher={Elsevier}
}
```

## Consolidated Takeaways

### Current V4 conclusions

- V4 mean fusion is the best measured FPGA quality configuration: Dice 0.7647 and segmentation-derived centroid TRE 2.025 mm on the DPU.
- The FPGA ARM reference reaches Dice 0.7638 and TRE 2.008 mm, so V4 INT8 quantization has a small measured quality effect on the nine pairs.
- The fixed `112x96x16` contract and canonical geometry corrected the pre-V4 result in which FPGA fusion performed worse than axial inference.
- Gaussian smoothing is not supported by the final quality results.
- Metric-free board timing is 15.302 s per pair for the current DPU path and 39.241 s for ARM CPU, but the first DPU pair is a warm-up outlier.
- Local inference-to-warp timing is 273.31 ms for 2.5D GPU and 114.73 ms for 3D GPU.
- The local 3D GPU path is faster and lower-energy on the desktop, but 3D has no FPGA implementation.
- Desktop CPU/GPU telemetry and FPGA rails do not cover equivalent hardware; use within-platform energy comparisons.
- Direct inference-only FPGA power is pending the latest board notebook run.

### Confirmed project failures and lessons

- Dynamic shapes are unsuitable for one DPU graph; make the fixed geometry part of training.
- Loading an approximate architecture can silently lose trained scaling behavior; verify complete-model or state-dict parity.
- Tensor names and fixed-point attributes are safer than positional assumptions at compiler boundaries.
- Training caches must be bounded in WSL; eager whole-dataset loading caused a crash.
- Board bundles must include every file referenced by deterministic pair lists.
- Empty benchmark outputs require explicit validation before aggregation.
- Quality metrics must not be included in a number labeled inference latency.
- Power sampling needs a sufficiently long separate window and a recorded idle baseline.
- TRE must be named according to its source; this project uses segmentation centroids, not manual landmarks.

### Historical material below this status

The TensorFlow and V2 sections remain in this file because they show how flow scaling, fixed shapes, quantization order, and reconstruction errors were discovered. Some historical notes disagree about a `10x` flow factor because they were written at different debugging stages. The active V4 code and measured parity results take precedence over those superseded notes.
