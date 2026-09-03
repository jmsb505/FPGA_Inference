# V4 Precision and Power Comparison

## Measurement status

| Platform | Precision | Latency (s) | Raw mean power (W) | Raw energy (J/inference) | Performance/W (inferences/s/W) |
|---|---:|---:|---:|---:|---:|
| FPGA DPU | INT8 | 13.088 | 1.848 | 24.243 | 0.04136 |
| ARM CPU | FP32 | 38.259 | 2.094 | 81.136 | 0.01248 |
| ARM CPU | INT8 | Pending | Pending | Pending | Pending |
| Local CPU | FP32 | 1.060 | 129.291 | 140.049 | 0.00730 |
| Local CPU | INT8 | 5.571 | 128.060 | 714.338 | 0.00140 |
| Local GPU | FP32 | 0.380 | 139.204 | 53.342 | 0.01893 |
| Local GPU | INT8 | 0.263 | 130.888 | 34.559 | 0.02900 |

All measured rows use the same nine-pair inference-to-warp boundary. Raw mean power is the workload mean without subtracting idle power.

## Sensor boundaries

- Desktop CPU: AMD CPU PPT/package sensor from LibreHardwareMonitor.
- Desktop GPU: NVIDIA board power from nvidia-smi.
- FPGA board: PSINTFP ARM/PS rail plus INT DPU/PL rail.
- None of these values is whole-system wall-plug power. Board and desktop watts have different physical boundaries.

## Quantized backends

- FPGA DPU: Vitis AI 2.5 INT8 xmodel.
- Local GPU: TensorRT 8.5.3.1 from the Vitis Q/DQ ONNX export. The engine audit found INT8 weights/tactics for all 17 convolutions, TF32 disabled, and no FP16 path.
- Local CPU: ONNX Runtime QOperator INT8, calibrated with the same 64 calibration samples. Its optimized graph contains 17 QLinearConv and 15 QLinearLeakyRelu operators.
- ARM CPU INT8: runner prepared but not measured because the ZCU104 was not connected.

The local CPU uses backend-specific ONNX Runtime scales. The local GPU reuses the Vitis NNDCT Q/DQ scales exported before xmodel compilation.

## ARM INT8 completion

The board must have onnxruntime with CPUExecutionProvider. Upload these files beside the board notebook:

- Vxm2p5dV4_ort_qoperator_int8.onnx
- measure_v4_arm_int8.py

After the V4 notebook setup and utility cells have run, import run_arm_int8_benchmark from measure_v4_arm_int8 and call:

    run_arm_int8_benchmark(
        globals(),
        model_path="Vxm2p5dV4_ort_qoperator_int8.onnx",
    )

The runner records three unmonitored latency repetitions per pair and a separate power pass of at least 10 seconds using the existing PSINTFP and INT rail monitor.

## Main observations

- Local GPU INT8 reduces latency by 30.6% and improves performance/W by 53.2% versus Local GPU FP32.
- Local CPU INT8 is 5.26 times slower than Local CPU FP32 on this runtime and reduces performance/W by 80.8%.
- FPGA DPU INT8 has the highest measured performance/W, but cross-platform power comparisons remain descriptive because the sensor boundaries differ.
