# VEK280 ARM CPU INT8 Measurement & Quantization Summary

## 1. Overview & Execution Scope
* **Target Hardware**: AMD Xilinx Versal VEK280 Board – Dual-core ARM Cortex-A72 Host CPU (`aarch64` Linux).
* **Pipeline Version**: VoxelMorph 2.5D (`v4-canonical-letterbox-int8-arm`).
* **Evaluation Scope**: Complete 9-pair volumetric inference benchmark on canonical $96 \times 112 \times 96$ MRI volumes.
* **Telemetry**: Monitored hardware power rails (`PSINTFP` + `INT` rails sampled via PYNQ PMBus/INA sensors at 0.1s intervals) with a 20-second idle baseline calibration.
* **Execution Status**: 100% completed across all 9 volume pairs.

---

## 2. Measurement Summary & Cross-Platform Comparison

| Configuration / Platform | Precision | Execution Backend | Latency (ms) | Mean Power (W) | Raw Energy (J) | Throughput (inf/s) | Perf / Watt (inf/s/W) |
| :--- | :---: | :--- | ---: | ---: | ---: | ---: | ---: |
| **ARM CPU** | **FP32** | PyTorch FP32 CPU | $38,258.93 \pm 213.38$ | $2.094 \pm 0.009$ | $81.14 \pm 0.37$ | 0.0261 | 0.0125 |
| **ARM CPU** | **INT8** | **ONNX Runtime (`ConvInteger`)** | **$33,040.29 \pm 86.44$** | **$2.067 \pm 0.003$** | **$69.23 \pm 0.37$** | **0.0303** | **0.0146** |
| **FPGA Board DPU** | **INT8** | **Vitis AI 2.5 DPU** | **$428.16 \pm 4.12$** | **$2.120 \pm 0.006$** | **$0.91 \pm 0.01$** | **2.3355** | **1.1017** |

### Key Benchmark Performance Gains:
* **Latency Reduction**: Software INT8 execution on the ARM CPU reduces runtime from **38.26 s down to 33.04 s** (**13.64% speedup** / ~5.22 s saved per volume).
* **Energy Savings**: Energy per volume drops from **81.14 J down to 69.23 J** (**14.67% reduction** / 11.91 J saved per volume).
* **Power Draw**: Mean active power draw remains virtually constant ($2.067 \text{ W}$ vs $2.094 \text{ W}$), reflecting the ARM CPU's active chip TDP during continuous 100% core load.
* **Hardware Acceleration Advantage**: Offloading the network to the dedicated **Vitis AI DPU** achieves an **89.4× speedup over ARM CPU FP32** and a **77.2× speedup over ARM CPU INT8**, with **76.1× greater energy efficiency**.

---

## 3. Quantization Process Details

### A. Export to ONNX Format
1. The trained PyTorch 2.5D VoxelMorph model (`2p5d_dense_pt_v4_canonical_best.pth`) was exported using `torch.onnx.export` with `opset_version=13`.
2. Input tensor dimensions: $(1, 16, 112, 96)$ per 2.5D slice batch.

### B. Why PyTorch Dynamic Quantization Failed
Initially, dynamic quantization was attempted via PyTorch's native `torch.quantization.quantize_dynamic`. However, PyTorch's CPU dynamic quantization engine **only supports 2D Linear (`nn.Linear`) and Recurrent (`nn.LSTM`/`nn.GRU`) layers**. Because VoxelMorph 2.5D is built entirely of 2D Convolutional layers (`nn.Conv2d`), PyTorch silently skipped quantization for all 17 convolutional layers, leaving the model executing entirely in FP32 and producing identical metrics to the FP32 baseline.

### C. True 8-Bit Integer Quantization with ONNX Runtime
To achieve true INT8 execution on the ARM CPU, we transitioned to **ONNX Runtime Quantization Tools**:
1. **Dynamic Quantization Pipeline**: Executed `onnxruntime.quantization.quantize_dynamic` targeting `QuantType.QUInt8` and `OpTypesToExclude=[]`.
2. **Quantized Graph Structure**: The exported graph (`2p5d_dense_v4_int8.onnx`, 195 KB) replaces standard FP32 `Conv` nodes with **`ConvInteger`** and **`QLinearConv`** 8-bit integer operators.
3. **Runtime Engine**: Implemented `OnnxRuntimeQuantizedModel` wrapper in `vxm_2p5d_export.py`, initializing an ONNX Runtime `InferenceSession` with `CPUExecutionProvider` on the ARM host CPU.
4. **Execution Protocol**: During slice-by-slice 2.5D inference, 8-bit integer matrix operations are executed natively across all 17 convolutional layers on the ARM Cortex-A72 cores.

---

## 4. Associated Files & Artifacts
* **Raw JSON Metrics**: [int8_arm_cpu_results_v4.json](file:///home/felipepuentec/FPGA_Inference/Voxelmorph/artifacts/results/int8_arm_cpu_v4_results/int8_arm_cpu_results_v4.json)
* **Model Loader & Engine**: [vxm_2p5d_export.py](file:///home/felipepuentec/FPGA_Inference/fpga_inference_int8_board_bundle/vxm_2p5d_export.py)
* **Execution Notebook**: [fpga_inference_int8.ipynb](file:///home/felipepuentec/FPGA_Inference/fpga_inference_int8_board_bundle/fpga_inference_int8.ipynb)
