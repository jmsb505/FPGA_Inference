FPGA Inference Bundle: INT8 Board Baseline (ARM CPU)
======================================================

This bundle measures the software-quantized INT8 reference model on the FPGA board's ARM CPU.

Contents:
- fpga_inference_int8.ipynb      : Self-contained notebook to run the INT8 ARM CPU benchmark.
- vxm_2p5d_export.py             : 2.5D model loader wrapper with ONNX Runtime INT8 quantized execution support.
- fpga_deployment_timing.py      : V4 metric-free timing helper module.
- fpga_deployment_power.py       : V4 power and PYNQ rail telemetry helper module.
- 2p5d_dense_pt_v4_canonical_best.pth : PyTorch canonical V4 model checkpoint.
- 2p5d_dense_v4_int8.onnx         : ONNX Runtime dynamic INT8 quantized model (true 8-bit ConvInteger ops).
- BUNDLE_MANIFEST.json           : Manifest listing all files in this bundle.

How to Run on Board:
1. Extract or copy this folder to the VEK280 Jupyter workspace.
2. Ensure test volume NPY files (e.g., 1BA125_mr.npy, 1BA220_ct.npy) are present in the directory or in ./data/test_data/.
3. Open and run all cells in `fpga_inference_int8.ipynb`.
4. The notebook will record latency, active power, energy per inference, and performance per watt across 9 volume pairs and save the results to `int8_arm_board_results_v4.json`.
