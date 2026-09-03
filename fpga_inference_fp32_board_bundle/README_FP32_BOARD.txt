FP32 ARM-CPU BOARD BUNDLE

Extract every file directly into the directory containing the current
FPGA inference notebook. Open fpga_inference_fp32.ipynb and run all
cells. The existing dataset is reused and is not included.

This runs FP32 PyTorch on the board ARM CPU. The DPU is INT8-only, so
this bundle needs no xmodel or bitstream.

Output: fp32_arm_board_results_v4.json

The full nine-pair run can take roughly 25-30 minutes. For a smoke
test set PAIR_LIMIT=1, LATENCY_REPETITIONS=1,
IDLE_CALIBRATION_S=3.0, and MINIMUM_POWER_WINDOW_S=1.0.
