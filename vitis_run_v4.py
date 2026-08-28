#!/usr/bin/env python3
"""Run the fixed-grid V4 2.5D model through the Vitis AI PyTorch pipeline."""

from vitis_toolkit import PipelineConfig, run_pytorch_pipeline


CANONICAL_WEIGHTS = "./Voxelmorph/trained_weights/2p5d_dense_pt_v4_canonical_best.pth"
CANONICAL_INPUT_SHAPE = "1,112,96,16"
CANONICAL_CALIB_DIR = "./Voxelmorph/2.5D/data/calibration_data_v4_canonical"
CANONICAL_OUTPUT_ROOT = "./out/v4_canonical/vxm_2p5d_pt_v4_canonical"
CANONICAL_NET_NAME = "vxm_2p5d_pt_v4_canonical"
CANONICAL_MODEL_MODULE = "vxm_2p5d_v4"
CANONICAL_MODEL_CLASS = "Vxm2p5dV4"


def main() -> None:
    config = PipelineConfig(
        float_model_path=CANONICAL_WEIGHTS,
        output_root=CANONICAL_OUTPUT_ROOT,
        net_name=CANONICAL_NET_NAME,
        calib_dir=CANONICAL_CALIB_DIR,
        calib_samples=64,
        custom_objects_module=CANONICAL_MODEL_MODULE,
        target_class=CANONICAL_MODEL_CLASS,
        input_shape=CANONICAL_INPUT_SHAPE,
        conda_env="vitis-ai-pytorch",
        vitis_version="2.5",
        visualize_graph=True,
    )
    raise SystemExit(run_pytorch_pipeline(config))


if __name__ == "__main__":
    main()
