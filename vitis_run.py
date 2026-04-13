#!/usr/bin/env python3
import sys
from pathlib import Path

# Add toolkit to path if running from root
sys.path.append(str(Path(__file__).parent))

from vitis_toolkit import PipelineConfig, run_tf_pipeline, run_pytorch_pipeline


CANONICAL_WEIGHTS = "./Voxelmorph/trained_weights/2p5d_dense_pt_best.pth"
CANONICAL_INPUT_SHAPE = "1,112,96,16"
CANONICAL_CALIB_DIR = "./Voxelmorph/2.5D/data/calibration_data"
CANONICAL_OUTPUT_ROOT = "./out/v2.5/vxm_2p5d_pt_v3"
CANONICAL_NET_NAME = "vxm_2p5d_pt_v3"
CANONICAL_MODEL_MODULE = "vxm_2p5d_export"
CANONICAL_MODEL_CLASS = "Vxm2p5dDenseCore"

def main():
    print("Vitis AI Cross-Version Test Suite")
    
    # Versions to test
    vitis_versions = ["2.5"]
    
    # Models to test
    models = [{
        "name": CANONICAL_NET_NAME,
        "framework": "pt",
        "weights": CANONICAL_WEIGHTS,
        "input_shape": CANONICAL_INPUT_SHAPE,
        "output_root": CANONICAL_OUTPUT_ROOT,
        "model_module": CANONICAL_MODEL_MODULE,
        "model_class": CANONICAL_MODEL_CLASS,
    }]

    for version in vitis_versions:
        print(f"\nTesting Vitis AI Version: {version}")

        for model_info in models:
            name = model_info["name"]
            output_path = model_info["output_root"]
            
            print(f"Processing Model: {name}")
            
            config = PipelineConfig(
                float_model_path=model_info["weights"],
                output_root=output_path,
                net_name=name,
                calib_dir=CANONICAL_CALIB_DIR,
                calib_samples=64,
                custom_objects_module=model_info["model_module"],
                target_class=model_info["model_class"],
                input_shape=model_info.get("input_shape"),
                conda_env="vitis-ai-pytorch" if model_info.get("framework") == "pt" else "vitis-ai-tensorflow2",
                vitis_version=version,
                visualize_graph=True
            )

            try:
                exit_code = run_pytorch_pipeline(config) if model_info.get("framework") == "pt" else run_tf_pipeline(config)
                
                if exit_code == 0:
                    print(f"Success: Results in {output_path}")
                elif exit_code == 125:
                    print(f"Skipped: Docker image not available")
                else:
                    print(f"Failed with exit code {exit_code}")
            except Exception as e:
                print(f"Critical error: {e}")

if __name__ == "__main__":
    main()
