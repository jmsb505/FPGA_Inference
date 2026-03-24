#!/usr/bin/env python3
import sys
from pathlib import Path

# Add toolkit to path if running from root
sys.path.append(str(Path(__file__).parent))

from vitis_toolkit import PipelineConfig, run_tf_pipeline, run_pytorch_pipeline

def main():
    print("Vitis AI Cross-Version Test Suite")
    
    # Versions to test
    vitis_versions = ["2.5"]
    
    # Models to test
    models = [
        {"name": "vxm_2p5d_tf", "framework": "tf", "weights": "./Voxelmorph/trained_weights/2p5d_dense_tf_best_fixed.h5"},
        {"name": "vxm_2p5d_pt", "framework": "pt", "weights": "./Voxelmorph/trained_weights/2p5d_dense_pt_best.pth", "input_shape": "1,14,256,256"}
    ]

    for version in vitis_versions:
        print(f"\nTesting Vitis AI Version: {version}")

        for model_info in models:
            name = model_info["name"]
            output_path = f"./out/v{version}/{name}"
            
            print(f"Processing Model: {name}")
            
            config = PipelineConfig(
                float_model_path=model_info["weights"],
                output_root=output_path,
                net_name=name,
                calib_dir="./Voxelmorph/2.5D/calibration_data",
                calib_samples=16, 
                custom_objects_module="brain_reg_models",
                target_class="Vxm2p5dDenseCore",
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
