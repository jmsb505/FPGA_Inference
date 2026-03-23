import argparse
import sys
import yaml
import json
from pathlib import Path
from .config import PipelineConfig
from .orchestrator import VitisDockerOrchestrator

def load_config(path: Path) -> PipelineConfig:
    with open(path, "r") as f:
        if path.suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    return PipelineConfig.from_dict(data)

def main():
    parser = argparse.ArgumentParser(prog="vitis-toolkit")
    subparsers = parser.add_subparsers(dest="command")

    # Pipeline command
    pipeline_parser = subparsers.add_parser("run", help="Run a quantization/compilation pipeline")
    pipeline_parser.add_argument("--config", type=Path, help="Path to YAML/JSON config file")
    pipeline_parser.add_argument("--host", action="store_true", help="Run via Docker from host")

    # Docker command
    docker_parser = subparsers.add_parser("docker", help="Launch Vitis AI Docker container")
    docker_parser.add_argument("--image", help="Docker image to use")

    # QAT prepare command
    qat_parser = subparsers.add_parser("qat-prepare", help="Create QAT-ready model with fake-quantize nodes")
    qat_parser.add_argument("--config", type=Path, help="Path to YAML/JSON config file")
    
    args = parser.parse_args()

    if args.command == "run":
        if not args.config:
            print("Error: --config is required for 'run'")
            sys.exit(1)
            
        config = load_config(args.config)
        
        if args.host:
             # Orchestrate from host
             orchestrator = VitisDockerOrchestrator()
             cmd = f"python toolkit/vitis_toolkit/cli.py run --config {args.config}"
             orchestrator.run(cmd, conda_env=config.conda_env)
        else:
             # Run locally (assumed to be inside container)
             # Determine framework and run
             if ".h5" in str(config.float_model_path):
                  from .tf_pipeline import TensorFlowPipeline
                  pipeline = TensorFlowPipeline(config)
             else:
                  from .pytorch_pipeline import PyTorchPipeline
                  pipeline = PyTorchPipeline(config)
             
             pipeline.run()

    elif args.command == "qat-prepare":
        if not args.config:
            print("Error: --config is required for 'qat-prepare'")
            sys.exit(1)
        config = load_config(args.config)
        from .tf_pipeline import TensorFlowPipeline
        pipeline = TensorFlowPipeline(config)
        qat_path = pipeline.qat_prepare()
        print(f"QAT model saved to: {qat_path}")

    elif args.command == "docker":
         orchestrator = VitisDockerOrchestrator(image=args.image)
         orchestrator.run(None) 

if __name__ == "__main__":
    main()
