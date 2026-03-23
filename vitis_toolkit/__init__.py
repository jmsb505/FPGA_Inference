from pathlib import Path
from typing import Optional, Any
from .config import PipelineConfig
from .base import Pipeline
from .orchestrator import VitisDockerOrchestrator

def run_tf_pipeline(config: PipelineConfig, **kwargs) -> int:
    """
    Programmatic entry point to run a TensorFlow pipeline from the host.
    """
    orchestrator = VitisDockerOrchestrator(
        workspace=Path.cwd(), 
        vitis_version=config.vitis_version,
        conda_env=config.conda_env
    )
    
    # Ensure output_root exists
    config.output_root.mkdir(parents=True, exist_ok=True)
    
    # Use JSON for internal config passing as it's in the standard library (no host dependencies)
    config_path = (config.output_root / "internal_config.json").resolve()
    
    import json
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f)
    
    # Run the internal CLI, ensuring we use relative path from workspace root
    rel_config_path = config_path.relative_to(orchestrator.workspace.resolve())
    cmd = f"python -m vitis_toolkit.cli run --config {rel_config_path}"
    return orchestrator.run(cmd, conda_env=config.conda_env or "vitis-ai-tensorflow2")

def run_pytorch_pipeline(config: PipelineConfig, **kwargs) -> int:
    """
    Programmatic entry point to run a PyTorch pipeline from the host.
    """
    orchestrator = VitisDockerOrchestrator(
        workspace=Path.cwd(), 
        vitis_version=config.vitis_version,
        conda_env=config.conda_env
    )
    
    config.output_root.mkdir(parents=True, exist_ok=True)
    config_path = (config.output_root / "internal_config.json").resolve()
    
    import json
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f)
        
    rel_config_path = config_path.relative_to(orchestrator.workspace.resolve())
    cmd = f"python -m vitis_toolkit.cli run --config {rel_config_path}"
    return orchestrator.run(cmd, conda_env=config.conda_env or "vitis-ai-pytorch")

def run_qat_prepare(config: PipelineConfig, **kwargs) -> int:
    """
    Programmatic entry point to prepare a QAT model from the host.
    Runs inside Docker to access VitisQuantizer.get_qat_model().
    """
    orchestrator = VitisDockerOrchestrator(
        workspace=Path.cwd(),
        vitis_version=config.vitis_version,
        conda_env=config.conda_env
    )

    config.output_root.mkdir(parents=True, exist_ok=True)
    config_path = (config.output_root / "internal_config.json").resolve()

    import json
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f)

    rel_config_path = config_path.relative_to(orchestrator.workspace.resolve())
    cmd = f"python -m vitis_toolkit.cli qat-prepare --config {rel_config_path}"
    return orchestrator.run(cmd, conda_env=config.conda_env or "vitis-ai-tensorflow2")

__all__ = ["PipelineConfig", "Pipeline", "VitisDockerOrchestrator", "run_tf_pipeline", "run_pytorch_pipeline", "run_qat_prepare"]
