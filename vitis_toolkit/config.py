from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, Dict, Any, List

@dataclass
class PipelineConfig:
    """
    Configuration for Vitis AI pipelines.
    """
    # Defaults for common ZCU104 usage
    target_dpu: str = "DPUCZDX8G_ISA1_B4096"
    arch_json: Path = Path("/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json")
    output_root: Path = Path("./out/vitis_toolkit")
    
    # Model Paths
    float_model_path: Optional[Path] = None
    net_name: str = "vitis_model"
    
    # Calibration
    calib_dir: Optional[Path] = None
    calib_batch_size: int = 8
    calib_samples: Optional[int] = None
    
    # Advanced
    custom_objects_module: Optional[str] = None
    builder_fn_name: Optional[str] = None
    input_shape: Optional[str] = None
    target_class: Optional[str] = None
    skip_compile: bool = False
    visualize_graph: bool = True
    extra_compiler_args: List[str] = field(default_factory=list)
    # Framework specific
    conda_env: Optional[str] = None # For orchestrator
    vitis_version: str = "2.5"      # For orchestrator image selection
    
    # XIR / Compilation specific
    input_names: Optional[List[str]] = None
    output_names: Optional[List[str]] = None

    def __post_init__(self):
        # Convert strings to Paths if necessary
        self.arch_json = Path(self.arch_json)
        self.output_root = Path(self.output_root)
        if self.float_model_path:
            self.float_model_path = Path(self.float_model_path)
        if self.calib_dir:
            self.calib_dir = Path(self.calib_dir)

    def to_dict(self):
        return {k: str(v) if isinstance(v, Path) else v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: dict):
        allowed = {item.name for item in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})
