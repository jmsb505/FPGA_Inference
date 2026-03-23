from abc import ABC, abstractmethod
from .config import PipelineConfig
from pathlib import Path

class Pipeline(ABC):
    """
    Base interface for Vitis AI pipelines.
    """
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.output_root.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def quantize(self, **kwargs) -> Path:
        """Run the quantization phase."""
        pass

    @abstractmethod
    def compile(self, quant_model_path: Path) -> Path:
        """Run the compilation phase."""
        pass

    def visualize(self, xmodel_path: Path) -> Path:
        """Generate an SVG visualization of the .xmodel graph."""
        try:
            import xir
        except ImportError:
            print("[PIPELINE] Warning: 'xir' library not found. Skipping visualization.")
            return xmodel_path

        viz_dir = self.config.output_root / "visualize"
        viz_dir.mkdir(parents=True, exist_ok=True)
        svg_out = viz_dir / f"{xmodel_path.stem}.svg"

        print(f"[PIPELINE] Visualizing graph to {svg_out}...")
        graph = xir.Graph.deserialize(str(xmodel_path))
        # Vitis AI 2.5 xir has a generic visualization helper
        # Usually it's available via xir.Graph.draw or a subprocess call to 'xir dump'
        # But most robust way is to use 'xir' utility command if available or direct api
        
        import subprocess
        try:
            cmd = ["xir", "svg", str(xmodel_path), str(svg_out)]
            subprocess.run(cmd, check=True)
            return svg_out
        except Exception as e:
            print(f"[PIPELINE] Warning: Failed to generate visualization: {e}")
            return xmodel_path

    def run(self, **kwargs) -> Path:
        """Execute the full pipeline: quantize + compile [+ visualize]."""
        quant_path = self.quantize(**kwargs)
        if self.config.skip_compile:
            return quant_path
        
        xmodel_path = self.compile(quant_path)
        
        if self.config.visualize_graph:
            self.visualize(xmodel_path)
            
        return xmodel_path
