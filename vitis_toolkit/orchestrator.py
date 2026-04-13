import os
import subprocess
import getpass
import shutil
from pathlib import Path
from typing import List, Optional

class VitisDockerOrchestrator:
    """
    Orchestrates Vitis AI Docker containers from the host.
    """
    IMAGE_MAP = {
        "2.5": "xilinx/vitis-ai-cpu:2.5",
        # For 3.0+, images are framework-specific (tensorflow2, pytorch, etc.)
        # If no framework is detected, we'll default to a generic guess or specific repo
    }

    def __init__(
        self,
        image: Optional[str] = None,
        workspace: Optional[Path] = None,
        vitis_ai_home: Optional[Path] = None,
        vitis_version: str = "2.5",
        conda_env: Optional[str] = None
    ):
        self.workspace = Path(workspace or os.getcwd()).resolve()
        # Vitis-AI home is only really needed for some older scripts, but we default to workspace
        self.vitis_ai_home = Path(vitis_ai_home or self.workspace).resolve()
        
        # Automatic image detection
        if image:
            self.image = image
        elif vitis_version == "2.5":
            self.image = self.IMAGE_MAP["2.5"]
        else:
            # 3.0+ logic: Framework-specific repos
            framework = "tensorflow2" # Default guess
            if conda_env:
                if "pytorch" in conda_env.lower():
                    framework = "pytorch"
                elif "tensorflow2" in conda_env.lower():
                    framework = "tensorflow2"
                elif "tensorflow" in conda_env.lower(): # TF1
                    framework = "tensorflow"
            
            # Map major versions to precise tags if needed
            version_tags = {
                "3.5": "latest",
                "3.0": {
                    "pytorch": "ubuntu2004-3.0.0.106",
                    "tensorflow2": "ubuntu2004-3.0.0.119",
                    "tensorflow": "ubuntu2004-3.0.0.091"
                }
            }
            
            ver_info = version_tags.get(vitis_version, vitis_version)
            if isinstance(ver_info, dict):
                tag = ver_info.get(framework, "latest")
            else:
                tag = ver_info
                
            self.image = f"xilinx/vitis-ai-{framework}-cpu:{tag}"
        
        self.user = getpass.getuser()
        self.uid = os.getuid()
        self.gid = os.getgid()
        self.docker_bin = self._resolve_docker_bin()

    def _resolve_docker_bin(self) -> str:
        for candidate in (
            shutil.which("docker"),
            shutil.which("docker.exe"),
            "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe",
            "/mnt/c/Program Files/Docker/Docker/resources/bin/docker",
        ):
            if candidate and Path(candidate).exists():
                return candidate
        return "docker"

    def _get_docker_devices(self) -> List[str]:
        """Replicate the device discovery logic from docker_run.sh"""
        devices = []
        # Find xclmgmt devices
        for dev in Path("/dev").glob("xclmgmt*"):
            devices.extend(["--device", str(dev)])
        
        # Find render devices
        dri_path = Path("/dev/dri")
        if dri_path.exists():
            for dev in dri_path.glob("renderD*"):
                devices.extend(["--device", str(dev)])
        
        return devices

    def build_command(
        self,
        command: Optional[str] = None,
        conda_env: Optional[str] = None,
        interactive: bool = True,
        extra_volumes: Optional[List[str]] = None
    ) -> List[str]:
        """
        Build the 'docker run' command.
        """
        docker_cmd = [self.docker_bin, "run", "--rm"]
        
        if interactive:
            docker_cmd.append("-it")
            
        docker_cmd.extend(self._get_docker_devices())
        
        # Standard volumes from docker_run.sh
        volumes = [
            "/dev/shm:/dev/shm",
            "/opt/xilinx/dsa:/opt/xilinx/dsa",
            "/opt/xilinx/overlaybins:/opt/xilinx/overlaybins",
            f"{self.workspace}:/workspace"
        ]
        if extra_volumes:
            volumes.extend(extra_volumes)
            
        for v in volumes:
            docker_cmd.extend(["-v", v])
            
        # Environment variables
        # We assume the library is at /workspace/vitis_toolkit
        docker_cmd.extend([
            "-e", f"USER={self.user}",
            "-e", f"UID={self.uid}",
            "-e", f"GID={self.gid}",
            "-e", "PYTHONPATH=/workspace",
            "-e", "TF_CPP_MIN_LOG_LEVEL=3",
            "-w", "/workspace",
            "--network=host"
        ])
        
        # GPU support
        if "gpu" in self.image:
             docker_cmd.append("--gpus all")

        docker_cmd.append(self.image)
        
        if command:
            # Wrap the command in conda run if environment is specified
            if conda_env:
                # In Vitis AI Docker, conda is usually at /opt/vitis_ai/conda
                # We source the profile to ensure conda is initialized in the shell
                conda_init = "source /opt/vitis_ai/conda/etc/profile.d/conda.sh"
                full_command = (
                    f"cd /workspace && export PYTHONPATH=/workspace && "
                    f"{conda_init} && conda run --cwd /workspace --no-capture-output -n {conda_env} {command}"
                )
            else:
                full_command = f"cd /workspace && export PYTHONPATH=/workspace && {command}"
            docker_cmd.extend(["bash", "-c", full_command])
        else:
            docker_cmd.append("bash")
        
        return docker_cmd

    def run(self, command: Optional[str] = None, conda_env: Optional[str] = None) -> int:
        """
        Execute a command inside the Vitis AI container.
        """
        cmd = self.build_command(command, conda_env=conda_env, interactive=False)
        print(f"[ORCHESTRATOR] Running Docker command...")
        
        try:
            # Use subprocess.run with shell=False and list of args for safety
            result = subprocess.run(cmd, check=False)
            return result.returncode
        except KeyboardInterrupt:
            print("\n[ORCHESTRATOR] Interrupted by user.")
            return 1
