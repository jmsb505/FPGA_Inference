import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

POWER_CELL = r'''
import platform
import urllib.request

POWER_SAMPLE_INTERVAL_S = float(os.environ.get('WINDOWS_METRICS_SAMPLE_INTERVAL_S', '0.25'))


class PowerMonitor:
    def __init__(self, device_name, sample_interval_s=POWER_SAMPLE_INTERVAL_S):
        self.device_name = device_name
        self.sample_interval_s = sample_interval_s
        self.gpu_samples_w = []
        self.cpu_samples_w = []
        self.memory_samples_mb = []
        self._stop = False
        self._gpu_thread = None
        self._cpu_thread = None
        self._memory_thread = None
        self._process = None
        self._start_rss_mb = None
        self._end_rss_mb = None
        self._start_wall = None
        self._end_wall = None
        self._start_cpu_j = None
        self._end_cpu_j = None

    @staticmethod
    def _rapl_energy_paths():
        base = Path('/sys/class/powercap')
        if not base.exists():
            return []
        paths = []
        for path in sorted(base.glob('intel-rapl:*')):
            if path.name.count(':') == 1 and (path / 'energy_uj').exists():
                paths.append(path / 'energy_uj')
        return paths

    @classmethod
    def _read_linux_cpu_energy_j(cls):
        values = []
        for path in cls._rapl_energy_paths():
            try:
                values.append(float(path.read_text().strip()) / 1e6)
            except Exception:
                pass
        return None if not values else float(sum(values))

    @staticmethod
    def _score_cpu_power_sensor_name(name):
        lower = str(name).lower()
        if 'cpu package' in lower:
            return 100
        if 'package' in lower and 'cpu' in lower:
            return 90
        if lower in {'package', 'cpu total'}:
            return 80
        if 'ppt' in lower:
            return 75
        if 'cpu' in lower and 'power' in lower:
            return 60
        if 'cpu' in lower:
            return 50
        return 0

    @classmethod
    def _query_lhm_cpu_power_w_wmi(cls):
        if platform.system().lower() != 'windows':
            return None
        try:
            import wmi
        except Exception:
            return None
        candidates = []
        for namespace in [r'root\LibreHardwareMonitor', r'root\OpenHardwareMonitor']:
            try:
                conn = wmi.WMI(namespace=namespace)
                sensors = conn.Sensor()
            except Exception:
                continue
            for sensor in sensors:
                try:
                    if str(getattr(sensor, 'SensorType', '')).lower() != 'power':
                        continue
                    name = str(getattr(sensor, 'Name', ''))
                    value = getattr(sensor, 'Value', None)
                    if value is None:
                        continue
                    score = cls._score_cpu_power_sensor_name(name)
                    if score:
                        candidates.append((score, namespace, name, float(value)))
                except Exception:
                    pass
        if not candidates:
            return None
        candidates.sort(reverse=True, key=lambda item: item[0])
        return float(candidates[0][3])

    @staticmethod
    def _query_lhm_cpu_power_w_powershell():
        if platform.system().lower() != 'windows':
            return None
        script = r"""
$ErrorActionPreference = 'SilentlyContinue'
$namespaces = @('root/LibreHardwareMonitor', 'root/OpenHardwareMonitor')
foreach ($ns in $namespaces) {
  $sensors = Get-CimInstance -Namespace $ns -ClassName Sensor | Where-Object { $_.SensorType -eq 'Power' }
  $preferred = $sensors | Where-Object { $_.Name -match 'CPU Package|CPU Total|Package|PPT' } | Select-Object -First 1
  if ($null -eq $preferred) { $preferred = $sensors | Where-Object { $_.Name -match 'CPU' } | Select-Object -First 1 }
  if ($null -ne $preferred) {
    [Console]::WriteLine(([double]$preferred.Value).ToString([Globalization.CultureInfo]::InvariantCulture))
    exit 0
  }
}
exit 1
"""
        try:
            proc = subprocess.run(
                ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', script],
                capture_output=True,
                text=True,
                timeout=4.0,
                check=False,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        text = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ''
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _parse_power_value_w(text):
        if text is None:
            return None
        raw = str(text).strip().replace(',', '.')
        if not raw:
            return None
        parts = raw.split()
        try:
            return float(parts[0])
        except Exception:
            return None

    @classmethod
    def _find_cpu_power_in_web_tree(cls, node, context=''):
        if not isinstance(node, dict):
            return []
        name = str(node.get('Text') or node.get('text') or node.get('Name') or node.get('name') or '')
        value = node.get('Value') if 'Value' in node else node.get('value')
        full = f'{context} {name}'.strip()
        found = []
        score = cls._score_cpu_power_sensor_name(full)
        watts = cls._parse_power_value_w(value)
        if score and watts is not None:
            found.append((score, watts))
        for child_key in ['Children', 'children']:
            children = node.get(child_key)
            if isinstance(children, list):
                for child in children:
                    found.extend(cls._find_cpu_power_in_web_tree(child, full))
        return found

    @classmethod
    def _query_lhm_cpu_power_w_web(cls):
        if platform.system().lower() != 'windows':
            return None
        for url in ['http://127.0.0.1:8085/data.json', 'http://localhost:8085/data.json']:
            try:
                with urllib.request.urlopen(url, timeout=2.0) as resp:
                    payload = json.loads(resp.read().decode('utf-8', errors='replace'))
            except Exception:
                continue
            candidates = cls._find_cpu_power_in_web_tree(payload)
            if candidates:
                candidates.sort(reverse=True, key=lambda item: item[0])
                return float(candidates[0][1])
        return None

    @classmethod
    def _query_windows_cpu_power_w(cls):
        value = cls._query_lhm_cpu_power_w_wmi()
        if value is not None:
            return value
        value = cls._query_lhm_cpu_power_w_powershell()
        if value is not None:
            return value
        return cls._query_lhm_cpu_power_w_web()

    @staticmethod
    def _query_gpu_power_w():
        try:
            proc = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits', '-i', '0'],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ''
        try:
            return float(line.strip())
        except ValueError:
            return None

    def _sample_cpu_power(self):
        while not self._stop:
            value = self._query_windows_cpu_power_w()
            if value is not None:
                self.cpu_samples_w.append(value)
            time.sleep(self.sample_interval_s)

    def _sample_gpu_power(self):
        while not self._stop:
            value = self._query_gpu_power_w()
            if value is not None:
                self.gpu_samples_w.append(value)
            time.sleep(self.sample_interval_s)

    def _sample_process_memory(self):
        if self._process is None:
            return
        while not self._stop:
            try:
                self.memory_samples_mb.append(float(self._process.memory_info().rss / (1024 ** 2)))
            except Exception:
                pass
            time.sleep(self.sample_interval_s)

    def __enter__(self):
        self._start_cpu_j = self._read_linux_cpu_energy_j()
        try:
            import psutil
            self._process = psutil.Process(os.getpid())
            self._start_rss_mb = float(self._process.memory_info().rss / (1024 ** 2))
            self.memory_samples_mb.append(self._start_rss_mb)
        except Exception:
            self._process = None
            self._start_rss_mb = None
        self._start_wall = time.perf_counter()
        if self._process is not None:
            self._memory_thread = threading.Thread(target=self._sample_process_memory, daemon=True)
            self._memory_thread.start()
        if platform.system().lower() == 'windows':
            self._cpu_thread = threading.Thread(target=self._sample_cpu_power, daemon=True)
            self._cpu_thread.start()
        if self.device_name == 'cuda':
            self._gpu_thread = threading.Thread(target=self._sample_gpu_power, daemon=True)
            self._gpu_thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._end_wall = time.perf_counter()
        self._end_cpu_j = self._read_linux_cpu_energy_j()
        if self._process is not None:
            try:
                self._end_rss_mb = float(self._process.memory_info().rss / (1024 ** 2))
                self.memory_samples_mb.append(self._end_rss_mb)
            except Exception:
                self._end_rss_mb = None
        self._stop = True
        if self._memory_thread is not None:
            self._memory_thread.join(timeout=1.0)
        if self._cpu_thread is not None:
            self._cpu_thread.join(timeout=1.0)
        if self._gpu_thread is not None:
            self._gpu_thread.join(timeout=1.0)
        return False

    def result(self):
        wall_time_s = None
        if self._start_wall is not None and self._end_wall is not None:
            wall_time_s = max(float(self._end_wall - self._start_wall), 0.0)

        cpu_power_mean_w = None
        cpu_power_peak_w = None
        cpu_energy_j = None
        if self.cpu_samples_w and wall_time_s is not None:
            cpu_power_mean_w = float(np.mean(self.cpu_samples_w))
            cpu_power_peak_w = float(np.max(self.cpu_samples_w))
            cpu_energy_j = float(cpu_power_mean_w * wall_time_s)
        elif self._start_cpu_j is not None and self._end_cpu_j is not None:
            delta = self._end_cpu_j - self._start_cpu_j
            if delta >= 0:
                cpu_energy_j = float(delta)
                if wall_time_s and wall_time_s > 0:
                    cpu_power_mean_w = float(cpu_energy_j / wall_time_s)

        gpu_power_mean_w = None
        gpu_power_peak_w = None
        gpu_energy_j = None
        if self.gpu_samples_w:
            gpu_power_mean_w = float(np.mean(self.gpu_samples_w))
            gpu_power_peak_w = float(np.max(self.gpu_samples_w))
            if wall_time_s is not None:
                gpu_energy_j = float(gpu_power_mean_w * wall_time_s)

        energy_parts = [v for v in [cpu_energy_j, gpu_energy_j] if v is not None]
        energy_j = float(sum(energy_parts)) if energy_parts else None
        power_mean_w = None
        if energy_j is not None and wall_time_s and wall_time_s > 0:
            power_mean_w = float(energy_j / wall_time_s)

        process_rss_peak_mb = None
        process_rss_delta_mb = None
        if self.memory_samples_mb:
            process_rss_peak_mb = float(np.max(self.memory_samples_mb))
        if self._start_rss_mb is not None and self._end_rss_mb is not None:
            process_rss_delta_mb = float(self._end_rss_mb - self._start_rss_mb)

        return {
            'power_wall_time_s': wall_time_s,
            'cpu_energy_j': cpu_energy_j,
            'cpu_power_mean_w': cpu_power_mean_w,
            'cpu_power_peak_w': cpu_power_peak_w,
            'cpu_power_samples': int(len(self.cpu_samples_w)),
            'gpu_energy_j': gpu_energy_j,
            'gpu_power_mean_w': gpu_power_mean_w,
            'gpu_power_peak_w': gpu_power_peak_w,
            'gpu_power_samples': int(len(self.gpu_samples_w)),
            'energy_j': energy_j,
            'power_mean_w': power_mean_w,
            'process_rss_start_mb': self._start_rss_mb,
            'process_rss_end_mb': self._end_rss_mb,
            'process_rss_peak_mb': process_rss_peak_mb,
            'process_rss_delta_mb': process_rss_delta_mb,
            'process_memory_samples': int(len(self.memory_samples_mb)),
        }


def combine_power_measurements(parts):
    parts = [p for p in parts if p]
    wall_time_s = sum(float(p.get('power_wall_time_s') or 0.0) for p in parts)

    def sum_known(key):
        vals = [p.get(key) for p in parts if p.get(key) is not None]
        return None if not vals else float(sum(vals))

    cpu_energy_j = sum_known('cpu_energy_j')
    gpu_energy_j = sum_known('gpu_energy_j')
    energy_j = sum_known('energy_j')
    cpu_peak_vals = [p.get('cpu_power_peak_w') for p in parts if p.get('cpu_power_peak_w') is not None]
    gpu_peak_vals = [p.get('gpu_power_peak_w') for p in parts if p.get('gpu_power_peak_w') is not None]
    cpu_samples = int(sum(int(p.get('cpu_power_samples') or 0) for p in parts))
    gpu_samples = int(sum(int(p.get('gpu_power_samples') or 0) for p in parts))
    rss_peak_vals = [p.get('process_rss_peak_mb') for p in parts if p.get('process_rss_peak_mb') is not None]
    rss_start_vals = [p.get('process_rss_start_mb') for p in parts if p.get('process_rss_start_mb') is not None]
    rss_end_vals = [p.get('process_rss_end_mb') for p in parts if p.get('process_rss_end_mb') is not None]
    memory_samples = int(sum(int(p.get('process_memory_samples') or 0) for p in parts))
    rss_start = None if not rss_start_vals else float(rss_start_vals[0])
    rss_end = None if not rss_end_vals else float(rss_end_vals[-1])

    return {
        'power_wall_time_s': float(wall_time_s),
        'cpu_energy_j': cpu_energy_j,
        'cpu_power_mean_w': None if cpu_energy_j is None or wall_time_s <= 0 else float(cpu_energy_j / wall_time_s),
        'cpu_power_peak_w': None if not cpu_peak_vals else float(max(cpu_peak_vals)),
        'cpu_power_samples': cpu_samples,
        'gpu_energy_j': gpu_energy_j,
        'gpu_power_mean_w': None if gpu_energy_j is None or wall_time_s <= 0 else float(gpu_energy_j / wall_time_s),
        'gpu_power_peak_w': None if not gpu_peak_vals else float(max(gpu_peak_vals)),
        'gpu_power_samples': gpu_samples,
        'energy_j': energy_j,
        'power_mean_w': None if energy_j is None or wall_time_s <= 0 else float(energy_j / wall_time_s),
        'process_rss_start_mb': rss_start,
        'process_rss_end_mb': rss_end,
        'process_rss_peak_mb': None if not rss_peak_vals else float(max(rss_peak_vals)),
        'process_rss_delta_mb': None if rss_start is None or rss_end is None else float(rss_end - rss_start),
        'process_memory_samples': memory_samples,
    }


def attach_power(summary, power):
    for key, value in power.items():
        summary[key] = value
    return summary
'''


def query_gpu_power_w():
    try:
        proc = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits', '-i', '0'],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip().splitlines()[0])
    except Exception:
        return None


def score_cpu_power_sensor_name(name):
    lower = str(name).lower()
    if 'cpu package' in lower:
        return 100
    if 'package' in lower and 'cpu' in lower:
        return 90
    if lower in {'package', 'cpu total'}:
        return 80
    if 'ppt' in lower:
        return 75
    if 'cpu' in lower and 'power' in lower:
        return 60
    if 'cpu' in lower:
        return 50
    return 0


def list_windows_power_sensors():
    if platform.system().lower() != 'windows':
        return []
    try:
        import wmi
    except Exception as exc:
        return [{'error': f'wmi import failed: {exc!r}'}]

    sensors_out = []
    for namespace in [r'root\LibreHardwareMonitor', r'root\OpenHardwareMonitor']:
        try:
            conn = wmi.WMI(namespace=namespace)
            sensors = conn.Sensor()
        except Exception as exc:
            sensors_out.append({'namespace': namespace, 'error': repr(exc)})
            continue
        for sensor in sensors:
            try:
                if str(getattr(sensor, 'SensorType', '')).lower() != 'power':
                    continue
                name = str(getattr(sensor, 'Name', ''))
                value = getattr(sensor, 'Value', None)
                sensors_out.append({
                    'namespace': namespace,
                    'name': name,
                    'value': None if value is None else float(value),
                    'identifier': str(getattr(sensor, 'Identifier', '')),
                    'hardware': str(getattr(sensor, 'Hardware', '')),
                    'cpu_score': int(score_cpu_power_sensor_name(name)),
                })
            except Exception as exc:
                sensors_out.append({'namespace': namespace, 'error': repr(exc)})
    return sensors_out


def parse_power_value_w(text):
    if text is None:
        return None
    raw = str(text).strip().replace(',', '.')
    if not raw:
        return None
    try:
        return float(raw.split()[0])
    except Exception:
        return None


def find_cpu_power_in_web_tree(node, context=''):
    if not isinstance(node, dict):
        return []
    name = str(node.get('Text') or node.get('text') or node.get('Name') or node.get('name') or '')
    value = node.get('Value') if 'Value' in node else node.get('value')
    full = f'{context} {name}'.strip()
    found = []
    score = score_cpu_power_sensor_name(full)
    watts = parse_power_value_w(value)
    if score and watts is not None:
        found.append({
            'name': full,
            'value': watts,
            'raw_value': value,
            'cpu_score': int(score),
            'backend': 'LibreHardwareMonitor/OpenHardwareMonitor web',
        })
    for child_key in ['Children', 'children']:
        children = node.get(child_key)
        if isinstance(children, list):
            for child in children:
                found.extend(find_cpu_power_in_web_tree(child, full))
    return found


def lhm_web_urls():
    urls = ['http://127.0.0.1:8085/data.json', 'http://localhost:8085/data.json']
    is_wsl = 'microsoft' in platform.release().lower() or 'wsl' in platform.release().lower()
    if is_wsl:
        try:
            for line in Path('/proc/net/route').read_text().splitlines()[1:]:
                fields = line.split()
                if len(fields) >= 3 and fields[1] == '00000000':
                    gateway_hex = fields[2]
                    octets = [str(int(gateway_hex[index:index + 2], 16)) for index in range(6, -1, -2)]
                    urls.append(f"http://{'.'.join(octets)}:8085/data.json")
                    break
        except Exception:
            pass
    return list(dict.fromkeys(urls))


def list_web_power_sensors():
    # Modern WSL reaches Windows services through its default gateway when
    # localhost forwarding is unavailable.
    found = []
    for url in lhm_web_urls():
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                payload = json.loads(resp.read().decode('utf-8', errors='replace'))
        except Exception as exc:
            found.append({'url': url, 'error': repr(exc)})
            continue
        matches = find_cpu_power_in_web_tree(payload)
        if matches:
            for item in matches:
                item['url'] = url
            return matches
        found.append({'url': url, 'note': 'web server responded, but no CPU power sensor was found'})
    return found


def query_cpu_power_w():
    candidates = []
    for sensor in list_windows_power_sensors():
        value = sensor.get('value')
        score = sensor.get('cpu_score') or 0
        if value is not None and score:
            candidates.append((score, value))
    for sensor in list_web_power_sensors():
        value = sensor.get('value')
        score = sensor.get('cpu_score') or 0
        if value is not None and score:
            candidates.append((score, value))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda item: item[0])
    return float(candidates[0][1])


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def linux_rapl_energy_paths():
    base = Path('/sys/class/powercap')
    if not base.exists():
        return []
    return [str(path) for path in sorted(base.glob('intel-rapl:*/energy_uj')) if path.parent.name.count(':') == 1]


def environment_payload(repo_root, devices):
    payload = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'platform': platform.platform(),
        'system': platform.system(),
        'python': sys.version,
        'python_executable': sys.executable,
        'repo_root': str(repo_root),
        'requested_devices': devices,
        'nvidia_smi_path': shutil.which('nvidia-smi'),
        'is_wsl': 'microsoft' in platform.release().lower() or 'wsl' in platform.release().lower(),
        'linux_rapl_energy_paths': linux_rapl_energy_paths(),
    }
    try:
        import torch
        payload['torch_version'] = torch.__version__
        payload['torch_cuda_available'] = bool(torch.cuda.is_available())
        payload['torch_cuda_device_count'] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            payload['torch_cuda_device_0'] = torch.cuda.get_device_name(0)
    except Exception as exc:
        payload['torch_error'] = repr(exc)
    return payload


def sensor_payload(repo_root, devices):
    out = environment_payload(repo_root, devices)
    power_sensors = list_windows_power_sensors()
    web_power_sensors = list_web_power_sensors()
    cpu_candidates = [s for s in power_sensors if s.get('value') is not None and (s.get('cpu_score') or 0) > 0]
    web_cpu_candidates = [s for s in web_power_sensors if s.get('value') is not None and (s.get('cpu_score') or 0) > 0]
    out['windows_power_sensors'] = power_sensors[:80]
    out['windows_web_power_sensors'] = web_power_sensors[:80]
    out['windows_cpu_power_candidates'] = (cpu_candidates + web_cpu_candidates)[:20]
    out['cpu_power_w'] = query_cpu_power_w()
    out['cpu_power_backend'] = 'LibreHardwareMonitor/OpenHardwareMonitor WMI or web' if out['cpu_power_w'] is not None else None
    if out['cpu_power_w'] is None:
        out['cpu_power_note'] = 'No CPU package power sensor was found. In WSL, enable LibreHardwareMonitor > Options > Remote Web Server > Run on Windows, or confirm Linux RAPL is exposed in /sys/class/powercap. The notebook can use RAPL energy even when this instantaneous check is null.'
    out['gpu_power_w'] = query_gpu_power_w()
    out['gpu_power_backend'] = 'nvidia-smi' if out['gpu_power_w'] is not None else None
    return out


V4_NOTEBOOK_RELATIVE = Path('Voxelmorph/compare_2p5d_v4_fusion_3d_v2_test.ipynb')
V4_OUTPUT_RELATIVE = Path('Voxelmorph/artifacts/results/compare_2p5d_v4_fusion_3d_v2_test')
V4_SUMMARY_NAME = 'benchmark_results_comparison_v4_local.json'


def available_device_names():
    names = ['cpu']
    try:
        import torch
        if torch.cuda.is_available():
            names.append('cuda')
    except Exception:
        pass
    return names


def execute_notebook_code(repo_root, devices, pair_limit):
    nb_path = repo_root / V4_NOTEBOOK_RELATIVE
    if not nb_path.exists():
        raise FileNotFoundError(f'Could not find {nb_path}')

    os.chdir(repo_root / 'Voxelmorph')
    data = json.loads(nb_path.read_text(encoding='utf-8'))
    ns = {'__name__': '__main__'}
    executed = []

    for cell_index, cell in enumerate(data.get('cells', [])):
        if cell.get('cell_type') != 'code':
            continue
        source = ''.join(cell.get('source', []))
        # The preview is useful interactively but repeats inference and is not a
        # benchmark stage. All V4 metric/artifact cells continue to execute.
        if '# Board-style single-slice V4 preview' in source:
            continue
        if 'class PowerMonitor' in source and 'POWER_SAMPLE_INTERVAL_S' in source:
            source = POWER_CELL
        if "power_fields = [" in source and "'cpu_power_mean_w'" in source:
            source = source.replace("    'cpu_power_mean_w',\n", "    'cpu_power_mean_w',\n    'cpu_power_peak_w',\n    'cpu_power_samples',\n")
            source = source.replace(
                'Power and energy fields are best-effort measurements. CPU values require Linux RAPL access; CUDA values require `nvidia-smi` power sampling.',
                'Power and energy fields are best-effort measurements. CPU values use Linux RAPL on Linux or LibreHardwareMonitor WMI on Windows; CUDA values use `nvidia-smi` power sampling.'
            )
        print(f'Executing notebook code cell {cell_index} ...', flush=True)
        exec(compile(source, str(nb_path) + f':cell{cell_index}', 'exec'), ns)
        executed.append(cell_index)
        if cell_index == 1:
            available = list(ns.get('AVAILABLE_DEVICE_NAMES', available_device_names()))
            requested = [device for device in devices if device in available]
            if not requested:
                raise RuntimeError(f'None of requested devices {devices} are available. Available: {available}')
            ns['REQUESTED_DEVICE_NAMES'] = requested
            if pair_limit is not None:
                ns['PAIR_LIMIT'] = int(pair_limit)
            print('Using devices:', requested, flush=True)
            if pair_limit is not None:
                print('Using pair limit:', pair_limit, flush=True)

    return executed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--devices', default='cpu,cuda')
    parser.add_argument('--pair-limit', type=int, default=None)
    parser.add_argument('--power-inference-repetitions', type=int, default=None)
    parser.add_argument('--power-postprocess-repetitions', type=int, default=None)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    devices = [device.strip() for device in args.devices.split(',') if device.strip()]
    output_root = repo_root / V4_OUTPUT_RELATIVE
    output_root.mkdir(parents=True, exist_ok=True)

    # The notebook reads these settings, preserving the same source of truth for
    # interactive use and scripted Windows benchmarks.
    os.environ['V4_MEASURE_LOCAL_POWER'] = '1'
    os.environ['V4_NONINTERACTIVE'] = '1'
    os.environ.setdefault('MPLBACKEND', 'Agg')
    if args.power_inference_repetitions is not None:
        os.environ['V4_POWER_INFERENCE_REPETITIONS'] = str(max(args.power_inference_repetitions, 1))
    if args.power_postprocess_repetitions is not None:
        os.environ['V4_POWER_POSTPROCESS_REPETITIONS'] = str(max(args.power_postprocess_repetitions, 1))

    if args.check_only:
        payload = sensor_payload(repo_root, devices)
        path = output_root / 'windows_sensor_check.json'
        write_json(path, payload)
        print(json.dumps(payload, indent=2))
        print(f'Wrote {path}')
        return

    sensor_before = sensor_payload(repo_root, devices)
    start = time.perf_counter()
    executed_cells = execute_notebook_code(repo_root, devices, args.pair_limit)
    elapsed_s = time.perf_counter() - start
    sensor_after = sensor_payload(repo_root, devices)

    summary_path = output_root / V4_SUMMARY_NAME
    if not summary_path.exists():
        raise FileNotFoundError(f'Benchmark finished but {summary_path} was not created')
    benchmark_summary = json.loads(summary_path.read_text(encoding='utf-8'))

    report = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'kind': 'windows_metrics_report_v4',
        'repo_root': str(repo_root),
        'elapsed_s': float(elapsed_s),
        'executed_notebook_cells': executed_cells,
        'pair_limit': args.pair_limit,
        'power_inference_repetitions': os.environ.get('V4_POWER_INFERENCE_REPETITIONS', '25'),
        'power_postprocess_repetitions': os.environ.get('V4_POWER_POSTPROCESS_REPETITIONS', '1'),
        'sensor_before': sensor_before,
        'sensor_after': sensor_after,
        'benchmark_summary_path': str(summary_path),
        'benchmark_summary': benchmark_summary,
    }
    report_path = output_root / 'windows_metrics_report_v4.json'
    write_json(report_path, report)
    print(f'Wrote {report_path}')


if __name__ == '__main__':
    main()
