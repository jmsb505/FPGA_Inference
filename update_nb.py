import json
import os
import shutil

NB_PATH = "/home/felipepuentec/FPGA_Inference/Voxelmorph/2.5D/fpga_inference.ipynb"
BACKUP_PATH = "/home/felipepuentec/FPGA_Inference/Voxelmorph/2.5D/fpga_inference_backup.ipynb"

# Restore from backup first to start clean
if os.path.exists(BACKUP_PATH):
    shutil.copy2(BACKUP_PATH, NB_PATH)
else:
    shutil.copy2(NB_PATH, BACKUP_PATH)

with open(NB_PATH, 'r') as f:
    nb = json.load(f)

# Helper function to convert raw multiline strings into Jupyter source line lists
def to_cell_source(text):
    # Split by newline and append \n to each line
    lines = text.strip().split('\n')
    return [line + '\n' for line in lines]

# 1. Update Imports (Cell 0)
imports_source = """
import os
import sys
import time
import json
import threading
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import map_coordinates, gaussian_filter
import pynq
from pynq import allocate
from pynq_dpu import DpuOverlay
import xir
import vart
"""
nb['cells'][0]['source'] = to_cell_source(imports_source)

# 2. Add Metrics, Fusion, and Telemetry Helpers (inserted at index 5)
helpers_source = """
SEG_LABELS = [2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 28]

def normalize_volume_contract(volume):
    arr = volume.astype(np.float32)
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max <= arr_min:
        return np.zeros_like(arr, dtype=np.float32)
    if arr_min >= -1.001 and arr_max <= 1.001:
        if arr_min >= -1e-4 and arr_max <= 1.0001:
            return (2.0 * arr - 1.0).astype(np.float32)
        return arr
    return (2.0 * (arr - arr_min) / (arr_max - arr_min) - 1.0).astype(np.float32)

def compute_dice_per_label(seg_a, seg_b, labels=SEG_LABELS):
    scores = []
    for lbl in labels:
        a = (seg_a == lbl).astype(np.float32)
        b = (seg_b == lbl).astype(np.float32)
        inter = float((a * b).sum())
        union = float(a.sum() + b.sum())
        scores.append(1.0 if union == 0.0 else 2.0 * inter / union)
    return np.array(scores, dtype=np.float32)

def mutual_information_np(a, b, bins=64, clip_range=(-1.0, 1.0)):
    hist_2d, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins, range=[clip_range, clip_range])
    pxy = hist_2d / np.maximum(hist_2d.sum(), 1.0)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    return float((pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])).sum())

def structural_similarity_np(a, b, data_range=2.0, sigma=1.5, truncate=3.5, eps=1e-8):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_a = gaussian_filter(a, sigma=sigma, mode='reflect', truncate=truncate)
    mu_b = gaussian_filter(b, sigma=sigma, mode='reflect', truncate=truncate)
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sigma_a_sq = gaussian_filter(a * a, sigma=sigma, mode='reflect', truncate=truncate) - mu_a_sq
    sigma_b_sq = gaussian_filter(b * b, sigma=sigma, mode='reflect', truncate=truncate) - mu_b_sq
    sigma_ab = gaussian_filter(a * b, sigma=sigma, mode='reflect', truncate=truncate) - mu_ab
    sigma_a_sq = np.maximum(sigma_a_sq, 0.0)
    sigma_b_sq = np.maximum(sigma_b_sq, 0.0)
    numerator = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    ssim_map = numerator / np.maximum(denominator, eps)
    return float(ssim_map.mean())

def warp_volume_3d_numpy(volume, flow, mode='linear'):
    D, H, W = volume.shape
    zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
    coords = np.stack([flow[0] + zz, flow[1] + yy, flow[2] + xx], axis=0)
    order = 1 if mode == 'linear' else 0
    return map_coordinates(volume.astype(np.float32), coords, order=order, mode='nearest')

def lift_axial(flow_stack, raw_shape):
    d, h, w = raw_shape
    out = np.zeros((3, d, h, w), dtype=np.float32)
    for idx, z in enumerate(range(WINDOW_RADIUS, d - WINDOW_RADIUS)):
        # Axial view: flow_stack[0]=X, flow_stack[1]=Y
        sx = w / float(flow_stack.shape[3])
        sy = h / float(flow_stack.shape[2])
        out[2, z] = cv2.resize(flow_stack[idx, 0], (w, h)) * sx
        out[1, z] = cv2.resize(flow_stack[idx, 1], (w, h)) * sy
    return out

def lift_coronal(flow_stack, raw_shape):
    d, h, w = raw_shape
    out = np.zeros((3, d, h, w), dtype=np.float32)
    for idx, y in enumerate(range(WINDOW_RADIUS, h - WINDOW_RADIUS)):
        # Coronal view: flow_stack[0]=X, flow_stack[1]=Z
        sz = d / float(flow_stack.shape[2])
        sx = w / float(flow_stack.shape[3])
        out[2, :, y, :] = cv2.resize(flow_stack[idx, 0], (w, d)) * sx
        out[0, :, y, :] = cv2.resize(flow_stack[idx, 1], (w, d)) * sz
    return out

def lift_sagittal(flow_stack, raw_shape):
    d, h, w = raw_shape
    out = np.zeros((3, d, h, w), dtype=np.float32)
    for idx, x in enumerate(range(WINDOW_RADIUS, w - WINDOW_RADIUS)):
        # Sagittal view: flow_stack[0]=Y, flow_stack[1]=Z
        sz = d / float(flow_stack.shape[2])
        sy = h / float(flow_stack.shape[3])
        out[1, :, :, x] = cv2.resize(flow_stack[idx, 0], (h, d)) * sy
        out[0, :, :, x] = cv2.resize(flow_stack[idx, 1], (h, d)) * sz
    return out

def smooth_field_3d(field, sigma=0.75):
    out = np.zeros_like(field)
    for c in range(field.shape[0]):
        out[c] = gaussian_filter(field[c], sigma=sigma).astype(np.float32)
    return out

def fuse_fields(axial_field, coronal_field, sagittal_field):
    out = np.zeros_like(axial_field)
    out[0] = 0.5 * axial_field[0] + 0.5 * coronal_field[0]
    out[1] = 0.5 * axial_field[1] + 0.5 * sagittal_field[1]
    out[2] = 0.5 * coronal_field[2] + 0.5 * sagittal_field[2]
    return out

def summarize_registration(moving_vol, fixed_vol, moving_seg, fixed_seg, flow, warped_vol, warped_seg):
    dice_before = compute_dice_per_label(moving_seg, fixed_seg)
    dice_after = compute_dice_per_label(warped_seg, fixed_seg)
    return {
        'mi_before': mutual_information_np(moving_vol, fixed_vol),
        'mi_after': mutual_information_np(warped_vol, fixed_vol),
        'ssim_deformed_fixed': structural_similarity_np(warped_vol, fixed_vol),
        'ssim_deformed_moving': structural_similarity_np(warped_vol, moving_vol),
        'dice_before': float(dice_before.mean()),
        'dice_after': float(dice_after.mean()),
        'flow_min': float(flow.min()),
        'flow_max': float(flow.max()),
        'flow_mean': float(flow.mean()),
        'flow_std': float(flow.std()),
    }

MODEL_CPU = None

def run_cpu_inference(moving_stack, fixed_stack, weights_path):
    global MODEL_CPU
    if MODEL_CPU is None:
        import sys
        from pathlib import Path
        sys.path.append(str(Path('/home/felipepuentec/FPGA_Inference')))
        from vxm_2p5d_export import load_model_for_export
        import torch
        print(f"Loading PyTorch CPU model from {weights_path}...")
        MODEL_CPU = load_model_for_export(weights_path)
        MODEL_CPU.to('cpu')
        MODEL_CPU.eval()
        print("Model loaded successfully on CPU.")

    moving_stack = pad_stack_to_n(moving_stack, N_STACK)
    fixed_stack = pad_stack_to_n(fixed_stack, N_STACK)

    combined = np.concatenate([moving_stack, fixed_stack], axis=0).astype(np.float32)
    input_data = combined.transpose(1, 2, 0)[np.newaxis, ...]

    import torch
    input_tensor = torch.from_numpy(input_data).float()

    start = time.time()
    with torch.no_grad():
        flow_tensor = MODEL_CPU(input_tensor)
    duration = time.time() - start

    flow = flow_tensor.squeeze(0).numpy()
    return flow.astype(np.float32), duration

def infer_full_volume(moving_vol, fixed_vol, moving_seg=None, axis=0, device='fpga', weights_path=None):
    D = moving_vol.shape[axis]
    wr = WINDOW_RADIUS

    warped_volume = np.zeros_like(moving_vol)
    if moving_seg is not None:
        warped_seg_volume = np.zeros_like(moving_seg)
    else:
        warped_seg_volume = None

    flow_volume = []
    inference_times = []

    for z in range(wr, D - wr):
        moving_stack = extract_slice_stack(moving_vol, axis, z, wr)
        fixed_stack = extract_slice_stack(fixed_vol, axis, z, wr)

        if device == 'fpga':
            flow, inf_time = run_dpu_inference(moving_stack, fixed_stack)
        else:
            flow, inf_time = run_cpu_inference(moving_stack, fixed_stack, weights_path)

        inference_times.append(inf_time)
        flow_volume.append(flow)

        if axis == 0:
            moving_center = moving_vol[z]
        elif axis == 1:
            moving_center = moving_vol[:, z, :]
        else:
            moving_center = moving_vol[:, :, z]

        warped_slice = apply_flow_2d(moving_center, flow)
        if moving_seg is not None:
            if axis == 0:
                seg_slice = moving_seg[z]
            elif axis == 1:
                seg_slice = moving_seg[:, z, :]
            else:
                seg_slice = moving_seg[:, :, z]
            warped_seg_slice = apply_flow_2d_nearest(seg_slice, flow)

            if axis == 0:
                warped_seg_volume[z] = warped_seg_slice
            elif axis == 1:
                warped_seg_volume[:, z, :] = warped_seg_slice
            else:
                warped_seg_volume[:, :, z] = warped_seg_slice

        if axis == 0:
            warped_volume[z] = warped_slice
        elif axis == 1:
            warped_volume[:, z, :] = warped_slice
        else:
            warped_volume[:, :, z] = warped_slice

    return warped_volume, np.array(flow_volume), inference_times, warped_seg_volume

class PowerMonitor:
    def __init__(self, device_name, sample_interval_s=0.10, idle_dpu_w=0.0, idle_cpu_w=0.0):
        self.device_name = device_name
        self.sample_interval_s = sample_interval_s
        self.idle_dpu_w = float(idle_dpu_w)
        self.idle_cpu_w = float(idle_cpu_w)
        self.gpu_samples_w = []
        self.cpu_samples_w = []
        self.memory_samples_mb = []
        self._stop = False
        self._thread = None
        self._start_rss_mb = None
        self._end_rss_mb = None
        self._start_wall = None
        self._end_wall = None
        self.rails = pynq.get_rails()

    @staticmethod
    def _read_rss_mb():
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            pass
        return 0.0

    def _sample_power_and_memory(self):
        while not self._stop:
            if 'INT' in self.rails and self.rails['INT'].power:
                self.gpu_samples_w.append(self.rails['INT'].power.value)
            if 'PSINTFP' in self.rails and self.rails['PSINTFP'].power:
                self.cpu_samples_w.append(self.rails['PSINTFP'].power.value)
            self.memory_samples_mb.append(self._read_rss_mb())
            time.sleep(self.sample_interval_s)

    def __enter__(self):
        self._start_rss_mb = self._read_rss_mb()
        self.memory_samples_mb.append(self._start_rss_mb)
        self._start_wall = time.perf_counter()
        self._stop = False
        self._thread = threading.Thread(target=self._sample_power_and_memory, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._end_wall = time.perf_counter()
        self._end_rss_mb = self._read_rss_mb()
        self.memory_samples_mb.append(self._end_rss_mb)
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return False

    def result(self):
        wall_time_s = 0.0
        if self._start_wall is not None and self._end_wall is not None:
            wall_time_s = max(float(self._end_wall - self._start_wall), 0.0)

        gpu_power_mean_w = None
        gpu_power_peak_w = None
        gpu_energy_j = None
        gpu_dynamic_energy_j = None
        if self.gpu_samples_w:
            gpu_power_mean_w = float(np.mean(self.gpu_samples_w))
            gpu_power_peak_w = float(np.max(self.gpu_samples_w))
            gpu_energy_j = float(gpu_power_mean_w * wall_time_s)
            gpu_dynamic_energy_j = float(max(gpu_energy_j - (self.idle_dpu_w * wall_time_s), 0.0))

        cpu_power_mean_w = None
        cpu_energy_j = None
        cpu_dynamic_energy_j = None
        if self.cpu_samples_w:
            cpu_power_mean_w = float(np.mean(self.cpu_samples_w))
            cpu_energy_j = float(cpu_power_mean_w * wall_time_s)
            cpu_dynamic_energy_j = float(max(cpu_energy_j - (self.idle_cpu_w * wall_time_s), 0.0))

        energy_parts = [v for v in [cpu_energy_j, gpu_energy_j] if v is not None]
        energy_j = float(sum(energy_parts)) if energy_parts else None
        
        dynamic_energy_parts = [v for v in [cpu_dynamic_energy_j, gpu_dynamic_energy_j] if v is not None]
        dynamic_energy_j = float(sum(dynamic_energy_parts)) if dynamic_energy_parts else None

        power_mean_w = None
        if energy_j is not None and wall_time_s > 0:
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
            'cpu_dynamic_energy_j': cpu_dynamic_energy_j,
            'cpu_power_mean_w': cpu_power_mean_w,
            'gpu_energy_j': gpu_energy_j,
            'gpu_dynamic_energy_j': gpu_dynamic_energy_j,
            'gpu_power_mean_w': gpu_power_mean_w,
            'gpu_power_peak_w': gpu_power_peak_w,
            'gpu_power_samples': int(len(self.gpu_samples_w)),
            'energy_j': energy_j,
            'dynamic_energy_j': dynamic_energy_j,
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
    cpu_dynamic_energy_j = sum_known('cpu_dynamic_energy_j')
    gpu_energy_j = sum_known('gpu_energy_j')
    gpu_dynamic_energy_j = sum_known('gpu_dynamic_energy_j')
    energy_j = sum_known('energy_j')
    dynamic_energy_j = sum_known('dynamic_energy_j')

    gpu_peak_vals = [p.get('gpu_power_peak_w') for p in parts if p.get('gpu_power_peak_w') is not None]
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
        'cpu_dynamic_energy_j': cpu_dynamic_energy_j,
        'cpu_power_mean_w': None if cpu_energy_j is None or wall_time_s <= 0 else float(cpu_energy_j / wall_time_s),
        'gpu_energy_j': gpu_energy_j,
        'gpu_dynamic_energy_j': gpu_dynamic_energy_j,
        'gpu_power_mean_w': None if gpu_energy_j is None or wall_time_s <= 0 else float(gpu_energy_j / wall_time_s),
        'gpu_power_peak_w': None if not gpu_peak_vals else float(max(gpu_peak_vals)),
        'gpu_power_samples': gpu_samples,
        'energy_j': energy_j,
        'dynamic_energy_j': dynamic_energy_j,
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
"""

metrics_fusion_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": to_cell_source(helpers_source)
}
# Cell will be placed in the final sequence builder at the end of the script


# 3. Telemetry 9-Pair Loop Execution Cell
execution_source = """
import gc
import json
import numpy as np

SUBJECTS = ['1BA111', '1BA116', '1BA125', '1BA131', '1BA141', '1BA143', '1BA151', '1BA158', '1BA159', '1BA164', '1BA172', '1BA175', '1BA184', '1BA185', '1BA189', '1BA201', '1BA206', '1BA220']
PAIRS = [(2, 17), (1, 12), (6, 0), (16, 4), (14, 8), (5, 9), (13, 11), (7, 10), (15, 3)]
WEIGHTS_PATH = resolve_file(
    './2p5d_dense_pt_v2_best.pth',
    [
        '../trained_weights/2p5d_dense_pt_v2_best.pth',
        '/home/xilinx/jupyter_notebooks/fpga_run_new/2p5d_dense_pt_v2_best.pth'
    ]
)

# 0. Calibration Phase (3 seconds)
print("Running power rail calibration (3s idle period)...")
idle_dpu_samples = []
idle_cpu_samples = []
rails = pynq.get_rails()
for _ in range(30):
    if 'INT' in rails and rails['INT'].power:
        idle_dpu_samples.append(rails['INT'].power.value)
    if 'PSINTFP' in rails and rails['PSINTFP'].power:
        idle_cpu_samples.append(rails['PSINTFP'].power.value)
    time.sleep(0.1)

IDLE_DPU_W = float(np.mean(idle_dpu_samples)) if idle_dpu_samples else 0.0
IDLE_CPU_W = float(np.mean(idle_cpu_samples)) if idle_cpu_samples else 0.0
print(f"Calibration completed:")
print(f"  Idle DPU Baseline Power: {IDLE_DPU_W:.4f} W")
print(f"  Idle CPU Baseline Power: {IDLE_CPU_W:.4f} W\\n")

# Run loop for both DPU and CPU
devices = ['fpga', 'cpu']
all_device_summaries = {}

for dev in devices:
    print(f"\\n" + "="*60)
    print(f" STARTING BENCHMARK FOR DEVICE: {dev.upper()}")
    print("="*60)
    
    pair_summaries = []
    
    for p_idx, (idx_m, idx_f) in enumerate(PAIRS):
        subj_m = SUBJECTS[idx_m]
        subj_f = SUBJECTS[idx_f]
        print(f"\\nPair {p_idx+1}/9: {subj_m} (Moving) -> {subj_f} (Fixed)")
        
        # Load Data
        path_m = resolve_file(f'./{subj_m}_mr.npy', [f'./data/test_data/{subj_m}_mr.npy'])
        path_f = resolve_file(f'./{subj_f}_ct.npy', [f'./data/test_data/{subj_f}_ct.npy'])
        path_m_seg = resolve_file(f'./{subj_m}_mr_seg.npy', [f'./data/test_data/{subj_m}_mr_seg.npy'])
        path_f_seg = resolve_file(f'./{subj_f}_ct_seg.npy', [f'./data/test_data/{subj_f}_ct_seg.npy'])
        
        try:
            moving_vol = np.load(path_m).astype(np.float32)
            fixed_vol = np.load(path_f).astype(np.float32)
            has_seg = os.path.exists(path_m_seg) and os.path.exists(path_f_seg)
            moving_seg = np.load(path_m_seg).astype(np.int16) if has_seg else np.zeros_like(moving_vol, dtype=np.int16)
            fixed_seg = np.load(path_f_seg).astype(np.int16) if has_seg else np.zeros_like(fixed_vol, dtype=np.int16)
        except Exception as e:
            print(f"Error loading data: {e}")
            continue
        
        # Normalization
        moving_vol = normalize_volume_contract(moving_vol)
        fixed_vol = normalize_volume_contract(fixed_vol)
        
        # 1. Axial Inference & Lifting
        with PowerMonitor(dev, idle_dpu_w=IDLE_DPU_W, idle_cpu_w=IDLE_CPU_W) as axial_infer_meter:
            w_vol_a, f_vol_a, run_a, s_vol_a = infer_full_volume(moving_vol, fixed_vol, moving_seg if has_seg else None, axis=0, device=dev, weights_path=WEIGHTS_PATH)
            field_a = lift_axial(f_vol_a, moving_vol.shape)
        axial_infer_power = axial_infer_meter.result()
        del w_vol_a, f_vol_a, s_vol_a
        gc.collect()
        
        # 2. Coronal Inference & Lifting
        with PowerMonitor(dev, idle_dpu_w=IDLE_DPU_W, idle_cpu_w=IDLE_CPU_W) as coronal_infer_meter:
            w_vol_c, f_vol_c, run_c, s_vol_c = infer_full_volume(moving_vol, fixed_vol, moving_seg if has_seg else None, axis=1, device=dev, weights_path=WEIGHTS_PATH)
            field_c = lift_coronal(f_vol_c, moving_vol.shape)
        coronal_infer_power = coronal_infer_meter.result()
        del w_vol_c, f_vol_c, s_vol_c
        gc.collect()
        
        # 3. Sagittal Inference & Lifting
        with PowerMonitor(dev, idle_dpu_w=IDLE_DPU_W, idle_cpu_w=IDLE_CPU_W) as sagittal_infer_meter:
            w_vol_s, f_vol_s, run_s, s_vol_s = infer_full_volume(moving_vol, fixed_vol, moving_seg if has_seg else None, axis=2, device=dev, weights_path=WEIGHTS_PATH)
            field_s = lift_sagittal(f_vol_s, moving_vol.shape)
        sagittal_infer_power = sagittal_infer_meter.result()
        del w_vol_s, f_vol_s, s_vol_s
        gc.collect()
        
        # 4. Method: 2p5d_axial
        with PowerMonitor(dev, idle_dpu_w=IDLE_DPU_W, idle_cpu_w=IDLE_CPU_W) as axial_post_meter:
            t_ax = time.perf_counter()
            axial_warped = warp_volume_3d_numpy(moving_vol, field_a, mode='linear')
            axial_warped_seg = warp_volume_3d_numpy(moving_seg.astype(np.float32), field_a, mode='nearest').astype(np.int16) if has_seg else np.zeros_like(fixed_seg)
            axial_post_ms = (time.perf_counter() - t_ax) * 1000.0
        axial_post_power = axial_post_meter.result()
        
        axial_summary = summarize_registration(moving_vol, fixed_vol, moving_seg, fixed_seg, field_a, axial_warped, axial_warped_seg)
        axial_summary['model_inference_ms'] = float(np.sum(run_a) * 1000.0)
        axial_summary['postprocess_ms'] = float(axial_post_ms)
        axial_summary['total_runtime_ms'] = float(axial_summary['model_inference_ms'] + axial_post_ms)
        attach_power(axial_summary, combine_power_measurements([axial_infer_power, axial_post_power]))
        
        del axial_warped, axial_warped_seg
        gc.collect()
        
        # 5. Method: 2p5d_mean_fused
        with PowerMonitor(dev, idle_dpu_w=IDLE_DPU_W, idle_cpu_w=IDLE_CPU_W) as mean_post_meter:
            t_fuse = time.perf_counter()
            mean_field = fuse_fields(field_a, field_c, field_s)
            mean_warped = warp_volume_3d_numpy(moving_vol, mean_field, mode='linear')
            mean_warped_seg = warp_volume_3d_numpy(moving_seg.astype(np.float32), mean_field, mode='nearest').astype(np.int16) if has_seg else np.zeros_like(fixed_seg)
            mean_post_ms = (time.perf_counter() - t_fuse) * 1000.0
        mean_post_power = mean_post_meter.result()
        
        mean_summary = summarize_registration(moving_vol, fixed_vol, moving_seg, fixed_seg, mean_field, mean_warped, mean_warped_seg)
        mean_summary['model_inference_ms'] = float(
            np.sum(run_a) * 1000.0 +
            np.sum(run_c) * 1000.0 +
            np.sum(run_s) * 1000.0
        )
        mean_summary['postprocess_ms'] = float(mean_post_ms)
        mean_summary['total_runtime_ms'] = float(mean_summary['model_inference_ms'] + mean_post_ms)
        attach_power(mean_summary, combine_power_measurements([axial_infer_power, coronal_infer_power, sagittal_infer_power, mean_post_power]))
        
        del mean_warped, mean_warped_seg
        gc.collect()
        
        # 6. Method: 2p5d_smoothed_0p75
        with PowerMonitor(dev, idle_dpu_w=IDLE_DPU_W, idle_cpu_w=IDLE_CPU_W) as smoothed_post_meter:
            t_smooth = time.perf_counter()
            smoothed_field = smooth_field_3d(mean_field, sigma=0.75)
            smoothed_warped = warp_volume_3d_numpy(moving_vol, smoothed_field, mode='linear')
            smoothed_warped_seg = warp_volume_3d_numpy(moving_seg.astype(np.float32), smoothed_field, mode='nearest').astype(np.int16) if has_seg else np.zeros_like(fixed_seg)
            smoothed_post_ms = (time.perf_counter() - t_smooth) * 1000.0
        smoothed_post_power = smoothed_post_meter.result()
        
        smoothed_summary = summarize_registration(moving_vol, fixed_vol, moving_seg, fixed_seg, smoothed_field, smoothed_warped, smoothed_warped_seg)
        smoothed_summary['model_inference_ms'] = float(mean_summary['model_inference_ms'])
        smoothed_summary['postprocess_ms'] = float(smoothed_post_ms)
        smoothed_summary['total_runtime_ms'] = float(smoothed_summary['model_inference_ms'] + smoothed_post_ms)
        attach_power(smoothed_summary, combine_power_measurements([axial_infer_power, coronal_infer_power, sagittal_infer_power, smoothed_post_power]))
        
        del field_a, field_c, field_s, mean_field, smoothed_field, smoothed_warped, smoothed_warped_seg
        gc.collect()
        
        pair_summary = {
            'pair_index': p_idx,
            'moving_idx': idx_m,
            'fixed_idx': idx_f,
            '2p5d_axial': axial_summary,
            '2p5d_mean_fused': mean_summary,
            '2p5d_smoothed_0p75': smoothed_summary
        }
        pair_summaries.append(pair_summary)
        
        print(f"  2p5d_axial           - Dice: {axial_summary['dice_after']:.4f}, Dynamic Energy: {axial_summary['dynamic_energy_j']:.2f} J, Peak RSS: {axial_summary['process_rss_peak_mb']:.1f} MB")
        print(f"  2p5d_mean_fused      - Dice: {mean_summary['dice_after']:.4f}, Dynamic Energy: {mean_summary['dynamic_energy_j']:.2f} J, Peak RSS: {mean_summary['process_rss_peak_mb']:.1f} MB")
        print(f"  2p5d_smoothed_0p75   - Dice: {smoothed_summary['dice_after']:.4f}, Dynamic Energy: {smoothed_summary['dynamic_energy_j']:.2f} J, Peak RSS: {smoothed_summary['process_rss_peak_mb']:.1f} MB")
        
        del moving_vol, fixed_vol, moving_seg, fixed_seg
        gc.collect()
        
    # Aggregate results for device
    methods = ['2p5d_axial', '2p5d_mean_fused', '2p5d_smoothed_0p75']
    aggregated = {}
    for method in methods:
        metrics = [p[method] for p in pair_summaries]
        aggregated[method] = {
            'dice_before': float(np.mean([m['dice_before'] for m in metrics])),
            'dice_after': float(np.mean([m['dice_after'] for m in metrics])),
            'mi_before': float(np.mean([m['mi_before'] for m in metrics])),
            'mi_after': float(np.mean([m['mi_after'] for m in metrics])),
            'ssim_deformed_fixed': float(np.mean([m['ssim_deformed_fixed'] for m in metrics])),
            'ssim_deformed_moving': float(np.mean([m['ssim_deformed_moving'] for m in metrics])),
            'model_inference_ms': float(np.mean([m['model_inference_ms'] for m in metrics])),
            'postprocess_ms': float(np.mean([m['postprocess_ms'] for m in metrics])),
            'total_runtime_ms': float(np.mean([m['total_runtime_ms'] for m in metrics])),
            'energy_j': float(np.mean([m['energy_j'] for m in metrics])),
            'dynamic_energy_j': float(np.mean([m['dynamic_energy_j'] for m in metrics])),
            'cpu_energy_j': float(np.mean([m['cpu_energy_j'] for m in metrics])),
            'cpu_dynamic_energy_j': float(np.mean([m['cpu_dynamic_energy_j'] for m in metrics])),
            'gpu_energy_j': float(np.mean([m['gpu_energy_j'] for m in metrics])),
            'gpu_dynamic_energy_j': float(np.mean([m['gpu_dynamic_energy_j'] for m in metrics])),
            'power_mean_w': float(np.mean([m['power_mean_w'] for m in metrics])),
            'gpu_power_peak_w': float(np.max([m['gpu_power_peak_w'] for m in metrics])) if dev == 'fpga' else 0.0,
            'process_rss_peak_mb': float(np.max([m['process_rss_peak_mb'] for m in metrics])),
            'process_rss_delta_mb': float(np.mean([m['process_rss_delta_mb'] for m in metrics])),
        }
    all_device_summaries[dev] = {
        'methods': aggregated,
        'pairs': pair_summaries
    }

# 7. Print Comparative Tables
print("\\n" + "="*80)
print(" COMPARATIVE SUMMARY TABLES (ARM CPU VS FPGA DPU) ")
print("="*80)

for m in ['2p5d_axial', '2p5d_mean_fused', '2p5d_smoothed_0p75']:
    print(f"\\nMethod: {m.upper()}")
    print("-"*60)
    print("| Device | Dice Before | Dice After | Model ms | Post ms | Total ms | Total Energy | Dyn Energy | Max RSS |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for dev in ['cpu', 'fpga']:
        agg = all_device_summaries[dev]['methods'][m]
        print(f"| {dev.upper():<6} | {agg['dice_before']:.4f} | {agg['dice_after']:.4f} | {agg['model_inference_ms']:.1f} | {agg['postprocess_ms']:.1f} | {agg['total_runtime_ms']:.1f} | {agg['energy_j']:.2f} J | {agg['dynamic_energy_j']:.2f} J | {agg['process_rss_peak_mb']:.1f} MB |")

output_payload = {
    'device': 'comparison_cpu_fpga',
    'calibration': {
        'idle_dpu_w': IDLE_DPU_W,
        'idle_cpu_w': IDLE_CPU_W
    },
    'pair_count': len(PAIRS),
    'fpga': all_device_summaries['fpga'],
    'cpu': all_device_summaries['cpu']
}

with open('benchmark_results_comparison.json', 'w') as f:
    json.dump(output_payload, f, indent=2)
print("\\nSaved comparison benchmark results to benchmark_results_comparison.json")
"""

execution_3d_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": to_cell_source(execution_source)
}

# 4. Telemetry Comparison Plotting Cell
visualization_source = """
import json
import matplotlib.pyplot as plt
import numpy as np

try:
    with open('benchmark_results_comparison.json', 'r') as f:
        res = json.load(f)
    
    methods = ['2p5d_axial', '2p5d_mean_fused', '2p5d_smoothed_0p75']
    devices = ['cpu', 'fpga']
    
    latencies = {dev: [res[dev]['methods'][m]['model_inference_ms'] for m in methods] for dev in devices}
    energy = {dev: [res[dev]['methods'][m]['dynamic_energy_j'] for m in methods] for dev in devices}
    
    dice_before = [res['fpga']['methods'][m]['dice_before'] for m in methods]
    dice_cpu = [res['cpu']['methods'][m]['dice_after'] for m in methods]
    dice_fpga = [res['fpga']['methods'][m]['dice_after'] for m in methods]
    
    x = np.arange(len(methods))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Latency Plot
    width_2 = 0.35
    rects1 = axes[0].bar(x - width_2/2, latencies['cpu'], width_2, label='ARM CPU', color='#E05A47')
    rects2 = axes[0].bar(x + width_2/2, latencies['fpga'], width_2, label='FPGA DPU', color='#479BE0')
    axes[0].set_ylabel('Model Latency (ms)')
    axes[0].set_title('Inference Latency (Lower is Better)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.upper() for m in methods])
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    for rect in rects1:
        h = rect.get_height()
        axes[0].annotate(f'{h:.1f}ms', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        axes[0].annotate(f'{h:.1f}ms', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
                    
    # 2. Energy Plot
    rects3 = axes[1].bar(x - width_2/2, energy['cpu'], width_2, label='ARM CPU', color='#E05A47')
    rects4 = axes[1].bar(x + width_2/2, energy['fpga'], width_2, label='FPGA DPU', color='#479BE0')
    axes[1].set_ylabel('Dynamic Energy (Joules)')
    axes[1].set_title('Dynamic Energy (Lower is Better)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.upper() for m in methods])
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    for rect in rects3:
        h = rect.get_height()
        axes[1].annotate(f'{h:.2f}J', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for rect in rects4:
        h = rect.get_height()
        axes[1].annotate(f'{h:.2f}J', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
                    
    # 3. Dice Accuracy Plot
    width_3 = 0.25
    rects5 = axes[2].bar(x - width_3, dice_before, width_3, label='Before Reg', color='#9E9E9E')
    rects6 = axes[2].bar(x, dice_cpu, width_3, label='CPU (FP32)', color='#E05A47')
    rects7 = axes[2].bar(x + width_3, dice_fpga, width_3, label='FPGA (INT8)', color='#479BE0')
    axes[2].set_ylabel('Dice Coefficient')
    axes[2].set_title('Dice Accuracy Comparison (Higher is Better)')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([m.upper() for m in methods])
    axes[2].set_ylim(0, 1.0)
    axes[2].legend(loc='lower right')
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)
    
    for rect in rects5:
        h = rect.get_height()
        axes[2].annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for rect in rects6:
        h = rect.get_height()
        axes[2].annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for rect in rects7:
        h = rect.get_height()
        axes[2].annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
                    
    plt.tight_layout()
    plt.savefig('benchmark_charts.png', dpi=150)
    plt.show()
    
    print("\\n" + "="*80)
    print(" ADDITIONAL DERIVED HARDWARE METRICS FOR PRESENTATION")
    print("="*80)
    for i, m in enumerate(methods):
        num_slices = 18 if 'axial' in m else 54
        
        cpu_sec = latencies['cpu'][i] / 1000.0
        fpga_sec = latencies['fpga'][i] / 1000.0
        
        cpu_throughput = num_slices / cpu_sec
        fpga_throughput = num_slices / fpga_sec
        
        cpu_efficiency = num_slices / energy['cpu'][i]
        fpga_efficiency = num_slices / energy['fpga'][i]
        
        print(f"\\nMethod: {m.upper()}")
        print(f"  - Throughput (Slices/sec):")
        print(f"      ARM CPU: {cpu_throughput:.2f} slices/s")
        print(f"      FPGA DPU: {fpga_throughput:.2f} slices/s ({(fpga_throughput/cpu_throughput):.1f}x higher)")
        print(f"  - Energy Efficiency (Slices/Joule):")
        print(f"      ARM CPU: {cpu_efficiency:.2f} slices/J")
        print(f"      FPGA DPU: {fpga_efficiency:.2f} slices/J ({(fpga_efficiency/cpu_efficiency):.1f}x more efficient)")
        print(f"  - Accuracy Retention:")
        print(f"      FPGA relative to CPU: {(dice_fpga[i]/dice_cpu[i]*100.0):.2f}%")
    print("="*80)
except Exception as e:
    print(f"Could not generate charts or report: {e}")
"""


visualization_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": to_cell_source(visualization_source)
}

# Construct the clean final cell sequence
new_cells = []
new_cells.extend(nb['cells'][:5])
new_cells.append(metrics_fusion_cell)
new_cells.extend(nb['cells'][5:7])
new_cells.append(execution_3d_cell)
new_cells.append(visualization_cell)
new_cells.append(nb['cells'][11])
nb['cells'] = new_cells


# Save modified notebook
with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Successfully updated {NB_PATH}.")

