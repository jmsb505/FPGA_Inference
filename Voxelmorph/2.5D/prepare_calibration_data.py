"""
Prepare CROSS-SUBJECT Calibration Dataset for Vitis AI Quantization

This script generates calibration data for quantizing the 2.5D VoxelMorph model.
It extracts slice stacks from DIFFERENT subjects (cross-subject pairs) matching
the training setup where moving MR and fixed CT come from different subjects.

Usage:
    python prepare_calibration_data.py --num_samples 100 --output_dir ./calibration_data
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path to import tuco_dataset
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import importlib.util
_pyc_dir = os.path.abspath(os.path.join('..', '__pycache__'))

def _load_pyc(name):
    pyc = os.path.join(_pyc_dir, f'{name}.cpython-310.pyc')
    if not os.path.exists(pyc):
        raise FileNotFoundError(f"Cannot find {pyc}. Please run the training notebook first to generate it.")
    spec = importlib.util.spec_from_file_location(name, pyc)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod

_load_pyc('tuco_dataset')
from tuco_dataset import TucoDataset
from torch.utils.data import Subset


# Configuration matching the training setup
DATA_ROOT = '../../Data/Tuco'
WINDOW_RADIUS = 3
N_STACK = 2 * WINDOW_RADIUS + 1  # 7 slices
DOWNSAMPLE = True

if DOWNSAMPLE:
    ORIENT_CONFIG = [
        (0, 96,  'axial',    (112, 96)),
        (1, 112, 'coronal',  (96,  96)),
        (2, 96,  'sagittal', (96,  112)),
    ]
else:
    ORIENT_CONFIG = [
        (0, 176, 'axial',    (208, 192)),
        (1, 208, 'coronal',  (176, 192)),
        (2, 192, 'sagittal', (176, 208)),
    ]


def load_training_volumes(data_root, max_subjects=None):
    """Load training volumes from TucoDataset"""
    print(f"Loading training volumes from {data_root}...")
    ds = TucoDataset(data_root, split='train')

    if max_subjects is not None:
        ds = Subset(ds, range(min(max_subjects, len(ds))))

    mr_vols, ct_vols = [], []
    for i, sample in enumerate(ds):
        if i % 10 == 0:
            print(f"  Loading volume {i+1}/{len(ds)}...")
        mr_vols.append(sample['moving'].squeeze(0).numpy().astype(np.float32))
        ct_vols.append(sample['fixed'].squeeze(0).numpy().astype(np.float32))

    print(f"Loaded {len(mr_vols)} training volumes")
    return mr_vols, ct_vols


def extract_slice_stack(volume, axis, z, window_radius):
    """Extract a stack of consecutive slices centered at position z"""
    wr = window_radius

    if axis == 0:  # Axial
        stack = volume[z-wr : z+wr+1]
    elif axis == 1:  # Coronal
        stack = volume[:, z-wr : z+wr+1, :].transpose(1, 0, 2)
    else:  # Sagittal (axis == 2)
        stack = volume[:, :, z-wr : z+wr+1].transpose(2, 0, 1)

    return np.ascontiguousarray(stack)


def generate_calibration_samples(mr_vols, ct_vols, num_samples=100, seed=42):
    """
    Generate CROSS-SUBJECT calibration samples by randomly sampling slice stacks.

    This matches the training setup where moving MR and fixed CT come from
    DIFFERENT subjects.

    Returns:
        List of (moving_stack, fixed_stack) tuples, each of shape (7, H, W)
    """
    np.random.seed(seed)
    samples = []
    n_subj = len(mr_vols)

    print(f"\nGenerating {num_samples} CROSS-SUBJECT calibration samples...")

    for i in range(num_samples):
        # Randomly select orientation
        axis, D, orient_name, (H, W) = ORIENT_CONFIG[np.random.randint(3)]

        # Randomly select TWO DIFFERENT subjects (cross-subject)
        mr_subj_idx = np.random.randint(n_subj)
        ct_subj_idx = np.random.randint(n_subj)
        while ct_subj_idx == mr_subj_idx:
            ct_subj_idx = np.random.randint(n_subj)

        # Randomly select slice positions (can be different for each subject)
        mr_z = np.random.randint(WINDOW_RADIUS, D - WINDOW_RADIUS)
        ct_z = np.random.randint(WINDOW_RADIUS, D - WINDOW_RADIUS)

        # Extract stacks from DIFFERENT subjects
        mr_stack = extract_slice_stack(mr_vols[mr_subj_idx], axis, mr_z, WINDOW_RADIUS)
        ct_stack = extract_slice_stack(ct_vols[ct_subj_idx], axis, ct_z, WINDOW_RADIUS)

        samples.append((mr_stack, ct_stack))

        if (i + 1) % 20 == 0:
            print(f"  Generated {i+1}/{num_samples} samples...")

    print(f"Generated {len(samples)} CROSS-SUBJECT calibration samples")
    return samples


def save_calibration_data(samples, output_dir):
    """
    Save calibration samples to disk in format expected by quantization.

    Saves both individual stacks and concatenated 14-channel inputs.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving calibration data to {output_dir}...")

    # Save individual stacks
    stacks_dir = output_path / 'stacks'
    stacks_dir.mkdir(exist_ok=True)

    for i, (mr_stack, ct_stack) in enumerate(samples):
        np.save(stacks_dir / f'moving_stack_{i:04d}.npy', mr_stack)
        np.save(stacks_dir / f'fixed_stack_{i:04d}.npy', ct_stack)

    # Save concatenated 14-channel inputs (ready for model input)
    inputs_dir = output_path / 'inputs'
    inputs_dir.mkdir(exist_ok=True)

    for i, (mr_stack, ct_stack) in enumerate(samples):
        # Concatenate: (14, H, W)
        combined = np.concatenate([mr_stack, ct_stack], axis=0).astype(np.float32)
        np.save(inputs_dir / f'input_{i:04d}.npy', combined)

    # Note: We don't create a single batch file because samples have different shapes
    # (different orientations = different H, W dimensions)

    print(f"  Saved {len(samples)} individual stacks to {stacks_dir}/")
    print(f"  Saved {len(samples)} concatenated inputs to {inputs_dir}/")
    print(f"  Note: Samples have varying shapes due to multi-orientation sampling")

    # Save metadata
    metadata = {
        'num_samples': len(samples),
        'window_radius': WINDOW_RADIUS,
        'n_stack': N_STACK,
        'input_channels': 14,
        'downsample': DOWNSAMPLE,
        'shape_info': f"Each input: (14, H, W) where H,W vary by orientation",
        'orientations': [c[2] for c in ORIENT_CONFIG]
    }

    import json
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved metadata to {output_path / 'metadata.json'}")

    return None  # No batch array due to varying shapes


def main():
    parser = argparse.ArgumentParser(description='Prepare calibration dataset for Vitis AI quantization')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of calibration samples to generate (default: 100)')
    parser.add_argument('--output_dir', type=str, default='./calibration_data',
                        help='Output directory for calibration data')
    parser.add_argument('--max_subjects', type=int, default=None,
                        help='Maximum number of training subjects to load (default: all)')
    parser.add_argument('--data_root', type=str, default=DATA_ROOT,
                        help=f'Path to Tuco dataset (default: {DATA_ROOT})')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Use the data_root from args
    data_root = args.data_root

    print("2.5D VoxelMorph - CROSS-SUBJECT Calibration Data Preparation")
    print(f"Configuration:")
    print(f"  Data root: {data_root}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Max subjects: {args.max_subjects if args.max_subjects else 'all'}")
    print(f"  Random seed: {args.seed}")
    print(f"  Window radius: {WINDOW_RADIUS}")
    print(f"  Stack size: {N_STACK} slices")
    print(f"  Downsample: {DOWNSAMPLE}")
    print(f"  Mode: CROSS-SUBJECT (MR and CT from different subjects)")

    # Load training volumes
    mr_vols, ct_vols = load_training_volumes(data_root, max_subjects=args.max_subjects)

    # Generate calibration samples
    samples = generate_calibration_samples(mr_vols, ct_vols,
                                          num_samples=args.num_samples,
                                          seed=args.seed)

    # Save calibration data
    save_calibration_data(samples, args.output_dir)

    print("\nCROSS-SUBJECT calibration data preparation complete!")
    print(f"\nGenerated files:")
    print(f"  - {args.output_dir}/stacks/moving_stack_*.npy  (individual moving stacks)")
    print(f"  - {args.output_dir}/stacks/fixed_stack_*.npy   (individual fixed stacks)")
    print(f"  - {args.output_dir}/inputs/input_*.npy         (concatenated 14-channel inputs)")
    print(f"  - {args.output_dir}/metadata.json              (dataset metadata)")

    print(f"\nTotal samples: {len(samples)}")
    print(f"Note: Samples have varying shapes due to multi-orientation (axial/coronal/sagittal)")
    print(f"Note: Each sample uses CROSS-SUBJECT pairs (MR != CT subject)")

    print("\nNext steps:")
    print("  1. Use this calibration data with your Vitis AI quantization pipeline")
    print("  2. Load individual samples from calibration_data/inputs/input_*.npy")
    print("  3. Each sample is a (14, H, W) array ready for the model")
    print("  4. This matches the training distribution (cross-subject registration)")


if __name__ == '__main__':
    main()
