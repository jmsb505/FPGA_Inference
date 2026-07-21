import os
import numpy as np
import nibabel as nib

# The exactly 18 subjects we need for the 9-pair loop
SUBJECTS = [
    '1BA111', '1BA116', '1BA125', '1BA131', '1BA141', '1BA143', 
    '1BA151', '1BA158', '1BA159', '1BA164', '1BA172', '1BA175', 
    '1BA184', '1BA185', '1BA189', '1BA201', '1BA206', '1BA220'
]

# Paths to your raw NIfTI datasets
VOLUMES_DIR = './Data/Tuco/Tuco/volumes_center'
SEG_DIR = './Data/Tuco/Tuco/seg_center'

# Where to save the FPGA-ready .npy arrays
OUTPUT_DIR = './benchmark_data_export'

def convert_to_npy():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Exporting 18 test subjects to {OUTPUT_DIR}...")
    
    for subj in SUBJECTS:
        for mod in ['mr', 'ct']:
            vol_nii_path = os.path.join(VOLUMES_DIR, f"{subj}_{mod}.nii.gz")
            seg_nii_path = os.path.join(SEG_DIR, f"{subj}_{mod}.nii.gz")
            
            if not os.path.exists(seg_nii_path):
                common_seg_path = os.path.join(SEG_DIR, f"{subj}.nii.gz")
                if os.path.exists(common_seg_path):
                    seg_nii_path = common_seg_path
                    
            if os.path.exists(vol_nii_path):
                # Load volume and save as float32 .npy
                vol = nib.load(vol_nii_path).get_fdata().astype(np.float32)
                vol_out = os.path.join(OUTPUT_DIR, f"{subj}_{mod}.npy")
                np.save(vol_out, vol)
                
                # Load segmentation and save as int16 .npy
                if os.path.exists(seg_nii_path):
                    seg = nib.load(seg_nii_path).get_fdata().astype(np.int16)
                    seg_out = os.path.join(OUTPUT_DIR, f"{subj}_{mod}_seg.npy")
                    np.save(seg_out, seg)
                else:
                    print(f"Warning: Missing segmentation for {subj}_{mod}")
            else:
                print(f"Warning: Missing volume for {subj}_{mod}")
                
        print(f"Processed {subj}")
        
    print(f"\\nDone! You can now zip the '{OUTPUT_DIR}' folder and upload its contents to the FPGA's ./data/test_data/ directory.")

if __name__ == '__main__':
    convert_to_npy()
