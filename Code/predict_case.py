"""
predict_case.py - run both trained models on one patient, offline on the laptop.

This is the starting point for the application: it shows how to load the weights downloaded from Google Drive
and produce detection, segmentation, localization and a report for new scans (see brats/inference.py).

Example (the four MRI files of one patient, BraTS format):
    python predict_case.py \
        --flair BraTS2021_00000_flair.nii.gz --t1 BraTS2021_00000_t1.nii.gz \
        --t1ce BraTS2021_00000_t1ce.nii.gz --t2 BraTS2021_00000_t2.nii.gz \
        --det-run runs/detection/effb0_v1 --seg-run runs/segmentation/segresnet_v1 \
        --out results/BraTS2021_00000
"""
import argparse
import json
from pathlib import Path

import torch

from brats.config import MODALITIES
from brats.env import get_device
from brats.inference import TumorAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for modality in MODALITIES:
        parser.add_argument(f'--{modality}', required=True, help=f'{modality} NIfTI file')
    parser.add_argument('--seg', help='optional expert label file, to compare against')
    parser.add_argument('--det-run', required=True, help='folder of a detection run (with checkpoints/best.pt)')
    parser.add_argument('--seg-run', required=True, help='folder of a segmentation run (with checkpoints/best.pt)')
    parser.add_argument('--out', required=True, help='output folder')
    parser.add_argument('--cpu', action='store_true', help='do not use the Apple GPU (MPS)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device('cpu') if args.cpu else get_device(allow_mps=True)
    files = {modality: Path(getattr(args, modality)) for modality in MODALITIES}
    if args.seg:
        files['seg'] = Path(args.seg)

    analyzer = TumorAnalyzer(Path(args.det_run), Path(args.seg_run), device)
    report = analyzer.analyze(files, out_dir=Path(args.out))

    detection = dict(report['detection'])
    tumor_slices = detection.pop('tumor_slices')
    detection['tumor_slices'] = (f'{len(tumor_slices)} slices ({min(tumor_slices)}-{max(tumor_slices)})'
                                 if tumor_slices else 'none')
    print(json.dumps({'detection': detection, 'summary': report['summary'], 'device': report['device'],
                      'seconds': report['seconds']}, indent=2))
    print('Saved to', args.out)


if __name__ == '__main__':
    main()
