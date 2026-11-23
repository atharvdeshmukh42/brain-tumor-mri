"""Constants shared by every part of the project."""
from __future__ import annotations

import re

import numpy as np

KAGGLE_DATASET = 'dschettler8845/brats-2021-task1'

# MRI sequences, in the channel order used by both models.
MODALITIES = ('flair', 't1', 't1ce', 't2')
CASE_FILE_KEYS = MODALITIES + ('seg',)

# Labels stored in the preprocessed files. BraTS uses 0, 1, 2, 4; we store 4 as 3 so labels are contiguous.
LABEL_BACKGROUND, LABEL_NECROTIC, LABEL_EDEMA, LABEL_ENHANCING = 0, 1, 2, 3
LABEL_NAMES = {0: 'Background', 1: 'Necrotic core (NCR)', 2: 'Edema (ED)', 3: 'Enhancing tumor (ET)'}
LABEL_SHORT = {0: 'BG', 1: 'NCR', 2: 'ED', 3: 'ET'}
INTERNAL_TO_BRATS_LABEL = np.array([0, 1, 2, 4], dtype=np.uint8)

# Evaluation regions of the BraTS challenge, in the channel order of the segmentation output.
REGIONS = ('WT', 'TC', 'ET')
REGION_NAMES = {'WT': 'Whole tumor', 'TC': 'Tumor core', 'ET': 'Enhancing tumor'}

EXPECTED_SHAPE = (240, 240, 155)   # BraTS volumes are 240 x 240 x 155 voxels of 1 mm
NORM_CLIP = 5.0                    # z-scored intensities are clipped to [-5, 5]
CROP_MARGIN = 2                    # voxels kept around the brain when cropping
DETECTION_CANVAS = 240             # the detection model sees full 240 x 240 axial slices
HD95_EMPTY_PENALTY_MM = 373.13     # BraTS convention when exactly one of the two masks is empty

RAW_HIST_BINS = np.logspace(0, 5, 101)                    # raw intensities (log scale)
NORM_HIST_BINS = np.linspace(-NORM_CLIP, NORM_CLIP, 101)  # after z-score normalization
FREQUENCY_MAP_DOWNSAMPLE = 2       # tumor-frequency maps are stored at half resolution

CASE_FILE_PATTERN = re.compile(r'^(?P<case>.+?)_(?P<mod>flair|t1ce|t1|t2|seg)\.nii(?:\.gz)?$', re.IGNORECASE)
ARCHIVE_SUFFIXES = ('.zip', '.tar', '.tar.gz', '.tgz')
