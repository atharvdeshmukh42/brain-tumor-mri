"""
brats - shared code for the BraTS 2021 brain tumor project.

Every notebook (and the offline application on the laptop) imports this package, so that
data preparation, normalization, prediction and measurement are always done in exactly the same way.

Modules (lowest level first):

    config        constants: modalities, labels, regions, normalization settings
    fileio        safe file writing on Google Drive (atomic writes, retries, verified copies, disk checks)
    env           Colab / Drive / device / seeds, project folders, reproducibility logs
    notebook      notebook start-up helpers (package installation); standard library only
    raw_data      Kaggle download and extraction of the BraTS archives
    case          one patient: loading NIfTI files, cropping, normalization, the PreparedCase container
    dataset       the whole dataset: resumable sharded preprocessing, statistics, patient split, staging
    metrics       Dice, IoU, HD95, detection metrics, threshold choice, calibration
    localization  tumor position, size, volume and hemisphere from a mask
    training      data loaders, mixed precision, learning-rate schedule, one training epoch, resumable runs
    detection     EfficientNet slice classifier: inputs, augmentation, datasets, prediction, evaluation
    segmentation  3D segmentation (SegResNet / U-Net): patches, datasets, loss, sliding-window prediction
    evaluation    resumable per-patient segmentation + localization evaluation and run comparison
    xai           Grad-CAM, Grad-CAM++, occlusion, uncertainty
    inference     run both trained models on new scans (used by predict_case.py and the app)
    plots         one visual style and every chart helper (PNG + PDF + CSV)

Modules that need PyTorch import it at the top, so `import brats.case` stays light for CPU-only work.
"""

__version__ = '2.0.0'
