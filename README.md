# CSA: Circle of Willis segmentation with connectivity-aware supervision and attention

Code for **Multiclass Circle of Willis Segmentation Enhanced by Connectivity-Aware Loss and Attention Mechanisms**.

**Authors:** Siyu Zhuang and Yueqi Zhu. Correspondence: zhuyueqi@sjtu.edu.cn.

CSA combines class-weighted centerline supervision (CAL), residual spatial attention (RSA) in skip connections, and efficient channel attention (ECA) at the bottleneck of a 3D nnU-Net.

## Release scope

This is a **code-only release**. It contains the local model, loss, and dataloader implementations, the uniform 2000-epoch trainer classes, manuscript CAL parameter comparisons, development-set configuration, and portable preparation/evaluation utilities. It includes **no pretrained weights, patient images, segmentation masks, or experimental result files**. Users must obtain the datasets from their providers and train their own models or supply compatible checkpoints.

The public entry points reproduce the training settings described in the manuscript. Internal base class names ending in `1000` are retained for compatibility; the supported entry points listed below override the budget to 2000 epochs. Only the components needed by these configurations are included.

Publication of this source snapshot is not evidence that a full training run or the manuscript's numerical results have been independently reproduced. This release extracts the formal method from the local implementation and provides portable preparation, inference and evaluation utilities. Checksums of the released source files are listed in `SOURCE_MANIFEST.json`.

## Installation

Use Python 3.11 in a fresh environment. Install the PyTorch build appropriate for your GPU first, then:

```bash
pip install -r requirements.txt
python scripts/install_overlay.py
```

The installer requires `nnunetv2==2.8.1` and copies the custom files into its Python package so nnU-Net can discover the trainer classes during training and inference. It refuses to replace differing existing files unless `--replace` is explicitly supplied. Use a dedicated environment, not a shared active training installation.

```bash
export nnUNet_raw="$PWD/work/nnUNet_raw"
export nnUNet_preprocessed="$PWD/work/nnUNet_preprocessed"
export nnUNet_results="$PWD/work/nnUNet_results"
export nnUNet_compile=false
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
python -m unittest discover -s tests -v
```

Dependencies include PyTorch, dynamic-network-architectures, NumPy, SciPy, SimpleITK, nibabel and scikit-image (installed through the pinned nnU-Net package and requirements). Exact GPU runtime selection is environment-dependent.

## Data and labels

Download the official [TopCoW release](https://doi.org/10.5281/zenodo.15692630). The development release has paired CTA and MRA volumes from 125 patients. The supplied `configs/splits_final.json` keeps both modalities of a patient in the same fold: 100 training and 25 validation patients per fold.

The release expects an unpacked development directory containing `imagesTr/` and `cow_seg_labelsTr/`. It maps original label 15 (3rd-A2) to contiguous label 13. Foreground labels are:

| ID | Class | ID | Class |
|---|---|---|---|
| 1 | BA | 8 | R-PCom |
| 2 | R-PCA | 9 | L-PCom |
| 3 | L-PCA | 10 | ACom |
| 4 | R-ICA | 11 | R-ACA |
| 5 | R-MCA | 12 | L-ACA |
| 6 | L-ICA | 13 | 3rd-A2 |
| 7 | L-MCA | 0 | Background |

Data are not licensed by this code repository. The training release requires attribution and permission for commercial use. The IXI source has a CC BY-SA 3.0 license; the Lausanne source is CC0. Follow the per-subset release terms and cite both TopCoW and the relevant original dataset sources:

- [IXI dataset](https://brain-development.org/ixi-dataset/)
- [Lausanne source paper](https://doi.org/10.1007/s12021-022-09597-0)

The 40-case evaluation cohort is excluded from training. As stated in the manuscript, it was used for CAL parameter selection and module comparisons; it is not an untouched confirmatory test set.

## Training workflow

### 1. Train modality-specific localizers

Prepare each modality from the official release; change paths to your downloaded data:

```bash
python scripts/prepare_data.py --mode localizer --modality ct \
  --source /path/to/TopCoW2024_Data_Release \
  --output "$nnUNet_raw/Dataset190_CSALocalizerCT" \
  --preprocessed-root "$nnUNet_preprocessed"
python scripts/prepare_data.py --mode localizer --modality mr \
  --source /path/to/TopCoW2024_Data_Release \
  --output "$nnUNet_raw/Dataset191_CSALocalizerMR" \
  --preprocessed-root "$nnUNet_preprocessed"
nnUNetv2_plan_and_preprocess -d 190 191 -c 3d_fullres --verify_dataset_integrity
```

Train folds 0 through 4 for each modality using the baseline trainer:

```bash
python scripts/train.py --dataset 190 --fold 0 --method Baseline
python scripts/train.py --dataset 191 --fold 0 --method Baseline
```

Repeat for folds 1–4. `--resume` resumes an existing training run. The budget is 2000 epochs, 250 training and 50 validation batches per epoch, foreground oversampling 0.33, no mirroring, and seed `20260922 + fold`. This does not guarantee bitwise reproducibility across CUDA hardware or multiworker augmentation.

### 2. Build a separate predicted-ROI dataset for each fold

For fold f, run its CTA localizer on the CTA development inputs and its MRA localizer on the MRA development inputs. Use `nnUNetv2_predict`, the corresponding dataset ID, `-tr nnUNetTrainerTopCoWCleanBaseline2000 -f f -chk checkpoint_best.pth --disable_tta`. Keep modalities in separate input directories. Combine the resulting masks in one fold-specific directory, retaining the original case names.

For fold 0:

```bash
python scripts/prepare_data.py --mode roi \
  --source /path/to/TopCoW2024_Data_Release \
  --masks /path/to/fold0_localizer_masks \
  --output "$nnUNet_raw/Dataset192_CSAROIFold0" \
  --preprocessed-root "$nnUNet_preprocessed"
nnUNetv2_preprocess -d 192 -c 3d_fullres
```

The utility expands the predicted bounding box by 16 voxels, uses the whole image for an empty localization, clips intensities to [-1000, 1800], and min-max normalizes each ROI. It retains the fixed `configs/nnUNetPlans.json` plan (80×160×192 patch, batch size 2, no additional normalization). **Do not re-plan the fine-model data.** Use new IDs 193–196 and the matching masks for folds 1–4. Train each dataset only with its corresponding fold index.

### 3. Train CSA or an ablation

```bash
python scripts/train.py --dataset 192 --fold 0 --method CSA
```

| Method | Trainer |
|---|---|
| Baseline | `nnUNetTrainerTopCoWCleanBaseline2000` |
| CAL | `nnUNetTrainerTopCoWCleanCAL2000` |
| RSA | `nnUNetTrainerTopCoWCleanRSA2000` |
| ECA | `nnUNetTrainerTopCoWCleanECA2000` |
| RSA_ECA | `nnUNetTrainerTopCoWCleanRSAECA2000` |
| CAL_RSA | `nnUNetTrainerTopCoWCleanCALRSA2000` |
| CAL_ECA | `nnUNetTrainerTopCoWCleanCALECA2000` |
| CSA | `nnUNetTrainerTopCoWCleanFull2000` |

All eight methods share the same 2000-epoch budget and sampling configuration. CAL gives labels 8/9/10/13 relative weight 1.3 (other foreground classes 1.0), with skeleton coefficient 0.5. The raw weight values are normalized within the loss. The five deep-supervision weights are 8/15, 4/15, 2/15, 1/15, 0. Class-preserving, undilated skeletons are generated after target transformation at each effective scale.

The manuscript's CAL comparisons are also available through `--method`: `CAL_uniform` (class weight 1.0), `CAL_weight1.6` (class weight 1.6), `CAL_skel0.25` (skeleton coefficient 0.25), and `CAL_skel1.0` (skeleton coefficient 1.0). Other settings remain those of CAL. These named entry points were added for the public release to expose the parameter settings in the manuscript.

## Inference and probability ensemble

Use each fold's modality-matched localizer and fine model. Prepare each evaluation ROI from that fold's localization masks; reference segmentation labels are not required:

```bash
python scripts/prepare_inference_roi.py --images /path/to/evaluation_images \
  --masks /path/to/fold0_localizer_masks --output /path/to/fold0_roi_images
```

This writes `crop_records.json` with the original image path, ROI path and crop offset used for restoration. For each fold-specific ROI input directory, use:

```bash
nnUNetv2_predict -i /path/to/fold0_roi_images -o /path/to/fold0_probabilities \
  -d 192 -c 3d_fullres -tr nnUNetTrainerTopCoWCleanFull2000 \
  -f 0 -chk checkpoint_best.pth --disable_tta --save_probabilities
```

Do not directly average arrays from different fold-specific crops. Restore their probabilities to the common full-image grid first; background probability is 1 outside each crop. `scripts/ensemble_probabilities.py` accepts a JSON list of cases. Each case must supply five fold entries:

```json
[
  {
    "case": "example",
    "reference_image": "/path/to/original_image.nii.gz",
    "folds": [
      {
        "probabilities": "/path/to/fold0/example.npz",
        "roi_image": "/path/to/fold0/example_0000.nii.gz",
        "start_xyz": [10, 20, 30]
      }
    ]
  }
]
```

The snippet illustrates one fold entry; add the other four before running. `reference_image` means the original input image defining the grid, not a segmentation label. Arrays must be nnU-Net SimpleITKIO-exported probabilities in C,Z,Y,X order. Geometry and probability normalization are checked before averaging.

```bash
python scripts/ensemble_probabilities.py --manifest /path/to/manifest.json --output-dir work/ensemble_raw
python scripts/postprocess.py --input-dir work/ensemble_raw --output-dir work/ensemble_post --case-prefix ""
```

Postprocessing keeps the largest PCA component and applies the fixed PCom component/bridge rules once after the ensemble. It uses 26-connectivity and the original index-grid thresholds (20 voxels for component size and endpoint distance), not millimeter thresholds.

## Evaluation

Prepare a reference directory with `labelsTr/<case>.nii.gz`, `dataset.json` (the label mapping), and `splits_final.json` listing evaluation IDs in a `val` list. Use the full reference geometry and remap label 15 to 13 beforehand. For a purely external cohort, the split may be `[{"train": [], "val": ["case1", "case2"]}]`.

```bash
python scripts/evaluate.py --raw-root /path/to/reference_directory \
  --pred-dir work/ensemble_post --out-dir work/metrics --case-prefix "" --fold 0
```

Dice and clDice are averaged over reference-positive cases within each class. Complete misses contribute zero to both. HD95 and ASSD are conditional on reference-positive, nonempty predictions. `class_summary.csv` contains per-class values; macro metrics are the unweighted mean of class means, not the average of per-case macros. Surface distances use physical spacing in mm.

The portable evaluator implements the manuscript's complete-miss convention (clDice = 0), computes all 13 classes by default, and rejects affine mismatches. It is a release adaptation, not a bundled archive of the original result-generation jobs. No precomputed results are included.

## Validation

The CPU tests cover patient-level split separation, all supported trainer settings, attention forward/backward passes and added parameter counts, CAL gradients, transformed skeleton targets, metric edge cases, ROI restoration, and fixed postprocessing. They use synthetic inputs only and do not establish reproduction of the reported clinical-data results.

## License and attribution

Code is distributed under Apache-2.0; see `LICENSE` and `NOTICE`. nnU-Net and dependencies retain their own authorship and licenses. Cite nnU-Net, the TopCoW challenge, the original datasets, and the methodological sources used in your research. No publication DOI has been assigned to this manuscript in this release.
