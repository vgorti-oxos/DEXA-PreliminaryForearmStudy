# Forearm DEXA Analysis

Python workflow for experimental dual-energy X-ray absorptiometry (DEXA) analysis of forearm images acquired at 50 kV and 70 kV with a Zn filter. The script registers the image pair, prompts for forearm ROI placement, automatically segments the radius and ulna inside the ROI, computes regional BMD and corresponding t-scores, and writes QC images, result tables, and presentation plots.

## In this repo

- `run_dexa.py` - main analysis and GUI workflow.
- `launch_dexa.bat` - Windows double-click launcher.
- `requirements.txt` - Python dependencies.
- `README.md` - this documentation.
- `.gitignore` - excludes patient data, outputs, virtual environments, and local backups.

## Input Folder Layout

Place data locally in a `data/` folder next to `run_dexa.py`:

```text
data/
  Patient age and gender.xlsx
  images_bg/
    ri_single_50_*.tif
    ri_single_70_*.tif
  images_<patient_id>/
    ri_single_50_*.tif
    ri_single_70_*.tif
```

Each patient folder must contain exactly one 50 kV TIFF and one 70 kV TIFF. The demographics spreadsheet must include patient identifier, age, and gender columns compatible with the current script.

## Running The Analysis

On Windows, double-click:

```text
launch_dexa.bat
```

The launcher creates a local `.venv`, installs packages from `requirements.txt`, and starts the GUI workflow.

The script can also be run from an existing Python environment:

```powershell
python run_dexa.py
```

To check that inputs are discoverable without running analysis:

```powershell
python run_dexa.py --check-inputs
```

## ROI Placement

For each patient, the GUI shows the 50 kV analysis image with a pre-sized forearm ROI box.

Place the ROI so that:

- The wrist is at the top.
- The red 0 mm top edge sits at the ulna styloid tip.
- The box extends down the forearm shaft.
- The box is just wide enough to cover the radius and ulna with a modest margin.

The script performs automatic bone segmentation inside the placed ROI. No manual segmentation is required.

## Analysis Summary

The workflow performs:

- Background correction for 50 kV and 70 kV images.
- Orientation standardization and optional edge cropping.
- Registration of 70 kV to 50 kV.
- Forearm masking.
- Edge/path-based radius and ulna segmentation inside the ROI.
- Local soft-tissue `k` estimation across ROI regions.
- BMD computation in `g/cm^2`.
- Regional summaries for:
  - `UD`: 10-25 mm from ulna styloid.
  - `MID`: 25-74 mm from ulna styloid.
  - `ONE_THIRD`: 74-94 mm from ulna styloid.
- Z-scores for age/sex-matched reference bins where available.
- T-scores using pooled peak-reference BMD.

Calibration constants are defined in `run_dexa.py`:

- `MU_B_L = 2.492`
- `MU_B_H = 1.141`
- `CALIBRATION_FACTOR_50_70 = 2.4329`
- `PIXEL_SPACING_MM = 0.1321`

## Outputs

Generated outputs are written to `outputs/`:

- `dexa_results.csv`
- `dexa_results.xlsx`
- Per-patient QC images:
  - `analysis_image.png`
  - `registration_before_blend.png`
  - `registration_after_blend.png`
  - `forearm_mask_overlay.png`
  - `bone_mask_overlay.png`
  - `roi_overlay.png`
  - `bmd_heatmap.png`
  - `analysis_metadata.json`
- Presentation plots in `outputs/plots/`.

These outputs may contain patient identifiers and image-derived patient data. Do not commit them unless they are de-identified and approved for sharing.

## Notes For Development

- Current segmentation is fully automatic and ROI-guided.
- Manual annotations, if created later, should be used only as development/evaluation masks, not as a required production step.
- Keep patient images and result files outside version control.
- For reproducibility, record the script version used to generate any reported results.

## Sample workflow
50kV
<img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/df28a978-47dd-4ecc-9b22-3c94893417f9" />
70kV with 0.5mm Zn filter
<img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/e6423dbe-4c5a-4185-8177-04433c20488d" />

