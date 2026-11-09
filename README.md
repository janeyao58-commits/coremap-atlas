# CoreMap atlas-view edition
Independent deployment; the research app and its repository remain unchanged.

## Start
Python 3.11: `streamlit run coremap_original.py`.
Baseline: coremap-research commit 4a11e8e8195c9b8f569b07d2ed5a0cf01ae1e4aa.
The seven processing/model modules and frozen model bundle are unchanged. Rigid to SyN, seed 20260921, single ITK thread and genericLabel are preserved.
Static templates, frozen model bundle and pre-aligned display atlases are packaged in deployment_assets.zip and extracted once at app startup, outside patient processing. No patient files are included.

## Added display
Registered core-mask intersections with AAL3, JHU or Brainnetome on an MNI background. Atlas switching and slice navigation run in the browser and never submit an imaging job. Physical-space checks permit lossless axis permutation only; incompatible grids fail visibly rather than silently resampling. Unlabelled lesion voxels remain visible. L/R follow RAS world orientation.
The five frozen-model ROI contributions are beta * (encoded input - development mean) / scale, in log-odds. They do not decompose the difference between two model probabilities. Manual registration quality review is preserved.

## Timing
The previously measured 20-case median of 152.8 seconds describes the existing imaging-processing and risk-computation benchmark. It excludes new browser asset downloads and rendering and is not a newly measured cloud latency. No new 20-case benchmark was run for this display change.

## Resources
Third-party templates retain their original terms. Display atlases were generated from predictor.resources() with the same labels and physical grid, then losslessly reoriented to RAS. AAL names use AAL3v1_1mm.xml; Brainnetome names use BN_Atlas_246_LUT.txt; JHU labels use the FSL JHU-labels.xml distributed in the FreeSurfer FSL 5.0.7 tutorial package.
