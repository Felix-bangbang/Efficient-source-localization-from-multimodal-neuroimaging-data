# Scripts for robust source localization from EEG-TMS-MRI data using MNE-Python.

## Overview of the Workflow
1. Load and Preprocess EEG Data: Loads raw EEG data and applies basic preprocessing;
2. Clean TMS Artifacts: Identifies and marks segments affected by TMS pulses;
3. Run ICA for Artifact Removal: Uses Independent Component Analysis (ICA) to remove additional artifacts like eye movements;
4. Prepare MRI Data: Converts MRI DICOM files to NIfTI format (if needed) and sets up source space and head models, ideally using individual MRI data (with a fallback to the fsaverage template);
5. Compute Forward Solution: Models how brain activity projects to EEG sensors, using digitized electrode positions when available;
6. Compute Covariance Matrices: Estimates data and noise covariance for beamforming;
7. Apply LCMV Beamformer: Reconstructs source activity from EEG data;
8. Visualize and Save Results: Saves source estimates and plots.
Note: It will fall back to templates (fsaverage) or standard montages when individual data is unavailable, ensuring the analysis completes even under suboptimal conditions.

For users who are not familiar with programming, you could directly run the main script ‘v2_TEM.py’ after modifying the path, but if you do not have the individual data, the results may not be accurate. 
For other users, you could use other scripts to do reconstruction, create BEM and Digitized Montage, perform coregistration, etc. to ensure the accuracy of source localization.
