
# This step creates a cortical reconstruction from your MRI data.

export FREESURFER_HOME=/Applications/freesurfer/dev # Replace with freesurfer path

export SUBJECTS_DIR=/Users/fvinci6/Documents/Python/Efficient-source-localization-from-multimodal-neuroimaging-data/Name_data_forsource_localiz # Set SUBJECTS_DIR to a writable directory

source $FREESURFER_HOME/SetUpFreeSurfer.sh

mkdir -p $SUBJECTS_DIR

recon-all -i /Users/fvinci6/Documents/Python/Efficient-source-localization-from-multimodal-neuroimaging-data/Name_data_forsource_localiz/nifti/sub003_T1.nii.gz -s sub003 -all -sd $SUBJECTS_DIR