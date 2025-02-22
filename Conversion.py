
"""
This step converts DICOM files to NIfTI format for FreeSurfer processing
"""

from v2_TE import convert_dicom_to_nifti

dicom_dir = "Name_data_forsource_localiz/OriginalImage"  # Path to DICOM folder
output_dir = "Name_data_forsource_localiz/nifti"         # Where to save NIfTI
subject_id = "sub003"                                     # Subject ID

# Install 'dcm2niix' before running 
nifti_path = convert_dicom_to_nifti(dicom_dir, output_dir, subject_id)
print(f"NIfTI saved to: {nifti_path}")