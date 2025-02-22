
"""
This script aims to build a robust automatical source localization process 
from TMS-EEG-MRI data using MNE-Python.
"""

import mne
import numpy as np
from mne.preprocessing import ICA
from mne.beamformer import make_lcmv, apply_lcmv
from mne.io import read_raw_brainvision
import os
import subprocess


def load_and_preprocess_eeg(vhdr_path):

    """
    Load and preprocess EEG data
    """

    # Load raw data with preload=True to ensure data is in memory
    raw = read_raw_brainvision(vhdr_path, preload=True)
    print(f"Loaded {len(raw.annotations)} annotations")
    
    # identify and mark EMG channels
    emg_channels = [ch for ch in raw.ch_names if any(x in ch for x in 
                   ['APB', 'FDI', 'ADM', 'EDC', 'BB', 'TA', 'HalBr', 'OrbOris'])]
    print(f"Identified EMG channels: {emg_channels}")
    
    for ch in emg_channels:
        raw.set_channel_types({ch: 'emg'})
    
    # pick EEG channels for further processing
    raw_eeg = raw.copy().pick_types(eeg=True, exclude=[])
    print(f"Using {len(raw_eeg.ch_names)} EEG channels for source analysis")
    
    # Basic filtering
    raw_eeg.filter(l_freq=1, h_freq=45)
    raw_eeg.notch_filter(freqs=[50, 100])
    
    # Setup average reference as a projector (but don't apply it yet)
    raw_eeg, _ = mne.set_eeg_reference(raw_eeg, ref_channels='average', projection=True)
    
    return raw_eeg


def clean_tms_artifacts(raw):

    """
    Clean TMS artifacts from the data using Stimulus/B annotations
    """

    if not raw.preload:
        raw.load_data()
        
    # Convert annotations to events, focusing on Stimulus/B
    events, event_id = mne.events_from_annotations(raw)
    
    if 'Stimulus/B' not in event_id:
        print("Warning: 'Stimulus/B' not found in events. Skipping TMS cleaning.")
        return raw
    
    # Find indices of Stimulus/B events
    stim_b_events = events[events[:, 2] == event_id['Stimulus/B']]
    print(f"Found {len(stim_b_events)} TMS pulses")
    
    # Define time window around TMS pulse to remove
    tmin, tmax = -0.002, 0.010  # ±2ms around pulse
    
    # Mark bad segments
    bad_segments = []
    for event in stim_b_events:
        onset = event[0] / raw.info['sfreq']  # Convert to seconds
        bad_segments.append((onset + tmin, onset + tmax))
    
    # interpolate bad segments
    cleaned_raw = raw.copy()
    cleaned_raw.annotations.append(
        onset=[x[0] for x in bad_segments],
        duration=[(x[1]-x[0]) for x in bad_segments],
        description=['bad_tms' for _ in bad_segments]
    )
    
    return cleaned_raw


def run_ica(raw, n_components=15, apply_threshold=False):

    """
    Run ICA for additional artifact removal with options to handle problematic data.
    """

    if not raw.preload:
        raw.load_data()

    raw_ica = raw.copy()
    print(f"Data shape: {raw_ica.get_data().shape}")

    # initialize bad index array
    try:
        bad_idx = np.zeros(raw_ica.get_data().shape[1], dtype=bool)
    except AttributeError as e:
        print(f"Error while creating bad_idx: {e}")
        return raw

    # Check annotations before processing
    if not raw_ica.annotations:
        print("No annotations found, skipping artifact removal.")

    for annot in raw_ica.annotations:
        print(f"Processing annotation: {annot}")
        if annot['description'] == 'bad_tms':
            start_idx = int(annot['onset'] * raw_ica.info['sfreq'])
            end_idx = start_idx + int(annot['duration'] * raw_ica.info['sfreq'])
            bad_idx[start_idx:end_idx] = True

    print("Finished marking bad segments.")

    # Run ICA
    try:
        print("Fitting ICA...")
        ica = ICA(n_components=n_components, random_state=42, max_iter='auto')
        reject = dict(eeg=500e-6) if apply_threshold else None
        ica.fit(raw_ica, reject=reject, reject_by_annotation=True)

        eog_indices, eog_scores = ica.find_bads_eog(raw_ica)
        if eog_indices:
            print(f"Found {len(eog_indices)} EOG components")
            ica.exclude = eog_indices

        cleaned_raw = raw.copy()
        ica.apply(cleaned_raw)

        return cleaned_raw

    except RuntimeError as e:
        print(f"ICA fitting failed: {e}")
        return raw


def convert_dicom_to_nifti(dicom_dir, output_dir, subject_id):

    """
    Convert DICOM files to NIfTI format for FreeSurfer processing
    """

    os.makedirs(output_dir, exist_ok=True)
    output_nifti = os.path.join(output_dir, f"{subject_id}_T1.nii.gz")
    
    # apply dcm2niix for DICOM to NIfTI conversion
    cmd = [
        'dcm2niix',
        '-z', 'y',           # Compress output
        '-f', f"{subject_id}_T1",  # Output filename
        '-o', output_dir,    # Output directory
        dicom_dir            # Input DICOM directory
    ]
    
    print(f"Running DICOM conversion: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return output_nifti
    except subprocess.CalledProcessError as e:
        print(f"DICOM conversion failed: {e}")
        return None
    except FileNotFoundError:
        print("dcm2niix not found. Please install it (e.g., 'brew install dcm2niix')")
        return None


def setup_individual_source_space_and_head_model(subject_id, subjects_dir):

    """
    Set up source space and head model using individual MRI
    Falls back to fsaverage template if BEM creation fails
    """

    try:
        # Check if subject exists in FreeSurfer subjects directory
        subject_path = os.path.join(subjects_dir, subject_id)
        if not os.path.exists(subject_path):
            raise FileNotFoundError(f"Subject directory not found: {subject_path}")

        # Setup source space
        src_path = os.path.join(subjects_dir, subject_id, 'bem', f'{subject_id}-oct-6-src.fif')
        os.makedirs(os.path.dirname(src_path), exist_ok=True)
        if os.path.exists(src_path):
            src = mne.read_source_spaces(src_path)
        else:
            src = mne.setup_source_space(subject_id, spacing='oct6',
                                        subjects_dir=subjects_dir, add_dist=False)
            mne.write_source_spaces(src_path, src, overwrite=True)

        # BEM path
        bem_path = os.path.join(subjects_dir, subject_id, 'bem', 
                               f'{subject_id}-5120-5120-5120-bem-sol.fif')
        
        # Try to create BEM if needed
        if not os.path.exists(bem_path):
            # Try watershed first
            try:
                mne.bem.make_watershed_bem(subject_id, subjects_dir, 
                                         overwrite=True, preflood=15, atlas=True)
            except Exception as e:
                print(f"Individual BEM creation failed: {e}")

            # Create BEM solution
            bem_model = mne.make_bem_model(subject_id, conductivity=[0.3, 0.006, 0.3], 
                                          subjects_dir=subjects_dir)
            bem = mne.make_bem_solution(bem_model)
            mne.write_bem_solution(bem_path, bem, overwrite=True)

        return src, mne.read_bem_solution(bem_path), subjects_dir

    except Exception as e:
        print("Falling back to fsaverage template BEM...")
        
        # Get fsaverage files
        fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
        subjects_dir = os.path.dirname(fs_dir)
        
        # Setup fsaverage source space
        src = mne.setup_source_space('fsaverage', spacing='oct6',
                                    subjects_dir=subjects_dir, add_dist=False)
        
        # Get fsaverage BEM
        bem_path = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
        bem = mne.read_bem_solution(bem_path)
        
        return src, bem, subjects_dir


def parse_electrode_digitization(digitization_file):

    """
    Parse electrode digitization file. Adapt this function based on your specific file format.
    """

    # Check file extension to determine format
    _, ext = os.path.splitext(digitization_file)
    
    if ext.lower() == '.nbe':
        try:
            # Try to read as EGI format
            return mne.channels.read_dig_egi(digitization_file)
        except Exception as e:
            print(f"Could not read .nbe file as EGI format: {e}")
            # It might be a custom format - implement custom parser here
            raise NotImplementedError("Custom .nbe parser not implemented")
    elif ext.lower() == '.elp':
        # Polhemus format
        return mne.channels.read_dig_polhemus_fastrack(digitization_file)
    elif ext.lower() == '.sfp' or ext.lower() == '.txt':
        # BESA or generic text format
        return mne.channels.read_dig_besa(digitization_file)
    elif ext.lower() == '.hsp':
        # Captrak format
        return mne.channels.read_dig_captrak(digitization_file)
    else:
        raise ValueError(f"Unsupported digitization file format: {ext}")


def compute_forward_solution_with_digitization(raw, bem, src, subject_id, subjects_dir, digitization_file):

    """
    Compute forward solution using individual head model with digitized electrode positions
    """

    try:
        # Load digitized electrode positions
        dig_montage = parse_electrode_digitization(digitization_file)
        
        # Apply the digitized montage to the raw data
        raw.set_montage(dig_montage, match_case=False, match_alias=True)
        print("Applied digitized electrode montage")
    except Exception as e:
        print(f"Error applying digitized montage: {e}")
        print("Falling back to standard 10-20 montage")
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False, match_alias=True)
    
    # Create MRI <-> Head transformation
    trans_path = os.path.join(subjects_dir, subject_id, 'bem', f'{subject_id}-trans.fif')
    
    # Check if transform file exists
    if not os.path.exists(trans_path):
        print(f"Transform file not found at {trans_path}")
        print("You need to run coregistration to properly align EEG and MRI.")
        print("For now, using an identity transform (this will affect accuracy)")
        
        # Create an identity transform for testing
        trans = mne.transforms.Transform(fro='head', to='mri', trans=np.eye(4))
        mne.write_trans(trans_path, trans, overwrite=True)
    
    # Compute forward solution with individual data
    fwd = mne.make_forward_solution(raw.info, trans=trans_path, src=src,
                                   bem=bem, meg=False, eeg=True,
                                   mindist=5.0, n_jobs=1)
    print(f"Forward solution contains {fwd['nsource']} source points")
    return fwd


def main():

    # Set up paths
    data_dir = 'DEXSTIM_SUB_003_TA_APB_Rest' # Replace
    mri_dir = 'Name_data_forsource_localiz' # Replace
    subject_id = 'fsaverage'  # Replace subject ID
    subjects_dir = os.path.join(os.getcwd(), 'Name_data_forsource_localiz')  # Replace path
    os.makedirs(subjects_dir, exist_ok=True)
    
    # Set up paths for MRI processing
    dicom_dir = os.path.join(mri_dir, 'OriginalImage') # Replace
    digitization_file = os.path.join(mri_dir, 'TextExport', 'Electrode-dig.fif') # Replace
    
    # Check if MRI data needs preprocessing
    subject_mri_dir = os.path.join(subjects_dir, subject_id)
    if not os.path.exists(subject_mri_dir):
        print(f"Subject {subject_id} not found in FreeSurfer directory.")
        print("Checking for DICOM data...")
        
        # Convert DICOM to NIfTI if needed
        if os.path.exists(dicom_dir):
            print(f"Found DICOM directory: {dicom_dir}")
            nifti_output_dir = os.path.join(mri_dir, 'nifti')
            nifti_file = convert_dicom_to_nifti(dicom_dir, nifti_output_dir, subject_id)
            
            if nifti_file and os.path.exists(nifti_file):
                print(f"Successfully converted DICOM to NIfTI: {nifti_file}")
                print("To process this data with FreeSurfer, run:")
                print(f"recon-all -i {nifti_file} -s {subject_id} -all")
                print("This will take several hours. For now, using fsaverage template.")
                
                # Fall back to fsaverage template
                subject_id = 'fsaverage'
                fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
                subjects_dir = os.path.dirname(fs_dir)
        else:
            print(f"DICOM directory not found: {dicom_dir}")
            print("Falling back to fsaverage template")
            subject_id = 'fsaverage'
            fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
            subjects_dir = os.path.dirname(fs_dir)
    
    # Load APB data - EEG channels only
    print("Processing APB data...")
    apb_raw = load_and_preprocess_eeg(
        os.path.join(data_dir, 'Dexstim_Sub_003_APB_hs_48%_rest.vhdr')
    )
    
    # Load TA data - EEG channels only
    print("Processing TA data...")
    ta_raw = load_and_preprocess_eeg(
        os.path.join(data_dir, 'Dexstim_Sub_003_TA_hs_67%_rest.vhdr')
    )
    
    # Clean TMS artifacts
    print("Cleaning TMS artifacts from APB data...")
    apb_clean = clean_tms_artifacts(apb_raw)
    print("Cleaning TMS artifacts from TA data...")
    ta_clean = clean_tms_artifacts(ta_raw)
    
    # Run ICA with fewer components
    print("Running ICA on APB data...")
    apb_clean = run_ica(apb_clean, n_components=15, apply_threshold=False)
    print("Running ICA on TA data...")
    ta_clean = run_ica(ta_clean, n_components=15, apply_threshold=False)
    
    # Setup source space and head model (template or individual)
    print("Setting up source space and head model...")
    try:
        src, bem, subjects_dir = setup_individual_source_space_and_head_model(subject_id, subjects_dir)
    except Exception as e:
        print(f"Error setting up individual source space: {e}")
        print("Falling back to fsaverage template")
        subject_id = 'fsaverage'
        fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
        subjects_dir = os.path.dirname(fs_dir)
        
        # Setup template source space and head model
        src = mne.setup_source_space(subject_id, spacing='oct6',
                                    subjects_dir=subjects_dir,
                                    add_dist=False)
        bem_path = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
        bem = mne.read_bem_solution(bem_path)
    
    # Ensure projections are applied before computing forward solutions
    print("Applying EEG reference projections...")
    apb_clean.apply_proj()
    ta_clean.apply_proj()
    
    # Compute forward solution for both conditions
    print("Computing forward solutions...")
    try:
        # Try to use digitized electrode positions
        fwd_apb = compute_forward_solution_with_digitization(
            apb_clean, bem, src, subject_id, subjects_dir, digitization_file
        )
        fwd_ta = compute_forward_solution_with_digitization(
            ta_clean, bem, src, subject_id, subjects_dir, digitization_file
        )
    except Exception as e:
        print(f"Error computing forward solution with digitization: {e}")
        print("Calculating forward solution with standard montage...")
        # Use standard montage as fallback
        montage = mne.channels.make_standard_montage('standard_1020')
        apb_clean.set_montage(montage, match_case=False, match_alias=True)
        ta_clean.set_montage(montage, match_case=False, match_alias=True)
        
        # Compute forward solution using standard approach
        fwd_apb = mne.make_forward_solution(apb_clean.info, trans='fsaverage', src=src,
                                          bem=bem, meg=False, eeg=True, mindist=5.0)
        fwd_ta = mne.make_forward_solution(ta_clean.info, trans='fsaverage', src=src,
                                         bem=bem, meg=False, eeg=True, mindist=5.0)
    
    # Compute covariance matrices
    print("Computing covariance matrices...")
    try:
        data_cov = mne.compute_raw_covariance(apb_clean)
    except Exception as e:
        print(f"Error computing data covariance: {e}")
        print("Using identity matrix as data covariance")
        data_cov = mne.make_ad_hoc_cov(apb_clean.info)
    
    try:
        noise_cov = mne.compute_raw_covariance(apb_clean, tmin=0, tmax=0.2)
    except Exception as e:
        print(f"Error computing noise covariance: {e}")
        print("Using diagonal noise covariance")
        noise_cov = mne.make_ad_hoc_cov(apb_clean.info, diag=True)
    
    # Compute beamformer
    print("Computing beamformers...")
    filters_apb = make_lcmv(apb_clean.info, fwd_apb, data_cov, reg=0.05,
                           noise_cov=noise_cov, pick_ori='max-power',
                           weight_norm='unit-noise-gain', rank=None)
    filters_ta = make_lcmv(ta_clean.info, fwd_ta, data_cov, reg=0.05,
                          noise_cov=noise_cov, pick_ori='max-power',
                          weight_norm='unit-noise-gain', rank=None)
    
    # apply beamformer to get source estimates
    print("Applying beamformers...")

    # define event markers for epochs
    events, event_id = mne.events_from_annotations(apb_clean)

    # create epochs
    epochs_apb = mne.Epochs(
        apb_clean, events, event_id=event_id, tmin=-0.1, tmax=0.4,
        baseline=(None, 0), reject=None, reject_by_annotation=False, preload=True 
    )
    evoked_apb = epochs_apb.average()

    # Repeat for TA data
    events_ta, event_id_ta = mne.events_from_annotations(ta_clean)
    epochs_ta = mne.Epochs(
        apb_clean, events_ta, event_id=event_id_ta, tmin=-0.1, tmax=0.4,
        baseline=(None, 0), reject=None, reject_by_annotation=False, preload=True 
    )
    evoked_ta = epochs_ta.average()

    # apply LCMV beamforming
    stc_apb = apply_lcmv(evoked_apb, filters_apb)
    stc_ta = apply_lcmv(evoked_ta, filters_ta)

    # Save source estimates
    stc_dir = os.path.join(os.getcwd(), 'source_estimates')
    os.makedirs(stc_dir, exist_ok=True)
    stc_apb.save(os.path.join(stc_dir, f'{subject_id}_apb-lh.stc'))
    stc_ta.save(os.path.join(stc_dir, f'{subject_id}_ta-lh.stc'))
    
    # Plot the results
    print("Plotting results...")
    try:
        brain_apb = stc_apb.plot(subjects_dir=subjects_dir,
                                subject=subject_id,
                                hemi='both',
                                time_viewer=True,
                                title=f"APB Source Activity ({subject_id})")
        
        brain_ta = stc_ta.plot(subjects_dir=subjects_dir,
                              subject=subject_id,
                              hemi='both',
                              time_viewer=True,
                              title=f"TA Source Activity ({subject_id})")
        
        # Save screenshots
        brain_apb.save_image(os.path.join(stc_dir, f'{subject_id}_apb_sources.png'))
        brain_ta.save_image(os.path.join(stc_dir, f'{subject_id}_ta_sources.png'))
        
    except Exception as e:
        print(f"Could not plot results: {e}")
        print("Source estimation completed, but visualization failed")
    
    print("Processing complete!")


if __name__ == "__main__":
    main()

    input("Press Enter to close the GUI...")