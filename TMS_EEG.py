
"""
Source localization with default MRI template 'fsaverage'
"""

import mne
import numpy as np
from mne.preprocessing import ICA
from mne.beamformer import make_lcmv, apply_lcmv
from mne.io import read_raw_brainvision
import os


def load_and_preprocess_eeg(vhdr_path):

    """
    Load and preprocess EEG data from BrainVision files
    """

    raw = read_raw_brainvision(vhdr_path, preload=True)
    print(f"Loaded {len(raw.annotations)} annotations")
    
    # Identify EMG channels
    emg_channels = [ch for ch in raw.ch_names if any(x in ch for x in 
                   ['APB', 'FDI', 'ADM', 'EDC', 'BB', 'TA', 'HalBr', 'OrbOris'])]
    print(f"Identified EMG channels: {emg_channels}")
    
    # Mark them as EMG type
    for ch in emg_channels:
        raw.set_channel_types({ch: 'emg'})
    
    # Only pick EEG channels for further processing
    raw_eeg = raw.copy().pick_types(eeg=True, exclude=[])
    print(f"Using {len(raw_eeg.ch_names)} EEG channels for source analysis")
    
    # Basic filtering
    raw_eeg.filter(l_freq=1, h_freq=45)
    
    # Remove power line noise
    raw_eeg.notch_filter(freqs=[50, 100])
    
    # Setup average reference as a projector (but don't apply it yet)
    raw_eeg, _ = mne.set_eeg_reference(raw_eeg, ref_channels='average', projection=True)
    
    return raw_eeg


def clean_tms_artifacts(raw):

    """
    Clean TMS artifacts from the data using Stimulus/B annotations
    """

    # Ensure data is loaded
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
    
    # Interpolate bad segments
    cleaned_raw = raw.copy()
    cleaned_raw.annotations.append(
        onset=[x[0] for x in bad_segments],
        duration=[(x[1]-x[0]) for x in bad_segments],
        description=['bad_tms' for _ in bad_segments]
    )
    
    return cleaned_raw


def run_ica(raw, n_components=15, apply_threshold=False):

    """
    Run ICA for additional artifact removal with options to handle problematic data
    """

    if not raw.preload:
        raw.load_data()

    raw_ica = raw.copy()

    # Initialize bad index array
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


def setup_source_space_and_head_model():

    """
    Set up source space and head model using template
    """

    # Download fsaverage files if they don't exist
    fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
    subjects_dir = os.path.dirname(fs_dir)

    # Setup source space using fsaverage template
    src = mne.setup_source_space('fsaverage', spacing='oct6',
                                subjects_dir=subjects_dir,
                                add_dist=False)

    # Get the BEM solution using fsaverage template
    bem_path = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    if not os.path.exists(bem_path):
        # Download if not available
        bem_path = mne.datasets.fetch_fsaverage_files(
            subjects_dir=subjects_dir,
            bem=True
        )['bem']

    bem = mne.read_bem_solution(bem_path)
    
    return src, bem, fs_dir


def compute_forward_solution(raw, bem, src):

    """
    Compute forward solution using template head model with standard 10-20 montage
    """

    # Ensure data is loaded
    if not raw.preload:
        raw.load_data()
        
    # Set standard 10-20 montage
    try:
        # First try standard_1020 montage
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False, match_alias=True)
    except ValueError:
        print("Could not set standard_1020 montage, trying standard_1005...")
        try:
            # Try more detailed standard_1005 montage
            montage = mne.channels.make_standard_montage('standard_1005')
            raw.set_montage(montage, match_case=False, match_alias=True)
        except ValueError:
            print("Could not set standard montages. Manually setting approximate positions...")
            # Get available EEG channel names
            ch_names = raw.ch_names
            
            # Create standard montage
            montage = mne.channels.make_standard_montage('standard_1020')
            
            # For channels that don't match standard names, pick closest standard position
            std_ch_names = montage.ch_names
            ch_pos = montage._get_ch_pos()
            
            # Create new pos dict with only the channels we have
            new_pos = {}
            for ch in ch_names:
                # Try to find match
                if ch in std_ch_names:
                    new_pos[ch] = ch_pos[ch]
                else:
                    # Assign to closest standard position
                    new_pos[ch] = ch_pos[std_ch_names[0]]  # Default to first channel
            
            # Create new montage
            new_montage = mne.channels.make_dig_montage(ch_pos=new_pos, coord_frame='head')
            raw.set_montage(new_montage)

    # Compute forward solution
    fwd = mne.make_forward_solution(raw.info, trans='fsaverage', src=src,
                                   bem=bem, meg=False, eeg=True,
                                   mindist=5.0, n_jobs=1)
    return fwd


def compute_beamformer(raw, fwd, data_cov, noise_cov):

    """
    Compute LCMV beamformer
    """

    filters = make_lcmv(raw.info, fwd, data_cov, reg=0.05,
                       noise_cov=noise_cov, pick_ori='max-power',
                       weight_norm='unit-noise-gain', rank=None)
    return filters


def main():

    data_dir = 'DEXSTIM_SUB_003_TA_APB_Rest'
    
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
    
    # Setup template source space and head model
    print("Setting up source space and head model...")
    src, bem, fs_dir = setup_source_space_and_head_model()
    
    # Ensure projections are applied before computing forward solutions
    print("Applying EEG reference projections...")
    apb_clean.apply_proj()
    ta_clean.apply_proj()
    
    # Compute forward solution for both conditions
    print("Computing forward solutions...")
    fwd_apb = compute_forward_solution(apb_clean, bem, src)
    fwd_ta = compute_forward_solution(ta_clean, bem, src)
    
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
    filters_apb = compute_beamformer(apb_clean, fwd_apb, data_cov, noise_cov)
    filters_ta = compute_beamformer(ta_clean, fwd_ta, data_cov, noise_cov)
    
    # Apply beamformer to get source estimates
    print("Applying beamformers...")

    # Define event markers for epochs
    events, event_id = mne.events_from_annotations(apb_clean)

    # Create epochs
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

    # Apply LCMV beamforming
    stc_apb = apply_lcmv(evoked_apb, filters_apb)
    stc_ta = apply_lcmv(evoked_ta, filters_ta)

    
    # Plot the results
    print("Plotting results...")
    try:
        brain_apb = stc_apb.plot(subjects_dir=os.path.dirname(fs_dir),
                                subject='fsaverage',
                                hemi='both',
                                time_viewer=True)
        
        brain_ta = stc_ta.plot(subjects_dir=os.path.dirname(fs_dir),
                              subject='fsaverage',
                              hemi='both',
                              time_viewer=True)
    except Exception as e:
        print(f"Could not plot results: {e}")
        print("Source estimation completed, but visualization failed")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()