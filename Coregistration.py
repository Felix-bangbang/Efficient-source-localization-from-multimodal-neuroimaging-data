
"""
This step runs Coregistration with MNE
"""

import mne

# Launch coregistration GUI with fsaverage
mne.gui.coregistration(
    subject="fsaverage", # Replace subject id
    subjects_dir="/Users/data_forsource_localiz", # Replace path
)

input("Press Enter to close the GUI...")
