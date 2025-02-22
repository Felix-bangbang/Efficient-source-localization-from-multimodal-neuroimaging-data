
"""
This step creates digitization montage with MNE
"""

import mne

# Define the digitization points from your data)
digitization_points = [
    [137.5, 185.4, 164.6],  # Point 1.1
    [115.3, 193.7, 184.4],  # Point 1.2
    [86.1, 197.5, 193.0],   # Point 1.3
    [52.7, 187.4, 189.1],   # Point 1.4
    [32.6, 174.3, 176.8],   # Point 1.5
    [18.5, 164.8, 152.3],   # Point 1.6
    [152.1, 173.9, 140.1],  # Point 1.7
    [141.4, 195.0, 139.2],  # Point 1.8
    [117.5, 209.6, 163.9],  # Point 1.9
    [84.5, 217.7, 169.7],   # Point 1.10
    [50.5, 206.2, 170.8],   # Point 1.11
    [27.5, 190.4, 151.2],   # Point 1.12
    [9.2, 156.3, 127.6],    # Point 1.13
    [156.1, 162.9, 115.7],  # Point 1.14
    [147.2, 189.8, 111.0],  # Point 1.15
    [130.0, 209.1, 107.6],  # Point 1.16
    [124.9, 212.7, 135.0],  # Point 1.17
    [104.1, 221.0, 134.3],  # Point 1.18
    [81.3, 225.2, 139.0],   # Point 1.19
    [59.4, 220.3, 140.6],   # Point 1.20
    [37.5, 206.9, 145.2],   # Point 1.21
    [105.2, 221.1, 107.4],  # Point 1.22
    [78.8, 224.8, 107.4],   # Point 1.23
    [53.3, 219.7, 110.4],   # Point 1.24
    [32.0, 204.8, 116.8],   # Point 1.25
    [157.2, 155.6, 90.2],   # Point 1.26
    [152.3, 182.1, 84.5],   # Point 1.27
    [133.6, 202.7, 78.5],   # Point 1.28
    [108.0, 216.2, 75.5],   # Point 1.29
    [78.4, 220.6, 74.7],    # Point 1.30
    [53.7, 215.5, 80.8],    # Point 1.31
    [29.0, 202.5, 85.6],    # Point 1.32
    [6.4, 145.0, 98.9],     # Point 1.33
    [16.2, 179.4, 120.8],   # Point 1.34
    [11.2, 183.2, 91.7],    # Point 1.35
    [6.2, 136.2, 68.3],     # Point 1.36
    [12.1, 165.7, 61.7],    # Point 1.37
    [29.6, 190.6, 55.6],    # Point 1.38
    [53.3, 205.3, 50.7],    # Point 1.39
    [78.2, 208.6, 47.4],    # Point 1.40
    [105.7, 204.8, 51.0],   # Point 1.41
    [130.0, 193.6, 54.6],   # Point 1.42
    [145.8, 173.8, 58.8],   # Point 1.43
    [152.6, 144.4, 62.8],   # Point 1.44
    [16.9, 130.7, 45.7],    # Point 1.45
    [23.0, 154.6, 38.9],    # Point 1.46
    [37.4, 172.8, 32.8],    # Point 1.47
    [54.9, 184.4, 33.1],    # Point 1.48
    [79.4, 188.7, 30.4],    # Point 1.49
    [100.8, 186.8, 30.5],   # Point 1.50
    [120.0, 178.0, 34.3],   # Point 1.51
    [134.7, 160.2, 34.5],   # Point 1.52
    [142.7, 140.5, 38.8],   # Point 1.53
    [124.9, 133.5, 19.6],   # Point 1.54
    [110.1, 156.2, 16.6],   # Point 1.55
    [80.2, 161.0, 13.6],    # Point 1.56
    [50.8, 153.3, 17.1],    # Point 1.57
    [35.5, 129.2, 25.0],    # Point 1.58
    [54.5, 128.3, 12.8],    # Point 1.59
    [80.1, 127.5, 8.7],     # Point 1.60
    [106.9, 126.9, 8.6],    # Point 1.61
    [79.1, 96.6, 11.3],     # Point 1.62
    [19.0, 77.6, 88.8],     # Point 1.63
    [152.5, 97.4, 100.3],   # Point 1.64 (LPA)
    [90.8, 135.5, 211.6],   # Point 1.65 (NAS)
    [18.7, 90.2, 108.0]     # Point 1.66 (RPA)
]

# Convert to meters (assuming mm units)
digitization_points = [[x / 1000, y / 1000, z / 1000] for x, y, z in digitization_points]

# Extract fiducials
lpa = digitization_points[63]  # Point 1.64
nas = digitization_points[64]  # Point 1.65
rpa = digitization_points[65]  # Point 1.66

# Extract electrode positions (Points 1.1 to 1.63)
electrode_positions = {}
for i in range(63):  # 0-based index for 1.1 to 1.63
    label = f"EEG{i+1:03d}"  # e.g., EEG001, EEG002, ..., EEG063
    electrode_positions[label] = digitization_points[i]

# Create DigMontage
dig_montage = mne.channels.make_dig_montage(
    ch_pos=electrode_positions,
    nasion=nas,
    lpa=lpa,
    rpa=rpa,
    coord_frame='head'  # Assuming digitizer space aligns with head coordinates
)

dig_montage.save('Electrode-dig.fif')

# Verify
dig = mne.channels.read_dig_fif('Electrode-dig.fif')
print(dig)