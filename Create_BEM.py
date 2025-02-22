
"""
This step creates BEM model (and source space)
"""

from v2_TE import setup_individual_source_space_and_head_model

subject_id = 'sub003'  # Replace with subject ID
subjects_dir = '/Users/Name_data_forsource_localiz'  # Replace with the actual path

# Call the function to create BEM (and source space)
src, bem, subjects_dir = setup_individual_source_space_and_head_model(subject_id, subjects_dir)

print("BEM creation complete!")