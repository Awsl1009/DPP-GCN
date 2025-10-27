import os
import numpy as np
import pandas as pd
import pickle
import logging


def get_raw_bodies_data_csv(group_data, ske_name, frames_drop_logger):
    """
    Get raw bodies data from the grouped CSV data for a particular ID/Cloth/View.

    Parameters:
      - group_data: The subset of data for a specific ID/Cloth/View (60 continuous rows).
      - ske_name: The skeleton name, derived from the ID.
      - frames_drop_logger: Logger to record any dropped frames.

    Return:
      A dictionary containing the skeleton data:
        - name: The skeleton filename.
        - data: A dict which stores raw data of each body.
        - num_frames: The number of valid frames.
    """
    num_frames = len(group_data)
    frames_drop = []
    bodies_data = dict()
    valid_frames = -1  # 0-based index

    # 17 joints, each with 3 columns (x, y, confidence)
    num_joints = 17
    joint_columns = num_joints * 3

    # Create arrays for 17 joints and their 2D coordinates (x, y)
    joints = np.zeros((num_frames, num_joints, 2), dtype=np.float32)

    # Exclude the 'ID', 'Cloth', 'View', and 'Image' columns (non-numeric)
    group_data_numeric = group_data.iloc[:, 4:]  # Start from column 4 (first three are non-numeric)

    for frame_idx in range(num_frames):
        # Extract only the joint data (x, y coordinates and ignore the confidence level)
        frame_data = group_data_numeric.iloc[frame_idx].values

        # Reshape the data into 17 joints, each with 3 values (x, y, confidence)
        xy_data = frame_data.reshape(num_joints, 3)[:, :2]  # Keep only x, y (first two columns)

        # Store x, y coordinates in the joints array
        joints[frame_idx, :, :] = xy_data
        valid_frames += 1

    # Store the joints data in the body_data dictionary
    body_data = {
        'joints': joints,  # ndarray: (num_frames, 17, 2)
        'interval': list(range(valid_frames + 1))  # store indices of valid frames
    }

    bodies_data['body'] = body_data  # Single body, so we use a simple key

    return {'name': ske_name, 'data': bodies_data, 'num_frames': num_frames}


def get_raw_skes_data_from_csv(csv_path, save_data_pkl, frames_drop_pkl):
    """
    Process the CSV dataset, classify by ID/Cloth/View, select 60 rows per group,
    and save the skeleton data.
    """
    # Read the CSV file
    data = pd.read_csv(csv_path, header=None)

    # Split the first column into ID, Cloth, View, and Image
    id_cloth_view_image = data.iloc[:, 0].str.split('/', expand=True)
    id_cloth_view_image.columns = ['ID', 'Cloth', 'View', 'Image']  # Assign names to the new columns

    # Concatenate these new columns with the original data (excluding the first column)
    data = pd.concat([id_cloth_view_image, data.iloc[:, 1:]], axis=1)

    # Filter by Cloth == 'nm' and View == 'A1-front'
    # data_filtered = data[(data['Cloth'] == 'nm') & (data['View'] == 'A1-front')]
    # Group the data by 'ID', 'Cloth', 'View'
    grouped = data.groupby(['ID', 'Cloth', 'View'])
    frames_drop_logger = logging.getLogger('frames_drop')
    frames_drop_logger.setLevel(logging.INFO)
    frames_drop_logger.addHandler(logging.FileHandler(os.path.join('./raw_data_test', 'frames_drop.log')))
    frames_drop_skes = dict()

    raw_skes_data = []

    # Iterate over each group
    for (group_key, group_data) in grouped:
        # Extract group information
        ID, Cloth, View = group_key

        # Use ID as ske_name
        ske_name = str(ID) + "_" + str(Cloth) + "_" + str(View)
        print(ske_name)

        # Select the middle 60 rows of data if possible
        if len(group_data) >= 60:
            # Calculate the starting position of the middle 60 lines
            start_idx = (len(group_data) - 60) // 2
            group_data_60 = group_data.iloc[start_idx:start_idx + 60]
        else:
            print(f"Warning: Group {group_key} has fewer than 60 rows.")
            continue  # Skip this group if less than 60 rows

        # Process the group data
        bodies_data = get_raw_bodies_data_csv(group_data_60, ske_name, frames_drop_logger)
        raw_skes_data.append(bodies_data)

    # Save the processed skeleton data
    with open(save_data_pkl, 'wb') as fw:
        pickle.dump(raw_skes_data, fw, pickle.HIGHEST_PROTOCOL)

    print(f'Saved raw bodies data into {save_data_pkl}')

    # Log dropped frames, if any
    with open(frames_drop_pkl, 'wb') as fw:
        pickle.dump(frames_drop_skes, fw, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    save_path = './raw_data_test'  # Directory to save the processed data

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Path to the CSV file
    csv_path = './test.csv'  # Update this path with your actual CSV file

    # Define the output paths for the processed data
    save_data_pkl = os.path.join(save_path, 'raw_skes_data.pkl')
    frames_drop_pkl = os.path.join(save_path, 'frames_drop_skes.pkl')

    # Process and save the skeleton data
    get_raw_skes_data_from_csv(csv_path, save_data_pkl, frames_drop_pkl)