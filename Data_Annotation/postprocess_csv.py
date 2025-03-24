# This is the post processing file for my csv files for ML code
import pandas as pd
from utils_module import helper_functions_module

# Load ROI data (assuming it's loaded from a JSON file or directly assigned)
roi_data = helper_functions_module.load_roi_data("ROI_values/qiyas_multicam.camera.json")  # Your ROI JSON data here

# CSV Path
csv_path = 'outputs\cam_1_chunk_02-03-25_15-35-desk1-2-3\csv\cam_1_chunk_02-03-25_15-35-desk1-2-3_data.csv'

# Output Path
output_path = 'processed_data.csv'

# Load CSV data into a DataFrame
df = pd.read_csv(csv_path)

# Function to get the ROI values for each camera and desk
def get_roi(camera_id, desk):
    camera_data = next(item for item in roi_data if item['_id'] == f'camera_{camera_id}')
    roi = camera_data['data'].get(str(desk))
    if roi:
        return roi['xmin'], roi['xmax'], roi['ymin'], roi['ymax']
    return None

# Normalize function
def normalize_keypoint_x(x, xmin, xmax):
    if xmin is None or xmax is None or x == 0:
        return x  # If ROI data is missing, return the original value
    normalized_value = (x - xmin) / (xmax - xmin) if xmin != xmax else 0
    return round(normalized_value, 2)

def normalize_keypoint_y(y, ymin, ymax):
    if ymin is None or ymax is None or y == 0:
        return y  # If ROI data is missing, return the original value
    normalized_value = (y - ymin) / (ymax - ymin) if ymin != ymax else 0
    return round(normalized_value, 2)

def convert_boolen(x):
    if x == True:
        return 1
    elif x == False:
        return 0



# Iterate through the DataFrame and normalize the keypoints
for idx, row in df.iterrows():
    camera_id = row['camera_id']
    desk = row['person_idx']
    inside_ROI = row['inside_ROI_flag']
    Hand_ear = row['hand_near_ear']

    xmin, xmax, ymin, ymax = get_roi(camera_id, desk)
    
    # Normalize each keypoint column (starting from keypoint columns)
    for key in row.index:
        # print(key)
        if key.startswith('keypoints_dict'):
            if key.endswith('_x'):
                x_col = key
                row[x_col] = normalize_keypoint_x(row[x_col], xmin, xmax)
            elif key.endswith('_y'):
                y_col = key 
                row[y_col] = normalize_keypoint_y(row[y_col], ymin, ymax)

        elif key == 'inside_ROI_flag':
            row[key] = convert_boolen(row[key])
            
        elif key == 'hand_near_ear':
            row[key] = convert_boolen(row[key])

    # Update the row in the DataFrame
    df.loc[idx] = row

df.to_csv(output_path, index=False)
print(f"Processed data saved to {output_path}")
