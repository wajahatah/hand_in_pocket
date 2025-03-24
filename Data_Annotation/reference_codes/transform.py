"""import pandas as pd
import ast

# Input CSV file path

# Frame dimensions (replace with your actual values)
frame_width = 1276  # Example width
frame_height = 720  # Example height

subtraction_values = {"person2": 370, "person3": 740, "person4":1080}

body_parts = ["head", "neck", "left_ear", "right_ear", "left_shoulder",
    "left_elbow", "left_hand", "right_shoulder", "right_elbow", "right_hand"]

# Load the CSV into a DataFrame

# Define the persons and corresponding column ranges
# persons = ['person1', 'person2', 'person3', 'person4']
# columns_per_person = 10  # Assuming columns a to j

def parse_and_normalize(value,person,subtraction_values):
    try:
        # Convert the string to a tuple (e.g., "(224.79, 514.90, 1.00)" -> (224.79, 514.90, 1.00))
        if value == 0 or value == "0":
                return (0, 0)
        
        x, y, confidence = ast.literal_eval(value)

        if x == 0 or y == 0 or confidence == 0:
            return (0, 0)

        # Apply subtraction if necessary
        subtraction = subtraction_values.get(person, 0)
        x -= subtraction
        # y -= subtraction

        
        # Normalize x and y values
        x_norm = x / frame_width
        y_norm = y / frame_height
        
        return (x_norm, y_norm)
    except (ValueError, SyntaxError):
        # Handle invalid or malformed data
        return (None, None)

# Generate a list to stack transformed data
df = pd.read_csv(input_csv_path)

# df['frame'] = df['Frame Name'].str.extract(r'(\d+)$').astype(int)

stacked_data = []

for person in ['person1', 'person2', 'person3', 'person4']:#persons:
    person_columns = [f"{person}{chr(i)}" for i in range(ord('a'), ord('a') + len(body_parts))]#columns_per_person)]
    person_data = df[person_columns]

    parsed_data = person_data.apply(lambda value: parse_and_normalize(value, person, subtraction_values))

    # frame_numbers = df["frame"]
    # Parse and normalize each column
    # parsed_data = person_data.applymap(lambda value: parse_and_normalize(value,person,subtraction_values)) #(parse_and_normalize)
    # parsed_data = person_data.apply(
    #     lambda col: col.map(lambda value: parse_and_normalize(value, person, subtraction_values))
    # )

    formatted_data = pd.DataFrame({
        "frame": df["Frame Name"].str.extract(r'(\d+)$').astype(int).iloc[:,0],
        "person": person,
        **{
            part: parsed_data[col].apply(lambda t: f"({t[0]:.2f},{t[1]:.2f})")
            for part, col in zip(body_parts, person_columns)
        }
    })

    # formatted_data = parsed_data.apply(
    #     lambda row: ",".join([f"({x:.2f},{y:.2f})" for x, y in row]), axis=1
    # )

    # # Create the final DataFrame for this person
    # normalized_df = pd.DataFrame({
    #     "frame": frame_numbers,
    #     "person": person,
    #     # "data": formatted_data
    #     **{part: parsed_data[col].apply(lambda t: f"({t[0]:.3f},{t[1]:.3f})") for part, col in zip(body_parts, person_columns)}

    # Split the parsed tuples back into separate columns
    # x_columns = [f"{col}_x" for col in person_columns]
    # y_columns = [f"{col}_y" for col in person_columns]
    # confidence_columns = [f"{col}_conf" for col in person_columns]
    
    # # Create new DataFrame with normalized x, y, and confidence columns
    # normalized_df = pd.DataFrame({
    #     **{x_col: parsed_data[col].apply(lambda t: t[0]) for x_col, col in zip(x_columns, person_columns)},
    #     **{y_col: parsed_data[col].apply(lambda t: t[1]) for y_col, col in zip(y_columns, person_columns)},
    #     **{conf_col: parsed_data[col].apply(lambda t: t[2]) for conf_col, col in zip(confidence_columns, person_columns)},
    # })

    # Add a person identifier column
    # normalized_df['person'] = person
    
    # Append to stacked data
    stacked_data.append(formatted_data)

# Combine all stacked data into a single DataFrame
result = pd.concat(stacked_data, ignore_index=True)

# Save to a new CSV file
result.to_csv(output_csv_path, index=False,header=["frmae","person"]+body_parts)

print(f"Transformed data saved to {output_csv_path}")


# Generate a list of new rows to stack
stacked_data = []

for person_idx, person in enumerate(persons, start=1):
    start_col = f"{person}a"
    end_col = f"{person}i"

    # Extract relevant columns for the current person
    person_data = df.loc[:, start_col:end_col]

    # Normalize by frame dimensions
    normalized_data = person_data.copy()
    for col in normalized_data.columns:
        # Assume x-coordinates (a, c, e, g, i) are normalized by width
        # and y-coordinates (b, d, f, h) are normalized by height
        if col[-1] in ['a', 'c', 'e', 'g', 'i']:  # x-coordinates
            normalized_data[col] = normalized_data[col] / frame_width
        elif col[-1] in ['b', 'd', 'f', 'h']:  # y-coordinates
            normalized_data[col] = normalized_data[col] / frame_height

    # Add person identifier to differentiate rows in the stacked data
    normalized_data['person'] = person

    # Append to stacked data
    stacked_data.append(normalized_data)

# Combine all stacked data into a single DataFrame
result = pd.concat(stacked_data, ignore_index=True)

# Save the transformed data to a new CSV
result.to_csv(output_csv_path, index=False)

print(f"Transformed data saved to {output_csv_path}")
"""

import pandas as pd
import ast

# Input and output file paths
# input_csv_path = "video2.csv"
# output_csv_path = "transformed_video2.csv"
v = "v1"
input_csv_path = f"C:/OsamaEjaz/Qiyas_Gaze_Estimation/Wajahat_Yolo_keypoint/frames_output/{v}/{v}.csv"# "C:/OsamaEjaz/Qiyas_Gaze_Estimation/Wajahat_Yolo_keypoint/frames_output/video1/n.csv"#"input.csv"
output_csv_path = f"C:/OsamaEjaz/Qiyas_Gaze_Estimation/Wajahat_Yolo_keypoint/frames_output/new_dataset/{v}.csv"#"output.csv"
additional_csv_path = f"C:/OsamaEjaz/Qiyas_Gaze_Estimation/Wajahat_Yolo_keypoint/frames_output/new_dataset/new_dataset/processed_{v}.csv"#"output.csv"

# Frame dimensions
frame_width = 1280
frame_height = 780

# Subtraction values for normalization
subtraction_values = {
    "person1": 60,
    "person2": 370,
    "person3": 680,
    "person4": 1030
}

# Body parts for column mapping
body_parts = [
    "head", "neck", "left_ear", "right_ear", "left_shoulder",
    "left_elbow", "left_hand", "right_shoulder", "right_elbow", "right_hand"
]

# Function to parse and normalize values
def parse_and_normalize(value):#, person, subtraction_values):
    try:
        # Handle missing points or zero entries
        # if value == 0 or value == "0":
        if value == "(0.00,0.00,0.00)":
            return (0, 0)
        
        # Convert string representation to a tuple
        x, y, confidence = ast.literal_eval(value)
        
        # Handle invalid or missing data
        if x == 0 or y == 0 or confidence == 0:
            return (0, 0)
        # print("X:" ,x)
        # Apply subtraction for specific persons
        # subtraction = subtraction_values.get(person.lower(), 0)
        # print(f"Person: {person}, Subtraction: {subtraction}")
        # x -= subtraction
        # y -= subtraction
        # print("XS:", x)
        
        # Normalize the coordinates
        x_norm = x / frame_width
        y_norm = y / frame_height
        
        return (x_norm, y_norm)
    except (ValueError, SyntaxError, TypeError):
        return (0, 0)

# Load the input CSV
df = pd.read_csv(input_csv_path)

# Prepare transformed data

for person in ["Person1", "Person2", "Person3", "Person4"]:
    for part in body_parts:
        column_name = f"{person}_{part}"
        if column_name in df.columns:
            df[column_name] = df[column_name].apply(parse_and_normalize)

# Load the additional CSV and select the columns to copy
additional_df = pd.read_csv(additional_csv_path)
columns_to_copy = ["Person1_hand_raised", "Person2_hand_raised", "Person3_hand_raised",	"Person4_hand_raised"]  # Replace with actual column names

# Check if the columns exist in the additional CSV
for col in columns_to_copy:
    if col in additional_df.columns:
        df[col] = additional_df[col]
    else:
        print(f"Column {col} does not exist in the additional CSV. Adding column filled with 0.")
        df[col] = 0
#     if col not in additional_df.columns:
#         raise ValueError(f"Column {col} does not exist in the additional CSV")

# # Copy the selected columns into the normalized dataframe
# for col in columns_to_copy:
#     df[col] = additional_df[col]

# Save the result to a new CSV file
df.to_csv(output_csv_path, index=False)

print(f"Normalized data saved to {output_csv_path}")

"""
# to seperate persons and add them in one coloumns in the csv 

stacked_data = []
for person in ["Person1", "Person2", "Person3", "Person4"]:
    # Extract relevant columns for this person
    # person_columns = [f"{person}{chr(i)}" for i in range(ord('a'), ord('a') + len(body_parts))]
    subtraction = subtraction_values.get(person.lower(), 0)
    print(f"Processing {person}: Subtraction value is {subtraction}")
    person_columns = [f"{person}_{part}" for part in body_parts]
    person_data = df[person_columns]
    
    # Normalize data for each column
    parsed_data = person_data.applymap(lambda value: parse_and_normalize(value, person, subtraction_values))
    
    # Prepare formatted data for this person
    formatted_data = pd.DataFrame({
        "frame": df["Frame"], #df["Frame Name"].str.extract(r'(\d+)$').astype(int).iloc[:, 0],
        "person": person,
        **{
            # part: parsed_data[col].apply(lambda t: f"({t[0]:.3f},{t[1]:.3f})")
            # for part, col in zip(body_parts, person_columns)
            part: parsed_data[f"{person}_{part}"].apply(lambda t: f"({t[0]:.3f},{t[1]:.3f})")
            for part in body_parts
        }
    })
    
    stacked_data.append(formatted_data)

# Combine all transformed person data
result = pd.concat(stacked_data, ignore_index=True)

# Save the result to a new CSV file
result.to_csv(output_csv_path, index=False)

print(f"Transformed data saved to {output_csv_path}")
"""