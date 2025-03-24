# This file is used to write the output values into a csv

import csv

# Function to flatten nested dictionaries
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Write CSV header if file doesn't exist
def initialize_csv(file_path, sample_data):
    with open(file_path, mode='w', newline='') as csv_file:
        fieldnames = ['camera_id'] + list(sample_data.keys()) + ['class_output']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

# Append a row to the CSV file
def write_to_csv(file_path, data):
    with open(file_path, mode='a', newline='') as csv_file:
        fieldnames = ['camera_id'] + list(data.keys()) + ['class_output']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow(data)