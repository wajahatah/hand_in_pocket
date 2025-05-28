# change the position values in a CSV file from -2,-1,0,1,2 to 1,0.5,0.5,0 

import pandas as pd

# Load CSV
input_file = 'your_input.csv'
output_file = 'your_output.csv'
df = pd.read_csv(input_file)

# Ensure required columns exist
required_columns = ['position_t0', 'position_t1', 'position_t2', 'position_t3', 'position_t4', 'source_file', 'hand_in_pocket']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# New columns to add
position_new_cols = ['position_a', 'position_b', 'position_c', 'position_d']

# Placeholder for new rows
new_rows = []

print("Processing rows...\nEnter values like: 1 0.5 0.5 0 or press Enter for default [0.5 0.5 0.5 0.5], or 's' to skip.\n")

for idx, row in df.iterrows():
    pos_values = [row[f'position_t{i}'] for i in range(5)]
    
    # Check if all values are the same
    if len(set(pos_values)) == 1:
        print(f"\nRow {idx} — Source File: {row['source_file']} — Position: {pos_values[0]}")
        user_input = input("Enter 4 values for position_a to position_d (or 's' to skip): ").strip()

        if user_input.lower() == 's':
            continue

        if not user_input:
            values = [0.5, 0.5, 0.5, 0.5]
        else:
            try:
                values = list(map(float, user_input.split()))
                if len(values) != 4:
                    print("Invalid input. Skipping row.")
                    continue
            except ValueError:
                print("Invalid input. Skipping row.")
                continue

        # Create a copy of the row and add new columns
        new_row = row.drop(labels=[f'position_t{i}' for i in range(5)]).to_dict()

        for col_name, val in zip(position_new_cols, values):
            new_row[col_name] = val

        new_rows.append(new_row)
    else:
        print(f"Row {idx} skipped — Position values not uniform: {pos_values}")

# Create final DataFrame
if new_rows:
    new_df = pd.DataFrame(new_rows)

    # Reorder columns: place new position_a-d before hand_in_pocket
    cols = list(new_df.columns)
    if 'hand_in_pocket' in cols:
        hip_index = cols.index('hand_in_pocket')
        reordered_cols = cols[:hip_index] + position_new_cols + cols[hip_index:]
        for col in position_new_cols:
            reordered_cols.remove(col)  # Remove duplicates
        new_df = new_df[reordered_cols]

    # Save to new file
    new_df.to_csv(output_file, index=False)
    print(f"\n✅ Output saved to: {output_file}")
else:
    print("⚠️ No valid rows processed.")
