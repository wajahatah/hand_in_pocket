import pandas as pd
import numpy as np

def generate_rnn_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # Extract source_file and label
    source_files = df['source_file'].values
    labels = df['hand_in_pocket'].values
    position_cols = ['position_a', 'position_b', 'position_c', 'position_d']

    # Dynamically extract columns per time step
    sequences = []
    for index, row in df.iterrows():
        frames = []
        for t in range(5):
            x_cols = [f'kp_{i}_x_t{t}' for i in range(10)]
            y_cols = [f'kp_{i}_y_t{t}' for i in range(10)]
            keypoints = row[x_cols + y_cols].values.astype(np.float32)

            positions = row[position_cols].values.astype(np.float32)  # (4,)
            frame_features = np.concatenate([keypoints, positions])   # (24,)
            frames.append(frame_features)

        sequence = np.concatenate(frames)  # Flatten to 1D (5 x 24 = 120)
        sequences.append(sequence)

    # Create final DataFrame
    output_df = pd.DataFrame(sequences)
    output_df.insert(0, 'source_file', source_files)
    output_df['label'] = labels

    # Save
    output_df.to_csv(output_csv, index=False)
    print(f"✅ RNN-ready CSV saved at: {output_csv}")


input_csv = "C:/wajahat/hand_in_pocket/dataset/split_keypoint/combined/temp_kp_l1_v2_norm_pos_gen.csv"
output_csv = "C:/wajahat/hand_in_pocket/dataset/split_keypoint/combined/temp_kp_l1_v2_norm_pos_gen_rnn.csv"
generate_rnn_csv(input_csv, output_csv)