import pandas as pd
import os 

# csv1 = "C:/wajahat/hand_in_pocket/dataset/without_kp/hp_augmented_annotations2.csv"
# csv2 = "C:/wajahat/hand_in_pocket/dataset/without_kp/no_hp_augmented_annotations2.csv"
csv2 = "C:/wajahat/hand_in_pocket/dataset/split_keypoint/combined/temp_kp_l1_v2_norm_sorted_pos_gen.csv"
output_csv = "C:/wajahat/hand_in_pocket/dataset/without_kp/temp_kp_l1_v2_norm_sorted_pos_gen_round.csv"

# df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)

combined_df = df2
# combined_df = pd.concat([df1, df2], ignore_index=True)

new_columns_order = ["camera",	"video",	"frame",	"desk",	"kp_0_x_t0",	"kp_0_x_t1",	"kp_0_x_t2",	"kp_0_x_t3",	"kp_0_x_t4",	"kp_0_y_t0",	"kp_0_y_t1",
                     	"kp_0_y_t2",	"kp_0_y_t3",	"kp_0_y_t4",	"kp_1_x_t0",	"kp_1_x_t1",	"kp_1_x_t2",	"kp_1_x_t3",	"kp_1_x_t4",
                        	"kp_1_y_t0",	"kp_1_y_t1",	"kp_1_y_t2",	"kp_1_y_t3",	"kp_1_y_t4",	"kp_2_x_t0",	"kp_2_x_t1",	"kp_2_x_t2",
                            	"kp_2_x_t3",	"kp_2_x_t4",	"kp_2_y_t0",	"kp_2_y_t1",	"kp_2_y_t2",	"kp_2_y_t3",	"kp_2_y_t4",	"kp_3_x_t0",
                                "kp_3_x_t1",	"kp_3_x_t2",	"kp_3_x_t3",	"kp_3_x_t4",	"kp_3_y_t0",	"kp_3_y_t1",	"kp_3_y_t2",	"kp_3_y_t3",
                            "kp_3_y_t4",	"kp_4_x_t0",	"kp_4_x_t1",	"kp_4_x_t2",	"kp_4_x_t3",	"kp_4_x_t4",	"kp_4_y_t0",	"kp_4_y_t1",	
                        "kp_4_y_t2",	"kp_4_y_t3",	"kp_4_y_t4",	"kp_5_x_t0",	"kp_5_x_t1",	"kp_5_x_t2",	"kp_5_x_t3",	"kp_5_x_t4",	
                    "kp_5_y_t0",	"kp_5_y_t1",	"kp_5_y_t2",	"kp_5_y_t3",	"kp_5_y_t4",	"kp_6_x_t0",	"kp_6_x_t1",	"kp_6_x_t2",	
                "kp_6_x_t3",	"kp_6_x_t4",	"kp_6_y_t0",	"kp_6_y_t1",	"kp_6_y_t2",	"kp_6_y_t3",	"kp_6_y_t4",	"kp_7_x_t0",	"kp_7_x_t1",
            	"kp_7_x_t2",	"kp_7_x_t3",	"kp_7_x_t4",	"kp_7_y_t0",	"kp_7_y_t1",	"kp_7_y_t2",	"kp_7_y_t3",	"kp_7_y_t4",	"kp_8_x_t0",
                "kp_8_x_t1",	"kp_8_x_t2",	"kp_8_x_t3",	"kp_8_x_t4",	"kp_8_y_t0",	"kp_8_y_t1",	"kp_8_y_t2",	"kp_8_y_t3",	"kp_8_y_t4",	
                "kp_9_x_t0",	"kp_9_x_t1",	"kp_9_x_t2",	"kp_9_x_t3",	"kp_9_x_t4",	"kp_9_y_t0",	"kp_9_y_t1",	"kp_9_y_t2",	"kp_9_y_t3",
                    	"kp_9_y_t4",	"position_a",	"position_b",	"position_c",	"position_d",	"hand_in_pocket"]

# filtered_columns = [col for col in new_columns_order if col in combined_df.columns]
# combined_df = combined_df[filtered_columns]
# combined_df = combined_df.astype(int)

for col in combined_df.columns:
    if 'x' in col.lower():
        # combined_df[col] = combined_df[col].astype(int) #for not normalized keypoint to convert them into numbers like 271,542
        # combined_df[col] = pd.to_numeric((combined_df[col] / 1280), errors='coerce')
        # combined_df[col] = combined_df[col].apply(lambda x: -1 if x == 0 else x)
        combined_df[col] = combined_df[col].round(3) # for notmalized keypoints to convert them till 3 decimal places 
    elif 'y' in col.lower():
        # combined_df[col] = combined_df[col].astype(int) #for not normalized keypoint to convert them into numbers like 271,542
        # combined_df[col] = pd.to_numeric((combined_df[col] / 720), errors='coerce')
        # combined_df[col] = combined_df[col].apply(lambda x: -1 if x == 0 else x)
        combined_df[col] = combined_df[col].round(3) # for notmalized keypoints to convert them till 3 decimal places 
    

combined_df.to_csv(output_csv, index=False)