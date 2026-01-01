import pandas as pd
import os 

csv1 = "C:/wajahat/hand_in_pocket/dataset/training3/old_hp_combine_pos.csv"
csv2 = "C:/wajahat/hand_in_pocket/dataset/training3/fp_combine_pos.csv"
csv3 = "C:/wajahat/hand_in_pocket/dataset/training3/moiz_fp_combine_pos.csv"
csv4 = "C:/wajahat/hand_in_pocket/dataset/training3/tp_combine_pos.csv"
csv5 = "C:/wajahat/hand_in_pocket/dataset/training3/fn_combine_pos.csv"
csv6 = "C:/wajahat/hand_in_pocket/dataset/training3/mudassir_hp_combine_pos.csv"
csv7 = "C:/wajahat/hand_in_pocket/dataset/training3/fp_s1_w1_pos.csv"
csv8 = "C:/wajahat/hand_in_pocket/dataset/training3/fp_s1_w2_pos.csv"
csv9 = "C:/wajahat/hand_in_pocket/dataset/training3/fp_s2_w1_pos.csv"
csv10 = "C:/wajahat/hand_in_pocket/dataset/training3/fp_s2_w2_pos.csv"
csv11 = "C:/wajahat/hand_in_pocket/dataset/training3/missing_s1_pos.csv"
csv12 = "C:/wajahat/hand_in_pocket/dataset/training3/missing_s2_pos.csv"
csv13 = "C:/wajahat/hand_in_pocket/dataset/training3/tp_s1_w1_pos.csv"
csv14 = "C:/wajahat/hand_in_pocket/dataset/training3/tp_s1_w2_pos.csv"
csv15 = "C:/wajahat/hand_in_pocket/dataset/training3/tp_s2_w1_pos.csv"
csv16 = "C:/wajahat/hand_in_pocket/dataset/training3/tp_s2_w2_pos.csv"
# csv17 =
# csv18 =
# csv19 =
# csv20 =

# csv2 = "C:/wajahat/hand_in_pocket/dataset/new_dataset/new_combined_sorted_balanced2.csv"
output_csv = "C:/wajahat/hand_in_pocket/dataset/training3/itteration3_temp_norm_balanced.csv"

df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)
df3 = pd.read_csv(csv3)
df4 = pd.read_csv(csv4)
df5 = pd.read_csv(csv5)
df6 = pd.read_csv(csv6)
df7 = pd.read_csv(csv7)
df8 = pd.read_csv(csv8)
df9 = pd.read_csv(csv9)
df10 = pd.read_csv(csv10)
df11 = pd.read_csv(csv11)
df12 = pd.read_csv(csv12)
df13 = pd.read_csv(csv13)
df14 = pd.read_csv(csv14)
df15 = pd.read_csv(csv15)
df16 = pd.read_csv(csv16)
# df17 = pd.read_csv(csv17)
# df18 = pd.read_csv(csv18)
# df19 = pd.read_csv(csv19)
# df20 = pd.read_csv(csv20)
# df = "C:/Users/LT/Downloads/new_combined_temp_balanced.csv"
# df = "C:/wajahat/hand_in_pocket/dataset/training2/new_combined_temp_balanced.csv"
# combined_df = pd.read_csv(df)
# df = "C:/wajahat/hand_in_pocket/dataset/training2/new_combined_temp_balanced_norm_without_seq2.csv"
combined_df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16], ignore_index=True)

new_columns_order = ["camera",	"video",	"frame",	"desk_no",	"kp_0_x_t0",	"kp_0_x_t1",	"kp_0_x_t2",	"kp_0_x_t3",	"kp_0_x_t4",	"kp_0_y_t0",	"kp_0_y_t1",
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

filtered_columns = [col for col in new_columns_order if col in combined_df.columns]
combined_df = combined_df[filtered_columns]
# combined_df = combined_df.astype(int)

for col in combined_df.columns:
    if 'x' in col.lower():
        combined_df[col] = combined_df[col].astype(int) #for not normalized keypoint to convert them into numbers like 271,542
        combined_df[col] = pd.to_numeric((combined_df[col] / 1280), errors='coerce')
        combined_df[col] = combined_df[col].apply(lambda x: -1 if x == 0 else x)
        combined_df[col] = combined_df[col].round(3) # for notmalized keypoints to convert them till 3 decimal places 
    elif 'y' in col.lower():
        combined_df[col] = combined_df[col].astype(int) #for not normalized keypoint to convert them into numbers like 271,542
        combined_df[col] = pd.to_numeric((combined_df[col] / 720), errors='coerce')
        combined_df[col] = combined_df[col].apply(lambda x: -1 if x == 0 else x)
        combined_df[col] = combined_df[col].round(3) # for notmalized keypoints to convert them till 3 decimal places 
    

combined_df.to_csv(output_csv, index=False)
# combined_df.to_csv(df, index=False)
print(f"✅ Combined CSV saved to: {output_csv}")