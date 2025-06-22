""" This script processes videos to extract frames where a hand is either in a pocket or not. 
Takes user input to label frames and saves them in respective folders."""

import os
import cv2

# Input and output folders
video_folder = "F:/Wajahat/hand_in_pocket/traiining"  # Folder containing input videos
output_folder_hip = "F:/Wajahat/hand_in_pocket/frames/hp"
output_folder_no_hip = "F:/Wajahat/hand_in_pocket/frames/no_hp"

# Create output directories if they don’t exist
os.makedirs(output_folder_hip, exist_ok=True)
os.makedirs(output_folder_no_hip, exist_ok=True)

# Get list of all video files
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

# Global frame counter for unique naming
global_frame_counter = 0

for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open video: {video_file}")
        continue

    print(f"\nProcessing video: {video_file}")
    video_name = os.path.splitext(video_file)[0]

    ret, first_frame = cap.read()
    if not ret:
        print(f"Failed to read first frame of video: {video_file}")
        cap.release()
        continue

    first_frame = cv2.resize(first_frame, (1280, 720))  # Resize for consistency

    cv2.imshow("To skip press 's' otherwise press enter", first_frame)
    key = cv2.waitKey(0)

    if key == ord('s') or key == 27:  # 's' or ESC to skip video
        print(f"Skipping video: {video_file}")
        cap.release()
        continue

    # else: 
    #     cv2.destroyWindow(first_frame)

    frame_history = [] 
    frame_index = 0

    while True:
        if frame_index < len(frame_history):
            # Return to previous frame
            frame, _, _ = frame_history[frame_index]
        else:
            ret, frame = cap.read()
            if not ret:
                break
            frame_history.append((frame.copy(), None, None))
        
        frame = cv2.resize(frame, (1280, 720))
        # Display the frame
        cv2.imshow("Frame (SPACE = HIP, ENTER = NO HIP, R = Undo, Q = Quit Video)", frame)
        print("label now")
        key = cv2.waitKey(0)

        if key == 32:  # SPACEBAR = hand_in_pocket
            # filename = f"f{global_frame_counter:04d}_{video_name}.jpg"
            filename = f"{video_name}_f{frame_index:04d}.jpg"
            filepath = os.path.join(output_folder_hip, filename)
            cv2.imwrite(filepath, frame)
            frame_history[frame_index] = (frame.copy(), filepath, "hip")
            print(f"Saved: {filename} -> hand_in_pocket")
            global_frame_counter += 1
            frame_index += 1

        elif key == 13:  # ENTER = no_hand_in_pocket
            # filename = f"f{global_frame_counter:04d}_{video_name}.jpg"
            filename = f"{video_name}_f{frame_index:04d}.jpg"
            filepath = os.path.join(output_folder_no_hip, filename)
            cv2.imwrite(filepath, frame)
            frame_history[frame_index] = (frame.copy(), filepath, "no_hip")
            print(f"Saved: {filename} -> no_hand_in_pocket")
            global_frame_counter += 1
            frame_index += 1

        elif key == ord('r') or key == ord('R'):  # Undo previous frame
            if frame_index > 0:
                frame_index -= 1
                _, last_path, last_label = frame_history[frame_index]

                if last_path and os.path.exists(last_path):
                    os.remove(last_path)
                    print(f"Removed previous saved frame: {last_path}")
                    global_frame_counter -= 1
                else:
                    print("Nothing to undo.")
            else:
                print("Already at the first frame.")

        elif key == ord('q') or key == 27:  # Q or ESC = quit current video
            print(f"Skipped rest of video: {video_file}")
            break

    cap.release()

cv2.destroyAllWindows()
print("All videos processed.")
