import cv2
import os

def extract_frames(video_path, save_folder, every_n_frames=10):
    # Create the save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    video = cv2.VideoCapture(video_path)
    frame_number = 0
    saved = 0

    while True:
        success, frame = video.read()
        if not success:
            break

        if frame_number % every_n_frames == 0:
            filename = f"{save_folder}/frame_{saved}.jpg"
            cv2.imwrite(filename, frame)
            saved += 1

        frame_number += 1

    video.release()
    print(f"Done! Saved {saved} frames from {video_path}")