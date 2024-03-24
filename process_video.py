import cv2
import os


def extract_frames(video_path, save_folder):
    video_name = video_path.split('/')[-1].split('.')[0]
    frame_name = f'{video_name}.jpg'

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # get total frame number
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        folder_name = os.path.join(save_folder, f'frame_{frame_count:04d}')
        os.makedirs(folder_name, exist_ok=True)
        frame_path = os.path.join(folder_name, frame_name)
        cv2.imwrite(frame_path, frame)
        frame_count += 1

        print(f'{video_name}, Processed {frame_count}/{total_frames} frames.')

    cap.release()



def process_videos(folder_path, save_folder):
    # List all files in the given folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):  # Check if the file is a video
            video_path = os.path.join(folder_path, filename)

            # Create a folder name based on the video file name
            extract_frames(video_path, save_folder)


video_folder = '/mnt/d/local_documentation/ChuangGan_3D/dataset/neural_3d_video/flame_steak'
save_folder = '/mnt/d/local_documentation/ChuangGan_3D/dataset/neural_3d_video/flame_steak/video_frames/'
process_videos(video_folder, save_folder)