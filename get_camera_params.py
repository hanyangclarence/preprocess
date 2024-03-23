import numpy as np
import glob
import torch
import os

# Specify the path to the .npy file
videos = [
    'coffee_martini',
]
for video in videos:
    video_path = f'/mnt/d/local_documentation/ChuangGan_3D/dataset/neural_3d_video/{video}'
    # Get the list of cam*.mp4 files in the directory
    cam_files = glob.glob(f'{video_path}/cam*.mp4')
    # Get the number of cam*.mp4 files
    num_cam_files = len(cam_files)
    print(f"Number of cam* folders in {video_path}: {num_cam_files}")

    cam_poses_file = f'{video_path}/poses_bounds.npy'
    # Load the cam_pose_file
    cam_poses = np.load(cam_poses_file)

    cam_dir = f'{video_path}/cams'
    os.makedirs(cam_dir, exist_ok=True)

    for i, cam_file in enumerate(cam_files):
        # Get the camera folder name
        cam_folder = cam_file.split('/')[-1].split('.')[0]
        cam_pose = cam_poses[i]
        cam2world = np.array(cam_pose[:15]).reshape(3, 5)[:,:4]
        cam2world = np.concatenate([cam2world, np.array([[0, 0, 0, 1]])], axis=0)
        intrinsics = np.array([
            [cam_pose[14], 0.0, cam_pose[14]/2],
            [0.0, cam_pose[14], cam_pose[14]/2],
            [0.0, 0.0, 1.0],
        ])

        depth_ranges = [0.1, 5.0]
        with open(os.path.join(cam_dir, "%08d_cam.txt" % i), "w") as f:
            f.write("extrinsic\n")
            for j in range(4):
                for k in range(4):
                    f.write(str(cam2world[j, k]) + " ")
                f.write("\n")
            f.write("\nintrinsic\n")
            for j in range(3):
                for k in range(3):
                    f.write(str(intrinsics[j, k]) + " ")
                f.write("\n")
            f.write("\n%f %f \n" % (depth_ranges[0], depth_ranges[1]))