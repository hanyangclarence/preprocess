import numpy as np
import glob
import torch
import os

# Specify the path to the .npy file
videos = [
    'flame_steak',
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
        cam2world = np.array(cam_pose[:15]).reshape(3, 5)

        # correct the rotation matrix order: llff to nerf
        cam2world = np.concatenate([cam2world[:, 1:2], -cam2world[:, 0:1], cam2world[:, 2:3], cam2world[:, 3:]], axis=1)

        extrinsic_mat = cam2world[:3, :4]
        extrinsic_mat = np.concatenate([extrinsic_mat, np.array([[0, 0, 0, 1]])], axis=0)  # (4, 4)

        h, w, f = cam2world[:, -1]
        intrinsic_mat = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])

        depth_ranges = cam_pose[15:17]

        with open(os.path.join(cam_dir, f"{cam_folder}.txt"), "w") as f:
            f.write("extrinsic\n")
            for j in range(4):
                for k in range(4):
                    f.write(str(extrinsic_mat[j, k]) + ", ")
                f.write("\n")
            f.write("\nintrinsic\n")
            for j in range(3):
                for k in range(3):
                    f.write(str(intrinsic_mat[j, k]) + ", ")
                f.write("\n")
            f.write("\n%f %f \n" % (depth_ranges[0], depth_ranges[1]))