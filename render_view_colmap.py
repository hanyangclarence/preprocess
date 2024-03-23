import numpy as np
from PIL import Image
from os.path import join as pjoin
from torchvision.transforms import ToPILImage, ToTensor
from utils import read_binary_array, quaterion_to_rotation, render_forward_splat
import torch

img_name = 'cam00.jpg'
depth_name = 'cam00.jpg.geometric.bin'

img_folder = './frame_0000'
depth_folder = 'colmap_res/dense/0/stereo/depth_maps'
camera_param_path = 'colmap_res/dense/0/sparse/cameras.txt'
image_param_path = 'colmap_res/dense/0/sparse/images.txt'

# read depth map
depth_map = read_binary_array(pjoin(depth_folder, depth_name))  # (H, W)
min_percentile = 5
max_percentile = 95
min_depth = np.percentile(depth_map, min_percentile)
max_depth = np.percentile(depth_map, max_percentile)
depth_map[depth_map < min_depth] = min_depth
depth_map[depth_map > max_depth] = max_depth
depth_map = torch.tensor(depth_map)

# read image
img = Image.open(pjoin(img_folder, img_name))

# match the size of depth map
img = img.resize((depth_map.shape[1], depth_map.shape[0]))
img = ToTensor()(img)[None]  # (1, 3, H, W)
img = img.permute(0, 2, 3, 1)  # (1, H, W, 3)

# read camera extrinsic parameters
extrinsic_line = None
with open(image_param_path, 'r') as f:
    for line in f:
        if img_name in line:
            extrinsic_line = line
            break
extrinsic_line = extrinsic_line.strip().split(' ')
camera_id = extrinsic_line[-2]
quaternion = torch.tensor([float(x) for x in extrinsic_line[1:5]])
R_src = quaterion_to_rotation(quaternion)
t_src = torch.tensor([float(x) for x in extrinsic_line[5:8]])

# read camera intrinsic parameters
intrinsic_line = None
with open(camera_param_path, 'r') as f:
    for line in f:
        if line.split(' ')[0] == camera_id:
            intrinsic_line = line
            break
intrinsic_line = intrinsic_line.strip().split(' ')
W, H, focal_length = int(intrinsic_line[2]), int(intrinsic_line[3]), float(intrinsic_line[4])
K = torch.tensor([[focal_length, 0, 0.5*W], [0, focal_length, 0.5*H], [0, 0, 1]])

# create new transformation matrix
R_tgt = torch.eye(3)
t_tgt = t_src.clone() - torch.tensor([-10.0, 10.0, 10.0])

# render view
warp_feature, warp_disp, warp_mask = render_forward_splat(img, depth_map[None], R_src, t_src, K, R_tgt, t_tgt, K)
feature = ToPILImage()(warp_feature[0])
disp = ToPILImage()(warp_disp[0])
mask = ToPILImage()(warp_mask[0])
feature.save('feature.png')
disp.save('disp.png')
mask.save('mask.png')


