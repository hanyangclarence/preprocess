from PIL import Image
import numpy as np
import torch
from torchvision.transforms import ToPILImage, ToTensor
from utils import read_pfm, get_camera_parameters, render_forward_splat
import os
from tqdm import tqdm


if __name__ == '__main__':
    img_path = '../PatchmatchNet_tool/frame_0005/images/00000000.jpg'
    depth_path = '../PatchmatchNet_tool/frame_0005/depth_est/00000000.pfm'
    source_cam_path = '../dataset/neural_3d_video/flame_steak/cams/cam00.txt'
    camera_pose_dir = '../dataset/neural_3d_video/flame_steak/cams'
    result_save_dir = 'rendered_images'
    os.makedirs(result_save_dir, exist_ok=True)

    dp, scale = read_pfm(depth_path)
    dp = dp.squeeze()  # [H, W]
    d_h, d_w = dp.shape
    min_depth, max_depth = np.percentile(dp, 5), np.percentile(dp, 95)
    dp[dp < min_depth] = min_depth
    dp[dp > max_depth] = max_depth
    depth_map = torch.tensor(dp.copy())

    img = Image.open(img_path)
    img = img.resize((d_w, d_h))
    img = ToTensor()(img)[None, ...]  # [1, 3, H, W]
    img = img.permute(0, 2, 3, 1)  # [1, H, W, 3]

    R_source, t_source, (h, w, f) = get_camera_parameters(source_cam_path)

    scale_factor = d_w / w
    w = int(w * scale_factor)
    h = int(h * scale_factor)
    f = f * scale_factor
    K = torch.tensor([
        [f, 0, w/2],
        [0, f, h/2],
        [0, 0, 1]
    ], dtype=torch.float32)

    cam_list = os.listdir(camera_pose_dir)
    cam_list = [s for s in cam_list if s != 'cam00.txt']

    rendered_img_list = []
    for i, cam_id in tqdm(enumerate(cam_list)):
        target_cam_path = os.path.join(camera_pose_dir, cam_id)
        R_target, t_target, _ = get_camera_parameters(target_cam_path)

        warp_feature, warp_disp, warp_mask = render_forward_splat(img, depth_map[None, ...], R_source, t_source, K, R_target, t_target, K)
        feature = ToPILImage()(warp_feature[0])
        disp = ToPILImage()(warp_disp[0])
        mask = ToPILImage()(warp_mask[0])
        feature.save(os.path.join(result_save_dir, f'feature_{cam_id}.png'))
        disp.save(os.path.join(result_save_dir, f'disp_{cam_id}.png'))
        mask.save(os.path.join(result_save_dir, f'mask_{cam_id}.png'))
