from PIL import Image
import torch
from torchvision.transforms import ToPILImage, ToTensor
from utils import read_pfm, get_camera_parameters_from_colmap, render_forward_splat
import os
from os.path import join as pjoin
from tqdm import tqdm

if __name__ == '__main__':
    folder_path = '../PatchmatchNet_tool/frame0001'
    reference_camera_id: str = '00000000'

    ref_img_path = pjoin(folder_path, 'images', f'{reference_camera_id}.jpg')
    ref_depth_path = pjoin(folder_path, 'depth_est', f'{reference_camera_id}.pfm')
    ref_cam_path = pjoin(folder_path, 'cams', f'{reference_camera_id}_cam.txt')
    camera_pose_dir = pjoin(folder_path, 'cams')
    result_save_dir = 'rendered_results'
    os.makedirs(result_save_dir, exist_ok=True)

    dp, scale = read_pfm(ref_depth_path)
    dp = dp.squeeze()  # [H, W]
    d_h, d_w = dp.shape
    depth_map = torch.tensor(dp.copy())

    img = Image.open(ref_img_path)
    img = img.resize((d_w, d_h))
    img = ToTensor()(img)[None, ...]  # [1, 3, H, W]
    img = img.permute(0, 2, 3, 1)  # [1, H, W, 3]

    R_source, t_source, (h, w, f) = get_camera_parameters_from_colmap(ref_cam_path)

    scale_factor = d_w / w
    w = int(w * scale_factor)
    h = int(h * scale_factor)
    f = f * scale_factor
    K = torch.tensor([
        [f, 0, w / 2],
        [0, f, h / 2],
        [0, 0, 1]
    ], dtype=torch.float32)

    cam_list = os.listdir(camera_pose_dir)
    cam_list = [s for s in cam_list if s != f'{reference_camera_id}_cam.txt']

    for i, cam_id in tqdm(enumerate(cam_list)):
        target_cam_path = pjoin(camera_pose_dir, cam_id)
        R_target, t_target, _ = get_camera_parameters_from_colmap(target_cam_path)

        warp_feature, warp_disp, warp_mask = render_forward_splat(img, depth_map[None, ...], R_source, t_source, K, R_target, t_target, K)
        feature = ToPILImage()(warp_feature[0])
        disp = ToPILImage()(warp_disp[0])
        mask = ToPILImage()(warp_mask[0])
        feature.save(pjoin(result_save_dir, f'feature_{cam_id}.jpg'))
        disp.save(pjoin(result_save_dir, f'disp_{cam_id}.jpg'))
        mask.save(pjoin(result_save_dir, f'mask_{cam_id}.jpg'))
