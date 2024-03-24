import numpy as np
import torch
from typing import Tuple
import re
from splatting import splatting_function
import cv2


def render_forward_splat(src_imgs, src_depths, R_src, t_src, K_src, R_dst, t_dst, K_dst, get_epipolar_mask=False):
    """3D render the image to the next viewpoint.
    The input transformation matrix should be in nerf space

    Returns:
      warp_feature: the rendered RGB feature map
      warp_disp: the rendered disparity
      warp_mask: the rendered mask
    """

    batch_size = src_imgs.shape[0]

    K_src_inv = K_src.inverse()
    R_dst_inv = R_dst.inverse()

    # convert nerf space to colmap space
    M = torch.eye(3)
    M[1, 1] = -1.0
    M[2, 2] = -1.0

    x = np.arange(src_imgs[0].shape[1])
    y = np.arange(src_imgs[0].shape[0])
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)
    coord = coord.astype(np.float32)
    coord = torch.as_tensor(coord, dtype=K_src.dtype, device=K_src.device)
    coord = coord[None, Ellipsis, None].repeat(batch_size, 1, 1, 1, 1)  # b,h,w,3,1

    depth = src_depths[:, :, :, None, None]

    # from reference to target viewpoint
    pts_3d_ref = depth * K_src_inv[None, None, None, Ellipsis] @ coord
    # get the relative rotation and transition
    R = M @ R_dst_inv @ R_src @ M
    t = M @ R_dst_inv @ (t_src - t_dst)
    pts_3d_tgt = R[None, None, None, Ellipsis] @ pts_3d_ref + t[None, None, None, :, None]

    points = K_dst[None, None, None, Ellipsis] @ pts_3d_tgt
    points = points.squeeze(-1)

    new_z = points[:, :, :, [2]].clone().permute(0, 3, 1, 2)  # b,1,h,w
    points = points / torch.clamp(points[:, :, :, [2]], 1e-8, None)

    src_ims_ = src_imgs.permute(0, 3, 1, 2)
    num_channels = src_ims_.shape[1]

    flow = points - coord.squeeze(-1)
    flow = flow.permute(0, 3, 1, 2)[:, :2, Ellipsis]

    importance = 1. / (new_z)
    importance_min = importance.amin((1, 2, 3), keepdim=True)
    importance_max = importance.amax((1, 2, 3), keepdim=True)
    weights = (importance - importance_min) / (importance_max - importance_min + 1e-6) * 20 - 10

    src_mask_ = torch.ones_like(new_z)
    input_data = torch.cat([src_ims_, (1. / (new_z)), src_mask_], 1)
    output_data = splatting_function('softmax', input_data, flow, weights.detach())

    warp_feature = output_data[:, 0:num_channels, Ellipsis]
    warp_disp = output_data[:, num_channels:num_channels + 1, Ellipsis]
    warp_mask = output_data[:, num_channels + 1:num_channels + 2, Ellipsis]

    if get_epipolar_mask:
        # calculate the source camera position in the target camera space
        cam_3d_ref = M @ torch.tensor([0, 0, 0], dtype=K_src.dtype, device=K_src.device)
        cam_3d_tgt = R_dst_inv @ R_src @ cam_3d_ref + R_dst_inv @ (t_src - t_dst)
        cam_3d_tgt = K_dst @ M @ cam_3d_tgt
        cam_2d_tgt = cam_3d_tgt / torch.clamp(cam_3d_tgt[2], 1e-8, None)

        # get the epipolar mask
        epip_mask = np.zeros((warp_mask.shape[-2], warp_mask.shape[-1], 1), dtype=np.uint8)  # h, w, 1

        start_coord = (int(cam_2d_tgt[0]), int(cam_2d_tgt[1]))  # (2,)
        color = 255
        thickness = 1
        for h in range(src_imgs.shape[1]):
            for w in range(src_imgs.shape[2]):
                mapped_coord = (int(points[0, h, w, 0]), int(points[0, h, w, 1]))  # (2,)
                # between start coord and mapped coord are marked as 1, other are 0
                try:
                    cv2.line(epip_mask, start_coord, mapped_coord, color, thickness)
                except Exception as e:
                    print(e)
                    print(start_coord, mapped_coord)

        return warp_feature, warp_disp, warp_mask, epip_mask
    else:
        return warp_feature, warp_disp, warp_mask


def read_binary_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def quaterion_to_rotation(q):
    w, x, y, z = q
    R = torch.tensor([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    return R


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def read_pfm(filename: str) -> Tuple[np.ndarray, float]:
    """Read a depth map from a .pfm file

    Args:
        filename: .pfm file path string

    Returns:
        data: array of shape (H, W, C) representing loaded depth map
        scale: float to recover actual depth map pixel values
    """
    file = open(filename, "rb")  # treat as binary and read-only

    header = file.readline().decode("utf-8").rstrip()
    if header == "PF":
        color = True
    elif header == "Pf": # depth is Pf
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width, 1)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def get_camera_parameters_from_opengl(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        c2w = torch.tensor([
            [float(x.strip()) for x in lines[1].strip().split(',') if x != ''],
            [float(x.strip()) for x in lines[2].strip().split(',') if x != ''],
            [float(x.strip()) for x in lines[3].strip().split(',') if x != '']
        ])  # 3x4
        R = c2w[:, :3]
        t = c2w[:, -1]

        f = float(lines[7].strip().split(', ')[0])
        half_w = float(lines[7].split(', ')[-2])
        half_h = float(lines[8].split(', ')[-2])
        hwf = torch.tensor([half_h * 2, half_w * 2, f])  # 3

        return R, t, hwf


def get_camera_parameters_from_colmap(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        c2w = torch.tensor([
            [float(x.strip()) for x in lines[1].strip().split(' ')],
            [float(x.strip()) for x in lines[2].strip().split(' ')],
            [float(x.strip()) for x in lines[3].strip().split(' ')]
        ])  # 3x4
        R = c2w[:, :3]
        t = c2w[:, -1]

        # transform camera space to nerf space
        R = torch.tensor([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ], dtype=torch.float32) @ R
        t = torch.tensor([0., -1., -1.]) * t

        f = float(lines[7].strip().split(' ')[0])
        half_w = float(lines[7].strip().split(' ')[-1])
        half_h = float(lines[8].strip().split(' ')[-1])
        hwf = torch.tensor([half_h * 2, half_w * 2, f])  # 3

        return R, t, hwf
