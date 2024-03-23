from PIL import Image
import numpy as np
from splatting import splatting_function
import torch
from torchvision.transforms import ToPILImage, ToTensor

def render_forward_splat(src_imgs, src_depths, R_src, t_src, K_src, R_dst, t_dst, K_dst):
    """3D render the image to the next viewpoint.

    Returns:
      warp_feature: the rendered RGB feature map
      warp_disp: the rendered disparity
      warp_mask: the rendered mask
    """
    batch_size = src_imgs.shape[0]

    K_src_inv = K_src.inverse()
    R_dst_inv = R_dst.inverse()

    x = np.arange(src_imgs[0].shape[1])
    y = np.arange(src_imgs[0].shape[0])
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)
    coord = coord.astype(np.float32)
    coord = torch.as_tensor(coord, dtype=K_src.dtype, device=K_src.device)
    coord = coord[None, Ellipsis, None].repeat(batch_size, 1, 1, 1, 1)

    depth = src_depths[:, :, :, None, None]

    # from reference to target viewpoint
    pts_3d_ref = depth * K_src_inv[None, None, None, Ellipsis] @ coord
    pts_3d_tgt = R_dst_inv[None, None, None, Ellipsis] @ R_src[None, None, None, Ellipsis] @ pts_3d_ref + \
                    R_dst_inv[None, None, None, Ellipsis] @ (t_src[None, None, None, :, None] - t_dst[None, None, None, :, None])
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
    weights = (importance - importance_min) / (importance_max - importance_min +
                                               1e-6) * 20 - 10
    src_mask_ = torch.ones_like(new_z)

    input_data = torch.cat([src_ims_, (1. / (new_z)), src_mask_], 1)

    output_data = splatting_function('softmax', input_data, flow,
                                     weights.detach())

    warp_feature = output_data[:, 0:num_channels, Ellipsis]
    warp_disp = output_data[:, num_channels:num_channels + 1, Ellipsis]
    warp_mask = output_data[:, num_channels + 1:num_channels + 2, Ellipsis]

    return warp_feature, warp_disp, warp_mask


def get_intrinsics(H,W):
    """
    Intrinsics for a pinhole camera model.
    Assume fov of 55 degrees and central principal point.
    """
    f = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H
    return torch.tensor([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]], dtype=torch.float32)


img = Image.open('cat.jpg')
img = ToTensor()(img)[None, ...]  # [1, 3, H, W]
img = img.permute(0, 2, 3, 1)  # [1, H, W, 3]
depth_map = torch.tensor(np.load('cat_depth_fp32.npy'))
depth_map = 1.0 + depth_map * 9.0

K = get_intrinsics(depth_map.shape[0], depth_map.shape[1])
R_source = torch.eye(3)
t_source = torch.tensor([0.0, 0.0, 0.0])

R_target = torch.eye(3)
t_target = torch.tensor([0.3, -0.5, -1.0])

warp_feature, warp_disp, warp_mask = render_forward_splat(img, depth_map[None, ...], R_source, t_source, K, R_target, t_target, K)
feature = ToPILImage()(warp_feature[0])
disp = ToPILImage()(warp_disp[0])
mask = ToPILImage()(warp_mask[0])
feature.save('feature.png')
disp.save('disp.png')
mask.save('mask.png')


