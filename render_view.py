import numpy as np
from PIL import Image


def get_intrinsics(H,W):
    """
    Intrinsics for a pinhole camera model.
    Assume fov of 55 degrees and central principal point.
    """
    f = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]])


def depth_to_points(depth, R=None, t=None):

    K = get_intrinsics(depth.shape[1], depth.shape[2])
    Kinv = np.linalg.inv(K)
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    # M converts from your coordinate to PyTorch3D's coordinate system
    M = np.eye(3)
    M[0, 0] = -1.0
    M[1, 1] = -1.0

    height, width = depth.shape[1:3]

    x = np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
    coord = coord.astype(np.float32)
    # coord = torch.as_tensor(coord, dtype=torch.float32, device=device)
    coord = coord[None]  # bs, h, w, 3

    D = depth[:, :, :, None, None]
    # print(D.shape, Kinv[None, None, None, ...].shape, coord[:, :, :, :, None].shape )
    pts3D_1 = D * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
    # pts3D_1 live in your coordinate system. Convert them to Py3D's
    pts3D_1 = M[None, None, None, ...] @ pts3D_1
    # from reference to targe tviewpoint
    pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]
    # pts3D_2 = pts3D_1
    # depth_2 = pts3D_2[:, :, :, 2, :]  # b,1,h,w
    return pts3D_2[:, :, :, :3, 0][0]


def map_points_to_screen(points, K, R, t):
    # points: (H, W, 3)
    # K: (3, 3)
    # R: (3, 3)
    # t: (3,)
    # Returns: (H, W, 2)
    H, W, _ = points.shape
    w2c = np.linalg.inv(R)

    # transform points to camera view
    pts3D_1 = w2c[None, None, ...] @ (points[:, :, :, None] - t[None, None, :, None])  # (H, W, 3, 1)

    # change the coordinate back
    M = np.eye(3)
    M[0, 0] = -1.0
    M[1, 1] = -1.0
    pts3D_1 = M[None, None, ...] @ pts3D_1  # (H, W, 3, 1)

    D = pts3D_1[:, :, 2, :]  # (H, W, 1)
    pts3D_2 = K[None, None, ...] @ (pts3D_1 / D[..., None])  # (H, W, 3, 1)
    pts2D = pts3D_2[:, :, :2, 0]  # (H, W, 2)

    # change the sequence of the last coordinate
    pts2D = pts2D[..., [1, 0]]
    return pts2D, D.squeeze()


img = Image.open('cat.jpg')
img = np.array(img)
depth_map = np.load('cat_depth_fp32.npy')

depth_map = 1.0 + depth_map * 10.0

R_source = np.eye(3)
t_source = np.asarray([0.0, 0.0, -2.0])

new_K = get_intrinsics(depth_map.shape[0], depth_map.shape[1])
new_R = np.eye(3)
new_t = np.asarray([-0.5, 0.5, -2.0])


pts3d = depth_to_points(depth_map[None], R_source, t_source)

mapped_2d, mapped_depth = map_points_to_screen(pts3d, new_K, new_R, new_t)

rendered = np.zeros_like(img)
depth_recorder = np.zeros((img.shape[0], img.shape[1])) + np.inf
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        i_mapped, j_mapped = mapped_2d[i, j]
        i_mapped = int(i_mapped)
        j_mapped = int(j_mapped)
        depth_mapped = mapped_depth[i, j]
        if 0 <= i_mapped < rendered.shape[0] and 0 <= j_mapped < rendered.shape[1]:
            if depth_mapped < depth_recorder[i_mapped, j_mapped]:
                rendered[i_mapped, j_mapped] = img[i, j]
                depth_recorder[i_mapped, j_mapped] = depth_mapped

print('here')
Image.fromarray(rendered).show()






