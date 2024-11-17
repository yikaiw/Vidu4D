from turtle import forward
import numpy as np
import torch
import random


# Reworked so this matches gluPerspective / glm::perspective, using fovy
def perspective(fovx=0.7854, aspect=1.0, n=0.1, f=1000.0, device=None):
    # y = np.tan(fovy / 2)
    x = np.tan(fovx / 2)
    return torch.tensor([[1/x,         0,            0,              0],
                         [  0, -aspect/x,            0,              0],
                         [  0,         0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                         [  0,         0,           -1,              0]], dtype=torch.float32, device=device)


def translate(x, y, z, device=None):
    return torch.tensor([[1, 0, 0, x],
                         [0, 1, 0, y],
                         [0, 0, 1, z],
                         [0, 0, 0, 1]], dtype=torch.float32, device=device)


def rotate_x(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[1, 0,  0, 0],
                         [0, c, -s, 0],
                         [0, s,  c, 0],
                         [0, 0,  0, 1]], dtype=torch.float32, device=device)


def rotate_y(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[ c, 0, s, 0],
                         [ 0, 1, 0, 0],
                         [-s, 0, c, 0],
                         [ 0, 0, 0, 1]], dtype=torch.float32, device=device)


def rotate_z(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[c, -s, 0, 0],
                         [s,  c, 0, 0],
                         [0,  0, 1, 0],
                         [0,  0, 0, 1]], dtype=torch.float32, device=device)

@torch.no_grad()
def batch_random_rotation_translation(b, t, device=None):
    m = np.random.normal(size=[b, 3, 3])
    m[:, 1] = np.cross(m[:, 0], m[:, 2])
    m[:, 2] = np.cross(m[:, 0], m[:, 1])
    m = m / np.linalg.norm(m, axis=2, keepdims=True)
    m = np.pad(m, [[0, 0], [0, 1], [0, 1]], mode='constant')
    m[:, 3, 3] = 1.0
    m[:, :3, 3] = np.random.uniform(-t, t, size=[b, 3])
    return torch.tensor(m, dtype=torch.float32, device=device)

@torch.no_grad()
def random_rotation_translation(t, device=None):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.random.uniform(-t, t, size=[3])
    return torch.tensor(m, dtype=torch.float32, device=device)


@torch.no_grad()
def random_rotation(device=None):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.array([0,0,0]).astype(np.float32)
    return torch.tensor(m, dtype=torch.float32, device=device)


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)


def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN


def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)


def lr_schedule(iter, warmup_iter, scheduler_decay):
    if iter < warmup_iter:
        return iter / warmup_iter
    return max(0.0, 10 ** (
            -(iter - warmup_iter) * scheduler_decay)) 


def trans_depth(depth):
    depth = depth[0].detach().cpu().numpy()
    valid = depth > 0
    depth[valid] -= depth[valid].min()
    depth[valid] = ((depth[valid] / depth[valid].max()) * 255)
    return depth.astype('uint8')


def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None):
    assert isinstance(input, torch.Tensor)
    if posinf is None:
        posinf = torch.finfo(input.dtype).max
    if neginf is None:
        neginf = torch.finfo(input.dtype).min
    assert nan == 0
    return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)


def load_item(filepath):
    with open(filepath, 'r') as f:
        items = [name.strip() for name in f.readlines()]
    return set(items)

def load_prompt(filepath):
    uuid2prompt = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            list_line = line.split(',')
            uuid2prompt[list_line[0]] = ','.join(list_line[1:]).strip()
    return uuid2prompt

def resize_and_center_image(image_tensor, scale=0.95, c = 0, shift = 0, rgb=False, aug_shift = 0):
    if scale == 1:
        return image_tensor
    B, C, H, W = image_tensor.shape
    new_H, new_W = int(H * scale), int(W * scale)
    resized_image = torch.nn.functional.interpolate(image_tensor, size=(new_H, new_W), mode='bilinear', align_corners=False).squeeze(0)
    background = torch.zeros_like(image_tensor) + c
    start_y, start_x = (H - new_H) // 2, (W - new_W) // 2
    if shift == 0:
        background[:, :, start_y:start_y + new_H, start_x:start_x + new_W] = resized_image
    else:
        for i in range(B):
            randx = random.randint(-shift, shift)
            randy = random.randint(-shift, shift)   
            if rgb == True:
                if i == 0 or i==2 or i==4:
                    randx = 0
                    randy = 0 
            background[i, :, start_y+randy:start_y + new_H+randy, start_x+randx:start_x + new_W+randx] = resized_image[i]
    if aug_shift == 0:
        return background  
    for i in range(B):
        for j in range(C):
            background[i, j, :, :] += (random.random() - 0.5)*2 * aug_shift / 255
    return background 
                               
def get_tri(triview_color, dim = 1, blender=True, c = 0, scale=0.95, shift = 0, fix = False, rgb=False, aug_shift = 0):
    # triview_color: [6,C,H,W]
    # rgb is useful when shift is not 0
    triview_color = resize_and_center_image(triview_color, scale=scale, c = c, shift=shift,rgb=rgb, aug_shift = aug_shift)
    if blender is False:
        triview_color0 = torch.rot90(triview_color[0],k=2,dims=[1,2])
        triview_color1 = torch.rot90(triview_color[4],k=1,dims=[1,2]).flip(2).flip(1)
        triview_color2 = torch.rot90(triview_color[5],k=1,dims=[1,2]).flip(2)
        triview_color3 = torch.rot90(triview_color[3],k=2,dims=[1,2]).flip(2)
        triview_color4 = torch.rot90(triview_color[1],k=3,dims=[1,2]).flip(1)
        triview_color5 = torch.rot90(triview_color[2],k=3,dims=[1,2]).flip(1).flip(2)
    else:
        triview_color0 = torch.rot90(triview_color[2],k=2,dims=[1,2])
        triview_color1 = torch.rot90(triview_color[4],k=0,dims=[1,2]).flip(2).flip(1)
        triview_color2 = torch.rot90(torch.rot90(triview_color[0],k=3,dims=[1,2]).flip(2), k=2,dims=[1,2])
        triview_color3 = torch.rot90(torch.rot90(triview_color[5],k=2,dims=[1,2]).flip(2), k=2,dims=[1,2])
        triview_color4 = torch.rot90(triview_color[1],k=2,dims=[1,2]).flip(1).flip(1).flip(2)
        triview_color5 = torch.rot90(triview_color[3],k=1,dims=[1,2]).flip(1).flip(2)
        if fix == True:
            triview_color0[1] = triview_color0[1] * 0
            triview_color0[2] = triview_color0[2] * 0
            triview_color3[1] = triview_color3[1] * 0
            triview_color3[2] = triview_color3[2] * 0

            triview_color1[0] = triview_color1[0] * 0
            triview_color1[1] = triview_color1[1] * 0
            triview_color4[0] = triview_color4[0] * 0
            triview_color4[1] = triview_color4[1] * 0

            triview_color2[0] = triview_color2[0] * 0
            triview_color2[2] = triview_color2[2] * 0
            triview_color5[0] = triview_color5[0] * 0
            triview_color5[2] = triview_color5[2] * 0
    color_tensor1_gt = torch.cat((triview_color0, triview_color1, triview_color2), dim=2)
    color_tensor2_gt = torch.cat((triview_color3, triview_color4, triview_color5), dim=2)
    color_tensor_gt = torch.cat((color_tensor1_gt, color_tensor2_gt), dim = dim)
    return color_tensor_gt




import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr


_FG_LUT = None


def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(
        attr.contiguous(), rast, attr_idx, rast_db=rast_db,
        diff_attrs=None if rast_db is None else 'all')


def xfm_points(points, matrix, use_python=True):
    '''Transform points.
    Args:
        points: Tensor containing 3D points with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)
    Returns:
        Transformed points in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''
    out = torch.matmul(torch.nn.functional.pad(points, pad=(0, 1), mode='constant', value=1.0), torch.transpose(matrix, 1, 2))
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_points contains inf or NaN"
    return out


class NeuralRender():
    def __init__(self, device='cuda'):
        super(NeuralRender, self).__init__()
        self.device = device
        self.ctx = None
        self.projection_mtx = None
        self.znear = 0.1
        self.zfar = 1000.0
        
    
    def forward(self):
        pass

    def render_mesh(
            self,
            mesh_v_pos_bxnx3,
            mesh_t_pos_idx_fx3,
            camera_mv_bx4x4,  # 外参
            mesh_v_feat_bxnxd, # 颜色
            resolution=256,
            spp=1,
            device='cuda',
            hierarchical_mask=False,
            Kinv=None, H=None, W=None,
    ):
        assert not hierarchical_mask

        if Kinv is not None:
            left = Kinv[0, 2] 
            right = Kinv[0, 2] + Kinv[0, 0] * W
            bottom = Kinv[1, 2]
            top = Kinv[1, 2] + Kinv[1, 1] * H
            P = torch.zeros(4, 4).to(self.data_device)

            znear=self.znear
            zfar=self.zfar
            z_sign = 1.0

            left = left * znear
            right = right * znear
            top = top * znear
            bottom = bottom * znear

            top_inv = -top
            bottom_inv = -bottom
            bottom = top_inv
            top = bottom_inv

            left_inv = -left
            right_inv = -right
            left = right_inv
            right = left_inv


            P[0, 0] = 2.0 * znear / (right - left)
            P[1, 1] = 2.0 * znear / (top - bottom)
            P[0, 2] = (right + left) / (right - left)
            P[1, 2] = (top + bottom) / (top - bottom)
            P[3, 2] = z_sign
            P[2, 2] = z_sign * (znear + zfar) / (zfar - znear)
            P[2, 3] = - 2 * (zfar * znear) / (zfar - znear)

            proj_mtx = P.transpose(0,1).to(self.data_device)
        
        else:
            fovy = 49.0
            focal = np.tan(fovy / 180.0 * np.pi * 0.5)
            proj_mtx = torch.from_numpy(projection(x=focal, f=1000.0, n=1.0, near_plane=0.1)).to(self.device).unsqueeze(dim=0)
        
        if self.ctx is None:
            self.ctx = dr.RasterizeGLContext(device=self.device)

        mtx_in = torch.tensor(camera_mv_bx4x4, dtype=torch.float32, device=device) if not torch.is_tensor(camera_mv_bx4x4) else camera_mv_bx4x4
        v_pos = xfm_points(mesh_v_pos_bxnx3, mtx_in)  # Rotate it to camera coordinates
        # v_pos_clip = self.camera.project(v_pos)  # Projection in the camera
        v_pos_clip = torch.matmul(
                    v_pos,
                    torch.transpose(proj_mtx, 1, 2))

        # Render the image,
        # Here we only return the feature (3D location) at each pixel, which will be used as the input for neural render
        num_layers = 1
        mask_pyramid = None
        assert mesh_t_pos_idx_fx3.shape[0] > 0  # Make sure we have shapes
        mesh_v_feat_bxnxd = torch.cat([mesh_v_feat_bxnxd, v_pos], dim=-1)  # Concatenate the pos  compute the supervision

        with dr.DepthPeeler(self.ctx, v_pos_clip, mesh_t_pos_idx_fx3, [resolution * spp, resolution * spp]) as peeler:
            for _ in range(num_layers):
                rast, db = peeler.rasterize_next_layer()
                gb_feat, _ = interpolate(mesh_v_feat_bxnxd, rast, mesh_t_pos_idx_fx3)

        hard_mask = torch.clamp(rast[..., -1:], 0, 1)
        antialias_mask = dr.antialias(
            hard_mask.clone().contiguous(), rast, v_pos_clip,
            mesh_t_pos_idx_fx3)

        depth = gb_feat[..., -2:-1]
        ori_mesh_feature = gb_feat[..., :-4]
        return ori_mesh_feature, antialias_mask, hard_mask, rast, v_pos_clip, mask_pyramid, depth


# ---- utils -----
def projection(x=0.1, n=1.0, f=50.0, near_plane=None):
    if near_plane is None:
        near_plane = n
    return np.array(
        [[n / x, 0, 0, 0],
         [0, n / -x, 0, 0],
         [0, 0, -(f + near_plane) / (f - near_plane), -(2 * f * near_plane) / (f - near_plane)],
         [0, 0, -1, 0]]).astype(np.float32)


class PerspectiveCamera(torch.nn.Module):
    def __init__(self, fovy=49.0, device='cuda'):
        super(PerspectiveCamera, self).__init__()
        self.device = device
        focal = np.tan(fovy / 180.0 * np.pi * 0.5)
        self.proj_mtx = torch.from_numpy(projection(x=focal, f=1000.0, n=1.0, near_plane=0.1)).to(self.device).unsqueeze(dim=0)

    def project(self, points_bxnx4):
        out = torch.matmul(
            points_bxnx4,
            torch.transpose(self.proj_mtx, 1, 2))
        return out