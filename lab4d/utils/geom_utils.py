# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R
from skimage import measure

from lab4d.utils.quat_transform import (
    dual_quaternion_apply,
    quaternion_translation_apply,
    dual_quaternion_apply,
    dual_quaternion_to_se3,
    dual_quaternion_to_quaternion_translation,
)


def pinhole_projection(Kmat, xyz_cam):
    """Project points from camera space to the image plane

    Args:
        Kmat: (M, 3, 3) Camera intrinsics
        xyz_cam: (M, ..., 3) Points in camera space
    Returns:
        hxy: (M, ..., 3) Homogeneous pixel coordinates on the image plane
    """
    shape = xyz_cam.shape
    Kmat = Kmat.view(shape[:1] + (1,) * (len(shape) - 2) + (3, 3))
    hxy = torch.einsum("...ij,...j->...i", Kmat, xyz_cam)
    hxy = hxy / (hxy[..., -1:] + 1e-6)
    return hxy


def rot_angle(mat):
    """Compute rotation angle of a rotation matrix

    Args:
        mat: (..., 3, 3) Rotation matrix
    Returns:
        angle: (...,) Rotation angle
    """
    eps = 1e-4
    cos = (mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2] - 1) / 2
    cos = cos.clamp(-1 + eps, 1 - eps)
    angle = torch.acos(cos)
    return angle


def dual_quaternion_skinning(dual_quat, pts, skin, return_qt=False):
    """Attach points to dual-quaternion bones according to skinning weights

    Args:
        dual_quat: ((M,B,4), (M,B,4)) per-bone SE(3) transforms,
            written as dual quaternions
        pts: (M, ..., 3) Points in object canonical space
        skin: (M, ..., B) Skinning weights from each point to each bone
    Returns:
        pts: (M, ..., 3) Articulated points
    """
    shape = pts.shape
    bs, B, _ = dual_quat[0].shape
    pts = pts.view(bs, -1, 3)
    skin = skin.view(bs, -1, B)  # M, N*D, B
    N = pts.shape[1]

    # (M, ND, B, 4)
    qr = dual_quat[0][:, None].repeat(1, N, 1, 1)
    qd = dual_quat[1][:, None].repeat(1, N, 1, 1)

    # make sure to blend in the same hemisphere
    anchor = skin.argmax(-1).view(shape[0], -1, 1, 1).repeat(1, 1, 1, 4)  # M, ND, 1, 4
    sign = (torch.gather(qr, 2, anchor) * qr).sum(-1) > 0  # M, ND, B
    sign = sign[..., None].float() * 2 - 1
    qr = sign * qr
    qd = sign * qd

    qr_w = torch.einsum("bnk,bnkl->bnl", skin, qr)
    qd_w = torch.einsum("bnk,bnkl->bnl", skin, qd)

    qr_mag_inv = qr_w.norm(p=2, dim=-1, keepdim=True).reciprocal()
    qr_w = qr_w * qr_mag_inv
    qd_w = qd_w * qr_mag_inv

    # import ipdb; ipdb.set_trace()
    if return_qt:
        q, t = dual_quaternion_to_quaternion_translation((qr_w, qd_w))
        return (q, t)

    else:
        # apply
        pts = dual_quaternion_apply((qr_w, qd_w), pts)
        pts = pts.view(*shape)
        return pts

# def dual_quaternion_skinning_return_qt(dual_quat, pts, skin):
#     """Attach points to dual-quaternion bones according to skinning weights

#     Args:
#         dual_quat: ((M,B,4), (M,B,4)) per-bone SE(3) transforms,
#             written as dual quaternions
#         pts: (M, ..., 3) Points in object canonical space
#         skin: (M, ..., B) Skinning weights from each point to each bone
#     Returns:
#         RT: (M, ..., 3) Articulated points
#     """
#     shape = pts.shape
#     bs, B, _ = dual_quat[0].shape
#     pts = pts.view(bs, -1, 3)
#     skin = skin.view(bs, -1, B)  # M, N*D, B
#     N = pts.shape[1]

#     # (M, ND, B, 4)
#     qr = dual_quat[0][:, None].repeat(1, N, 1, 1)
#     qd = dual_quat[1][:, None].repeat(1, N, 1, 1)

#     # make sure to blend in the same hemisphere
#     anchor = skin.argmax(-1).view(shape[0], -1, 1, 1).repeat(1, 1, 1, 4)  # M, ND, 1, 4
#     sign = (torch.gather(qr, 2, anchor) * qr).sum(-1) > 0  # M, ND, B
#     sign = sign[..., None].float() * 2 - 1
#     qr = sign * qr
#     qd = sign * qd

#     qr_w = torch.einsum("bnk,bnkl->bnl", skin, qr)
#     qd_w = torch.einsum("bnk,bnkl->bnl", skin, qd)

#     qr_mag_inv = qr_w.norm(p=2, dim=-1, keepdim=True).reciprocal()
#     qr_w = qr_w * qr_mag_inv
#     qd_w = qd_w * qr_mag_inv
#     # apply
#     # pts = dual_quaternion_apply((qr_w, qd_w), pts)
    
#     q, t = dual_quaternion_to_se3((qr_w, qd_w))
#     return (q, t)


def hat_map(v):
    """Returns the skew-symmetric matrix corresponding to the last dimension of
    a PyTorch tensor.

    Args:
        v: (..., 3) Input vector
    Returns:
        V: (..., 3, 3) Output matrix
    """
    # Reshape the input tensor to have shape (..., 3)
    v = v.reshape(-1, 3)
    # Compute the skew-symmetric matrix using a vectorized implementation
    V = torch.zeros(v.shape[0], 3, 3, dtype=v.dtype, device=v.device)
    V[:, 0, 1] = -v[:, 2]
    V[:, 0, 2] = v[:, 1]
    V[:, 1, 0] = v[:, 2]
    V[:, 1, 2] = -v[:, 0]
    V[:, 2, 0] = -v[:, 1]
    V[:, 2, 1] = v[:, 0]
    # Reshape the output tensor to match the shape of the input tensor
    V = V.reshape(v.shape[:-1] + (3, 3))
    return V


def so3_to_exp_map(so3, eps=1e-6):
    """Converts a PyTorch tensor of shape (..., 3) representing an element of
    SO(3) to a PyTorch tensor of shape (..., 3, 3) representing the
    corresponding exponential map.

    Args:
        so3: (..., 3) Element of SO(3)
        eps (float): Small value to avoid division by zero
    Returns:
        exp_V: (..., 3, 3) Exponential map
    """
    shape = so3.shape
    so3 = so3.reshape(-1, 3)

    # Compute the magnitude and direction of the rotation vector
    theta = torch.norm(so3, p=2, dim=-1, keepdim=True)
    theta = torch.clamp(theta, eps)
    v = so3 / theta
    # Compute the skew-symmetric matrix of the rotation vector
    V = hat_map(v)
    # Broadcast theta along the last two dimensions
    theta = theta.unsqueeze(-1)
    # Compute the exponential map of the rotation vector
    exp_V = (
        torch.eye(3, dtype=V.dtype, device=V.device)
        + torch.sin(theta) * V
        + (1 - torch.cos(theta)) * torch.matmul(V, V)
    )
    # Reshape the output tensor to match the shape of the input tensor
    exp_V = exp_V.reshape(shape[:-1] + (3, 3))
    return exp_V


def compute_crop_params(mask, crop_factor=1.2, crop_size=256, use_full=False):
    """Compute camera intrinsics transform from cropped to raw images

    Args:
        mask: (H, W) segmentation mask
        crop_factor (float): Ratio between crop size and size of a tight crop
        crop_size (int): Target size of cropped images
        use_full (bool): If True, return a full image
    """
    if use_full or mask.min() < 0:  # no crop if no mask
        mask = np.ones_like(mask)
        crop_factor = 1
    # ss=time.time()
    indices = np.where(mask > 0)
    xid = indices[1]
    yid = indices[0]
    center = ((xid.max() + xid.min()) // 2, (yid.max() + yid.min()) // 2)
    length = (
        (xid.max() - xid.min()) // 2,
        (yid.max() - yid.min()) // 2,
    )  # half length
    length = (int(crop_factor * length[0]), int(crop_factor * length[1]))

    # print('center:%f'%(time.time()-ss))
    # transformation from augmented image to original image
    fls = [2 * length[0] / crop_size, 2 * length[1] / crop_size]
    pps = np.asarray([float(center[0] - length[0]), float(center[1] - length[1])])
    crop2raw = np.asarray([fls[0], fls[1], pps[0], pps[1]])
    return crop2raw


def se3_vec2mat(vec):
    """Convert an SE(3) quaternion or axis-angle vector into 4x4 matrix.

    Args:
        vec: (..., 7) quaternion real-last or (..., 6) axis angle
    Returns:
        mat: (..., 4, 4) SE(3) matrix
    """
    shape = vec.shape[:-1]
    if torch.is_tensor(vec):
        mat = torch.zeros(shape + (4, 4)).to(vec.device)
        if vec.shape[-1] == 6:
            rmat = transforms.axis_angle_to_matrix(vec[..., 3:6])
        else:
            vec = vec[..., [0, 1, 2, 6, 3, 4, 5]]  # xyzw => wxyz
            rmat = transforms.quaternion_to_matrix(vec[..., 3:7])
        tmat = vec[..., :3]
    else:
        mat = np.zeros(shape + (4, 4))
        vec = vec.reshape((-1, vec.shape[-1]))
        if vec.shape[-1] == 6:
            rmat = R.from_axis_angle(vec[..., 3:6]).as_matrix()  # xyzw
        else:
            rmat = R.from_quat(vec[..., 3:7]).as_matrix()  # xyzw
        tmat = np.asarray(vec[..., :3])
        rmat = rmat.reshape(shape + (3, 3))
        tmat = tmat.reshape(shape + (3,))
    mat[..., :3, :3] = rmat
    mat[..., :3, 3] = tmat
    mat[..., 3, 3] = 1
    return mat


def apply_se3mat(se3, pts):
    """Apply an SE(3) rotation and translation to points.

    Note:
        `se3` and `pts` have the same number of batch dimensions.
        During skinning there could be an additional dimension B
    Args:
        se3: (M,1,1,(B),4) Real-first quaternion and (M,1,1,(B),3) Translation
        pts: (M,N,D,(1),3) Points to transform
    Returns:
        pts_out: (M,N,D,(B),3) Transformed points
    """
    quat, trans = se3
    pts_out = quaternion_translation_apply(quat, trans, pts)
    return pts_out


def se3_mat2rt(mat):
    """Convert an SE(3) 4x4 matrix into rotation matrix and translation.

    Args:
        mat: (..., 4, 4) SE(3) matrix
    Returns:
        rmat: (..., 3, 3) Rotation
        tmat: (..., 3) Translation
    """
    rmat = mat[..., :3, :3]
    tmat = mat[..., :3, 3]
    return rmat, tmat


def se3_mat2vec(mat, outdim=7):
    """Convert SE(3) 4x4 matrix into a quaternion or axis-angle vector
    Args:
        mat: (..., 4, 4) SE(3) matrix
        outdim (int): 7 to output quaternion vector, 6 to output axis-angle
    Returns:
        vec: (..., outdim): Quaternion or axis-angle vector
    """
    shape = mat.shape[:-2]
    assert torch.is_tensor(mat)
    tmat = mat[..., :3, 3]
    quat = transforms.matrix_to_quaternion(mat[..., :3, :3])
    if outdim == 7:
        rot = quat[..., [1, 2, 3, 0]]  # xyzw <= wxyz
    elif outdim == 6:
        rot = transforms.quaternion_to_axis_angle(quat)
    else:
        print("error")
        exit()
    vec = torch.cat([tmat, rot], -1)
    return vec


def K2mat(K):
    """Convert camera intrinsics tuple into matrix

    Args:
        K: (..., 4) Camera intrinsics (fx, fy, cx, cy)

    Returns:
        Kmat: (..., 3, 3) Camera intrinsics matrix
    """
    if torch.is_tensor(K):
        Kmat = torch.zeros(K.shape[:-1] + (3, 3), device=K.device)
    else:
        Kmat = np.zeros(K.shape[:-1] + (3, 3))

    Kmat[..., 0, 0] = K[..., 0]
    Kmat[..., 1, 1] = K[..., 1]
    Kmat[..., 0, 2] = K[..., 2]
    Kmat[..., 1, 2] = K[..., 3]
    Kmat[..., 2, 2] = 1
    return Kmat


def mat2K(Kmat):
    """Convert camera intrinsics matrix into tuple

    Args:
        Kmat: (..., 3, 3) Camera intrinsics matrix

    Returns:
        K: (..., 4) Camera intrinsics (fx, fy, cx, cy)
    """
    shape = Kmat.shape[:-2]
    Kmat = Kmat.reshape((-1, 3, 3))
    bs = Kmat.shape[0]

    if torch.is_tensor(Kmat):
        K = torch.zeros(bs, 4, device=Kmat.device)
    else:
        K = np.zeros((bs, 4))
    K[:, 0] = Kmat[:, 0, 0]
    K[:, 1] = Kmat[:, 1, 1]
    K[:, 2] = Kmat[:, 0, 2]
    K[:, 3] = Kmat[:, 1, 2]
    K = K.reshape(shape + (4,))
    return K


def Kmatinv(Kmat):
    """Invert camera intrinsics matrix

    Args:
        Kmat: (..., 3, 3) Camera intrinsics matrix

    Returns:
        Kmatinv: (..., 3, 3) Inverse camera intrinsics matrix
    """
    K = mat2K(Kmat)
    Kmatinv = K2inv(K)
    Kmatinv = Kmatinv.reshape(Kmat.shape)
    return Kmatinv


def K2inv(K):
    """Compute the inverse camera intrinsics matrix from tuple

    Args:
        K: (..., 4) Camera intrinsics (fx, fy, cx, cy)

    Returns:
        Kmat: (..., 3, 3) Inverse camera intrinsics matrix
    """
    if torch.is_tensor(K):
        Kmat = torch.zeros(K.shape[:-1] + (3, 3), device=K.device)
    else:
        Kmat = np.zeros(K.shape[:-1] + (3, 3))
    Kmat[..., 0, 0] = 1.0 / K[..., 0]
    Kmat[..., 1, 1] = 1.0 / K[..., 1]
    Kmat[..., 0, 2] = -K[..., 2] / K[..., 0]
    Kmat[..., 1, 2] = -K[..., 3] / K[..., 1]
    Kmat[..., 2, 2] = 1
    return Kmat


def get_near_far(pts, rtmat, tol_fac=1.5):
    """
    Args:
        pts:        Point coordinate, (N,3), torch
        rtmat:      Object to camera transform, (M,4,4), torch
        tol_fac:    Tolerance factor
    """
    pts = obj_to_cam(pts, rtmat)

    pmax = pts[..., -1].max(-1)[0]
    pmin = pts[..., -1].min(-1)[0]
    delta = (pmax - pmin) * (tol_fac - 1)

    near = pmin - delta
    far = pmax + delta

    near_far = torch.stack([near, far], -1)
    near_far = torch.clamp(near_far, min=1e-3)
    return near_far


def obj_to_cam(pts, rtmat):
    """
    Args:
        pts:        Point coordinate, (M,N,3) or (N,3), torch or numpy
        rtmat:      Object to camera transform, M,4,4, torch or numpy
    Returns:
        verts:      Transformed points, (M,N,3), torch or numpy
    """
    if torch.is_tensor(pts):
        is_tensor = True
    else:
        is_tensor = False

    if not is_tensor:
        pts = torch.tensor(pts, device="cuda")
        rtmat = torch.tensor(rtmat, device="cuda")  # M,4,4

    pts = pts.view(-1, pts.shape[-2], 3)  # -1,N,3
    pts = torch.cat([pts, torch.ones_like(pts[..., :1])], -1)  # -1,N,4
    pts = torch.einsum("mij,mjk->mik", pts, rtmat.permute(0, 2, 1))  # M,N,4
    pts = pts[..., :3]

    if not is_tensor:
        pts = pts.cpu().numpy()
    return pts


def sample_grid(aabb, grid_size):
    """Densely sample points in a 3D grid

    Args:
        aabb: (2,3) Axis-aligned bounding box
        grid_size (int): Points to sample along each axis
    Returns:
        query_xyz: (grid_size^3,3) Dense xyz grid
    """
    device = aabb.device
    ptx = torch.linspace(aabb[0][0], aabb[1][0], grid_size, device=device)
    pty = torch.linspace(aabb[0][1], aabb[1][1], grid_size, device=device)
    ptz = torch.linspace(aabb[0][2], aabb[1][2], grid_size, device=device)
    query_xyz = torch.cartesian_prod(ptx, pty, ptz)  # (x,y,z)
    return query_xyz


def extend_aabb(aabb, factor=0.1):
    """Extend aabb along each side by factor of the previous size.
    If aabb = [-1,1] and factor = 1, the extended aabb will be [-3,3]

    Args:
        aabb: Axis-aligned bounding box, (2,3)
        factor (float): Amount to extend on each side
    Returns:
        aabb_new: Extended aabb, (2,3)
    """
    aabb_new = aabb.clone()
    aabb_new[0] = aabb[0] - (aabb[1] - aabb[0]) * factor
    aabb_new[1] = aabb[1] + (aabb[1] - aabb[0]) * factor
    return aabb_new


def eval_func_chunk(func, xyz, chunk_size):
    """Evaluate a function in chunks to avoid OOM.

    Args:
        func: (M,x) -> (M,y)
        xyz: (M,x)
        chunk_size: int
    Returns:
        vals: (M,y)
    """
    vals = []
    for i in range(0, xyz.shape[0], chunk_size):
        vals.append(func(xyz[i : i + chunk_size]))
    vals = torch.cat(vals, dim=0)
    return vals


@torch.no_grad()
def marching_cubes(
    sdf_func,
    aabb,
    visibility_func=None,
    grid_size=64,
    level=0,
    chunk_size=64**3,
    apply_connected_component=False,
):
    """Extract a mesh from a signed-distance function using marching cubes.
    For the multi-instance case, we use the mean shape/visibility

    Args:
        sdf_func (Function): Signed distance function
        aabb: (2,3) Axis-aligned bounding box
        visibility_func (Function): Returns visibility of each point from camera
        grid_size (int): Marching cubes resolution
        level (float): Contour value to search for isosurfaces on the signed
            distance function
        chunk_size (int): Chunk size to evaluate the sdf function
        apply_connected_component (bool): Whether to apply connected component
    Returns:
        mesh (Trimesh): Output mesh
    """
    # sample grid
    grid = sample_grid(aabb, grid_size)

    # evaluate sdf
    sdf = eval_func_chunk(sdf_func, grid, chunk_size=chunk_size)
    sdf = sdf.cpu().numpy().reshape(grid_size, grid_size, grid_size)

    # evaluate visibility: # ignore the points that are not sampled during optimization
    if visibility_func is not None:
        vis = eval_func_chunk(visibility_func, grid, chunk_size=chunk_size)
        vis = vis.cpu().numpy().reshape(grid_size, grid_size, grid_size)
    else:
        vis = np.ones_like(sdf).astype(bool)

    # extract mesh from sdf: 0-1
    try:
        verts, faces, _, _ = measure.marching_cubes(
            sdf,
            level=level,
            spacing=(1.0 / grid_size, 1.0 / grid_size, 1.0 / grid_size),
            mask=vis,
        )
    except:
        print("marching cubes failed")
        return trimesh.Trimesh()

    # transform from 0-1 to bbox
    aabb = aabb.cpu().numpy()
    verts = verts * (aabb[1:] - aabb[:1]) + aabb[:1]

    mesh = trimesh.Trimesh(verts, faces)
    if apply_connected_component:
        # keep the largest connected component
        mesh = [i for i in mesh.split(only_watertight=False)]
        mesh = sorted(mesh, key=lambda x: x.vertices.shape[0])
        mesh = mesh[-1]
    return mesh


def check_inside_aabb(xyz, aabb):
    """Return a mask of whether the input poins are inside the aabb

    Args:
        xyz: (N,3) Points in object canonical space to query
        aabb: (2,3) axis-aligned bounding box
    Returns:
        inside_aabb: (N) Inside mask, bool
    """
    # check whether the point is inside the aabb
    inside_aabb = ((xyz > aabb[:1]) & (xyz < aabb[1:])).all(-1)
    return inside_aabb
