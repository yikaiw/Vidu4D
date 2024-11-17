# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh

from lab4d.nnutils.base import CondMLP, BaseMLP, ScaleLayer
from lab4d.nnutils.time import TimeMLP
from lab4d.utils.geom_utils import so3_to_exp_map
from lab4d.utils.quat_transform import (
    axis_angle_to_quaternion,
    matrix_to_quaternion,
    quaternion_mul,
    quaternion_translation_to_dual_quaternion,
    dual_quaternion_to_quaternion_translation,
    quaternion_translation_to_se3,
)
from lab4d.utils.skel_utils import (
    fk_se3,
    get_predefined_skeleton,
    rest_joints_to_local,
    shift_joints_to_bones_dq,
    shift_joints_to_bones,
)
from lab4d.utils.vis_utils import draw_cams


class CameraMLP(TimeMLP):
    """Encode camera pose over time (rotation + translation) with an MLP

    Args:
        rtmat: (N,4,4) Object to camera transform
        frame_info (Dict): Metadata about the frames in a dataset
        D (int): Number of linear layers
        W (int): Number of hidden units in each MLP layer
        num_freq_t (int): Number of frequencies in time Fourier embedding
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
    """

    def __init__(
        self,
        rtmat,
        frame_info=None,
        D=5,
        W=256,
        num_freq_t=6,
        skips=[],
        activation=nn.ReLU(True),
    ):
        if frame_info is None:
            num_frames = len(rtmat)
            frame_info = {
                "frame_offset": np.asarray([0, num_frames]),
                "frame_mapping": list(range(num_frames)),
                "frame_offset_raw": np.asarray([0, num_frames]),
            }
        # xyz encoding layers
        super().__init__(
            frame_info,
            D=D,
            W=W,
            num_freq_t=num_freq_t,
            skips=skips,
            activation=activation,
        )

        # output layers
        self.trans = nn.Sequential(
            nn.Linear(W, W // 2),
            activation,
            nn.Linear(W // 2, 3),
        )
        self.quat = nn.Sequential(
            nn.Linear(W, W // 2),
            activation,
            nn.Linear(W // 2, 4),
        )

        # camera pose: field to camera
        self.base_quat = nn.Parameter(torch.zeros(self.time_embedding.num_vids, 4))
        self.register_buffer(
            "init_vals", torch.tensor(rtmat, dtype=torch.float32), persistent=False
        )

        # override the loss function
        def loss_fn(gt):
            quat, trans = self.get_vals()
            pred = quaternion_translation_to_se3(quat, trans)
            loss = F.mse_loss(pred, gt)
            return loss

        self.loss_fn = loss_fn

    def base_init(self):
        """Initialize base camera rotations from initial camera trajectory"""
        rtmat = self.init_vals
        frame_offset = self.get_frame_offset()
        base_rmat = rtmat[frame_offset[:-1], :3, :3]
        base_quat = matrix_to_quaternion(base_rmat)
        self.base_quat.data = base_quat

    def mlp_init(self):
        """Initialize camera SE(3) transforms from external priors"""
        self.base_init()
        super().mlp_init()

        # with torch.no_grad():
        #     os.makedirs("tmp", exist_ok=True)
        #     draw_cams(rtmat.cpu().numpy()).export("tmp/cameras_gt.obj")
        #     quat, trans = self.get_vals()
        #     rtmat_pred = quaternion_translation_to_se3(quat, trans)
        #     draw_cams(rtmat_pred.cpu()).export("tmp/cameras_pred.obj")

    def forward(self, t_embed):
        """
        Args:
            t_embed: (M, self.W) Input Fourier time embeddings
        Returns:
            quat: (M, 4) Output camera rotation quaternions
            trans: (M, 3) Output camera translations
        """
        t_feat = super().forward(t_embed)
        trans = self.trans(t_feat)
        quat = self.quat(t_feat)
        quat = F.normalize(quat, dim=-1)
        return quat, trans

    def get_vals(self, frame_id=None):
        """Compute camera pose at the given frames.

        Args:
            frame_id: (M,) Frame id. If None, compute values at all frames
        Returns:
            quat: (M, 4) Output camera rotations
            trans: (M, 3) Output camera translations
        """
        t_embed = self.time_embedding(frame_id)
        quat, trans = self.forward(t_embed)
        if frame_id is None:
            inst_id = self.time_embedding.frame_to_vid
        else:
            inst_id = self.time_embedding.raw_fid_to_vid[frame_id]

        # multiply with per-instance base rotation
        base_quat = self.base_quat[inst_id]
        base_quat = F.normalize(base_quat, dim=-1)
        quat = quaternion_mul(quat, base_quat)
        return quat, trans


class ArticulationBaseMLP(TimeMLP):
    """Base class for bone articulation model (bag-of-bones or skeleton)

    Args:
        frame_info (FrameInfo): Metadata about the frames in a dataset
        num_se3 (int): Number of bones
        D (int): Number of linear layers
        W (int): Number of hidden units in each MLP layer
        num_freq_t (int): Number of frequencies in time Fourier embedding
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
    """

    def __init__(
        self,
        frame_info,
        num_se3,
        D=5,
        W=256,
        num_freq_t=6,
        skips=[],
        activation=nn.ReLU(True),
    ):
        # xyz encoding layers
        super().__init__(
            frame_info,
            D=D,
            W=W,
            num_freq_t=num_freq_t,
            skips=skips,
            activation=activation,
        )
        self.edges = None

        self.num_se3 = num_se3

    def forward(self, t_embed):
        """
        Args:
            t_embed: (M, self.W) Time Fourier embedding
        Returns:
            t_feat: (M, self.W) Time-dependent features
        """
        t_feat = super().forward(t_embed)
        return t_feat

    def get_vals(self, frame_id=None):
        """Compute articulation parameters at the given frames.

        Args:
            frame_id: (M,) Frame id. If None, compute values at all frames
        Returns:
            pred: ((M,B,4), (M,B,4)) Predicted bone-to-object transforms at each
                frame, written as dual quaternions
        """
        if frame_id is None:
            inst_id = self.time_embedding.frame_to_vid
        else:
            inst_id = self.time_embedding.raw_fid_to_vid[frame_id]
        t_embed = self.time_embedding(frame_id)

        pred = self.forward(t_embed, inst_id)
        return pred

    def get_mean_vals(self, inst_id=None):
        """Compute bone-to-object transforms for the rest shape

        Args:
            inst_id: (M,) Instance id. If None, compute values at the mean instance
        Returns:
            pred: ((1,B,4), (1,B,4)) Predicted bone-to-object transform for the rest
                shape, written as dual quaternions
        """
        device = self.parameters().__next__().device
        t_embed = self.time_embedding.get_mean_embedding(device)
        pred = self.forward(t_embed, inst_id)
        return pred

    def get_vals_and_mean(self, frame_id=None):
        """Compute bone-to-object transforms at the given frames, and also for
        the rest shape

        Args:
            frame_id: (M,) Frame id. If None, compute values at all frames
        """
        raise NotImplementedError


class ArticulationFlatMLP(ArticulationBaseMLP):
    """Encode a bag of bones over time using an MLP

    Args:
        frame_info (FrameInfo): Metadata about the frames in a dataset
        num_se3 (int): Number of bones
        D (int): Number of linear layers
        W (int): Number of hidden units in each MLP layer
        num_freq_t (int): Number of frequencies in time Fourier embedding
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
    """

    def __init__(
        self,
        frame_info,
        num_se3,
        D=5,
        W=256,
        num_freq_t=6,
        skips=[],
        activation=nn.ReLU(True),
    ):
        # xyz encoding layers
        super().__init__(
            frame_info,
            num_se3,
            D=D,
            W=W,
            num_freq_t=num_freq_t,
            skips=skips,
            activation=activation,
        )

        # output layers
        self.trans = nn.Sequential(
            nn.Linear(W, W // 2),
            activation,
            nn.Linear(W // 2, 3 * num_se3),
            ScaleLayer(0.1),
        )
        self.so3 = nn.Sequential(
            nn.Linear(W, W // 2),
            activation,
            nn.Linear(W // 2, 3 * num_se3),
        )

    def forward(self, t_embed, inst_id):
        """
        Args:
            t_embed: (M, num_freq_t) Time Fourier embedding
            inst_id: (M,) Instance id. If None, evaluate for the mean instance
        Returns:
            out: ((M,B,4), (M,B,4)): Predicted bone-to-object transforms for
                each bone, written as dual quaternions
        """
        t_feat = super().forward(t_embed)
        trans = self.trans(t_feat).reshape(*t_embed.shape[:-1], self.num_se3, 3)
        so3 = self.so3(t_feat).reshape(*t_embed.shape[:-1], self.num_se3, 3)
        # convert to rigid transformation
        qr = axis_angle_to_quaternion(so3)
        dq = quaternion_translation_to_dual_quaternion(qr, trans)
        return dq

    def get_vals_and_mean(self, frame_id=None):
        """Compute bone-to-object transforms at the given frames, and also for
        the rest shape

        Args:
            frame_id: (M,) Frame id. If None, compute values at all frames
        Returns:
            pred_t: ((M,B,4), (M,B,4)) Predicted bone-to-object transforms at
                each frame, written as dual quaternions
            pred_mean: ((M,B,4), (M,B,4)) Predicted bone-to-object transforms
                for the rest shape
        """
        pred_t = self.get_vals(frame_id)  # (M,K,4,4)
        pred_mean = self.get_mean_vals()  # (M,K,4,4)
        pred_mean = (
            pred_mean[0].expand_as(pred_t[0]).contiguous(),
            pred_mean[1].expand_as(pred_t[1]).contiguous(),
        )
        return pred_t, pred_mean


class ArticulationSkelMLP(ArticulationBaseMLP):
    """Encode a skeleton over time using an MLP

    Args:
        frame_info (FrameInfo): Metadata about the frames in a dataset
        skel_type (str): Skeleton type ("human" or "quad")
        joint_angles: (B, 3) If provided, initial joint angles
        num_se3 (int): Number of bones
        D (int): Number of linear layers
        W (int): Number of hidden units in each MLP layer
        num_freq_t (int): Number of frequencies in time Fourier embedding
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
    """

    def __init__(
        self,
        frame_info,
        skel_type,
        joint_angles,
        D=5,
        W=256,
        num_freq_t=6,
        skips=[],
        activation=nn.ReLU(True),
    ):
        # get skeleton
        rest_joints, edges, symm_idx = get_predefined_skeleton(skel_type)
        num_se3 = len(rest_joints)

        # xyz encoding layers
        super(ArticulationSkelMLP, self).__init__(
            frame_info,
            num_se3,
            D=D,
            W=W,
            num_freq_t=num_freq_t,
            skips=skips,
            activation=activation,
        )

        # register the skeleton
        self.edges, self.symm_idx = edges, symm_idx
        self.register_buffer("rest_joints", rest_joints)  # K, 3

        # output layers
        self.so3 = nn.Sequential(
            nn.Linear(W, W // 2),
            activation,
            nn.Linear(W // 2, self.num_se3 * 3),
        )

        self.logscale = nn.Parameter(torch.zeros(1))
        self.shift = nn.Parameter(torch.zeros(3))
        # instance bone length
        num_inst = len(frame_info["frame_offset"]) - 1
        self.log_bone_len = CondMLP(
            num_inst,
            in_channels=0,
            D=2,
            W=64,
            out_channels=self.num_se3,
        )

        # initialize with per-frame pose
        if joint_angles is not None:
            self.register_buffer(
                "init_vals",
                torch.tensor(joint_angles, dtype=torch.float32),
                persistent=False,
            )

        # override the loss function
        def loss_fn(gt):
            inst_id = self.time_embedding.frame_to_vid
            t_embed = self.time_embedding(frame_id=None)
            pred = self.forward(t_embed, inst_id, return_so3=True)
            loss = F.mse_loss(pred, gt)
            return loss

        self.loss_fn = loss_fn

    def mlp_init(self):
        """For skeleton fields, initialize bone lengths and rest joint angles
        from an external skeleton
        """
        if not hasattr(self, "init_vals"):
            return

        super().mlp_init()

    def forward(
        self,
        t_embed,
        inst_id,
        return_so3=False,
        override_so3=None,
        override_log_bone_len=None,
        override_local_rest_joints=None,
    ):
        """
        Args:
            t_embed: (M, self.W) Time Fourier embedding
            inst_id: (M,) Instance id. If None, evaluate for the mean instance
            return_so3 (bool): If True, return computed joint angles instead
            override_so3: (M,K,3) If given, override computed joint angles from
                inputs. Used during reanimation
            override_log_bone_len: If given, override computed bone lengths
                from inputs
            override_local_rest_joints: If given, override local rest joints
                from inputs
        Returns:
            out: ((M,B,4), (M,B,4)) Predicted bone-to-object transforms for
                each bone, written as dual quaternions
        """
        # compute so3
        if override_so3 is None:
            t_feat = super(ArticulationSkelMLP, self).forward(t_embed)
            so3 = self.so3(t_feat).reshape(
                *t_embed.shape[:-1], self.num_se3, 3
            )  # joint angles, so3 exp
        else:
            so3 = override_so3

        if return_so3:
            return so3

        # get relative joints
        if override_local_rest_joints is None:
            local_rest_joints = self.compute_rel_rest_joints(
                inst_id=inst_id, override_log_bone_len=override_log_bone_len
            )
        else:
            local_rest_joints = override_local_rest_joints

        # run forward kinematics
        out = fk_se3(local_rest_joints, so3, self.edges)
        out = shift_joints_to_bones_dq(out, self.edges, shift=self.shift)
        return out

    def compute_rel_rest_joints(self, inst_id=None, override_log_bone_len=None):
        """Compute relative position difference from parent to child bone
        coordinate frames, without scale

        Args:
            inst_id: (M,) instance id. If None, compute for the mean instance
            override_log_bone_len: If provided, override computed bone lengths
                from inputs
        Returns:
            rel_rest_joints: Translations from parent to child joints
        """
        # get relative joints
        rel_rest_joints = rest_joints_to_local(self.rest_joints, self.edges)

        # match the shape
        rel_rest_joints = rel_rest_joints[None]
        if inst_id is not None:
            rel_rest_joints = rel_rest_joints.repeat(inst_id.shape[0], 1, 1)

        # update bone length
        empty_feat = torch.zeros_like(rel_rest_joints[..., 0, :0])  # (B, 0)
        if override_log_bone_len is not None:
            log_bone_len_inc = override_log_bone_len
        else:
            log_bone_len_inc = self.log_bone_len(empty_feat, inst_id)
        bone_length = (log_bone_len_inc + self.logscale).exp()
        bone_length = (bone_length + bone_length[..., self.symm_idx]) / 2
        rel_rest_joints = rel_rest_joints * bone_length[..., None]
        return rel_rest_joints

    def get_vals(self, frame_id=None, return_so3=False, override_so3=None):
        """Compute articulation parameters at the given frames.

        Args:
            frame_id: (M,) Frame id. If None, compute values at all frames
            return_so3 (bool): If True, return computed joint angles instead
            override_so3: (M,K,3) If given, override computed joint angles with
                inputs. Used during reanimation
        Returns:
            pred: ((M,B,4), (M,B,4)) Predicted bone-to-object transforms at each
                frame, written as duql quaternions
        """
        if frame_id is None:
            inst_id = self.time_embedding.frame_to_vid
        else:
            inst_id = self.time_embedding.raw_fid_to_vid[frame_id]
        t_embed = self.time_embedding(frame_id)
        pred = self.forward(
            t_embed, inst_id, return_so3=return_so3, override_so3=override_so3
        )
        return pred

    def get_vals_and_mean(self, frame_id=None):
        """Compute bone-to-object transforms at the given frames, and also for
        the rest shape. Faster than calling forward kinematics separately

        Args:
            frame_id: (M,) Frame id. If None, compute values at all frames
        Returns:
            pred_t: ((M,B,4), (M,B,4)) Predicted bone-to-object transforms at
                each frame, written as dual quaternions
            pred_mean: ((M,B,4), (M,B,4)) Predicted bone-to-object transforms
                for the rest shape
        """
        if frame_id is None:
            inst_id = self.time_embedding.frame_to_vid
        else:
            inst_id = self.time_embedding.raw_fid_to_vid[frame_id]
        bs = inst_id.shape[0]
        # t embedding
        t_embed = self.time_embedding(frame_id)
        # mean embedding
        device = self.parameters().__next__().device
        t_embed_mean = self.time_embedding.get_mean_embedding(device)
        t_embed_mean = t_embed_mean.expand(bs, *t_embed_mean.shape[1:])
        # concat
        t_embed = torch.cat([t_embed, t_embed_mean], dim=0)

        # forward
        rel_rest_joints_c = self.compute_rel_rest_joints()  # canonical skel
        rel_rest_joints_c = rel_rest_joints_c.repeat(bs, 1, 1)
        rel_rest_joints_i = self.compute_rel_rest_joints(inst_id=inst_id)  # inst skel
        rel_rest_joints = torch.cat([rel_rest_joints_i, rel_rest_joints_c], dim=0)
        pred = self.forward(t_embed, None, override_local_rest_joints=rel_rest_joints)
        # split
        pred_t = pred[0][:bs], pred[1][:bs]
        pred_mean = pred[0][bs:], pred[1][bs:]

        # # sanity check
        # pred_tt = self.get_vals(frame_id)
        # pred_mm = self.get_mean_vals()
        # pred_mm = (
        #     pred_mm[0].expand_as(pred_tt[0]).contiguous(),
        #     pred_mm[1].expand_as(pred_tt[1]).contiguous(),
        # )

        # print((pred_t[0] - pred_tt[0]).abs().max())
        # print((pred_t[1] - pred_tt[1]).abs().max())
        # print((pred_mean[0] - pred_mm[0]).abs().max())
        # print((pred_mean[1] - pred_mm[1]).abs().max())

        return pred_t, pred_mean

    def skel_prior_loss(self):
        """Encourage the skeleton rest pose to be near the pose initialization.
        Computes L2 loss on joint axis-angles and bone lengths

        Returns:
            loss: (0,) Skeleton prior loss
        """
        # get rest joint angles increment
        device = self.parameters().__next__().device
        t_embed = self.time_embedding.get_mean_embedding(device)
        so3 = self.forward(t_embed, None, return_so3=True)  # 1, num_channels
        loss_so3 = so3.pow(2).mean()

        # get average log bone length increment
        empty_feat = torch.zeros_like(so3[..., 0, :0])  # (1, 0)
        log_bone_len_inc = self.log_bone_len(empty_feat, None)
        loss_bone = 0.02 * log_bone_len_inc.pow(2).mean()

        loss = loss_so3 + loss_bone

        # # alternative: minimize joint location difference
        # device = self.parameters().__next__().device
        # t_embed = self.time_embedding.get_mean_embedding(device)
        # bones_dq = self.forward(t_embed, None)
        # bones_pred = dual_quaternion_to_quaternion_translation(bones_dq)[1][0]  # B,3

        # joints_gt = self.rest_joints * self.logscale.exp() + self.shift[None]
        # bones_gt = shift_joints_to_bones(joints_gt, self.edges)

        # loss = (bones_gt - bones_pred).norm(2, -1).mean()
        # loss = loss * 0.2
        return loss
