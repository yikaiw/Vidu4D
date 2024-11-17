# Decorate all modules with @record_function and @record_class
import lab4d.dataloader.data_utils
import lab4d.dataloader.vidloader
import lab4d.engine.model
import lab4d.engine.train_utils
import lab4d.engine.trainer
import lab4d.nnutils.appearance
import lab4d.nnutils.base
import lab4d.nnutils.deformable
import lab4d.nnutils.embedding
import lab4d.nnutils.feature
import lab4d.nnutils.intrinsics
import lab4d.nnutils.multifields
import lab4d.nnutils.nerf
import lab4d.nnutils.pose
import lab4d.nnutils.skinning
import lab4d.nnutils.time
import lab4d.nnutils.visibility
import lab4d.nnutils.warping
import lab4d.utils.cam_utils
import lab4d.utils.camera_utils
import lab4d.utils.geom_utils
import lab4d.utils.io
import lab4d.utils.loss_utils
import lab4d.utils.numpy_utils
import lab4d.utils.quat_transform
import lab4d.utils.render_utils
import lab4d.utils.skel_utils
import lab4d.utils.torch_utils
import lab4d.utils.transforms
import lab4d.utils.vis_utils
from lab4d.utils.profile_utils import decorate_module

decorate_module(lab4d.dataloader.data_utils)
decorate_module(lab4d.dataloader.vidloader)
decorate_module(lab4d.engine.model)
decorate_module(lab4d.engine.trainer)
decorate_module(lab4d.engine.train_utils)
decorate_module(lab4d.nnutils.appearance)
decorate_module(lab4d.nnutils.base)
decorate_module(lab4d.nnutils.deformable)
decorate_module(lab4d.nnutils.embedding)
decorate_module(lab4d.nnutils.feature)
decorate_module(lab4d.nnutils.intrinsics)
decorate_module(lab4d.nnutils.multifields)
decorate_module(lab4d.nnutils.nerf)
decorate_module(lab4d.nnutils.pose)
decorate_module(lab4d.nnutils.skinning)
decorate_module(lab4d.nnutils.time)
decorate_module(lab4d.nnutils.visibility)
decorate_module(lab4d.nnutils.warping)
decorate_module(lab4d.utils.camera_utils)
decorate_module(lab4d.utils.cam_utils)
decorate_module(lab4d.utils.geom_utils)
decorate_module(lab4d.utils.io)
decorate_module(lab4d.utils.loss_utils)
decorate_module(lab4d.utils.numpy_utils)
decorate_module(lab4d.utils.quat_transform)
decorate_module(lab4d.utils.render_utils)
decorate_module(lab4d.utils.skel_utils)
decorate_module(lab4d.utils.torch_utils)
decorate_module(lab4d.utils.transforms)
decorate_module(lab4d.utils.vis_utils)
