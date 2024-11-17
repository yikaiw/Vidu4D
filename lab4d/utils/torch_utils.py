# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import torch

@torch.enable_grad()
def compute_gradient(fn, x):
    """
    gradient of mlp params wrt pts
    """
    x.requires_grad_(True)
    y = fn(x)

    # get gradient for each size-1 output
    gradients = []
    for i in range(y.shape[-1]):
        y_sub = y[..., i : i + 1]
        d_output = torch.ones_like(y_sub, requires_grad=False, device=y.device)
        gradient = torch.autograd.grad(
            outputs=y_sub,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients.append(gradient[..., None])
    gradients = torch.cat(gradients, -1)  # ...,input-dim, output-dim
    return gradients

def frameid_to_vid(fid, frame_offset):
    """Given absolute frame ids [0, ..., N], compute the video id of each frame.

    Args:
        fid: (nframes,) Absolute frame ids
          e.g. [0, 1, 2, 3, 100, 101, 102, 103, 200, 201, 202, 203]
        frame_offset: (nvideos + 1,) Offset of each video
          e.g., [0, 100, 200, 300]
    Returns:
        vid: (nframes,) Maps idx to video id
        tid: (nframes,) Maps idx to relative frame id
    """
    vid = torch.zeros_like(fid)
    for i in range(frame_offset.shape[0] - 1):
        assign = torch.logical_and(fid >= frame_offset[i], fid < frame_offset[i + 1])
        vid[assign] = i
    return vid


def remove_ddp_prefix(state_dict):
    """Remove distributed data parallel prefix from model checkpoint

    Args:
        state_dict (Dict): Model checkpoint
    Returns:
        new_state_dict (Dict): New model checkpoint
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[7:]  # Remove 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def remove_state_startwith(state_dict, prefix):
    """Remove model parameters that start with a prefix

    Args:
        state_dict (Dict): Model checkpoint
        prefix (str): Prefix to filter
    Returns:
        new_state_dict (Dict): New model checkpoint
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            continue
        else:
            new_state_dict[key] = value
    return new_state_dict


def remove_state_with(state_dict, string):
    """Remove model parameters that contain a string

    Args:
        state_dict (Dict): Model checkpoint
        string (str): String to filter
    Returns:
        new_state_dict (Dict): New model checkpoint
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if string in key:
            continue
        else:
            new_state_dict[key] = value
    return new_state_dict


def compress_state_with(state_dict, string):
    """Initialize model parameters with the mean of the instance embedding if
    the parameter name contains a string

    Args:
        state_dict (Dict): Model checkpoint, modified in place
        string (str): String to filter
    """
    # init with the mean of inst_embedding
    for key, value in state_dict.items():
        if string in key:
            state_dict[key] = value.mean(dim=0, keepdim=True)
