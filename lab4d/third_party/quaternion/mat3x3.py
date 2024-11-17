import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    import _quaternion as _backend
except ImportError:
    from .backend import _backend


class _Mat3x3_det(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float)
    def forward(ctx, inputs:torch.Tensor):
        B = inputs.shape[0]
        assert(inputs.shape[1] == 9)
        dtype = inputs.dtype
        device = inputs.device
        
        outputs = torch.empty(B, dtype=dtype, device=device)

        _backend.mat3x3_det_forward(inputs, outputs, B)
        ctx.save_for_backward(inputs)

        return outputs

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad):
        return None

_mat3x3_det = _Mat3x3_det.apply
def mat3x3_det(inputs:torch.Tensor):
    rt_size = inputs.shape[:-2]
    outputs = _mat3x3_det(inputs.contiguous().view(-1,9))
    return outputs.view(rt_size)


class _Mat3x3_scale_adjoint(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, inputs:torch.Tensor, scales:torch.Tensor):
        B = inputs.shape[0]
        assert(inputs.shape[1] == 9)
        dtype = inputs.dtype
        device = inputs.device    
        outputs = torch.empty(B, 9, dtype=dtype, device=device)
        _backend.mat3x3_scale_adjoint_forward(inputs, scales, outputs, B)
        ctx.save_for_backward(inputs, scales)
        return outputs

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, *grad_outputs):
        return None

_mat3x3_scale_adjoint = _Mat3x3_scale_adjoint.apply
def mat3x3_scale_adjoint(inputs:torch.Tensor, scales:torch.Tensor):
    rt_size = inputs.shape
    outputs = _mat3x3_scale_adjoint(inputs.contiguous().view(-1,9), scales.contiguous().view(-1))
    return outputs.view(rt_size)


class _Mat3x3_inv(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float)
    def forward(ctx, inputs:torch.Tensor):
        B = inputs.shape[0]
        assert(inputs.shape[1] == 9)
        dtype = inputs.dtype
        device = inputs.device    
        scales = torch.empty(B, dtype=dtype, device=device)
        outputs = torch.empty(B, 9, dtype=dtype, device=device)
        _backend.mat3x3_inv_forward(inputs, outputs, scales, B)
        ctx.save_for_backward(outputs, scales)
        # print(scales)
        return outputs

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad):
        inv_mats, _ = ctx.saved_tensors
        B = inv_mats.shape[0]
        assert(inv_mats.shape[1] == 9)
        dtype = inv_mats.dtype
        device = inv_mats.device  
        grad_inputs = torch.empty(B, 9, dtype=dtype, device=device)
        _backend.mat3x3_inv_backward(grad, inv_mats, grad_inputs, B)
        return grad_inputs



_mat3x3_inv = _Mat3x3_inv.apply
def mat3x3_inv(inputs:torch.Tensor):
    rt_size = inputs.shape
    outputs = _mat3x3_inv(inputs.contiguous().view(-1,9))
    return outputs.view(rt_size)  

def _test_mat3x3_inv_backward(x:torch.Tensor):
    x_inv = mat3x3_inv(x)
    loss = x_inv.mean()
    loss.backward()

def _test():
    import torch.utils.benchmark as benchmark
    N = 4096*128
    # N = 100
    x = torch.randn(N, 3, 3, requires_grad=True).float().cuda()
    x_det = mat3x3_det(x)

    # torch.autograd.gradcheck(mat3x3_inv, x)

    T = 100
    t = benchmark.Timer(
        stmt='mat3x3_det(x)',
        setup='from __main__ import mat3x3_det',
        globals={'x': x})
    print(t.timeit(T))

    x_adj = mat3x3_scale_adjoint(x, x_det)
    T = 100
    t = benchmark.Timer(
        stmt='mat3x3_scale_adjoint(x, x_det)',
        setup='from __main__ import mat3x3_scale_adjoint',
        globals={'x': x, 'x_det':x_det})
    print(t.timeit(T))

    # check correctness
    print(x @ x_adj)

    x_inv = mat3x3_inv(x)
    print(x @ x_inv)
    T = 100
    t = benchmark.Timer(
        stmt='mat3x3_inv(x)',
        setup='from __main__ import mat3x3_inv',
        globals={'x': x})
    print(t.timeit(T))

    T = 100
    t = benchmark.Timer(
        stmt='_test_mat3x3_inv_backward(x)',
        setup='from __main__ import _test_mat3x3_inv_backward',
        globals={'x': x})
    print(t.timeit(T))


if __name__ == '__main__':
    _test()



