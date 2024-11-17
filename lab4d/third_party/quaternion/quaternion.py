import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    import _quaternion as _backend
except ImportError:
    from .backend import _backend

class _Quaternion_mul_backward(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, grad, inputs_1, inputs_2):
        B = inputs_1.shape[0] # batch size, coord dim
        D1 = inputs_1.shape[1]
        D2 = inputs_2.shape[1]  
        dtype, device = inputs_1.dtype, inputs_1.device
        grad_inputs_1 = torch.empty(B, D1, device=device, dtype=dtype)
        grad_inputs_2 = torch.empty(B, D2, device=device, dtype=dtype)
        _backend.quaternion_mul_backward(grad, B, D1, D2, inputs_1, inputs_2, grad_inputs_1, grad_inputs_2)
        ctx.save_for_backward(grad, inputs_1, inputs_2)
        return grad_inputs_1, grad_inputs_2  

    @staticmethod
    @once_differentiable
    @custom_bwd  
    def backward(ctx, *grad_outputs):
        grad_out_1, grad_out_2 = grad_outputs
        grad, inputs_1, inputs_2 = ctx.saved_tensors
        B = inputs_1.shape[0] # batch size, coord dim
        D1 = inputs_1.shape[1]
        D2 = inputs_2.shape[1]  
        dtype, device = inputs_1.dtype, inputs_1.device
        grad_grad = torch.empty(B, 4, device=device, dtype=dtype)
        grad_grad_inputs_1 = torch.empty(B, D1, device=device, dtype=dtype)
        grad_grad_inputs_2 = torch.empty(B, D2, device=device, dtype=dtype)
        _backend.quaternion_mul_backward_backward(grad_out_1, grad_out_2,
            B, D1, D2,
            grad, inputs_1, inputs_2,
            grad_grad, grad_grad_inputs_1, grad_grad_inputs_2)
        return grad_grad, grad_grad_inputs_1, grad_grad_inputs_2

_quaternion_mul_backward = _Quaternion_mul_backward.apply

class _Quaternion_mul(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, inputs_1:torch.Tensor, inputs_2:torch.Tensor):
        # inputs: [B, input_dim], float in [-1, 1]
        # RETURN: [B, F], float
        calc_grad_inputs = inputs_1.requires_grad or inputs_2.requires_grad

        inputs_1 = inputs_1.contiguous()
        inputs_2 = inputs_2.contiguous()
        
        B = inputs_1.shape[0] # batch size, coord dim
        D1 = inputs_1.shape[1]
        D2 = inputs_2.shape[1]

        dtype = inputs_1.dtype
        device = inputs_1.device
        
        outputs = torch.empty(B, 4, dtype=dtype, device=device)


        _backend.quaternion_mul_forward(inputs_1, inputs_2, outputs, B, D1, D2)

        ctx.save_for_backward(inputs_1, inputs_2)
        # ctx.dims = [B, D1, D2]
        # ctx.calc_grad_inputs = calc_grad_inputs

        return outputs
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        # grad: [B, C * C]

        # if ctx.calc_grad_inputs:
        grad = grad.contiguous()
        inputs_1, inputs_2 = ctx.saved_tensors

        grad_inputs_1, grad_inputs_2 = _quaternion_mul_backward(grad, inputs_1, inputs_2)

        # B = inputs_1.shape[0] # batch size, coord dim
        # D1 = inputs_1.shape[1]
        # D2 = inputs_2.shape[1]

        # dtype, device = inputs_1.dtype, inputs_1.device
        # grad_inputs_1 = torch.empty(B, D1, device=device, dtype=dtype)
        # grad_inputs_2 = torch.empty(B, D2, device=device, dtype=dtype)
        # _backend.quaternion_mul_backward(grad, B, D1, D2, inputs_1, inputs_2, grad_inputs_1, grad_inputs_2)
        # # print('inputs_1', inputs_1)
        # # print('dy_dx1', dy_dx1)
        # # print('grad_inputs_1', grad_inputs_1)
        return grad_inputs_1, grad_inputs_2
        # else:
        #     return None, None



quaternion_mul = _Quaternion_mul.apply


class _Quaternion_conjugate(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, inputs:torch.Tensor):
        B = inputs.shape[0] # batch size, coord dim
        outputs = torch.empty_like(inputs)
        _backend.quaternion_conjugate(inputs.contiguous(), B, outputs)
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        return _Quaternion_conjugate.apply(grad)


quaternion_conjugate = _Quaternion_conjugate.apply

