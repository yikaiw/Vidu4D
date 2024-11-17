# pragma once

#include <stdint.h>
#include <torch/torch.h>

// inputs: [B, D], float, in [-1, 1]
// outputs: [B, F], float

// encode_forward(inputs, outputs, B, input_dim, degree, calc_grad_inputs, dy_dx)
void quaternion_mul_forward(at::Tensor inputs_1, at::Tensor inputs_2, at::Tensor outputs, const uint32_t B, const uint32_t D1, const uint32_t D2);

// sh_encode_backward(grad, inputs, B, input_dim, degree, ctx.calc_grad_inputs, dy_dx, grad_inputs)
void quaternion_mul_backward(at::Tensor grad, const uint32_t B, const uint32_t D1, const uint32_t D2, at::Tensor inputs_1, at::Tensor inputs_2,  at::Tensor grad_inputs_1, at::Tensor grad_inputs_2);


void quaternion_mul_backward_backward(
    at::Tensor grad_out_1, at::Tensor grad_out_2, 
    const uint32_t B, const uint32_t D1, const uint32_t D2, 
    at::Tensor grad, at::Tensor inputs_1, at::Tensor inputs_2, 
    at::Tensor grad_grad, at::Tensor grad_grad_inputs_1, at::Tensor grad_grad_inputs_2);


void quaternion_conjugate(at::Tensor inputs, const uint32_t B, at::Tensor outputs);