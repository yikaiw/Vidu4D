# pragma once

#include <stdint.h>
#include <torch/torch.h>

void mat3x3_det_forward(at::Tensor inputs, at::Tensor outputs,const uint32_t B);
void mat3x3_scale_adjoint_forward(at::Tensor inputs, at::Tensor scales, at::Tensor outputs, const uint32_t B);
void mat3x3_inv_forward(at::Tensor inputs, at::Tensor outputs, at::Tensor output_scales, const uint32_t B);

void mat3x3_inv_backward(at::Tensor grad, at::Tensor inv_mats, at::Tensor grad_inputs, const uint32_t B);