#include <stdint.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <algorithm>
#include <stdexcept>

#include <cstdio>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")

#define CHECK_IS_CONTIGUOUS_FLOAT_CUDA(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_IS_FLOATING(x)

#define I00 0
#define I01 1
#define I02 2

#define I10 3
#define I11 4
#define I12 5

#define I20 6
#define I21 7
#define I22 8


template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

template <typename scalar_t>
__global__ void kernel_mat3x3_det(
    const scalar_t * __restrict__ inputs, 
    scalar_t * outputs, 
    uint32_t B
) {
	const uint32_t b = threadIdx.x + blockIdx.x * blockDim.x;
	if (b >= B) return;

    inputs += b*9;
    scalar_t x00 = inputs[0], x01 = inputs[1], x02 = inputs[2];
    scalar_t x10 = inputs[3], x11 = inputs[4], x12 = inputs[5];
    scalar_t x20 = inputs[6], x21 = inputs[7], x22 = inputs[8];

    outputs[b] = x00 * x11 * x22 
            + x10 * x21 * x02
            + x20 * x12 * x01
            - x02 * x11 * x20
            - x12 * x21 * x00
            - x22 * x10 * x01;
}
/*
SCALE_ADJOINT_3X3
#define SCALE_ADJOINT_3X3(a,s,m)				\
{								\
    a[0][0] = (s) * (m[1][1] * m[2][2] - m[1][2] * m[2][1]);	\
    a[0][1] = (s) * (m[0][2] * m[2][1] - m[0][1] * m[2][2]);	\
    a[0][2] = (s) * (m[0][1] * m[1][2] - m[0][2] * m[1][1]);	\

    a[1][0] = (s) * (m[1][2] * m[2][0] - m[1][0] * m[2][2]);	\
    a[1][1] = (s) * (m[0][0] * m[2][2] - m[0][2] * m[2][0]);	\
    a[1][2] = (s) * (m[0][2] * m[1][0] - m[0][0] * m[1][2]);	\
    
    a[2][0] = (s) * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);	\
	a[2][1] = (s) * (m[0][1] * m[2][0] - m[0][0] * m[2][1]);	\
	a[2][2] = (s) * (m[0][0] * m[1][1] - m[0][1] * m[1][0]);	\
}
*/
template <typename scalar_t>
__global__ void kernel_mat3x3_scale_adjoint(
    const scalar_t * __restrict__ inputs,
    const scalar_t * __restrict__ scales,  
    scalar_t * outputs, 
    uint32_t B
) {
	const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t b = t / 9;
	if (b >= B) return;

    // locate
    inputs += 9*b;

    const uint32_t d = t - b*9;
    const scalar_t s = 1 / scales[b];

    if (d==0) {
        outputs[t] = s * (inputs[4] * inputs[8] - inputs[5] * inputs[7]);
    } else if (d==1) {
        outputs[t] = s * (inputs[2] * inputs[7] - inputs[1] * inputs[8]);
    } else if (d==2) {
        outputs[t] = s * (inputs[1] * inputs[5] - inputs[2] * inputs[4]);
    } else if (d==3) {
        outputs[t] = s * (inputs[5] * inputs[6] - inputs[3] * inputs[8]);
    } else if (d==4) {
        outputs[t] = s * (inputs[0] * inputs[8] - inputs[2] * inputs[6]);
    } else if (d==5) {
        outputs[t] = s * (inputs[2] * inputs[3] - inputs[0] * inputs[5]);
    } else if (d==6) {
        outputs[t] = s * (inputs[3] * inputs[7] - inputs[4] * inputs[6]);
    } else if (d==7) {
        outputs[t] = s * (inputs[1] * inputs[6] - inputs[0] * inputs[7]);
    } else  {
        outputs[t] = s * (inputs[0] * inputs[4] - inputs[1] * inputs[3]);
    } 
}


template <typename scalar_t>
__global__ void kernel_mat3x3_inv_backward(
    const scalar_t * __restrict__ grad,
    const scalar_t * __restrict__ inv_mats,
    scalar_t * grad_inputs, 
    uint32_t B
) {
	const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t b = t / 9;
	if (b >= B) return;

    // locate
    inv_mats += 9*b;
    grad += 9*b;
    const uint32_t d = t - b*9;
    
    // H d_ij H = [0 ... H[:,i],...]H = H[:,i] @ H[j, :]
    // dy_dx_ij = sum((H[:,i] @ H[j, :])*grad)

    if (d==I00) {
        grad_inputs[t] = -(
                grad[I00] * inv_mats[I00] * inv_mats[I00]
            +   grad[I01] * inv_mats[I00] * inv_mats[I01]
            +   grad[I02] * inv_mats[I00] * inv_mats[I02]
            +   grad[I10] * inv_mats[I10] * inv_mats[I00]
            +   grad[I11] * inv_mats[I10] * inv_mats[I01]
            +   grad[I12] * inv_mats[I10] * inv_mats[I02]
            +   grad[I20] * inv_mats[I20] * inv_mats[I00]
            +   grad[I21] * inv_mats[I20] * inv_mats[I01]
            +   grad[I22] * inv_mats[I20] * inv_mats[I02]);   
    } else if (d==I01) {
        grad_inputs[t] = -(
                grad[I00] * inv_mats[I00] * inv_mats[I10]
            +   grad[I01] * inv_mats[I00] * inv_mats[I11]
            +   grad[I02] * inv_mats[I00] * inv_mats[I12]
            +   grad[I10] * inv_mats[I10] * inv_mats[I10]
            +   grad[I11] * inv_mats[I10] * inv_mats[I11]
            +   grad[I12] * inv_mats[I10] * inv_mats[I12]
            +   grad[I20] * inv_mats[I20] * inv_mats[I10]
            +   grad[I21] * inv_mats[I20] * inv_mats[I11]
            +   grad[I22] * inv_mats[I20] * inv_mats[I12]);        
    } else if (d==I02) {
            grad_inputs[t] = -(
                grad[I00] * inv_mats[I00] * inv_mats[I20]
            +   grad[I01] * inv_mats[I00] * inv_mats[I21]
            +   grad[I02] * inv_mats[I00] * inv_mats[I22]
            +   grad[I10] * inv_mats[I10] * inv_mats[I20]
            +   grad[I11] * inv_mats[I10] * inv_mats[I21]
            +   grad[I12] * inv_mats[I10] * inv_mats[I22]
            +   grad[I20] * inv_mats[I20] * inv_mats[I20]
            +   grad[I21] * inv_mats[I20] * inv_mats[I21]
            +   grad[I22] * inv_mats[I20] * inv_mats[I22]);  
    } else if (d==I10) {        
            grad_inputs[t] = -(
                grad[I00] * inv_mats[I01] * inv_mats[I00]
            +   grad[I01] * inv_mats[I01] * inv_mats[I01]
            +   grad[I02] * inv_mats[I01] * inv_mats[I02]
            +   grad[I10] * inv_mats[I11] * inv_mats[I00]
            +   grad[I11] * inv_mats[I11] * inv_mats[I01]
            +   grad[I12] * inv_mats[I11] * inv_mats[I02]
            +   grad[I20] * inv_mats[I21] * inv_mats[I00]
            +   grad[I21] * inv_mats[I21] * inv_mats[I01]
            +   grad[I22] * inv_mats[I21] * inv_mats[I02]);
    } else if (d==I11) {        
            grad_inputs[t] = -(
                grad[I00] * inv_mats[I01] * inv_mats[I10]
            +   grad[I01] * inv_mats[I01] * inv_mats[I11]
            +   grad[I02] * inv_mats[I01] * inv_mats[I12]
            +   grad[I10] * inv_mats[I11] * inv_mats[I10]
            +   grad[I11] * inv_mats[I11] * inv_mats[I11]
            +   grad[I12] * inv_mats[I11] * inv_mats[I12]
            +   grad[I20] * inv_mats[I21] * inv_mats[I10]
            +   grad[I21] * inv_mats[I21] * inv_mats[I11]
            +   grad[I22] * inv_mats[I21] * inv_mats[I12]);
    } else if (d==I12) {        
            grad_inputs[t] = -(
                grad[I00] * inv_mats[I01] * inv_mats[I20]
            +   grad[I01] * inv_mats[I01] * inv_mats[I21]
            +   grad[I02] * inv_mats[I01] * inv_mats[I22]
            +   grad[I10] * inv_mats[I11] * inv_mats[I20]
            +   grad[I11] * inv_mats[I11] * inv_mats[I21]
            +   grad[I12] * inv_mats[I11] * inv_mats[I22]
            +   grad[I20] * inv_mats[I21] * inv_mats[I20]
            +   grad[I21] * inv_mats[I21] * inv_mats[I21]
            +   grad[I22] * inv_mats[I21] * inv_mats[I22]);
    }  else if (d==I20) {        
            grad_inputs[t] = -(
                grad[I00] * inv_mats[I02] * inv_mats[I00]
            +   grad[I01] * inv_mats[I02] * inv_mats[I01]
            +   grad[I02] * inv_mats[I02] * inv_mats[I02]
            +   grad[I10] * inv_mats[I12] * inv_mats[I00]
            +   grad[I11] * inv_mats[I12] * inv_mats[I01]
            +   grad[I12] * inv_mats[I12] * inv_mats[I02]
            +   grad[I20] * inv_mats[I22] * inv_mats[I00]
            +   grad[I21] * inv_mats[I22] * inv_mats[I01]
            +   grad[I22] * inv_mats[I22] * inv_mats[I02]);
    } else if (d==I21) {        
            grad_inputs[t] = -(
                grad[I00] * inv_mats[I02] * inv_mats[I10]
            +   grad[I01] * inv_mats[I02] * inv_mats[I11]
            +   grad[I02] * inv_mats[I02] * inv_mats[I12]
            +   grad[I10] * inv_mats[I12] * inv_mats[I10]
            +   grad[I11] * inv_mats[I12] * inv_mats[I11]
            +   grad[I12] * inv_mats[I12] * inv_mats[I12]
            +   grad[I20] * inv_mats[I22] * inv_mats[I10]
            +   grad[I21] * inv_mats[I22] * inv_mats[I11]
            +   grad[I22] * inv_mats[I22] * inv_mats[I12]);
    } else {        
            grad_inputs[t] = -(
                grad[I00] * inv_mats[I02] * inv_mats[I20]
            +   grad[I01] * inv_mats[I02] * inv_mats[I21]
            +   grad[I02] * inv_mats[I02] * inv_mats[I22]
            +   grad[I10] * inv_mats[I12] * inv_mats[I20]
            +   grad[I11] * inv_mats[I12] * inv_mats[I21]
            +   grad[I12] * inv_mats[I12] * inv_mats[I22]
            +   grad[I20] * inv_mats[I22] * inv_mats[I20]
            +   grad[I21] * inv_mats[I22] * inv_mats[I21]
            +   grad[I22] * inv_mats[I22] * inv_mats[I22]);
    } 
}



template <typename scalar_t>
void mat3x3_det_forward_cuda(const scalar_t *inputs, scalar_t *outputs, const uint32_t B) {
    static constexpr uint32_t N_THREADS = 256;
   	kernel_mat3x3_det<scalar_t><<<div_round_up(B, N_THREADS), N_THREADS>>>(inputs, outputs, B); 
}


template <typename scalar_t>
void mat3x3_scale_adjoint_forward_cuda(const scalar_t *inputs, const scalar_t *scales, scalar_t *outputs, const uint32_t B) {
    static constexpr uint32_t N_THREADS = 256;
    kernel_mat3x3_scale_adjoint<scalar_t><<<div_round_up(B*9, N_THREADS), N_THREADS>>>(inputs, scales, outputs, B); 
}


template <typename scalar_t>
void mat3x3_inv_forward_cuda(const scalar_t *inputs, scalar_t *outputs, scalar_t *output_scales, const uint32_t B) {
    static constexpr uint32_t N_THREADS = 256;
   	kernel_mat3x3_det<scalar_t><<<div_round_up(B, N_THREADS), N_THREADS>>>(inputs, output_scales, B); 
    kernel_mat3x3_scale_adjoint<scalar_t><<<div_round_up(B*9, N_THREADS), N_THREADS>>>(inputs, output_scales, outputs, B); 
}

template <typename scalar_t>
void mat3x3_inv_backward_cuda(const scalar_t * __restrict__ grad, const scalar_t * __restrict__ inv_mats, scalar_t * grad_inputs, uint32_t B) {
    static constexpr uint32_t N_THREADS = 256;
    kernel_mat3x3_inv_backward<scalar_t><<<div_round_up(B*9, N_THREADS), N_THREADS>>>(grad, inv_mats, grad_inputs, B);
}


void mat3x3_det_forward(at::Tensor inputs, at::Tensor outputs,const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(inputs);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(outputs);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    inputs.scalar_type(), "mat3x3_det_forward_cuda", ([&] {
		mat3x3_det_forward_cuda<scalar_t>(
			inputs.data_ptr<scalar_t>(),
			outputs.data_ptr<scalar_t>(), 
			B);
    }));	
} 

void mat3x3_scale_adjoint_forward(at::Tensor inputs, at::Tensor scales, at::Tensor outputs, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(inputs);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(scales);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(outputs);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    inputs.scalar_type(), "mat3x3_scale_adjoint_forward_cuda", ([&] {
		mat3x3_scale_adjoint_forward_cuda<scalar_t>(
			inputs.data_ptr<scalar_t>(),
            scales.data_ptr<scalar_t>(),
			outputs.data_ptr<scalar_t>(), 
			B);
    }));	
} 


void mat3x3_inv_forward(at::Tensor inputs, at::Tensor outputs, at::Tensor output_scales, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(inputs);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(outputs);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(output_scales);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    inputs.scalar_type(), "mat3x3_inv_forward_cuda", ([&] {
		mat3x3_inv_forward_cuda<scalar_t>(
			inputs.data_ptr<scalar_t>(),
			outputs.data_ptr<scalar_t>(), 
            output_scales.data_ptr<scalar_t>(), 
			B);
    }));	
} 

void mat3x3_inv_backward(at::Tensor grad, at::Tensor inv_mats, at::Tensor grad_inputs, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(inv_mats);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(grad_inputs);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    inv_mats.scalar_type(), "mat3x3_inv_backward_cuda", ([&] {
		mat3x3_inv_backward_cuda<scalar_t>(
			grad.data_ptr<scalar_t>(),
			inv_mats.data_ptr<scalar_t>(), 
            grad_inputs.data_ptr<scalar_t>(), 
			B);
    }));	
} 






