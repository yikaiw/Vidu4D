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


template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

template <typename scalar_t>
__global__ void kernel_quaternion_mul(
    const scalar_t * __restrict__ inputs_1, 
	const scalar_t * __restrict__ inputs_2, 
    scalar_t * outputs, 
    uint32_t B,
	uint32_t D1,
	uint32_t D2
) {
	const uint32_t b = threadIdx.x + blockIdx.x * blockDim.x;
	if (b >= B) return;

	// locate
	inputs_1 += b * D1;
	inputs_2 += b * D2;
	outputs += b * 4;

	scalar_t aw, ax, ay, az, bw, bx, by, bz = 0;
	if (D1 == 3) {
		aw = 0, ax = inputs_1[0], ay = inputs_1[1], az = inputs_1[2];
	} else {
		aw = inputs_1[0], ax = inputs_1[1], ay = inputs_1[2], az = inputs_1[3];
	}
	
	if (D2 == 3) {
		bw = 0, bx = inputs_2[0], by = inputs_2[1], bz = inputs_2[2];
	} else {
		bw = inputs_2[0], bx = inputs_2[1], by = inputs_2[2], bz = inputs_2[3];
	}

	outputs[0] = aw * bw - ax * bx - ay * by - az * bz;
    outputs[1] = aw * bx + ax * bw + ay * bz - az * by;
    outputs[2] = aw * by - ax * bz + ay * bw + az * bx;
    outputs[3] = aw * bz + ax * by - ay * bx + az * bw;

}


template <typename scalar_t>
__global__ void kernel_quaternion_mul_backward(
    const scalar_t * __restrict__ grad,
    uint32_t B,
	uint32_t D1,
	uint32_t D2,
    const scalar_t * __restrict__ inputs_1,
	const scalar_t * __restrict__ inputs_2,
    scalar_t * grad_inputs_1,
	scalar_t * grad_inputs_2
) {
	const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t b = t / 4;
	if (b >= B) return;

	const uint32_t d = t - b * 4;
	// locate
	grad += b * 4;
	inputs_1 += b * D1;
	inputs_2 += b * D2;

	scalar_t aw, ax, ay, az, bw, bx, by, bz = 0;
	if (D1 == 3) {
		aw = 0, ax = inputs_1[0], ay = inputs_1[1], az = inputs_1[2];
	} else {
		aw = inputs_1[0], ax = inputs_1[1], ay = inputs_1[2], az = inputs_1[3];
	}
	
	if (D2 == 3) {
		bw = 0, bx = inputs_2[0], by = inputs_2[1], bz = inputs_2[2];
	} else {
		bw = inputs_2[0], bx = inputs_2[1], by = inputs_2[2], bz = inputs_2[3];
	}

	grad_inputs_1 += D1 * b + d + D1 - 4;
	grad_inputs_2 += D2 * b + d + D2 - 4;
	if (d==0) {
		if (D1 > 3){
			grad_inputs_1[0] = grad[0] * bw + grad[1] * bx + grad[2] * by + grad[3] * bz;
		}
		if (D2 > 3){
			grad_inputs_2[0] = grad[0] * aw + grad[1] * ax + grad[2] * ay + grad[3] * az;
		}		
	} else if (d==1)
	{
		grad_inputs_1[0] = grad[0] * (-bx) + grad[1] * bw + grad[2] * (-bz) + grad[3] * by;
		grad_inputs_2[0] = grad[0] * (-ax) + grad[1] * aw + grad[2] * az + grad[3] * (-ay);
	} else if (d==2)
	{
		grad_inputs_1[0] = grad[0] * (-by) + grad[1] * bz + grad[2] * bw + grad[3] * (-bx);
		grad_inputs_2[0] = grad[0] * (-ay) + grad[1] * (-az) + grad[2] * aw + grad[3] * ax;
	} else 
	{
		grad_inputs_1[0] = grad[0] * (-bz) + grad[1] * (-by) + grad[2] * bx + grad[3] * bw;
		grad_inputs_2[0] = grad[0] * (-az) + grad[1] * ay + grad[2] * (-ax) + grad[3] * aw;
	}

	// if (d==0) {
	// 	grad_inputs_1[t] = grad[0] * bw + grad[1] * bx + grad[2] * by + grad[3] * bz;
	// 	grad_inputs_2[t] = grad[0] * aw + grad[1] * ax + grad[2] * ay + grad[3] * az;
	// } else if (d==1)
	// {
	// 	grad_inputs_1[t] = grad[0] * (-bx) + grad[1] * bw + grad[2] * (-bz) + grad[3] * by;
	// 	grad_inputs_2[t] = grad[0] * (-ax) + grad[1] * aw + grad[2] * az + grad[3] * (-ay);
	// } else if (d==2)
	// {
	// 	grad_inputs_1[t] = grad[0] * (-by) + grad[1] * bz + grad[2] * bw + grad[3] * (-bx);
	// 	grad_inputs_2[t] = grad[0] * (-ay) + grad[1] * (-az) + grad[2] * aw + grad[3] * ax;
	// } else 
	// {
	// 	grad_inputs_1[t] = grad[0] * (-bz) + grad[1] * (-by) + grad[2] * bx + grad[3] * bw;
	// 	grad_inputs_2[t] = grad[0] * (-az) + grad[1] * ay + grad[2] * (-ax) + grad[3] * aw;
	// }

}


template <typename scalar_t>
__global__ void kernel_quaternion_mul_backward_backward(
    const scalar_t * __restrict__ grad_out_1,
	const scalar_t * __restrict__ grad_out_2,
    uint32_t B,
	uint32_t D1,
	uint32_t D2,
	const scalar_t * __restrict__ grad,
	const scalar_t * __restrict__ inputs_1,
	const scalar_t * __restrict__ inputs_2,
	scalar_t * grad_grad,
    scalar_t * grad_grad_inputs_1,
	scalar_t * grad_grad_inputs_2
) {
	const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t b = t / 4;
	if (b >= B) return;

	const uint32_t d = t - b * 4;
	// locate
	grad += b * 4;
	inputs_1 += b*D1;
	inputs_2 += b*D2;
	grad_out_1 += b * D1;
	grad_out_2 += b * D2;
	
	scalar_t aw, ax, ay, az, bw, bx, by, bz = 0;
	scalar_t d_aw, d_ax, d_ay, d_az, d_bw, d_bx, d_by, d_bz = 0;
	if (D1 == 3) {
		aw = 0, ax = inputs_1[0], ay = inputs_1[1], az = inputs_1[2];
		d_aw = 0, d_ax = grad_out_1[0], d_ay = grad_out_1[1], d_az = grad_out_1[2];
	} else {
		aw = inputs_1[0], ax = inputs_1[1], ay = inputs_1[2], az = inputs_1[3];
		d_aw = grad_out_1[0], d_ax = grad_out_1[1], d_ay = grad_out_1[2], d_az = grad_out_1[3];
	}
	
	if (D2 == 3) {
		bw = 0, bx = inputs_2[0], by = inputs_2[1], bz = inputs_2[2];
		d_bw = 0, d_bx = grad_out_2[0], d_by = grad_out_2[1], d_bz = grad_out_2[2];
	} else {
		bw = inputs_2[0], bx = inputs_2[1], by = inputs_2[2], bz = inputs_2[3];
		d_bw = grad_out_2[0], d_bx = grad_out_2[1], d_by = grad_out_2[2], d_bz = grad_out_2[3];
	}

	grad_grad_inputs_1 += D1 * b + d + D1 - 4;
	grad_grad_inputs_2 += D2 * b + d + D2 - 4;

	if (d==0) {
		if (D1 > 3){
			grad_grad_inputs_1[0] = d_bw * grad[0] + d_bx * grad[1] + d_by * grad[2] + d_bz * grad[3];
		}
		if (D2 > 3){
			grad_grad_inputs_2[0] = d_aw * grad[0] + d_ax * grad[1] + d_ay * grad[2] + d_az * grad[3];
		}
		grad_grad[t] = d_aw * bw + d_bw * aw - d_ax * bx - d_bx * ax - d_ay * by - d_by * ay - d_az * bz - d_bz * az;
	}  else if (d==1){
		grad_grad_inputs_1[0] = d_bw * grad[1] - d_bx * grad[0] + d_by * grad[3] - d_bz * grad[2];
		grad_grad_inputs_2[0] = d_aw * grad[1] - d_ax * grad[0] - d_ay * grad[3] + d_az * grad[2];

		grad_grad[t] = d_aw * bx + d_bw * ax + d_ax * bw + d_bx * aw + d_ay * bz - d_by * az - d_az * by + d_bz * ay;
	}  else if (d==2){
		grad_grad_inputs_1[0] = d_bw * grad[2] - d_bx * grad[3] - d_by * grad[0] + d_bz * grad[1];
		grad_grad_inputs_2[0] = d_aw * grad[2] + d_ax * grad[3] - d_ay * grad[0] - d_az * grad[1];

		grad_grad[t] = d_aw * by + d_bw * ay - d_ax * bz + d_bx * az + d_ay * bw + d_by * aw + d_az * bx - d_bz * ax;
	}  else {
		grad_grad_inputs_1[0] = d_bw * grad[3] + d_bx * grad[2] - d_by * grad[1] - d_bz * grad[0];
		grad_grad_inputs_2[0] = d_aw * grad[3] - d_ax * grad[2] + d_ay * grad[1] - d_az * grad[0];

		grad_grad[t] = d_aw * bz + d_bw * az + d_ax * by - d_bx * ay - d_ay * bx + d_by * ax + d_az * bw + d_bz * aw;
	}
}

// template <typename scalar_t>
// inline scalar_t compute_sin_over_half_angle(scalar_t angle) {
// 	scalar_t half_angle = 0.5 * angle;

// } 

/*
template <typename scalar_t>
__global__ void kernel_axis_angle_to_quaternion_forward(
    const scalar_t * __restrict__ inputs,
    uint32_t B,
	scalar_t * angles,
	scalar_t * sin_over_half_angles,
	scalar_t * outputs
) {
	const uint32_t b = threadIdx.x + blockIdx.x * blockDim.x;
	if (b >= B) return;

	inputs += b*3;
	outputs += b*4;
	
	scalar_t angle = sqrt(inputs[0]**2 + inputs[1]**2 + inputs[2]**2);
	scalar_t half_angle = 0.5 * angle;
	outputs[0] = cos(half_angle);
	scalar_t sin_over_half_angle = 0
	if (angle > 1e-6) {
		scalar_t sin_over_half_angle = sin(half_angle) / angle;

	} else {
		scalar_t sin_over_half_angle = 0.5 - (angle**2) / 48.0

	}
	outputs[1] = inputs[0] * sin_over_half_angle;
	outputs[2] = inputs[1] * sin_over_half_angle;
	outputs[3] = inputs[2] * sin_over_half_angle;

	angles[b] = angle;
	sin_over_half_angles[b] = sin_over_half_angle;
}
*/
/*
template <typename scalar_t>
__global__ void kernel_axis_angle_to_quaternion_backward(
    const scalar_t * __restrict__ grad,
	const scalar_t * __restrict__ inputs,
	const scalar_t * __restrict__ angles,
	const scalar_t * __restrict__ sin_over_half_angles
    uint32_t B,
	scalar_t * grad_inputs
) {
	const uint32_t b = threadIdx.x + blockIdx.x * blockDim.x;
	if (b >= B) return;

	inputs += b*3;
	grad += b*4;
	
	scalar_t angle = angles[b];
	scalar_t half_angle = 0.5 * angle;
	scalar_t sin_over_half_angle = sin_over_half_angles[b];

	grad_inputs += 3*b;

	scalar_t dw_dx = -0.5 * sin_over_half_angle;

	scalar_t J = 0;
	if (angle > 1e-6) {
		J = 0.5 * cos(half_angle) - sin_over_half_angle;
		
 	}

}
*/

template <typename scalar_t>
__global__ void kernel_quaternion_conjugate(
	const scalar_t * __restrict__ inputs,
    uint32_t B,
	scalar_t * outputs
) {
	const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t b = t / 4;
	if (b >= B) return;
	uint32_t d = t - b * 4;
	if (d == 0) {
		outputs[t] = inputs[t];
	} else {
		outputs[t] = -inputs[t];
	} 
}

// template <typename scalar_t>
// __global__ void kernel_quaternion_conjugate_backward(
// 	const scalar_t * __restrict__ grad,
//     uint32_t B,
// 	scalar_t * grad_inputs
// ) {
// 	const uint32_t b = threadIdx.x + blockIdx.x * blockDim.x;
// 	if (b >= B) return;
// 	grad += 4*b;
// 	grad_inputs += 4*b;
// 	grad_inputs[0] = grad[0];
// 	grad_inputs[1] = -grad[1];
// 	grad_inputs[2] = -grad[2];
// 	grad_inputs[3] = -grad[3];
// }

// inputs: [B, D], float, in [0, 1]
// outputs: [B, L * C], float
template <typename scalar_t>
void quaternion_mul_forward_cuda(const scalar_t *inputs_1, const scalar_t *inputs_2, scalar_t *outputs, const uint32_t B, const uint32_t D1, const uint32_t D2) {
	static constexpr uint32_t N_THREADS = 256;
	kernel_quaternion_mul<scalar_t><<<div_round_up(B, N_THREADS), N_THREADS>>>(inputs_1, inputs_2, outputs, B, D1, D2);
}


template <typename scalar_t>
void quaternion_mul_backward_cuda(const scalar_t *grad, const uint32_t B, const uint32_t D1, const uint32_t D2, const scalar_t *inputs_1, const scalar_t *inputs_2, scalar_t *grad_inputs_1, scalar_t *grad_inputs_2) {
	static constexpr uint32_t N_THREADS = 256;
	kernel_quaternion_mul_backward<scalar_t><<<div_round_up(B * 4, N_THREADS), N_THREADS>>>(grad, B, D1, D2, inputs_1, inputs_2, grad_inputs_1, grad_inputs_2);
}

template <typename scalar_t>
void quaternion_mul_backward_backward_cuda(
	const scalar_t *grad_out_1, const scalar_t *grad_out_2, 
	const uint32_t B, const uint32_t D1, const uint32_t D2, 
	const scalar_t *grad, const scalar_t *inputs_1, const scalar_t *inputs_2, 
	scalar_t *grad_grad, scalar_t *grad_grad_inputs_1, scalar_t *grad_grad_inputs_2) {
	static constexpr uint32_t N_THREADS = 256;
	kernel_quaternion_mul_backward_backward<scalar_t><<<div_round_up(B * 4, N_THREADS), N_THREADS>>>(
		grad_out_1, grad_out_2, B, D1, D2, 
		grad, inputs_1, inputs_2,
		grad_grad, grad_grad_inputs_1, grad_grad_inputs_2);
}

template <typename scalar_t>
void quaternion_conjugate_cuda(
	const scalar_t *inputs,
	const uint32_t B,
	scalar_t *outputs
) {
	static constexpr uint32_t N_THREADS = 256;
	kernel_quaternion_conjugate<scalar_t><<<div_round_up(B*4, N_THREADS), N_THREADS>>>(inputs, B, outputs);
}








void quaternion_mul_forward(at::Tensor inputs_1, at::Tensor inputs_2, at::Tensor outputs, const uint32_t B, const uint32_t D1, const uint32_t D2) {
    CHECK_CUDA(inputs_1);
	CHECK_CUDA(inputs_2);
    CHECK_CUDA(outputs);

    
    CHECK_CONTIGUOUS(inputs_1);
	CHECK_CONTIGUOUS(inputs_2);
    CHECK_CONTIGUOUS(outputs);


    CHECK_IS_FLOATING(inputs_1);
	CHECK_IS_FLOATING(inputs_2);
    CHECK_IS_FLOATING(outputs);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    inputs_1.scalar_type(), "quaternion_mul_forward_cuda", ([&] {
		quaternion_mul_forward_cuda<scalar_t>(
			inputs_1.data_ptr<scalar_t>(),
			inputs_2.data_ptr<scalar_t>(),
			outputs.data_ptr<scalar_t>(), 
			B, D1, D2);
    }));	
}

void quaternion_mul_backward_backward(
	at::Tensor grad_out_1, at::Tensor grad_out_2, 
	const uint32_t B,  const uint32_t D1,  const uint32_t D2, 
	at::Tensor grad, at::Tensor inputs_1, at::Tensor inputs_2,
	at::Tensor grad_grad, at::Tensor grad_grad_inputs_1,  at::Tensor grad_grad_inputs_2) {    

    CHECK_CUDA(grad_out_1);
    CHECK_CUDA(grad_out_2);
	CHECK_CUDA(grad);
	CHECK_CUDA(inputs_1);
	CHECK_CUDA(inputs_2);
	CHECK_CUDA(grad_grad);
    CHECK_CUDA(grad_grad_inputs_1);
	CHECK_CUDA(grad_grad_inputs_2);

    CHECK_CONTIGUOUS(grad_out_1);
    CHECK_CONTIGUOUS(grad_out_2);
	CHECK_CONTIGUOUS(grad);
	CHECK_CONTIGUOUS(inputs_1);
	CHECK_CONTIGUOUS(inputs_2);
	CHECK_CONTIGUOUS(grad_grad);
    CHECK_CONTIGUOUS(grad_grad_inputs_1);
	CHECK_CONTIGUOUS(grad_grad_inputs_2);

    CHECK_IS_FLOATING(grad_out_1);
    CHECK_IS_FLOATING(grad_out_2);
	CHECK_IS_FLOATING(grad);
	CHECK_IS_FLOATING(inputs_1);
	CHECK_IS_FLOATING(inputs_2);
	CHECK_IS_FLOATING(grad_grad);
    CHECK_IS_FLOATING(grad_grad_inputs_1);
	CHECK_IS_FLOATING(grad_grad_inputs_2);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad.scalar_type(), "quaternion_mul_backward_backward_cuda", ([&] {
    	quaternion_mul_backward_backward_cuda<scalar_t>(
			grad_out_1.data_ptr<scalar_t>(),
			grad_out_2.data_ptr<scalar_t>(),
			B, D1, D2,
			grad.data_ptr<scalar_t>(),
			inputs_1.data_ptr<scalar_t>(),
			inputs_2.data_ptr<scalar_t>(),
			grad_grad.data_ptr<scalar_t>(),
			grad_grad_inputs_1.data_ptr<scalar_t>(),
			grad_grad_inputs_2.data_ptr<scalar_t>());
    }));	
}



void quaternion_mul_backward(at::Tensor grad, const uint32_t B,  const uint32_t D1,  const uint32_t D2, at::Tensor inputs_1, at::Tensor inputs_2,  at::Tensor grad_inputs_1, at::Tensor grad_inputs_2) {    
    CHECK_CUDA(grad);
    CHECK_CUDA(inputs_1);
	CHECK_CUDA(inputs_2);
    CHECK_CUDA(grad_inputs_1);
	CHECK_CUDA(grad_inputs_2);

    CHECK_CONTIGUOUS(grad);
    CHECK_CONTIGUOUS(inputs_1);
	CHECK_CONTIGUOUS(inputs_2);
    CHECK_CONTIGUOUS(grad_inputs_1);
	CHECK_CONTIGUOUS(grad_inputs_2);

    CHECK_IS_FLOATING(grad);
    CHECK_IS_FLOATING(inputs_1);
	CHECK_IS_FLOATING(inputs_2);
    CHECK_IS_FLOATING(grad_inputs_1);
	CHECK_IS_FLOATING(grad_inputs_2);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad.scalar_type(), "quaternion_mul_backward_cuda", ([&] {
    	quaternion_mul_backward_cuda<scalar_t>(
			grad.data_ptr<scalar_t>(),
			B, D1, D2,
			inputs_1.data_ptr<scalar_t>(), 
			inputs_2.data_ptr<scalar_t>(), 
			grad_inputs_1.data_ptr<scalar_t>(),
			grad_inputs_2.data_ptr<scalar_t>());
    }));	
}


void quaternion_conjugate(at::Tensor inputs, const uint32_t B, at::Tensor outputs) {
	CHECK_IS_CONTIGUOUS_FLOAT_CUDA(inputs);
	CHECK_IS_CONTIGUOUS_FLOAT_CUDA(outputs);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    inputs.scalar_type(), "quaternion_conjugate_cuda", ([&] {
    	quaternion_conjugate_cuda<scalar_t>(
			inputs.data_ptr<scalar_t>(),
			B,
			outputs.data_ptr<scalar_t>());
    }));	

}