#include <torch/extension.h>

#include "quaternion.h"
#include "matinv.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quaternion_mul_forward", &quaternion_mul_forward, "quaternion multiplication forward (CUDA)");
    m.def("quaternion_mul_backward", &quaternion_mul_backward, "quaternion multiplication backward (CUDA)");
    m.def("quaternion_mul_backward_backward", &quaternion_mul_backward_backward, "quaternion multiplication backward (CUDA)");
    m.def("quaternion_conjugate", &quaternion_conjugate, "quaternion_conjugate (CUDA)");
    // mat3x3 inverse
    m.def("mat3x3_det_forward", &mat3x3_det_forward, "mat3x3_det_forward (CUDA)");
    m.def("mat3x3_scale_adjoint_forward", &mat3x3_scale_adjoint_forward, "mat3x3_scale_adjoint_forward (CUDA)");
    m.def("mat3x3_inv_forward", &mat3x3_inv_forward, "mat3x3_inv_forward (CUDA)");
    m.def("mat3x3_inv_backward", &mat3x3_inv_backward, "mat3x3_inv_backward (CUDA)");
}