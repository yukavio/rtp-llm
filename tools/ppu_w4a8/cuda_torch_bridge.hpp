#pragma once
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAFunctions.h>
#include <hggc_runtime_api.h>
#include <acblasLt.h>
#include <map>
namespace dg_cuda_torch {
inline const hggcDeviceProp* getDeviceProperties(int device) {
    static thread_local std::map<int, hggcDeviceProp> props;
    auto it = props.find(device);
    if (it == props.end()) {
        hggcDeviceProp prop;
        TORCH_CHECK(hggcGetDeviceProperties(&prop, device) == hggcSuccess);
        it = props.emplace(device, prop).first;
    }
    return &it->second;
}
inline const hggcDeviceProp* getCurrentDeviceProperties() {
    return getDeviceProperties(c10::cuda::current_device());
}
inline hggcDataType scalarType(at::ScalarType type) {
    switch (type) {
        case at::kFloat: return HGGC_R_32F;
        case at::kHalf: return HGGC_R_16F;
        case at::kBFloat16: return HGGC_R_16BF;
        case at::kChar: return HGGC_R_8I;
        case at::kInt: return HGGC_R_32I;
        default: TORCH_CHECK(false, "Unsupported native acBLAS scalar type");
    }
}
}
