#pragma once

#include <iostream>
#include <sycl/sycl.hpp>

namespace sycl = cl::sycl;

class CudaSelector : public sycl::device_selector
{
public:
    int operator()(const sycl::device& device) const override
    {
        const std::string driver_version = device.get_info<sycl::info::device::driver_version>();

        if(device.is_gpu() && driver_version.find("CUDA") != std::string::npos)
            return sycl::gpu_selector()(device);

        return -1;
    }
};