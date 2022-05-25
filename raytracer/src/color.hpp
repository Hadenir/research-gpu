#pragma once

#include <sycl/sycl.hpp>

using Color = cl::sycl::float4;

SYCL_EXTERNAL Color clamp(Color color);
