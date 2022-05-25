#pragma once

#include <sycl/sycl.hpp>

#include "color.hpp"

namespace sycl = cl::sycl;

class Ray
{
public:
    SYCL_EXTERNAL Ray() {}
    SYCL_EXTERNAL Ray(sycl::float3 origin, sycl::float3 direction);

    SYCL_EXTERNAL sycl::float3 get_origin() const;
    SYCL_EXTERNAL sycl::float3 get_direction() const;

    SYCL_EXTERNAL sycl::float3 point_at(float t) const;

private:
    sycl::float3 _origin;
    sycl::float3 _direction;
};

struct HitResult
{
    float t;
    sycl::float3 hit_point;
    sycl::float3 normal;
    Color color;
};

SYCL_EXTERNAL float length_sq(sycl::float3 vec);
SYCL_EXTERNAL float dot(sycl::float3 v1, sycl::float3 v2);
SYCL_EXTERNAL sycl::float3 cross(sycl::float3 v1, sycl::float3 v2);
SYCL_EXTERNAL sycl::float3 normalize(sycl::float3 vec);
// SYCL_EXTERNAL sycl::float4 operator*(sycl::float4 c1, sycl::float4 c2);
float to_radians(float degrees);
