#include "ray.hpp"

Ray::Ray(sycl::float3 origin, sycl::float3 direction)
    : _origin(origin), _direction(direction)
{}

sycl::float3 Ray::get_origin() const
{
    return _origin;
}

sycl::float3 Ray::get_direction() const
{
    return _direction;
}

sycl::float3 Ray::point_at(float t) const
{
    return _origin + t * _direction;
}

float length_sq(sycl::float3 vec)
{
    return vec.x() * vec.x() + vec.y() * vec.y() + vec.z() * vec.z();
}

float dot(sycl::float3 v1, sycl::float3 v2)
{
    return v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z();
}

sycl::float3 cross(sycl::float3 v1, sycl::float3 v2)
{
    return {
        v1.y() * v2.z() - v1.z() * v2.y(),
        v1.z() * v2.x() - v1.x() * v2.z(),
        v1.x() * v2.y() - v1.y() * v2.x()
    };
}

sycl::float3 normalize(sycl::float3 vec)
{
    float len = sqrtf(length_sq(vec));
    if (len == 0) return vec;

    return {vec.x() / len, vec.y() / len, vec.z() / len};
}

sycl::float4 operator*(sycl::float4 c1, sycl::float4 c2)
{
    return {c1.x() * c2.x(), c1.y() * c2.y(), c1.z() * c2.z(), c1.w() * c2.w()};
}

float to_radians(float degrees)
{
    return degrees / 180.f * CL_M_PI;
}
