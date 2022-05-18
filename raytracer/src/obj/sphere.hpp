#pragma once

#include <sycl/sycl.hpp>

#include "../ray.hpp"

namespace sycl = cl::sycl;

namespace obj
{
    class Sphere
    {
    public:
        Sphere(sycl::float3 center, float radius, sycl::float4 color);

        sycl::float3 get_center() const;
        float get_radius() const;
        sycl::float4 get_color() const;

        SYCL_EXTERNAL bool hit(const Ray& ray, float t_min, float t_max, HitResult& result) const;

    private:
        sycl::float3 _center;
        float _radius;
        sycl::float4 _color;
    };
}
