#pragma once

#include <sycl/sycl.hpp>

#include "../vec3.hpp"
#include "../ray.hpp"
#include "../color.hpp"

namespace sycl = cl::sycl;

namespace obj
{
    class Sphere
    {
    public:
        Sphere(sycl::float3 center, float radius, Color color);

        sycl::float3 get_center() const;
        float get_radius() const;
        Color get_color() const;

        SYCL_EXTERNAL bool hit(const Ray& ray, float t_min, float t_max, HitResult& result) const;

    private:
        sycl::float3 _center;
        float _radius;
        Color _color;
    };
}
