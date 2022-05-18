#include "sphere.hpp"

namespace obj
{
    Sphere::Sphere(sycl::float3 center, float radius, sycl::float4 color)
        : _center(center), _radius(radius), _color(color)
    {}

    sycl::float3 Sphere::get_center() const
    {
        return _center;
    }

    float Sphere::get_radius() const
    {
        return _radius;
    }

    sycl::float4 Sphere::get_color() const
    {
        return _color;
    }

    bool Sphere::hit(const Ray& ray, float t_min, float t_max, HitResult& result) const
    {
        sycl::float3 oc = ray.get_origin() - _center;
        float a = length_sq(ray.get_direction());
        float half_b = dot(oc, ray.get_direction());
        float c = length_sq(oc) - _radius * _radius;

        float discriminant = half_b * half_b - a * c;
        if(discriminant > 0)
        {
            float t = (-half_b - sqrtf(discriminant) / a);
            sycl::float3 hit_point = ray.point_at(t);
            if(t >= t_min && t <= t_max)
            {
                result.t = t;
                result.hit_point = hit_point;
                result.normal = (hit_point - _center) / _radius;
                result.color = _color;
                return true;
            }

            t = (-half_b + sqrtf(discriminant) / a);
            hit_point = ray.point_at(t);
            if(t >= t_min && t <= t_max)
            {
                result.t = t;
                result.hit_point = hit_point;
                result.normal = (hit_point - _center) / _radius;
                result.color = _color;
                return true;
            }
        }

        return false;
    }
}
