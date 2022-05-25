#include "sphere.hpp"

namespace obj
{
    Sphere::Sphere(sycl::float3 center, float radius, Color color)
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

    Color Sphere::get_color() const
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
                result.normal = normalize(hit_point - _center);
                result.color = _color;
                return true;
            }

            t = (-half_b + sqrtf(discriminant) / a);
            hit_point = ray.point_at(t);
            if(t >= t_min && t <= t_max)
            {
                result.t = t;
                result.hit_point = hit_point;
                result.normal = normalize(hit_point - _center);
                result.color = _color;
                return true;
            }
        }

        return false;
    }

    // bool Sphere::hit(const Ray& ray, float t_min, float t_max, HitResult& result) const
    // {
    //     float radius2 = _radius * _radius;

    //     Vec3 L = _center - ray.get_origin();
    //     float tca = dot(L, ray.get_direction());
    //     if(tca < 0) return false;

    //     float d2 = dot(L, L) - tca * tca;
    //     if(d2 > radius2) return false;

    //     float thc = sycl::sqrt(radius2 - d2);
    //     float t0 = tca - thc;
    //     float t1 = tca + thc;

    //     if(t0 > t1)
    //     {
    //         float tmp = t0;
    //         t0 = t1;
    //         t1 = tmp;
    //     }

    //     if(t0 < 0)
    //     {
    //         t0 = t1;
    //         if (t0 < 0) return false;
    //     }

    //     if(t0 < t_min || t0 > t_max) return false;

    //     result.t = t0;
    //     result.color = _color;
    //     result.hit_point = ray.point_at(t0);
    //     result.normal = normalize(result.hit_point - _center);
    //     return true;
    // }
}
