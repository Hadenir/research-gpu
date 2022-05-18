#pragma once

#include <sycl/sycl.hpp>

#include "../ray.hpp"

namespace sycl = cl::sycl;

namespace gfx
{
    class Camera
    {
    public:
        Camera(float distance, sycl::float3 look_at, float fov, float aspect_ratio);

        SYCL_EXTERNAL sycl::float3 get_position() const;

        SYCL_EXTERNAL Ray calculate_ray(float u, float v) const;

    private:
        sycl::float3 _origin;
        sycl::float3 _target;
        float _x_angle = 0;
        float _y_angle = 0;
        float _distance;

        float _theta;
        float _aspect_ratio;

        sycl::float3 _lower_left;
        sycl::float3 _horizontal;
        sycl::float3 _vertical;

        void update();
    };
}
