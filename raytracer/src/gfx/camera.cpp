#include "camera.hpp"

namespace gfx
{
    Camera::Camera(float distance, sycl::float3 look_at, float fov, float aspect_ratio)
        : _target(look_at), _aspect_ratio(aspect_ratio), _distance(distance)
    {
        _theta = to_radians(fov);
        update();
    }

    sycl::float3 Camera::get_position() const
    {
        return _origin;
    }

    Ray Camera::calculate_ray(float u, float v) const
    {
        return {
            _origin,
            _lower_left + u * _horizontal + v * _vertical - _origin
        };
    }

    void Camera::update()
    {
        float h = tan(_theta / 2);
        float viewport_height = 2 * h;
        float viewport_width = _aspect_ratio * viewport_height;

        _origin = _target + sycl::float3(_distance * sin(_x_angle) * cos(_y_angle), _distance * sin(_y_angle), _distance * cos(_y_angle) * cos(_x_angle));

        sycl::float3 w = normalize(_origin - _target);
        sycl::float3 u = normalize(cross(sycl::float3(0, 1, 0), w));
        sycl::float3 v = cross(w, u);

        _horizontal = viewport_width * u;
        _vertical = viewport_height * v;
        _lower_left = _origin - _horizontal / 2 - _vertical / 2 - w;
    }
}
