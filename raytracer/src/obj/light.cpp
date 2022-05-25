#include "light.hpp"

namespace obj
{
    Light::Light(sycl::float3 position, Color color)
        : _position(position), _color(color)
    {}

    sycl::float3 Light::get_position() const
    {
        return _position;
    }

    Color Light::get_color() const
    {
        return _color;
    }
}
