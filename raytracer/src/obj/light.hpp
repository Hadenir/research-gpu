#pragma once

#include <sycl/sycl.hpp>

#include "../color.hpp"

namespace obj
{
    class Light
    {
    public:
        Light(sycl::float3 position, Color color);

        SYCL_EXTERNAL sycl::float3 get_position() const;
        SYCL_EXTERNAL Color get_color() const;

    private:
        sycl::float3 _position;
        Color _color;
    };
}
