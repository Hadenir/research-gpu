#include "color.hpp"

Color clamp(Color color)
{
    return sycl::clamp(color, Color(0), Color(1));
}
