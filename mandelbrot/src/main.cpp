#include <complex>
#include <iostream>
#include <sycl/sycl.hpp>

#include "cuda_selector.hpp"
#include "gfx/renderer.hpp"
#include "gfx/display.hpp"

namespace sycl = cl::sycl;

using complex = std::complex<double>;

auto handle_async_error = [](sycl::exception_list elist) {
    for (auto &e : elist)
    {
        try
        {
            std::rethrow_exception(e);
        }
        catch(sycl::exception& e)
        {
            std::cout << e.what() << "\n";
        }
    }
};

int main()
{
    size_t window_width = 1400;
    size_t window_height = 700;
    size_t render_width = window_width;
    size_t render_height = window_height;
    gfx::Display display("SYCL Mandelbrot", window_width, window_height);

    gfx::Renderer renderer(render_width, render_height);

    // CudaSelector device_selector;
    sycl::host_selector device_selector;
    sycl::queue queue(device_selector, handle_async_error);

    auto device = queue.get_device();
    std::cout << "Running on " << device.get_info<sycl::info::device::name>() << std::endl;

    const size_t MAX_ITERS = 32;

    // const double viewport_min_x = -1.0;
    // const double viewport_max_x = -0.5;
    // const double viewport_min_y = 0;
    // const double viewport_max_y = 0.25;
    const double viewport_min_x = -2.0;
    const double viewport_max_x = 1.0;
    const double viewport_min_y = -1.0;
    const double viewport_max_y = 1.0;
    const double viewport_width = viewport_max_x - viewport_min_x;
    const double viewport_height = viewport_max_y - viewport_min_y;

    while(!display.should_close())
    {
        renderer.clear();

        auto& framebuffer = renderer.get_framebuffer();

        queue.wait_and_throw();
        queue.submit([&](sycl::handler& cgh)
        {
            auto pixels = framebuffer.get_access<sycl::access::mode::discard_write>(cgh);

            auto range = framebuffer.get_range();
            cgh.parallel_for<class mandelbrot_render>(range, [=](sycl::item<2> item)
            {
                double x = item[1];
                double y = item[0];

                x = viewport_min_x + (x / render_width) * viewport_width;
                y = viewport_min_y + (y / render_height) * viewport_height;
                complex c(x, y);

                complex z(0, 0);
                size_t i;
                for(i = 0; i < MAX_ITERS; i++)
                {
                    z = z * z + c;

                    if(std::norm(z) > 4)
                        break;
                }

                if(std::norm(z) > 4)
                    pixels[item] = sycl::float4(0.0f, 0.0f, 0.0f, 1.0f);
                else
                    pixels[item] = sycl::float4(1.0f, 0.0f, 0.0f, 1.0f);
            });
        });

        renderer.blit();
        renderer.draw();
        display.update();
    }

    return 0;
}
