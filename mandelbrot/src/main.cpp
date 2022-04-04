#include <complex>
#include <iostream>
#include <sycl/sycl.hpp>

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
    auto& framebuffer = renderer.get_framebuffer();

    sycl::gpu_selector selector;
    sycl::queue queue(selector, handle_async_error);

    auto device = queue.get_device();
    std::cout << "Running on " << device.get_info<sycl::info::device::name>();

    while(!display.should_close())
    {
        renderer.clear();

        queue.submit([&](sycl::handler& cgh)
        {
            auto pixels = framebuffer.get_access<sycl::access::mode::discard_write>(cgh);

            cgh.parallel_for(framebuffer.get_range(), [=](sycl::id<2> id)
            {
                pixels[id] = sycl::float4(1.0f, 0.0f, 0.0f, 1.0f);
            });
        });
        queue.wait_and_throw();

        renderer.blit();
        renderer.draw();
        display.update();
    }

    return 0;
}
