#include <iostream>
#include <sycl/sycl.hpp>

namespace sycl = cl::sycl;

int main()
{
    sycl::float4 a = {1.0, 1.0, 1.0, 1.0};
    sycl::float4 b = {1.0, 1.0, 1.0, 1.0};
    sycl::float4 c = {0.0, 0.0, 0.0, 0.0};

    sycl::gpu_selector selector;

    sycl::queue queue(selector);
    std::cout << "Running on " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

    {
        sycl::buffer<sycl::float4, 1> a_buf(&a, 1);
        sycl::buffer<sycl::float4, 1> b_buf(&b, 1);
        sycl::buffer<sycl::float4, 1> c_buf(&c, 1);

        queue.submit([&](sycl::handler& h) {
            auto a_acc = a_buf.get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.get_access<sycl::access::mode::read>(h);
            auto c_acc = c_buf.get_access<sycl::access::mode::discard_write>(h);

            h.single_task<class vector_addition>([=]() {
                c_acc[0] = a_acc[0] + b_acc[0];
            });
        });
    }

    std::cout << "  A { " << a.x() << ", " << a.y() << ", " << a.z() << ", " << a.w() << " }\n"
        << "+ B { " << b.x() << ", " << b.y() << ", " << b.z() << ", " << b.w() << " }\n"
        << "------------------\n"
        << "= C { " << c.x() << ", " << c.y() << ", " << c.z() << ", " << c.w() << " }"
        << std::endl;

    return 0;
}
