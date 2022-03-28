#include <iostream>
#include <sycl/sycl.hpp>

namespace sycl = cl::sycl;

int main()
{
    auto exception_handler = [](sycl::exception_list exceptions) {
        for(const std::exception_ptr& e : exceptions)
        {
            try
            {
                std::rethrow_exception(e);
            }
            catch(const sycl::exception& e)
            {
                std::cout << "Caught async SYCL exception: " << e.what() << std::endl;
            }
        }
    };

    sycl::gpu_selector selector;

    sycl::queue queue(selector, exception_handler);
    std::cout << "Running on " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

    queue.submit([&](sycl::handler& h) {
        auto range = sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(10));
        h.parallel_for<class invalid_kernel>(range, [=](sycl::nd_item<1>) {});
    });

    try
    {
        queue.wait_and_throw();
    }
    catch(const sycl::exception& e)
    {
        std::cout << "Caught sync SYCL exception: " << e.what() << std::endl;
    }

    return 0;
}
