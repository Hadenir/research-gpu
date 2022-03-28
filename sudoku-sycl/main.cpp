#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

namespace sycl = cl::sycl;

int main()
{
    std::array<int32_t, 1 << 15> arr;
    std::mt19937 mt_engine(std::random_device{}());
    std::uniform_int_distribution<int32_t> idist(0, 10);

    for(auto& el : arr)
        el = idist(mt_engine);

    sycl::gpu_selector selector;
    sycl::queue queue(selector);
    sycl::buffer<int32_t, 1> buf(arr.data(), arr.size());

    auto device = queue.get_device();

    size_t wgroup_size = device.get_info<sycl::info::device::max_work_group_size>();
    auto part_size = wgroup_size * 2;

    auto has_local_mem = device.is_host() || device.get_info<sycl::info::device::local_mem_type>() != sycl::info::local_mem_type::none;
    auto local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
    if(!has_local_mem || local_mem_size < (wgroup_size * sizeof(int32_t)))
    {
        throw "Device doesn't have enough local memory";
    }

    size_t len = arr.size();
    while(len != 1)
    {
        auto n_wgroups = (len + part_size - 1) / part_size;

        queue.submit([&](sycl::handler& h) {
            auto global_mem = buf.get_access<sycl::access::mode::read_write>(h);
            sycl::accessor<int32_t, 1, sycl::access::mode::read_write, sycl::access::target::local> local_mem(wgroup_size, h);

            auto range = sycl::nd_range<1>(n_wgroups * wgroup_size, wgroup_size);
            h.parallel_for<class reduction_kernel>(range, [=](sycl::nd_item<1> item) {
                size_t local_id = item.get_local_linear_id();
                size_t global_id = item.get_global_linear_id();

                local_mem[local_id] = 0;

                if(2 * global_id < len)
                    local_mem[local_id] = global_mem[2 * global_id] + global_mem[2 * global_id + 1];

                item.barrier(sycl::access::fence_space::local_space);

                for(size_t stride = 1; stride < wgroup_size; stride *= 2)
                {
                    auto idx = 2 * stride * local_id;
                    if(idx < wgroup_size)
                        local_mem[idx] = local_mem[idx] + local_mem[idx + stride];

                    item.barrier(sycl::access::fence_space::local_space);
                }

                if(local_id == 0)
                    global_mem[item.get_group_linear_id()] = local_mem[0];
            });
        });

        queue.wait_and_throw();
        len = n_wgroups;
    }

    auto acc = buf.get_access<sycl::access::mode::read>();
    std::cout << "Sum: " << acc[0] << std::endl;

    return 0;
}
