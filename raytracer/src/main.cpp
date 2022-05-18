#include <iostream>
#include <array>
#include <sycl/sycl.hpp>

#include "ray.hpp"
#include "cuda_selector.hpp"
#include "gfx/renderer.hpp"
#include "gfx/display.hpp"
#include "gfx/camera.hpp"
#include "obj/sphere.hpp"

namespace sycl = cl::sycl;

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

sycl::float4 calculate_color(
    const Ray& ray,
    sycl::accessor<obj::Sphere, 1, sycl::access_mode::read> spheres,
    sycl::float3 camera_position)
{
    float k_s = 0.2f;
    float k_d = 0.9f;
    float k_a = 0.1f;
    float alpha = 100.0f;
    sycl::float4 ambient(0.1f, 0.1f, 0.1f, 1.0f);

    HitResult result;

    HitResult tmp_result;
    bool hit_anything = false;
    float t_max = FLT_MAX;
    for(size_t i = 0; i < spheres.get_size(); i++)
    {
        auto& sphere = spheres[i];
        if(sphere.hit(ray, 0, t_max, tmp_result))
        {
            hit_anything = true;
            result = tmp_result;
            t_max = tmp_result.t;
        }
    }

    if(hit_anything)
    {
        // LightSource& light = lights[0];
        auto N = result.normal;
        // auto L_m = (light.position() - result.hit_point).normalized(); // direction from surface to light
        // auto R_m = 2.0f * L_m.dot(N) * N - L_m; // direction of perfectly reflected ray
        auto V = normalize(camera_position - result.hit_point); // direction from surface to the camera

        auto light_color = k_a * ambient;
        // auto diffuse_intensity = L_m.dot(N);
        // if(diffuse_intensity > 0)
        //     light_color += k_d * diffuse_intensity * light.color();
        // auto specular_intensity = R_m.dot(V);
        // if(specular_intensity > 0)
        //     light_color += k_s * powf(specular_intensity, alpha) * light.color();

        auto color = sycl::clamp(light_color * result.color, sycl::float4(0), sycl::float4(1));
        return color;
    }
    else
    {
        sycl::float3 direction = ray.get_direction();
        float t = 0.5f * (direction.y() + 1.0f);
        return (1.0f - t) * sycl::float4(1) + t * sycl::float4(0.5f, 0.7f, 1.0f, 1.0f);
    }
}

void render(
    sycl::queue& queue,
    sycl::buffer<obj::Sphere>& spheres_buf,
    gfx::Renderer& renderer,
    gfx::Camera& camera)
{
    renderer.clear();
    auto& framebuffer = renderer.get_framebuffer();

    queue.wait_and_throw();
    queue.submit([&](sycl::handler& cgh)
    {
        auto range = framebuffer.get_range();
        size_t render_width = range[1];
        size_t render_height = range[0];

        auto spheres = spheres_buf.get_access<sycl::access_mode::read>(cgh);

        cgh.parallel_for<class raytracer_renderer>(range, [=](sycl::item<2> item)
        {
            size_t x = item[1];
            size_t y = item[0];

            float u = (float) x / (render_width - 1);
            float v = (float) y / (render_height - 1);
            Ray ray = camera.calculate_ray(u, v);

            sycl::float4 color = calculate_color(ray, spheres, camera.get_position());
        });
    });
}

int main(int argc, char* argv[])
{
    size_t window_width = 1400;
    size_t window_height = 700;
    size_t render_width = window_width;
    size_t render_height = window_height;
    gfx::Display display("SYCL Raytracer", window_width, window_height);

    gfx::Renderer renderer(render_width, render_height);
    gfx::Camera camera(3, sycl::float3(0, 0, -1), 100, (float) render_width / render_height);

    CudaSelector device_selector;
    // sycl::host_selector device_selector;
    sycl::queue queue(device_selector, handle_async_error);

    auto device = queue.get_device();
    std::cout << "Running on " << device.get_info<sycl::info::device::name>() << std::endl;

    std::array<obj::Sphere, 3> spheres = {
        obj::Sphere(sycl::float3(0, 0, 0), 0.5f, sycl::float4(1, 0, 0, 1)),
        obj::Sphere(sycl::float3(0, 1, 0), 0.5f, sycl::float4(0, 1, 0, 1)),
        obj::Sphere(sycl::float3(0, -1, 0), 0.5f, sycl::float4(0, 0, 1, 1))
    };

    sycl::buffer<obj::Sphere> spheres_buf(spheres);

    while(!display.should_close())
    {
        render(queue, spheres_buf, renderer, camera);

        renderer.blit();
        renderer.draw();
        display.update();
    }

    return 0;
}
