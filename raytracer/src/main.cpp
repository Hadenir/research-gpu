#include <iostream>
#include <array>
#include <sycl/sycl.hpp>
#include <chrono>

#include "ray.hpp"
#include "cuda_selector.hpp"
#include "gfx/renderer.hpp"
#include "gfx/display.hpp"
#include "gfx/camera.hpp"
#include "obj/sphere.hpp"
#include "obj/light.hpp"

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

Color calculate_color(
    const Ray& ray,
    sycl::accessor<obj::Sphere, 1, sycl::access_mode::read> spheres,
    sycl::accessor<obj::Light, 1, sycl::access_mode::read> lights,
    sycl::float3 camera_position,
    const sycl::stream& dbg)
{
    float k_a = 0.3f;
    float k_s = 0.4f;
    float k_d = 0.9f;
    float alpha = 10.0f;
    Color ambient(1, 1, 1, 1);

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
        auto light_color = k_a * ambient;

        auto N = result.normal;

        for(size_t i = 0; i < lights.get_size(); i++)
        {
            auto& light = lights[i];
            auto L_m = normalize(light.get_position() - result.hit_point); // direction from surface to light
            auto R_m = normalize(2 * dot(L_m, N) * N - L_m); // direction of perfectly reflected ray
            auto V = normalize(camera_position - result.hit_point); // direction from surface to the camera

            auto diffuse_intensity = dot(L_m, N);
            if(diffuse_intensity > 0)
                light_color += k_d * diffuse_intensity * light.get_color();
            auto specular_intensity = dot(R_m, V);
            if(specular_intensity > 0)
                light_color += k_s * sycl::pow(specular_intensity, alpha) * light.get_color();
        }

        auto color = clamp(light_color * result.color);
        return color;
    }
    else
    {
        sycl::float3 direction = ray.get_direction();
        float t = 0.5f * (direction.y() + 1.0f);
        return (1.0f - t) * Color(1) + t * Color(0.5f, 0.7f, 1.0f, 1.0f);
    }
}

void render(
    sycl::queue& queue,
    sycl::buffer<obj::Sphere>& spheres_buf,
    sycl::buffer<obj::Light>& lights_buf,
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
        auto lights = lights_buf.get_access<sycl::access_mode::read>(cgh);
        auto pixels = framebuffer.get_access<sycl::access_mode::discard_write>(cgh);

        sycl::stream out(65535, 256, cgh);
        cgh.parallel_for<class raytracer_renderer>(range, [=](sycl::item<2> item)
        {
            size_t x = item[1];
            size_t y = item[0];

            float u = (float) x / (render_width - 1);
            float v = (float) y / (render_height - 1);
            Ray ray = camera.calculate_ray(u, v);

            Color color = calculate_color(ray, spheres, lights, camera.get_position(), out);
            pixels[item] = color;
        });
    });
}

int main(int argc, char* argv[])
{
    size_t window_width = 1000;
    size_t window_height = 600;
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

    std::array<obj::Sphere, 1> spheres = {
        obj::Sphere(sycl::float3(0, 2, 0), 0.5f, Color(0.8, 0.8, 0.8, 1)),
        // obj::Sphere(sycl::float3(0, 1.2, 0), 0.5f, Color(0.8, 0.8, 0.8, 1)),
        // obj::Sphere(sycl::float3(0, -1.2, 0), 0.5f, Color(0.8, 0.8, 0.8, 1)),
    };
    std::array<obj::Light, 3> lights = {
        obj::Light(sycl::float3(5, 0, 0), Color(1, 0, 0, 1)),
        obj::Light(sycl::float3(0, 5, 0), Color(0, 1, 0, 1)),
        obj::Light(sycl::float3(-2.5, -2.5, 5), Color(0, 0, 1, 1)),
    };

    sycl::buffer<obj::Sphere> spheres_buf(spheres);
    sycl::buffer<obj::Light> lights_buf(lights);

    while(!display.should_close())
    {
        auto frame_start = std::chrono::high_resolution_clock::now();

        render(queue, spheres_buf, lights_buf, renderer, camera);

        auto copy_start = std::chrono::high_resolution_clock::now();
        renderer.blit();
        auto copy_end = std::chrono::high_resolution_clock::now();
        float copy_duration_ms = std::chrono::duration(copy_end - copy_start).count() / 1000000.0f;
        std::cout << "Copying took " << copy_duration_ms << "ms" << std::endl;

        renderer.draw();
        display.update();

        auto frame_end = std::chrono::high_resolution_clock::now();
        float frame_duration_ms = std::chrono::duration(frame_end - frame_start).count() / 1000000.0f;
        std::cout << "Frame   took " << frame_duration_ms << "ms" << std::endl;
    }

    return 0;
}
