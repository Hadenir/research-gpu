#pragma once

#include <sycl/sycl.hpp>
#include <glad.h>

#include "../color.hpp"

namespace gfx
{
    class Renderer
    {
    public:
        Renderer(size_t render_width, size_t render_height);
        ~Renderer();

        size_t get_render_width() const { return _render_width; }
        size_t get_render_height() const { return _render_height; }

        sycl::buffer<Color, 2>& get_framebuffer() { return _framebuffers[_backbuffer_id]; }

        void clear();

        void blit();

        void draw();

    private:
        const float CLEAR_COLOR[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        const float VERTICES_DATA[30] = {
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
            -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
             1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
             1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
             1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
        };

        static const std::string VERTEX_SHADER_SOURCE;
        static const std::string FRAGMENT_SHADER_SOURCE;

        size_t _render_width;
        size_t _render_height;

        size_t _backbuffer_id = 0;
        sycl::buffer<Color, 2> _framebuffers[2];

        GLuint _texture;
        GLuint _shader_program;

        void init_shaders();
        void init_texture();
        void init_vertex_objects();
    };
}
