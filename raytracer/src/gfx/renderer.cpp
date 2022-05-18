#include <sstream>
#include <stdexcept>

#include "renderer.hpp"

namespace gfx
{
    Renderer::Renderer(size_t render_width, size_t render_height)
        : _render_width(render_width), _render_height(render_height),
          _framebuffers{sycl::range<2>(render_height, render_width), sycl::range<2>(render_height, render_width)}
    {
        init_shaders();
        init_texture();
        init_vertex_objects();
    }

    Renderer::~Renderer()
    {}

    void Renderer::clear()
    {
        glClearColor(CLEAR_COLOR[0], CLEAR_COLOR[1], CLEAR_COLOR[2], CLEAR_COLOR[3]);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    void Renderer::blit()
    {
        size_t backbuffer_id = _backbuffer_id;
        size_t frontbuffer_id = (backbuffer_id + 1) % 2;
        auto& frontbuffer = _framebuffers[frontbuffer_id];

        // TODO: There is an expensive host<->device copy here. Can it be avoided?
        auto pixels = frontbuffer.get_access<sycl::access::mode::read>();
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, _render_width, _render_height, 0, GL_RGBA, GL_FLOAT, pixels.get_pointer());
        GLenum err;
        while((err = glGetError()) != GL_NO_ERROR)
        {
            std::cerr << std::hex << err << std::endl;
        }

        _backbuffer_id = frontbuffer_id; 
    }

    void Renderer::draw()
    {
        glUseProgram(_shader_program);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    void Renderer::init_shaders()
    {
        int success;
        char infoLog[512];

        const char* vertex_src = VERTEX_SHADER_SOURCE.c_str();
        GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex_shader, 1, &vertex_src, nullptr);
        glCompileShader(vertex_shader);

        glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
        if (success != GL_TRUE)
        {
            glGetShaderInfoLog(vertex_shader, 512, NULL, infoLog);
            std::stringstream ss;
            ss << "Compilation of vertex shader failed: " << infoLog;
            throw std::runtime_error(ss.str());
        }

        const char* fragment_src = FRAGMENT_SHADER_SOURCE.c_str();
        GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment_shader, 1, &fragment_src, nullptr);
        glCompileShader(fragment_shader);

        glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
        if (success != GL_TRUE)
        {
            glGetShaderInfoLog(fragment_shader, 512, NULL, infoLog);
            std::stringstream ss;
            ss << "Compilation of fragment shader failed: " << infoLog;
            throw std::runtime_error(ss.str());
        }

        _shader_program = glCreateProgram();
        glAttachShader(_shader_program, vertex_shader);
        glAttachShader(_shader_program, fragment_shader);
        glLinkProgram(_shader_program);

        glGetProgramiv(_shader_program, GL_LINK_STATUS, &success);
        if(success != GL_TRUE) {
            glGetProgramInfoLog(_shader_program, 512, NULL, infoLog);
            std::stringstream ss;
            ss << "Linking of shader program failed: " << infoLog;
            throw std::runtime_error(ss.str());
        }

        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
    }

    void Renderer::init_texture()
    {
        glGenTextures(1, &_texture);
        glBindTexture(GL_TEXTURE_2D, _texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, _render_width, _render_height, 0, GL_RGBA, GL_FLOAT, NULL);
    }

    void Renderer::init_vertex_objects()
    {
        GLuint VAO;
        glGenVertexArrays(1, &VAO);
        glBindVertexArray(VAO);

        GLuint VBO;
        glGenBuffers(1, &VBO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(VERTICES_DATA), VERTICES_DATA, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
    }

    const std::string Renderer::VERTEX_SHADER_SOURCE =
        "#version 330 core\n"
        "layout (location = 0) in vec3 inPos;\n"
        "layout (location = 1) in vec2 inTexCoord;\n"
        "out vec2 texCoord;\n"
        "\n"
        "void main()\n"
        "{\n"
        "    gl_Position = vec4(inPos.x, inPos.y, inPos.z, 1.0);\n"
        "    texCoord = inTexCoord;\n"
        "}\n";

    const std::string Renderer::FRAGMENT_SHADER_SOURCE =
        "#version 330 core\n"
        "uniform sampler2D texImage;\n"
        "in vec2 texCoord;\n"
        "out vec4 outColor;\n"
        "\n"
        "void main()\n"
        "{\n"
        "    vec4 c = texture(texImage, texCoord);\n"
        "    outColor = c;\n"
        "}\n";
}
