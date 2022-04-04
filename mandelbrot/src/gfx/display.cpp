#include <stdexcept>
#include <sstream>
#include <glad.h>

#include "display.hpp"

namespace gfx
{
    Display::Display(const std::string& title, size_t width, size_t height)
        : _width(width), _height(height)
    {
        int result = glfwInit();
        if(result != GLFW_TRUE)
            throw std::runtime_error("Failed to initialize GLFW");

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        _window = glfwCreateWindow((int) width, (int) height, title.c_str(), nullptr, nullptr);
        if(_window == nullptr)
            throw std::runtime_error("Failed to create GLFW window");

        glfwSetWindowUserPointer(_window, this);
        glfwSetErrorCallback(glfw_on_error);

        glfwMakeContextCurrent(_window);

        result = gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);
        if(result == 0)
            throw std::runtime_error("Failed to load GLAD");

        glViewport(0, 0, (GLsizei) width, (GLsizei) height);
    }

    Display::~Display()
    {
        glfwDestroyWindow(_window);
        glfwTerminate();
    }

    void Display::update()
    {
        glfwSwapBuffers(_window);
        glfwPollEvents();
    }

    void Display::glfw_on_error(int error_code, const char* error_message)
    {
        std::stringstream ss;
        ss << "GLFW error (" << error_code << ") occured: " << error_message;
        throw std::runtime_error(ss.str());
    }
}
