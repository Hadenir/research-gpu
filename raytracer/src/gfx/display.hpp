#pragma once

#include <string>
#include <GLFW/glfw3.h>

namespace gfx
{
    class Display
    {
    public:
        Display(const std::string& title, size_t width, size_t height);
        ~Display();

        size_t get_width() const { return _width; }
        size_t get_height() const { return _height; }

        bool should_close() { return glfwWindowShouldClose(_window); }

        void update();

        Display(const Display&) = delete;
        Display& operator=(const Display&) = delete;

    private:
        size_t _width;
        size_t _height;

        GLFWwindow* _window = nullptr;

        // GLFW callbacks
        static void glfw_on_error(int error_code, const char* error_message);
    };
}
