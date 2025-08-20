#include <GLFW/glfw3.h>
#include <stdio.h>

#include <memory>
#include <stdexcept>

#include "core/video/v4l2/v4l2_camera.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

static void glfw_error_callback(int error, const char* description) {
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

int main(int, char**) {
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) return 1;

  const char* glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

  // create window based on monitor size
  GLFWmonitor* primary = glfwGetPrimaryMonitor();
  const GLFWvidmode* mode = glfwGetVideoMode(primary);
  int window_width = mode->width - 100;
  int window_height = mode->height - 100;

  GLFWwindow* window = glfwCreateWindow(window_width, window_height,
                                        "Camera GUI", nullptr, nullptr);
  if (window == nullptr) return 1;

  // center window
  glfwSetWindowPos(window, 50, 50);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  ImGui::StyleColorsDark();

  // scale UI for better readability
  float scale = 5.0f;
  ImGuiStyle& style = ImGui::GetStyle();
  style.ScaleAllSizes(scale);
  io.FontGlobalScale = scale;

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  std::unique_ptr<core::video::v4l2::Camera> camera;
  unsigned int cam_tex = 0;
  int cam_w = 0, cam_h = 0;

  try {
    core::video::v4l2::Config config;
    config.device = "/dev/video0";
    config.width = 640;
    config.height = 480;
    config.fps = 30;
    config.pixfmt = V4L2_PIX_FMT_YUYV;

    camera = std::make_unique<core::video::v4l2::Camera>(config);
    cam_w = camera->width();
    cam_h = camera->height();

    glGenTextures(1, &cam_tex);
    glBindTexture(GL_TEXTURE_2D, cam_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, cam_w, cam_h, 0, GL_RGB, GL_FLOAT,
                 nullptr);
  } catch (const std::exception& e) {
    fprintf(stderr, "Camera init failed: %s\n", e.what());
  }

  // Main loop
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    // Start frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (camera) {
      if (auto rgb_opt = camera->try_grab_rgb()) {
        const core::Mat& cam_rgb = rgb_opt.value();
        glBindTexture(GL_TEXTURE_2D, cam_tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, cam_w, cam_h, GL_RGB, GL_FLOAT,
                        cam_rgb.data());
      }

      ImGui::Begin("Camera Feed");
      ImGui::Text("Resolution: %d x %d", cam_w, cam_h);
      if (cam_tex != 0) {
        ImGui::Image((ImTextureID)(intptr_t)cam_tex,
                     ImVec2((float)cam_w, (float)cam_h));
      }
      ImGui::Text("%.1f FPS", io.Framerate);
      ImGui::End();
    }

    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
  }

  if (cam_tex) glDeleteTextures(1, &cam_tex);
  camera.reset();

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
