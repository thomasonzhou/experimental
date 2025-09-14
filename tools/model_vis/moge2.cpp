#include <GLFW/glfw3.h>
#include <stdio.h>

#include <chrono>
#include <memory>
#include <stdexcept>

#include "core/inference/onnx_infer.hpp"
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

  // webcamera
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

  //   // realsense 405
  //   std::unique_ptr<core::video::v4l2::Camera> realsense_camera;
  //   unsigned int realsense_tex = 1;
  //   int realsense_w = 0, realsense_h = 0;

  //   try {
  //     core::video::v4l2::Config config;
  //     config.device = "/dev/video8";
  //     config.width = 640;
  //     config.height = 480;
  //     config.fps = 30;
  //     config.pixfmt = V4L2_PIX_FMT_YUYV;

  //     realsense_camera = std::make_unique<core::video::v4l2::Camera>(config);
  //     realsense_w = realsense_camera->width();
  //     realsense_h = realsense_camera->height();

  //     glGenTextures(1, &realsense_tex);
  //     glBindTexture(GL_TEXTURE_2D, realsense_tex);
  //     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  //     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  //     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, realsense_w, realsense_h, 0,
  //     GL_RGB,
  //                  GL_FLOAT, nullptr);
  //   } catch (const std::exception& e) {
  //     fprintf(stderr, "Camera init failed: %s\n", e.what());
  //   }

  core::inference::onnx::CUDAModel model(
      "/home/thchzh/src/experimental/weights/moge-2-vits-normal.onnx");
  core::Mat model_output;
  bool model_output_initialized = false;
  unsigned int model_tex = 2;
  glGenTextures(1, &model_tex);
  glBindTexture(GL_TEXTURE_2D, model_tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, cam_w, cam_h, 0, GL_RGB, GL_FLOAT,
               nullptr);

  // depth channel!
  GLint swizzle[4] = {GL_BLUE, GL_BLUE, GL_BLUE, GL_ONE};
  glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzle);

  int video_fps = 30;
  int video2_fps = 30;
  int inference_fps = 5;
  auto last_video1_time = std::chrono::steady_clock::now();
  auto last_video2_time = std::chrono::steady_clock::now();
  auto last_inference_time = std::chrono::steady_clock::now();

  // Main loop
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    // Start frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (camera) {
      auto current_time = std::chrono::steady_clock::now();
      auto video_interval = std::chrono::duration<float>(1.0f / video_fps);
      auto inference_interval =
          std::chrono::duration<float>(1.0f / inference_fps);

      bool should_update_video1 =
          (current_time - last_video1_time) >= video_interval;
      bool should_update_video2 =
          (current_time - last_video2_time) >= video_interval;
      bool should_run_inference =
          (current_time - last_inference_time) >= inference_interval;

      // webcam
      std::optional<std::reference_wrapper<const core::Mat>> cam_rgb;

      // Always grab the latest frame for potential use
      if (auto rgb_opt = camera->try_grab_rgb()) {
        cam_rgb = rgb_opt.value();

        // Update video texture only at the specified video frame rate
        if (should_update_video1) {
          glBindTexture(GL_TEXTURE_2D, cam_tex);
          glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, cam_w, cam_h, GL_RGB,
                          GL_FLOAT, cam_rgb->get().data());
          last_video1_time = current_time;
        }
      }

      ImGui::Begin("Camera Feed");
      ImGui::Text("Resolution: %d x %d", cam_w, cam_h);

      if (cam_tex != 0) {
        ImGui::Image((ImTextureID)(intptr_t)cam_tex,
                     ImVec2((float)cam_w, (float)cam_h));
      }
      ImGui::Text("%.1f FPS", io.Framerate);
      ImGui::Separator();
      ImGui::SliderInt("Video FPS", &video_fps, 1, 20, "%d");
      ImGui::End();

      //   // realsense 405

      //   std::optional<std::reference_wrapper<const core::Mat>> realsense_rgb;

      //   // Always grab the latest frame for potential use
      //   if (auto rgb_opt = realsense_camera->try_grab_rgb()) {
      //     realsense_rgb = rgb_opt.value();

      //     // Update video texture only at the specified video frame rate
      //     if (should_update_video2) {
      //       glBindTexture(GL_TEXTURE_2D, realsense_tex);
      //       glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, realsense_w, realsense_h,
      //                       GL_RGB, GL_FLOAT, realsense_rgb->get().data());
      //       last_video1_time = current_time;
      //     }
      //   }

      //   ImGui::Begin("Realsense Camera Feed");
      //   ImGui::Text("Resolution: %d x %d", realsense_w, realsense_h);

      //   if (realsense_tex != 0) {
      //     ImGui::Image((ImTextureID)(intptr_t)realsense_tex,
      //                  ImVec2((float)realsense_w, (float)realsense_h));
      //   }
      //   ImGui::Text("%.1f FPS", io.Framerate);
      //   ImGui::Separator();
      //   ImGui::SliderInt("Video FPS", &video2_fps, 1, 20, "%d");
      //   ImGui::End();

      ImGui::Begin("Camera Inference");

      if (model_tex != 0 && should_run_inference && cam_rgb) {
        if (!model_output_initialized) {
          model_output = core::Mat(cam_rgb->get().shape());
          model_output_initialized = true;
        }
        model.infer_inplace(cam_rgb->get(), model_output);

        glBindTexture(GL_TEXTURE_2D, model_tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, cam_w, cam_h, GL_RGB, GL_FLOAT,
                        model_output.data());

        last_inference_time = current_time;
      }

      if (model_tex != 0 && model_output_initialized) {
        ImGui::Image((ImTextureID)(intptr_t)model_tex,
                     ImVec2((float)cam_w, (float)cam_h));
      }
      ImGui::Separator();
      ImGui::SliderInt("Inference FPS", &inference_fps, 1, 10, "%d");

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
