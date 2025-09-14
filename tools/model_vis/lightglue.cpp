#include <GLFW/glfw3.h>
#include <stdio.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>

#include "core/inference/feature_matcher.hpp"
#include "core/io/load_bazel_runfile.hpp"
#include "core/video/v4l2/v4l2_camera.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

static void glfw_error_callback(int error, const char* description) {
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

int main(int, char** argv) {
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

  // realsense 405
  std::unique_ptr<core::video::v4l2::Camera> realsense_camera;
  unsigned int realsense_tex = 1;
  int realsense_w = 0, realsense_h = 0;

  try {
    core::video::v4l2::Config config;
    config.device = "/dev/video8";
    config.width = 640;
    config.height = 480;
    config.fps = 30;
    config.pixfmt = V4L2_PIX_FMT_YUYV;

    realsense_camera = std::make_unique<core::video::v4l2::Camera>(config);
    realsense_w = realsense_camera->width();
    realsense_h = realsense_camera->height();

    glGenTextures(1, &realsense_tex);
    glBindTexture(GL_TEXTURE_2D, realsense_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, realsense_w, realsense_h, 0, GL_RGB,
                 GL_FLOAT, nullptr);
  } catch (const std::exception& e) {
    fprintf(stderr, "Camera init failed: %s\n", e.what());
  }

  std::unique_ptr<core::inference::onnx::FeatureMatcher> feature_matcher;
  core::inference::onnx::FeatureMatchingResult last_matches;
  bool matches_initialized = false;

  std::string feature_matcher_model_path = core::io::find_runfile_path(
      argv, "model_weights/superpoint-lightglue.onnx");
  try {
    feature_matcher = std::make_unique<core::inference::onnx::FeatureMatcher>(
        feature_matcher_model_path);
  } catch (const std::exception& e) {
    fprintf(stderr, "Feature matcher init failed: %s\n", e.what());
  }

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

      // realsense 405

      std::optional<std::reference_wrapper<const core::Mat>> realsense_rgb;

      // Always grab the latest frame for potential use
      if (auto rgb_opt = realsense_camera->try_grab_rgb()) {
        realsense_rgb = rgb_opt.value();

        // Update video texture only at the specified video frame rate
        if (should_update_video2) {
          glBindTexture(GL_TEXTURE_2D, realsense_tex);
          glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, realsense_w, realsense_h,
                          GL_RGB, GL_FLOAT, realsense_rgb->get().data());
          last_video2_time = current_time;
        }
      }

      ImGui::Begin("Realsense Camera Feed");
      ImGui::Text("Resolution: %d x %d", realsense_w, realsense_h);

      if (realsense_tex != 0) {
        ImGui::Image((ImTextureID)(intptr_t)realsense_tex,
                     ImVec2((float)realsense_w, (float)realsense_h));
      }
      ImGui::Text("%.1f FPS", io.Framerate);
      ImGui::Separator();
      ImGui::SliderInt("Video FPS", &video2_fps, 1, 20, "%d");
      ImGui::End();

      // Feature Matching between cameras
      ImGui::Begin("Feature Matching");

      if (feature_matcher && should_run_inference && cam_rgb && realsense_rgb) {
        // Run feature matching between the two cameras
        last_matches = feature_matcher->match_features(cam_rgb->get(),
                                                       realsense_rgb->get());
        matches_initialized = true;
        last_inference_time = current_time;
      }

      if (matches_initialized) {
        ImGui::Text("Left keypoints: %zu", last_matches.left_keypoints.size());
        ImGui::Text("Right keypoints: %zu",
                    last_matches.right_keypoints.size());
        ImGui::Text("Matches: %zu", last_matches.matches.size());

        // Create side-by-side visualization - make images bigger for better
        // visibility
        ImVec2 image_size((float)cam_w * 2.0f, (float)cam_h * 2.0f);

        // Store positions for match line drawing
        ImVec2 left_img_pos, right_img_pos;

        // Left image with keypoints
        ImGui::Text("Webcam with keypoints:");
        if (cam_tex != 0) {
          left_img_pos = ImGui::GetCursorScreenPos();
          ImGui::Image((ImTextureID)(intptr_t)cam_tex, image_size);

          // Draw keypoints on left image
          ImDrawList* draw_list = ImGui::GetWindowDrawList();
          for (const auto& kp : last_matches.left_keypoints) {
            ImVec2 center(left_img_pos.x + kp.x * image_size.x / cam_w,
                          left_img_pos.y + kp.y * image_size.y / cam_h);
            draw_list->AddCircle(center, 3.0f, IM_COL32(0, 255, 0, 255), 12,
                                 2.0f);
          }
        }

        ImGui::SameLine();

        // Right image with keypoints
        ImGui::Text("RealSense with keypoints:");
        if (realsense_tex != 0) {
          right_img_pos = ImGui::GetCursorScreenPos();
          ImGui::Image((ImTextureID)(intptr_t)realsense_tex, image_size);

          // Draw keypoints on right image
          ImDrawList* draw_list = ImGui::GetWindowDrawList();
          for (const auto& kp : last_matches.right_keypoints) {
            ImVec2 center(right_img_pos.x + kp.x * image_size.x / realsense_w,
                          right_img_pos.y + kp.y * image_size.y / realsense_h);
            draw_list->AddCircle(center, 3.0f, IM_COL32(0, 255, 0, 255), 12,
                                 2.0f);
          }
        }

        // Draw matches as lines between the images
        if (cam_tex != 0 && realsense_tex != 0 &&
            !last_matches.matches.empty()) {
          ImDrawList* draw_list = ImGui::GetWindowDrawList();

          // Draw match lines (limit to top N matches for clarity)
          int max_matches = std::min(50, (int)last_matches.matches.size());
          for (int i = 0; i < max_matches; ++i) {
            const auto& match = last_matches.matches[i];

            if (match.left_idx < last_matches.left_keypoints.size() &&
                match.right_idx < last_matches.right_keypoints.size()) {
              const auto& left_kp = last_matches.left_keypoints[match.left_idx];
              const auto& right_kp =
                  last_matches.right_keypoints[match.right_idx];

              // Calculate actual screen positions
              ImVec2 left_pt(left_img_pos.x + left_kp.x * image_size.x / cam_w,
                             left_img_pos.y + left_kp.y * image_size.y / cam_h);
              ImVec2 right_pt(
                  right_img_pos.x + right_kp.x * image_size.x / realsense_w,
                  right_img_pos.y + right_kp.y * image_size.y / realsense_h);

              // Color by match score - higher score = more red, lower = more
              // blue
              float normalized_score = std::min(1.0f, match.score);
              ImU32 color =
                  IM_COL32((int)(255 * normalized_score),           // Red
                           0,                                       // Green
                           (int)(255 * (1.0f - normalized_score)),  // Blue
                           150                                      // Alpha
                  );

              draw_list->AddLine(left_pt, right_pt, color, 1.5f);
            }
          }
        }
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
