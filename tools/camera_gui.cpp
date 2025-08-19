#include <GLFW/glfw3.h>
#include <stdio.h>

#include "core/video/color_conversion.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#ifdef __linux__
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace v4l2cam {
// Robust ioctl with EINTR handling.
static int xioctl(int fd, unsigned long req, void* arg) noexcept {
  for (;;) {
    int r = ::ioctl(fd, req, arg);
    if (r == -1 && errno == EINTR) continue;
    return r;
  }
}

struct MappedBuffer {
  void* data;
  std::size_t size;
  MappedBuffer() : data(NULL), size(0) {}
};

class Camera {
 public:
  struct Config {
    std::string dev{"/dev/video0"};
    int w{640};
    int h{480};
    int fps{30};
  };

  explicit Camera(const Config& cfg) : cfg_(cfg) {
    fd_ = ::open(cfg_.dev.c_str(), O_RDWR | O_NONBLOCK);
    if (fd_ < 0)
      throw std::runtime_error("open(" + cfg_.dev + ") failed: " +
                               std::string(std::strerror(errno)));

    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = cfg_.w;
    fmt.fmt.pix.height = cfg_.h;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;
    if (xioctl(fd_, VIDIOC_S_FMT, &fmt) == -1)
      throw std::runtime_error("VIDIOC_S_FMT failed");
    w_ = (int)fmt.fmt.pix.width;
    h_ = (int)fmt.fmt.pix.height;

    v4l2_requestbuffers req{};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd_, VIDIOC_REQBUFS, &req) == -1 || req.count < 2)
      throw std::runtime_error("VIDIOC_REQBUFS failed");

    bufs_.resize(req.count);
    for (std::size_t i = 0; i < bufs_.size(); ++i) {
      v4l2_buffer b{};
      b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      b.memory = V4L2_MEMORY_MMAP;
      b.index = (uint32_t)i;
      if (xioctl(fd_, VIDIOC_QUERYBUF, &b) == -1)
        throw std::runtime_error("VIDIOC_QUERYBUF failed");
      void* m = ::mmap(nullptr, b.length, PROT_READ | PROT_WRITE, MAP_SHARED,
                       fd_, b.m.offset);
      if (m == MAP_FAILED) throw std::runtime_error("mmap failed");
      bufs_[i].data = m;
      bufs_[i].size = b.length;
    }

    for (std::size_t i = 0; i < bufs_.size(); ++i) {
      v4l2_buffer b{};
      b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      b.memory = V4L2_MEMORY_MMAP;
      b.index = (uint32_t)i;
      if (xioctl(fd_, VIDIOC_QBUF, &b) == -1)
        throw std::runtime_error("VIDIOC_QBUF failed");
    }

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_, VIDIOC_STREAMON, &type) == -1)
      throw std::runtime_error("VIDIOC_STREAMON failed");
  }

  ~Camera() {
    try {
      v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      if (fd_ >= 0) (void)xioctl(fd_, VIDIOC_STREAMOFF, &type);
    } catch (...) {
    }
    for (auto& b : bufs_)
      if (b.data && b.size) ::munmap(b.data, b.size);
    if (fd_ >= 0) ::close(fd_);
  }

  std::optional<core::Mat> try_grab_rgb() {
    struct pollfd pfd{fd_, POLLIN, 0};
    int pr = ::poll(&pfd, 1, 0);
    if (pr <= 0) return std::nullopt;

    v4l2_buffer buf;
    std::memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd_, VIDIOC_DQBUF, &buf) == -1) {
      if (errno == EAGAIN) return std::nullopt;
      throw std::runtime_error("VIDIOC_DQBUF failed");
    }

    const std::uint8_t* yuyv =
        static_cast<const std::uint8_t*>(bufs_[buf.index].data);
    core::Mat rgb = core::video::convert_yuyv_to_rgb_f32(yuyv, w_, h_);

    if (xioctl(fd_, VIDIOC_QBUF, &buf) == -1)
      throw std::runtime_error("VIDIOC_QBUF failed");
    return rgb;
  }

  int w() const { return w_; }
  int h() const { return h_; }

 private:
  Config cfg_{};
  int fd_{-1};
  int w_{0}, h_{0};
  std::vector<MappedBuffer> bufs_{};
};
}  // namespace v4l2cam
#endif  // __linux__

static void glfw_error_callback(int error, const char* description) {
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

int main(int, char**) {
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) return 1;

  // Simple OpenGL 3.0 setup
  const char* glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

  // Create window - get monitor size and make it fit screen
  GLFWmonitor* primary = glfwGetPrimaryMonitor();
  const GLFWvidmode* mode = glfwGetVideoMode(primary);
  int window_width = mode->width - 100;    // Leave some margin
  int window_height = mode->height - 100;  // Leave some margin

  GLFWwindow* window = glfwCreateWindow(window_width, window_height,
                                        "Camera GUI", nullptr, nullptr);
  if (window == nullptr) return 1;

  // Center the window
  glfwSetWindowPos(window, 50, 50);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  // Setup Dear ImGui
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  ImGui::StyleColorsDark();

  // Scale UI for better readability
  float scale = 5.0f;
  ImGuiStyle& style = ImGui::GetStyle();
  style.ScaleAllSizes(scale);
  io.FontGlobalScale = scale;

  // Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  // Camera setup
#ifdef __linux__
  std::unique_ptr<v4l2cam::Camera> cam{};
  unsigned int cam_tex = 0;
  int cam_w = 0, cam_h = 0;
  core::Mat cam_rgb;

  try {
    v4l2cam::Camera::Config cfg;
    cam = std::make_unique<v4l2cam::Camera>(cfg);
    cam_w = cam->w();
    cam_h = cam->h();

    glGenTextures(1, &cam_tex);
    glBindTexture(GL_TEXTURE_2D, cam_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, cam_w, cam_h, 0, GL_RGB, GL_FLOAT,
                 nullptr);
  } catch (const std::exception& e) {
    fprintf(stderr, "Camera init failed: %s\n", e.what());
  }
#endif

  // Main loop
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    // Start frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

#ifdef __linux__
    // Camera frame capture
    if (cam) {
      if (auto rgb = cam->try_grab_rgb()) {
        cam_rgb = std::move(*rgb);
        glBindTexture(GL_TEXTURE_2D, cam_tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, cam_w, cam_h, GL_RGB, GL_FLOAT,
                        cam_rgb.data());
      }

      // Simple camera window
      ImGui::Begin("Camera");
      ImGui::Text("Resolution: %d x %d", cam_w, cam_h);
      if (cam_tex != 0) {
        ImGui::Image((ImTextureID)(intptr_t)cam_tex,
                     ImVec2((float)cam_w, (float)cam_h));
      }
      ImGui::Text("%.1f FPS", io.Framerate);
      ImGui::End();
    }
#endif

    // Render
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
  }

  // Cleanup
#ifdef __linux__
  if (cam_tex) glDeleteTextures(1, &cam_tex);
  cam.reset();
#endif

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
