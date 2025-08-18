#include <stdio.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h>  // Will drag system OpenGL headers

// Select OpenGL loader headers if the build defines one (matches official
// examples)
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
#include <GL/gl3w.h>
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
#include <GL/glew.h>
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
#include <glad/glad.h>
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD2)
#include <glad/gl.h>
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING2)
#include <glbinding/Binding.h>
#include <glbinding/gl/gl.h>
using namespace gl;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING3)
#include <glbinding/gl/gl.h>
#include <glbinding/glbinding.h>
using namespace gl;
#else
// IMGUI_IMPL_OPENGL_LOADER_CUSTOM may be defined by the build; otherwise system
// GL will be used via GLFW.
#endif

#ifdef __EMSCRIPTEN__
#include "../libs/emscripten/emscripten_mainloop_stub.h"
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Linux V4L2 camera capture (YUYV → RGB888) + OpenGL texture upload
// Wrapped for Linux builds only; harmless on other platforms.
#ifdef __linux__
#include <fcntl.h>            // open()
#include <linux/videodev2.h>  // V4L2 (linux kernel header)
#include <poll.h>             // poll()
#include <sys/ioctl.h>        // ioctl()
#include <sys/mman.h>         // mmap()
#include <sys/stat.h>         // open flags
#include <unistd.h>           // close()

#include <cerrno>   // errno
#include <cstdint>  // std::uint8_t
#include <cstring>  // std::memset, std::strerror
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace v4l2cam {  // isolate symbols (namespacing requested)
// Robust ioctl with EINTR handling.
static int xioctl(int fd, unsigned long req, void* arg) {
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

// YUYV (Y0 U0 Y1 V0) -> RGB888 (BT.601 approx)
static void yuyv_to_rgb(const std::uint8_t* src, int w, int h,
                        std::vector<std::uint8_t>& out_rgb) {
  const int n = w * h;
  out_rgb.resize(n * 3);
  auto clamp = [](int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); };
  const std::uint8_t* s = src;
  std::uint8_t* d = out_rgb.data();
  for (int i = 0; i < n; i += 2) {
    int y0 = s[0];
    int u = s[1] - 128;
    int y1 = s[2];
    int v = s[3] - 128;
    auto y2rgb = [&](int y, std::uint8_t* o) {
      int c = y - 16;
      int r = (298 * c + 409 * v + 128) >> 8;
      int g = (298 * c - 100 * u - 208 * v + 128) >> 8;
      int b = (298 * c + 516 * u + 128) >> 8;
      o[0] = (std::uint8_t)clamp(r);
      o[1] = (std::uint8_t)clamp(g);
      o[2] = (std::uint8_t)clamp(b);
    };
    y2rgb(y0, d + 0);
    y2rgb(y1, d + 3);
    s += 4;
    d += 6;
  }
}

class Camera {
 public:
  struct Config {
    std::string dev{"/dev/video0"};
    int w{800};
    int h{600};
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
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;  // request YUYV
    fmt.fmt.pix.field = V4L2_FIELD_ANY;
    if (xioctl(fd_, VIDIOC_S_FMT, &fmt) == -1)
      throw std::runtime_error("VIDIOC_S_FMT failed");
    w_ = (int)fmt.fmt.pix.width;
    h_ = (int)fmt.fmt.pix.height;
    fps_ = cfg_.fps;

    v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = cfg_.fps;
    (void)xioctl(fd_, VIDIOC_S_PARM, &parm);  // best-effort

    v4l2_requestbuffers req{};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd_, VIDIOC_REQBUFS, &req) == -1 || req.count < 2)
      throw std::runtime_error("VIDIOC_REQBUFS failed or insufficient buffers");

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
        throw std::runtime_error("VIDIOC_QBUF (prime) failed");
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
  std::optional<std::vector<std::uint8_t>> try_grab_rgb() {
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
    std::vector<std::uint8_t> rgb;
    yuyv_to_rgb(yuyv, w_, h_, rgb);
    if (xioctl(fd_, VIDIOC_QBUF, &buf) == -1)
      throw std::runtime_error("VIDIOC_QBUF failed");
    return rgb;
  }
  int w() const { return w_; }
  int h() const { return h_; }
  int fps() const { return fps_; }

 private:
  Config cfg_{};
  int fd_{-1};
  int w_{0}, h_{0}, fps_{30};
  std::vector<MappedBuffer> bufs_{};
};
}  // namespace v4l2cam
#endif  // __linux__

static void glfw_error_callback(int error, const char* description) {
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// Main code
int main(int, char**) {
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) return 1;

  // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
  // GL ES 2.0 + GLSL 100 (WebGL 1.0)
  const char* glsl_version = "#version 100";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(IMGUI_IMPL_OPENGL_ES3)
  // GL ES 3.0 + GLSL 300 es (WebGL 2.0)
  const char* glsl_version = "#version 300 es";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
  // GL 3.2 + GLSL 150
  const char* glsl_version = "#version 150";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);  // Required on Mac
#else
  // GL 3.0 + GLSL 130
  const char* glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+
  // only glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // 3.0+ only
#endif

  // Create window with graphics context
  float main_scale = 2.5f;  // Increased default scale for larger text
#if GLFW_VERSION_MAJOR >= 3 && GLFW_VERSION_MINOR >= 3
  // Try to get content scale for monitor if available
  GLFWmonitor* primary = glfwGetPrimaryMonitor();
  if (primary) {
    float xscale, yscale;
    glfwGetMonitorContentScale(primary, &xscale, &yscale);
    main_scale =
        xscale * 2.5f;  // Apply additional 1.5x multiplier to monitor scale
  }
#endif
  GLFWwindow* window =
      glfwCreateWindow((int)(1280 * main_scale), (int)(800 * main_scale),
                       "Dear ImGui GLFW+OpenGL3 example", nullptr, nullptr);
  if (window == nullptr) return 1;
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);  // Enable vsync

  // Initialize chosen GL loader if needed (matches upstream example pattern)
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
  if (gl3wInit() != 0) {
    fprintf(stderr, "Failed to initialize OpenGL loader (gl3w)\n");
    return 1;
  }
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
  if (glewInit() != GLEW_OK) {
    fprintf(stderr, "Failed to initialize OpenGL loader (GLEW)\n");
    return 1;
  }
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
  if (!gladLoadGL()) {
    fprintf(stderr, "Failed to initialize OpenGL loader (glad)\n");
    return 1;
  }
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD2)
  if (!gladLoadGL(glfwGetProcAddress)) {
    fprintf(stderr, "Failed to initialize OpenGL loader (glad2)\n");
    return 1;
  }
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING2)
  glbinding::Binding::initialize();
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING3)
  glbinding::initialize([](const char* name) {
    return (glbinding::ProcAddress)glfwGetProcAddress(name);
  });
#endif

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  (void)io;
  io.ConfigFlags |=
      ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
  io.ConfigFlags |=
      ImGuiConfigFlags_NavEnableGamepad;             // Enable Gamepad Controls
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;  // Enable Docking
  io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;  // Enable Multi-Viewport
                                                       // / Platform Windows
  // io.ConfigViewportsNoAutoMerge = true;
  // io.ConfigViewportsNoTaskBarIcon = true;

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();
  // ImGui::StyleColorsLight();

  // Setup scaling
  ImGuiStyle& style = ImGui::GetStyle();
  style.ScaleAllSizes(main_scale);  // Bake a fixed style scale.
  // Note: FontScaleDpi was removed in newer ImGui versions
#if GLFW_VERSION_MAJOR >= 3 && GLFW_VERSION_MINOR >= 3
  // These config options are experimental and may not be available
  // io.ConfigDpiScaleFonts = true;          // [Experimental]
  // io.ConfigDpiScaleViewports = true;      // [Experimental]
#endif

  // When viewports are enabled we tweak WindowRounding/WindowBg so platform
  // windows can look identical to regular ones.
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    style.WindowRounding = 0.0f;
    style.Colors[ImGuiCol_WindowBg].w = 1.0f;
  }

  // Load and scale fonts
  io.Fonts->AddFontDefault();
  io.FontGlobalScale = main_scale;  // Scale fonts globally

  // Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(window, true);
#ifdef __EMSCRIPTEN__
  ImGui_ImplGlfw_InstallEmscriptenCallbacks(window, "#canvas");
#endif
  ImGui_ImplOpenGL3_Init(glsl_version);

  // -------------------------------------------------------------------------
  // Camera state (created after GL is ready so we can allocate a texture)
#ifdef __linux__
  std::unique_ptr<v4l2cam::Camera> cam{};
  unsigned int cam_tex = 0;
  int cam_w = 0, cam_h = 0;
  std::vector<std::uint8_t> cam_rgb;  // CPU RGB buffer
  try {
    v4l2cam::Camera::Config cfg;  // defaults: /dev/video0, 640x480@30
    cam = std::make_unique<v4l2cam::Camera>(cfg);
    cam_w = cam->w();
    cam_h = cam->h();
    glGenTextures(1, &cam_tex);
    glBindTexture(GL_TEXTURE_2D, cam_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, cam_w, cam_h, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, nullptr);
  } catch (const std::exception& e) {
    fprintf(stderr, "Camera init failed: %s\n", e.what());
  }
#endif

  // Load Fonts (optional – keep defaults)
  // io.Fonts->AddFontDefault();

  // Main loop
#ifdef __EMSCRIPTEN__
  // For an Emscripten build we are disabling file-system access, so let's not
  // attempt to do a fopen() of the imgui.ini file. You may manually call
  // LoadIniSettingsFromMemory() to load settings from your own storage.
  io.IniFilename = nullptr;
  EMSCRIPTEN_MAINLOOP_BEGIN
#else
  while (!glfwWindowShouldClose(window))
#endif
  {
    // Poll and handle events (inputs, window resize, etc.)
    // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to
    // tell if dear imgui wants to use your inputs.
    // - When io.WantCaptureMouse is true, do not dispatch mouse input data to
    // your main application, or clear/overwrite your copy of the mouse data.
    // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input
    // data to your main application, or clear/overwrite your copy of the
    // keyboard data. Generally you may always pass all inputs to dear imgui,
    // and hide them from your application based on those two flags.
    glfwPollEvents();
    if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0) {
      ImGui_ImplGlfw_Sleep(10);
      continue;
    }

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

#ifdef __linux__
    // Try to grab a camera frame (non-blocking) and upload to GL
    if (cam) {
      if (auto rgb = cam->try_grab_rgb()) {
        cam_rgb = std::move(*rgb);
        glBindTexture(0x0DE1 /*GL_TEXTURE_2D*/, cam_tex);
        glPixelStorei(0x0CF5 /*GL_UNPACK_ALIGNMENT*/, 1);
        glTexSubImage2D(0x0DE1, 0, 0, 0, cam_w, cam_h, 0x1907 /*GL_RGB*/,
                        0x1401 /*GL_UNSIGNED_BYTE*/, cam_rgb.data());
      }
      // Dockable camera window
      ImGui::Begin("Camera");
      ImGui::TextUnformatted("Source: /dev/video0  (YUYV -> RGB888, CPU)");
      ImGui::Text("Resolution: %d x %d", cam_w, cam_h);
      ImGui::SameLine();
      ImGui::Text("Camera frame rate: %d FPS", cam->fps());
      if (cam_tex != 0) {
        ImGui::Image((ImTextureID)(intptr_t)cam_tex,
                     ImVec2((float)cam_w, (float)cam_h));
      } else {
        ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1),
                           "No texture. Camera not initialized.");
      }

      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                  1000.0f / io.Framerate, io.Framerate);
      ImGui::End();
    }
#endif

    // Rendering
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // Update and Render additional Platform Windows
    // (Platform functions may change the current OpenGL context, so we
    // save/restore it to make it easier to paste this code elsewhere.
    //  For this specific demo app we could also call
    //  glfwMakeContextCurrent(window) directly)
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
      GLFWwindow* backup_current_context = glfwGetCurrentContext();
      ImGui::UpdatePlatformWindows();
      ImGui::RenderPlatformWindowsDefault();
      glfwMakeContextCurrent(backup_current_context);
    }

    glfwSwapBuffers(window);
  }
#ifdef __EMSCRIPTEN__
  EMSCRIPTEN_MAINLOOP_END;
#endif

  // Cleanup
#ifdef __linux__
  if (cam_tex) {
    glDeleteTextures(1, &cam_tex);
    cam_tex = 0;
  }
  cam.reset();
#endif
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
