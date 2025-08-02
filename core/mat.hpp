#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#include <memory>
#include <string>

namespace core{
    class Mat{
    public:
        Mat(const std::string filename);
    private:
        std::unique_ptr<unsigned char, decltype(&stbi_image_free)> img_ptr;
        int width, height, channels;
    };
};
