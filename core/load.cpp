

#include "mat.hpp"

#include <stdio.h>
#include <memory>
#include <string>



int main(int argc, char* argv[]){
    const std::string filename = "image.png";
    core::Mat mat(filename);

    return 0;
}