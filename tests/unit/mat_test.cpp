#include "core/mat.hpp"

#include <array>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Mat class") {
  constexpr int kHeight = 10;
  constexpr int kWidth = 10;
  constexpr int kChannels = 1;
  constexpr std::array<unsigned char, 4> values = {0, 1, 2, 3};

  core::Mat mat1(kWidth, kHeight, kChannels, 0);
  REQUIRE(mat1.width() == kWidth);
  REQUIRE(mat1.height() == kHeight);
  REQUIRE(mat1.channels() == kChannels);
  REQUIRE(mat1.size() == 1 * kWidth * kHeight * kChannels);

  for (size_t i = 0; i < values.size(); ++i) {
    mat1(0, 0, i) = values[i];
    REQUIRE(mat1(0, 0, i) == values[i]);
  }

  core::Mat mat2 = mat1.clone();

  REQUIRE(mat1 == mat2);

  constexpr int kNewValue = 5;
  REQUIRE(mat1(0, 0, 0) != kNewValue);
  mat2(0, 0) = kNewValue;
  REQUIRE(mat2(0, 0) == kNewValue);
  REQUIRE(mat1(0, 0) != kNewValue);
  REQUIRE(mat1 != mat2);

  REQUIRE(mat1 + mat2 == mat2 + mat1);

  core::Mat mat3 = mat2.clone() + 1;

  REQUIRE(mat3 != mat2);
  REQUIRE((mat1 + mat2) + mat3 == mat1 + (mat2 + mat3));
}
