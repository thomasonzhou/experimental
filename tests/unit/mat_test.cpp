#include "core/mat.hpp"

#include <array>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Mat class") {
  constexpr int kHeight = 3;
  constexpr int kWidth = 5;
  constexpr int kChannels = 1;
  constexpr std::array<float, 4> values = {0.0f, 1.0f, 2.0f, 3.0f};

  core::Mat mat1(kHeight, kWidth, kChannels, 0.0f);
  REQUIRE(mat1.width() == kWidth);
  REQUIRE(mat1.height() == kHeight);
  REQUIRE(mat1.channels() == kChannels);
  REQUIRE(mat1.size() == 1 * kWidth * kHeight * kChannels);

  for (size_t i = 0; i < values.size(); ++i) {
    mat1(0, 0, i) = values[i];
    REQUIRE(mat1(0, 0, i) == values[i]);
  }

  // successful clone
  core::Mat mat2 = mat1.clone();
  REQUIRE(mat1 == mat2);

  // update cloned matrices and verify source matrix is unchanged
  constexpr float kNewValue = 5.0f;
  REQUIRE(mat1(0, 0, 0) != kNewValue);
  mat2(0, 0) = kNewValue;
  REQUIRE(mat2(0, 0) == kNewValue);
  REQUIRE(mat1(0, 0) != kNewValue);
  REQUIRE(mat1 != mat2);

  core::Mat mat3 = mat2.clone() + 1.0f;
  REQUIRE(mat3 != mat2);

  REQUIRE(mat1 + mat2 == mat2 + mat1);  // A + B == B + A
  REQUIRE((mat1 + mat2) + mat3 ==
          mat1 + (mat2 + mat3));  // (A + B) + C == A + (B + C)
  REQUIRE(mat1 + core::zeros(kHeight, kWidth, kChannels) ==
          mat1);  // A + 0 == A
  REQUIRE(mat1 + (-mat1) ==
          core::zeros(kHeight, kWidth, kChannels));  // A + (-A) == 0

  constexpr double a = 2.0;
  constexpr double b = 3.0;
  REQUIRE(a * (b * mat1) == (a * b) * mat1);  // a * (b * A) == (a * b) * A
  REQUIRE(1.0 * mat1 == mat1);                // 1 * A == A
  REQUIRE((a + b) * mat1 ==
          a * mat1 + b * mat1);  // (a + b) * A == a * A + b * A
  REQUIRE(a * (mat1 + mat2) ==
          a * mat1 + a * mat2);  // a * (A + B) == a * A + a * B
}
