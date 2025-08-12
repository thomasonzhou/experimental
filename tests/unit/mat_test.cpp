#include "core/mat.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Mat class") {
  constexpr int kHeight = 10;
  constexpr int kWidth = 10;
  constexpr int kChannels = 1;
  constexpr int a = 1;
  constexpr int b = 2;
  constexpr int c = 3;
  constexpr int d = 4;
  constexpr int e = 5;

  core::Mat mat1(kWidth, kHeight, kChannels, 0);
  REQUIRE(mat1.width() == kWidth);
  REQUIRE(mat1.height() == kHeight);
  REQUIRE(mat1.channels() == kChannels);
  REQUIRE(mat1.size() == 1 * kWidth * kHeight * kChannels);

  mat1(0, 0) = a;
  REQUIRE(mat1(0, 0) == a);
  mat1(0, 1) = b;
  REQUIRE(mat1(0, 1) == b);
  mat1(1, 0) = c;
  REQUIRE(mat1(1, 0) == c);
  mat1(1, 1) = d;
  REQUIRE(mat1(1, 1) == d);

  core::Mat mat2 = mat1.clone();

  REQUIRE(mat1 == mat2);

  mat2(0, 0) = e;
  REQUIRE(mat2(0, 0) == e);
  REQUIRE(mat1 != mat2);
}
