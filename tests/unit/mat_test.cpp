#include "core/mat.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Mat class") {
  constexpr int kHeight = 10;
  constexpr int kWidth = 10;
  constexpr int kChannels = 3;
  constexpr int a = 1;
  constexpr int b = 2;
  constexpr int c = 3;
  constexpr int d = 4;

  core::Mat mat(kWidth, kHeight, kChannels, 0);
  REQUIRE(mat.batch_size() == 1);
  REQUIRE(mat.width() == kWidth);
  REQUIRE(mat.height() == kHeight);
  REQUIRE(mat.channels() == kChannels);
  REQUIRE(mat.size() == 1 * kWidth * kHeight * kChannels);

  REQUIRE(mat(0, 0, 0, 0) == 0);
  mat(0, 0, 0, 0) = 128;
  REQUIRE(mat(0, 0, 0, 0) == 128);
}
