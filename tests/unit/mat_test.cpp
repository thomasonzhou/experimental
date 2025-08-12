#include "core/mat.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Mat class") {
  core::Mat mat(100, 100, 3, 255);
  REQUIRE(mat.batch_size() == 1);
  REQUIRE(mat.width() == 100);
  REQUIRE(mat.height() == 100);
  REQUIRE(mat.channels() == 3);
  REQUIRE(mat.size() == 30000);  // 100 * 100 * 3

  REQUIRE(mat(0, 0, 0, 0) == 255);
  mat(0, 0, 0, 0) = 128;
  REQUIRE(mat(0, 0, 0, 0) == 128);
}
