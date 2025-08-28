#include "core/mat/mat.hpp"

#include <catch2/catch_test_macros.hpp>

#include "core/mat/mat.pb.h"
#include "core/mat/mat_io.hpp"

namespace core {
TEST_CASE("Mat construction and basic properties", "[mat]") {
  Mat mat(MatShape::make_3d(3, 5, 2), 1.5f);
  REQUIRE(mat.rows() == 3);
  REQUIRE(mat.cols() == 5);
  REQUIRE(mat.channels() == 2);
  REQUIRE(mat.size() == 30);  // 3 * 5 * 2 = 30
  REQUIRE(mat(0, 0, 0) == 1.5f);
  REQUIRE(mat(2, 4, 1) == 1.5f);

  Mat tiny(MatShape::make_3d(1, 1, 1), 42.0f);
  REQUIRE(tiny.size() == 1);
  REQUIRE(tiny(0, 0) == 42.0f);
}

TEST_CASE("Mat element access", "[mat]") {
  Mat mat(MatShape::make_3d(2, 3, 2), 0.0f);

  mat(1, 2, 0) = 3.14f;
  mat(1, 2, 1) = 2.71f;
  REQUIRE(mat(1, 2, 0) == 3.14f);
  REQUIRE(mat(1, 2, 1) == 2.71f);

  Mat single(MatShape::make_3d(2, 2, 1), 0.0f);
  single(1, 1) = 5.0f;
  REQUIRE(single(1, 1) == single(1, 1, 0));
}

TEST_CASE("Mat cloning and equality", "[mat]") {
  Mat original(MatShape::make_3d(2, 2, 1), 1.0f);
  original(0, 0) = 42.0f;

  Mat cloned = original.clone();
  REQUIRE(cloned == original);

  cloned(0, 0) = 99.0f;
  REQUIRE(original(0, 0) == 42.0f);
  REQUIRE(cloned(0, 0) == 99.0f);
  REQUIRE(original != cloned);

  Mat different(MatShape::make_3d(2, 3, 1), 1.0f);
  REQUIRE(original != different);
}

TEST_CASE("Mat arithmetic operations", "[mat]") {
  Mat base(MatShape::make_3d(2, 2, 1), 0.0f);
  base(0, 0) = 1.0f;
  base(0, 1) = 2.0f;
  base(1, 0) = 3.0f;
  base(1, 1) = 4.0f;

  Mat add_result = base + 10.0f;
  REQUIRE(add_result(0, 0) == 11.0f);
  REQUIRE(add_result(1, 1) == 14.0f);

  Mat mul_result = base * 2.0f;
  REQUIRE(mul_result(0, 0) == 2.0f);
  REQUIRE(mul_result(1, 1) == 8.0f);

  REQUIRE(3.0f * base == base * 3.0f);
  REQUIRE(3.0 * base == base * 3.0);  // double

  Mat neg = -base;
  REQUIRE(neg(0, 0) == -1.0f);
  REQUIRE(neg(1, 1) == -4.0f);
}

TEST_CASE("Mat helper functions and edge cases", "[mat]") {
  Mat zeros_mat = zeros(2, 3, 2);
  Mat ones_mat = ones(2, 3, 2);
  REQUIRE(zeros_mat(0, 0, 0) == 0.0f);
  REQUIRE(ones_mat(1, 2, 1) == 1.0f);

  Mat negative(MatShape::make_3d(1, 1, 1), -5.0f);
  Mat tiny(MatShape::make_3d(1, 1, 1), 1e-6f);
  Mat large(MatShape::make_3d(1, 1, 1), 1e6f);
  REQUIRE(negative(0, 0) == -5.0f);
  REQUIRE(tiny(0, 0) == 1e-6f);
  REQUIRE(large(0, 0) == 1e6f);
}

TEST_CASE("Mat layout memory verification", "[mat]") {
  Mat hwc(MatShape::make_3d(2, 2, 2), 0.0f, MatLayout::NHWC);
  Mat chw(MatShape::make_3d(2, 2, 2), 0.0f, MatLayout::NCHW);

  hwc(0, 0, 0) = 1.0f;
  hwc(0, 0, 1) = 2.0f;
  hwc(0, 1, 0) = 3.0f;
  hwc(1, 0, 0) = 4.0f;

  chw(0, 0, 0) = 1.0f;
  chw(0, 0, 1) = 2.0f;
  chw(0, 1, 0) = 3.0f;
  chw(1, 0, 0) = 4.0f;

  REQUIRE(hwc.layout() == MatLayout::NHWC);
  REQUIRE(chw.layout() == MatLayout::NCHW);

  REQUIRE(hwc == chw);

  const float *hwc_data = hwc.data();
  const float *chw_data = chw.data();

  REQUIRE(hwc_data[0] == 1.0f);  // 0,0,0
  REQUIRE(hwc_data[1] == 2.0f);  // 0,0,1
  REQUIRE(hwc_data[2] == 3.0f);  // 0,1,0
  REQUIRE(hwc_data[3] == 0.0f);  // 0,1,1
  REQUIRE(hwc_data[4] == 4.0f);  // 1,0,0

  REQUIRE(chw_data[0] == 1.0f);  // 0, 0, 0
  REQUIRE(chw_data[1] == 3.0f);  // 0, 0, 1
  REQUIRE(chw_data[2] == 4.0f);  // 0, 1, 0
  REQUIRE(chw_data[3] == 0.0f);  // 0, 1, 1
  REQUIRE(chw_data[4] == 2.0f);  // 1, 0, 0

  bool layouts_different = false;
  for (size_t i = 0; i < hwc.size(); ++i) {
    if (hwc_data[i] != chw_data[i]) {
      layouts_different = true;
      break;
    }
  }
  REQUIRE(layouts_different);
}

TEST_CASE("Mat layout conversion", "[mat]") {
  Mat hwc(MatShape::make_3d(2, 4, 3), 0.0f, MatLayout::NHWC);
  Mat chw(MatShape::make_3d(2, 4, 3), 0.0f, MatLayout::NCHW);
  hwc(1, 2, 2) = 1.0f;
  chw(1, 2, 2) = 1.0f;

  REQUIRE(hwc.layout() == MatLayout::NHWC);
  REQUIRE(chw.layout() == MatLayout::NCHW);

  REQUIRE(hwc == chw);
  Mat hwc_from_chw = chw.to_layout(MatLayout::NHWC);
  REQUIRE(hwc_from_chw == chw);
  REQUIRE(hwc_from_chw.layout() == MatLayout::NHWC);

  // Test no-op conversion
  Mat chw_from_chw = chw.to_layout(MatLayout::NCHW);
  REQUIRE(chw_from_chw == chw);
  REQUIRE(chw_from_chw.layout() == MatLayout::NCHW);
}

TEST_CASE("Mat layout access patterns", "[mat]") {
  Mat hwc(MatShape::make_3d(3, 4, 2), 0.0f, MatLayout::NHWC);
  Mat chw(MatShape::make_3d(3, 4, 2), 0.0f, MatLayout::NCHW);

  for (size_t r = 0; r < 3; ++r) {
    for (size_t c = 0; c < 4; ++c) {
      for (size_t ch = 0; ch < 2; ++ch) {
        float value = r * 100 + c * 10 + ch;
        hwc(r, c, ch) = value;
        chw(r, c, ch) = value;
      }
    }
  }

  REQUIRE(hwc == chw);

  REQUIRE(hwc(2, 3, 1) == 231.0f);  // 2*100 + 3*10 + 1 = 231
  REQUIRE(chw(2, 3, 1) == 231.0f);
  REQUIRE(hwc(0, 0, 0) == 0.0f);
  REQUIRE(chw(0, 0, 0) == 0.0f);
}

TEST_CASE("Mat proto serialization and deserialization", "[mat][proto]") {
  Mat original(MatShape::make_3d(2, 3, 2), 0.0f);
  original(0, 0, 0) = 1.5f;
  original(0, 1, 1) = 2.7f;
  original(1, 2, 0) = 3.14f;
  original(1, 0, 1) = -1.0f;

  const v1::Mat proto = to_proto(original);

  REQUIRE(proto.rows() == 2);
  REQUIRE(proto.cols() == 3);
  REQUIRE(proto.channels() == 2);
  REQUIRE(proto.data().size() == 12 * sizeof(float));  // 2 * 3 * 2

  const std::string &byte_data = proto.data();
  const float *float_data = reinterpret_cast<const float *>(byte_data.data());

  constexpr int index_0_0_0 = 0 * 3 * 2 + 0 * 2 + 0;
  constexpr int index_0_1_1 = 0 * 3 * 2 + 1 * 2 + 1;
  constexpr int index_1_2_0 = 1 * 3 * 2 + 2 * 2 + 0;
  constexpr int index_1_0_1 = 1 * 3 * 2 + 0 * 2 + 1;

  REQUIRE(float_data[index_0_0_0] == 1.5f);
  REQUIRE(float_data[index_0_1_1] == 2.7f);
  REQUIRE(float_data[index_1_2_0] == 3.14f);
  REQUIRE(float_data[index_1_0_1] == -1.0f);

  auto expect_reconstructed = from_proto(proto);
  REQUIRE(expect_reconstructed.has_value());
  Mat reconstructed = expect_reconstructed.value();

  REQUIRE(reconstructed.rows() == original.rows());
  REQUIRE(reconstructed.cols() == original.cols());
  REQUIRE(reconstructed.channels() == original.channels());
  REQUIRE(reconstructed == original);

  REQUIRE(reconstructed(0, 0, 0) == 1.5f);
  REQUIRE(reconstructed(0, 1, 1) == 2.7f);
  REQUIRE(reconstructed(1, 2, 0) == 3.14f);
  REQUIRE(reconstructed(1, 0, 1) == -1.0f);

  reconstructed(1, 1, 1) = 99.9f;
  v1::Mat updated_proto = to_proto(reconstructed);

  constexpr int index_1_1_1 = 1 * 3 * 2 + 1 * 2 + 1;

  const std::string &updated_byte_data = updated_proto.data();
  const float *updated_float_data =
      reinterpret_cast<const float *>(updated_byte_data.data());
  REQUIRE(updated_float_data[index_1_1_1] == 99.9f);
}
}  // namespace core
