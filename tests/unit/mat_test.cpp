#include "core/mat.hpp"

#include <catch2/catch_test_macros.hpp>

#include "core/mat.pb.h"

TEST_CASE("Mat construction and basic properties", "[mat]") {
  core::Mat mat(3, 5, 2, 1.5f);
  REQUIRE(mat.rows() == 3);
  REQUIRE(mat.cols() == 5);
  REQUIRE(mat.channels() == 2);
  REQUIRE(mat.size() == 30);  // 3 * 5 * 2 = 30
  REQUIRE(mat(0, 0, 0) == 1.5f);
  REQUIRE(mat(2, 4, 1) == 1.5f);

  core::Mat tiny(1, 1, 1, 42.0f);
  REQUIRE(tiny.size() == 1);
  REQUIRE(tiny(0, 0) == 42.0f);
}

TEST_CASE("Mat element access", "[mat]") {
  core::Mat mat(2, 3, 2, 0.0f);

  mat(1, 2, 0) = 3.14f;
  mat(1, 2, 1) = 2.71f;
  REQUIRE(mat(1, 2, 0) == 3.14f);
  REQUIRE(mat(1, 2, 1) == 2.71f);

  core::Mat single(2, 2, 1, 0.0f);
  single(1, 1) = 5.0f;
  REQUIRE(single(1, 1) == single(1, 1, 0));

  // out of bounds access
  REQUIRE_THROWS_AS(mat(2, 0, 0), std::runtime_error);
  REQUIRE_THROWS_AS(mat(0, -1, 0), std::runtime_error);
  REQUIRE_THROWS_AS(single(0, 0, 1), std::runtime_error);  // single channel
}

TEST_CASE("Mat cloning and equality", "[mat]") {
  core::Mat original(2, 2, 1, 1.0f);
  original(0, 0) = 42.0f;

  core::Mat cloned = original.clone();
  REQUIRE(cloned == original);

  cloned(0, 0) = 99.0f;
  REQUIRE(original(0, 0) == 42.0f);
  REQUIRE(cloned(0, 0) == 99.0f);
  REQUIRE(original != cloned);

  core::Mat different(2, 3, 1, 1.0f);
  REQUIRE(original != different);
}

TEST_CASE("Mat arithmetic operations", "[mat]") {
  core::Mat base(2, 2, 1, 0.0f);
  base(0, 0) = 1.0f;
  base(0, 1) = 2.0f;
  base(1, 0) = 3.0f;
  base(1, 1) = 4.0f;

  core::Mat add_result = base + 10.0f;
  REQUIRE(add_result(0, 0) == 11.0f);
  REQUIRE(add_result(1, 1) == 14.0f);

  core::Mat mul_result = base * 2.0f;
  REQUIRE(mul_result(0, 0) == 2.0f);
  REQUIRE(mul_result(1, 1) == 8.0f);

  REQUIRE(3.0f * base == base * 3.0f);
  REQUIRE(3.0 * base == base * 3.0);  // double

  core::Mat other(2, 2, 1, 1.0f);
  core::Mat mat_add = base + other;
  core::Mat mat_sub = base - other;
  REQUIRE(mat_add(0, 0) == 2.0f);
  REQUIRE(mat_sub(0, 0) == 0.0f);

  core::Mat neg = -base;
  REQUIRE(neg(0, 0) == -1.0f);
  REQUIRE(neg(1, 1) == -4.0f);
}

TEST_CASE("Mat algebraic properties", "[mat]") {
  core::Mat a(2, 2, 1, 1.0f);
  a(0, 0) = 2.0f;
  core::Mat b(2, 2, 1, 3.0f);
  b(0, 0) = 4.0f;
  core::Mat c(2, 2, 1, 5.0f);

  // addition properties
  REQUIRE(a + b == b + a);                    // commutativity
  REQUIRE((a + b) + c == a + (b + c));        // associativity
  REQUIRE(a + core::zeros(2, 2, 1) == a);     // identity
  REQUIRE(a + (-a) == core::zeros(2, 2, 1));  // inverse

  // scalar multiplication properties
  constexpr double x = 2.0, y = 3.0;
  REQUIRE(x * (y * a) == (x * y) * a);    // associativity
  REQUIRE(1.0 * a == a);                  // identity
  REQUIRE((x + y) * a == x * a + y * a);  // distributivity
  REQUIRE(x * (a + b) == x * a + x * b);  // distributivity
}

TEST_CASE("Mat helper functions and edge cases", "[mat]") {
  core::Mat zeros = core::zeros(2, 3, 2);
  core::Mat ones = core::ones(2, 3, 2);
  REQUIRE(zeros(0, 0, 0) == 0.0f);
  REQUIRE(ones(1, 2, 1) == 1.0f);

  core::Mat negative(1, 1, 1, -5.0f);
  core::Mat tiny(1, 1, 1, 1e-6f);
  core::Mat large(1, 1, 1, 1e6f);
  REQUIRE(negative(0, 0) == -5.0f);
  REQUIRE(tiny(0, 0) == 1e-6f);
  REQUIRE(large(0, 0) == 1e6f);
}

TEST_CASE("Mat proto serialization and deserialization", "[mat][proto]") {
  core::Mat original(2, 3, 2, 0.0f);
  original(0, 0, 0) = 1.5f;
  original(0, 1, 1) = 2.7f;
  original(1, 2, 0) = 3.14f;
  original(1, 0, 1) = -1.0f;

  const core::v1::Mat proto = original.to_proto();

  REQUIRE(proto.rows() == 2);
  REQUIRE(proto.cols() == 3);
  REQUIRE(proto.channels() == 2);
  REQUIRE(proto.data().size() == 12 * sizeof(float));  // 2 * 3 * 2

  const std::string& byte_data = proto.data();
  const float* float_data = reinterpret_cast<const float*>(byte_data.data());

  constexpr int index_0_0_0 = 0 * 3 * 2 + 0 * 2 + 0;
  constexpr int index_0_1_1 = 0 * 3 * 2 + 1 * 2 + 1;
  constexpr int index_1_2_0 = 1 * 3 * 2 + 2 * 2 + 0;
  constexpr int index_1_0_1 = 1 * 3 * 2 + 0 * 2 + 1;

  REQUIRE(float_data[index_0_0_0] == 1.5f);
  REQUIRE(float_data[index_0_1_1] == 2.7f);
  REQUIRE(float_data[index_1_2_0] == 3.14f);
  REQUIRE(float_data[index_1_0_1] == -1.0f);

  core::Mat reconstructed(1, 1, 1, 0.0f);
  reconstructed.from_proto(proto);

  REQUIRE(reconstructed.rows() == original.rows());
  REQUIRE(reconstructed.cols() == original.cols());
  REQUIRE(reconstructed.channels() == original.channels());
  REQUIRE(reconstructed == original);

  REQUIRE(reconstructed(0, 0, 0) == 1.5f);
  REQUIRE(reconstructed(0, 1, 1) == 2.7f);
  REQUIRE(reconstructed(1, 2, 0) == 3.14f);
  REQUIRE(reconstructed(1, 0, 1) == -1.0f);

  reconstructed(1, 1, 1) = 99.9f;
  const core::v1::Mat updated_proto = reconstructed.to_proto();
  constexpr int index_1_1_1 = 1 * 3 * 2 + 1 * 2 + 1;

  const std::string& updated_byte_data = updated_proto.data();
  const float* updated_float_data =
      reinterpret_cast<const float*>(updated_byte_data.data());
  REQUIRE(updated_float_data[index_1_1_1] == 99.9f);
}
