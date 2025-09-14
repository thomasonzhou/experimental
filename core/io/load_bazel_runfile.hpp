#include <iostream>
#include <memory>

#include "absl/log/log.h"
#include "rules_cc/cc/runfiles/runfiles.h"

namespace core::io {
std::string find_runfile_path(char** argv, const std::string& file_path) {
  std::string error;
  std::unique_ptr<rules_cc::cc::runfiles::Runfiles> runfiles(
      rules_cc::cc::runfiles::Runfiles::Create(argv[0], &error));

  if (runfiles == nullptr) {
    LOG(FATAL) << "Error initializing runfiles: " << error;
    return "";
  }
  std::string runfile_path = runfiles->Rlocation(file_path);
  LOG(INFO) << "Loading model from: " << runfile_path;
  return runfile_path;
}
}  // namespace core::io
