/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gloo/common/logging.h"

#include <algorithm>
#include <numeric>

namespace gloo {

std::shared_ptr<spdlog::logger> get_logger(
    const std::string& logger_name,
    const std::unordered_set<spdlog::sink_ptr>& logger_sinks) {
  auto logger = spdlog::get(logger_name);
  if (logger == nullptr) {
    logger = std::make_shared<spdlog::logger>(logger_name);
    logger->set_level(spdlog::level::trace);
    logger->flush_on(spdlog::level::info);
  }

  auto& sinks = logger->sinks();
  for (const auto& sink : logger_sinks) {
    if (std::find(sinks.begin(), sinks.end(), sink) == sinks.end()) {
      sinks.push_back(sink);
    }
  }
  return logger;
}

EnforceNotMet::EnforceNotMet(
    const char* file,
    const int line,
    const char* condition,
    const std::string& msg)
    : msg_stack_{MakeString(
          "[enforce fail at ",
          file,
          ":",
          line,
          "] ",
          condition,
          ". ",
          msg)} {
  full_msg_ = this->msg();
}

std::string EnforceNotMet::msg() const {
  return std::accumulate(msg_stack_.begin(), msg_stack_.end(), std::string(""));
}

const char* EnforceNotMet::what() const noexcept {
  return full_msg_.c_str();
}

} // namespace gloo
