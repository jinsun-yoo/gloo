/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>
#include <iostream>

namespace gloo {
namespace transport {

class Buffer {
 public:
  explicit Buffer(int slot, void* ptr, size_t size)
      : slot_(slot), ptr_(ptr), size_(size), debug_(false) {}
  virtual ~Buffer() = 0;

  virtual void setDebug(bool debug) {
    debug_ = debug;
  }

  virtual void send(size_t offset, size_t length, size_t roffset = 0, int imm_data = -1) = 0;

  // Send entire buffer by default
  void send() {
    send(0, size_);
  }

  virtual void recv(int wr_id) = 0;
  
  virtual void waitRecv() = 0;
  virtual void waitSend() = 0;
  
  // Default implementations for polling - can be overridden by derived classes
  virtual bool pollSend() {
    // Default behavior: try to wait with no timeout (non-blocking check)
    // This is a simple fallback - derived classes should provide better implementations
    std::cerr << "pollSend not implemented, returning false by default." << std::endl;
    return false;  // Conservatively return false by default
  }
  
  virtual bool pollRecv() {
    // Default behavior: try to wait with no timeout (non-blocking check)  
    // This is a simple fallback - derived classes should provide better implementations
    return false;  // Conservatively return false by default
  }
  virtual int pollQP() {
    // Default behavior: try to wait with no timeout (non-blocking check)  
    // This is a simple fallback - derived classes should provide better implementations
    return false;  // Conservatively return false by default
  }

 protected:
  int slot_;
  void* ptr_;
  size_t size_;
  bool debug_;
};

} // namespace transport
} // namespace gloo
