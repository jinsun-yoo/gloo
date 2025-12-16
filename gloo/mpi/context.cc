/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gloo/mpi/context.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <iostream>
#include <unistd.h>


#include "gloo/common/error.h"
#include "gloo/common/logging.h"
#include "gloo/transport/address.h"

namespace gloo {
namespace mpi {

static int MPICommSize(const MPI_Comm& comm) {
  int comm_size;
  auto error = MPI_Comm_size(comm, &comm_size);
  GLOO_ENFORCE(error == MPI_SUCCESS, "MPI_Comm_size: ", error);
  return comm_size;
}

static int MPICommRank(const MPI_Comm& comm) {
  int comm_rank;
  auto error = MPI_Comm_rank(comm, &comm_rank);
  GLOO_ENFORCE(error == MPI_SUCCESS, "MPI_Comm_rank: ", error);
  return comm_rank;
}

MPIScope::MPIScope() {
  auto rv = MPI_Init(nullptr, nullptr);
  GLOO_ENFORCE_EQ(rv, MPI_SUCCESS);
}

MPIScope::~MPIScope() {
  auto rv = MPI_Finalize();
  GLOO_ENFORCE_EQ(rv, MPI_SUCCESS);
}

namespace {

std::shared_ptr<MPIScope> getMPIScope() {
  static std::once_flag once;

  // Use weak pointer so that the initializer is destructed when the
  // last context referring to it is destructed, not when statics
  // are destructed on program termination.
  static std::weak_ptr<MPIScope> wptr;
  std::shared_ptr<MPIScope> sptr;

  // Create MPIScope only once
  std::call_once(once, [&]() {
    sptr = std::make_shared<MPIScope>();
    wptr = sptr;
  });

  // Create shared_ptr<MPIScope> from weak_ptr
  sptr = wptr.lock();
  GLOO_ENFORCE(sptr, "Cannot create MPI context after MPI_Finalize()");
  return sptr;
}

} // namespace

std::shared_ptr<Context> Context::createManaged() {
  auto mpiScope = getMPIScope();
  auto context = std::make_shared<Context>(MPI_COMM_WORLD);
  context->mpiScope_ = std::move(mpiScope);
  return context;
}

Context::Context(const MPI_Comm& comm, int nchannels)
    : ::gloo::Context(MPICommRank(comm), MPICommSize(comm), nchannels) {
  auto error = MPI_Comm_dup(comm, &comm_);
  GLOO_ENFORCE(error == MPI_SUCCESS, "MPI_Comm_dup: ", error);
}

Context::~Context() {
  MPI_Comm_free(&comm_);
}

void Context::connectFullMesh(std::shared_ptr<transport::Device>& dev) {
  std::vector<std::vector<char>> addresses(size * nchannels);
  // std::vector<std::vector<char>> addresses(size);
  unsigned long maxLength = 0;
  int rv;

  // Create pair to connect to every other node in the collective
  auto transportContext = dev->createContext(rank, size, nchannels);
  std::cout << "Create context" << std::endl;
  transportContext->setTimeout(getTimeout());
  std::cout << "Set timeout " << std::endl;
  for (int i = 0; i < size; i++) {
    if (i == rank) {
      continue;
    }

    for (int j = 0; j < nchannels; j++) {
      auto& pair = transportContext->createPair(i, j);

      // Store address for pair for this rank
      auto address = pair->address().bytes();
      maxLength = std::max(maxLength, address.size());
      // addresses[i] = std::move(address);
      addresses[i * nchannels + j] = std::move(address);
    }
  }
  std::cout << "Create pair" << std::endl;

  // Agree on maximum length so we can prepare buffers
  rv = MPI_Allreduce(
      MPI_IN_PLACE, &maxLength, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm_);
  if (rv != MPI_SUCCESS) {
    GLOO_THROW_IO_EXCEPTION("MPI_Allreduce: ", rv);
  }

  // Prepare input and output
  std::vector<char> in(size * nchannels * maxLength);
  std::vector<char> out(size * size * nchannels * maxLength);
  for (int i = 0; i < size; i++) {
    if (i == rank) {
      continue;
    }

    for (int j = 0; j < nchannels; j++) {
      auto& address = addresses[i * nchannels + j];
      memcpy(in.data() + (i * nchannels + j) * maxLength, address.data(), address.size());
      std::cout << "(" << rank << "," << i << "," << j << "," << i * nchannels + j << ")" << std::endl;
      // std::cout << "Rank " << rank << " peer rank " << i << " and slot " << j << " into addres idx " << i * nchannels + j <<std::endl;
    }
  }

  // Allgather to collect all addresses of all pairs
  rv = MPI_Allgather(
      in.data(), in.size(), MPI_BYTE, out.data(), in.size(), MPI_BYTE, comm_);
  if (rv != MPI_SUCCESS) {
    GLOO_THROW_IO_EXCEPTION("MPI_Allgather: ", rv);
  }

  // Connect every pair
  for (int i = 0; i < size; i++) {
    if (i == rank) {
      continue;
    }

    for (int j = 0; j < nchannels; j++) {
      //   auto offset = (rank + (i * size + j) * size) * maxLength;
      auto offset = (rank * nchannels + i * size * nchannels + j) * maxLength;
      std::vector<char> address(maxLength);
      memcpy(address.data(), out.data() + offset, maxLength);
      transportContext->getPair(i, j)->connect(address);
      std::string addr_str(address.begin(), address.end());
      std::cout << rank << "," << i << "," << j << "," << rank * nchannels + i * size * nchannels + j << std::endl;
      // std::cout << "Rank " << rank << " connect with rank " << i << " slot " << j << " get out from index " << rank * nchannels + i * size + j  << std::endl;
    }
  }

  device_ = dev;
  transportContext_ = std::move(transportContext);
  std::cout << "Exit" << std::endl;
}

} // namespace mpi
} // namespace gloo
