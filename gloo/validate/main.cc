/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>
#include <string>
#include <stdlib.h>
#include <malloc.h>
#include <unistd.h>

#include "gloo/allgather.h"
#include "gloo/allgather_ring.h"
#include "gloo/allgatherv.h"
#include "gloo/allreduce.h"
#include "gloo/allreduce_bcube.h"
#include "gloo/allreduce_halving_doubling.h"
#include "gloo/allreduce_local.h"
#include "gloo/allreduce_ring.h"
#include "gloo/allreduce_ring_chunked.h"
#include "gloo/alltoall.h"
#include "gloo/alltoallv.h"
#include "gloo/barrier_all_to_all.h"
#include "gloo/barrier_all_to_one.h"
#include "gloo/broadcast.h"
#include "gloo/broadcast_one_to_all.h"
#include "gloo/common/aligned_allocator.h"
#include "gloo/common/common.h"
#include "gloo/common/logging.h"
#include "gloo/context.h"
#include "gloo/pairwise_exchange.h"
#include "gloo/reduce.h"
#include "gloo/reduce_scatter.h"
#include "gloo/scatter.h"
#include "gloo/types.h"
#include "gloo/rendezvous/redis_store.h"
#include "gloo/rendezvous/context.h"
#include "gloo/transport/ibverbs/device.h"
#include <getopt.h>


using namespace gloo;

enum Mode {
  NONE,
  PINGPONG,
  RING,
};

struct ParsedArgs {
  int rank;
  std::string rdma_driver;
  int rdma_port;
  std::string redis_ip;
  int peer_rank;
  long int buff_size = 1 << 30; // 1
  int post_list = 1;
  int tx_depth = 128;
  int cq_mod = 1;
  bool skip_ack = false;
  int iter = -1;
  int send_peer;
  int recv_peer;
  Mode mode = PINGPONG;
  int world_size = 2;
bool is_client = false;
};

ParsedArgs parse_arguments(int argc, char *argv[]) {
	ParsedArgs args;
	const char* const short_opts = "r:b:s:i:m:w:c:";
	const option long_opts[] = {
		{"rank", required_argument, nullptr, 'r'},
		{"buff_size", required_argument, nullptr, 'b'},
		{"skip_ack", no_argument, nullptr, 's'},
		{"iter", required_argument, nullptr, 'i'},
		{"mode", required_argument, nullptr, 'm'},
		{"world_size", required_argument, nullptr, 'w'},
{"client", no_argument, nullptr, 'c'},
	};
	int opt;
	while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) != -1) {
		switch (opt) {
			case 'r':
				args.rank = std::stoi(optarg);
				std::cout << "Parse rank " << args.rank << " " << optarg << std::endl;
				break;
			case 'b':
				std::cout << "Parse buffer " << optarg << std::endl;
				args.buff_size = std::stol(optarg);
				break;
			case 's':
				args.skip_ack = true;
				break;
			case 'i':
				args.iter = std::stoi(optarg);
				std::cout << "Parse iter " << args.iter << " " << optarg << std::endl;
				break;
			case 'm':
				if (strcmp(optarg, "pingpong") == 0) {
					args.mode = PINGPONG;
				} else if (strcmp(optarg, "ring") == 0) {
					args.mode = RING;
				} else {
					std::cerr << "Unknown mode. Should be 'pingpong' or 'ring': " << optarg << std::endl;
					exit(1);
				}
				break;
			case 'w':
				args.world_size = std::stoi(optarg);
				std::cout << "Parse world size " << args.world_size << " " << optarg << std::endl;
				break;
case 'c':
				args.is_client = true;
				break;
			default:
				std::cerr << "Usage: " << argv[0]
				          << " --rank <rank> --buff_size <buffer_size>"
				          << std::endl;
				exit(1);
		}
	} 
	args.peer_rank = args.rank ^ 1;	// Assuming 2 ranks for simplicity
	args.send_peer = (args.rank + 1) % args.world_size;
	args.recv_peer = (args.rank + args.world_size - 1) % args.world_size;
	args.rdma_driver = "mlx5_" + std::to_string(args.rank);
	args.rdma_port = 1; // Assuming port 1 for simplicity
	args.redis_ip = "127.0.0.1"; // Assuming localhost for simplicity
	

	std::cout << "Parsed arguments:" << std::endl;
	std::cout << "  Rank: " << args.rank << std::endl;
	std::cout << "  RDMA Driver: " << args.rdma_driver << std::endl;
	std::cout << "  RDMA Port: " << args.rdma_port << std::endl;
	std::cout << "  Redis IP: " << args.redis_ip << std::endl;
	std::cout << "  Peer Rank: " << args.peer_rank << std::endl;
	std::cout << "  Buffer Size: " << args.buff_size << std::endl;
	std::cout << "  Post List: " << args.post_list << std::endl;
	std::cout << "  TX Depth: " << args.tx_depth << std::endl;
	std::cout << "  CQ Mod: " << args.cq_mod << std::endl;
	std::cout << "  Skip Ack: " << (args.skip_ack ? "true" : "false") << std::endl;
	std::cout << "  Iter: " << args.iter << std::endl;
	std::cout << "  Send Peer: " << args.send_peer << std::endl;
	std::cout << "  Recv Peer: " << args.recv_peer << std::endl;
	std::cout << "  Mode: " << (args.mode == PINGPONG ? "PINGPONG" : "RING") << std::endl;
	std::cout << "  World Size: " << args.world_size << std::endl;
std::cout << "  Is Client: " << args.is_client << std::endl;
	return args;
}

struct TestContext {
  std::shared_ptr<gloo::transport::Buffer> mr;
  std::shared_ptr<gloo::transport::Buffer> ack;
  std::shared_ptr<gloo::transport::Buffer> send_mr;
  std::shared_ptr<gloo::transport::Buffer> recv_mr;
  std::shared_ptr<Context> gloo_context;
  void *buf_start_addr;
};

int establish_connection(TestContext *test_ctx, ParsedArgs args) {
  std::cout << "Hello, world!" << std::endl;
  //Device name obtained by running 'rdma dev' on command line
  //Port from 'rdma link'
  auto ibv_attr = transport::ibverbs::attr{args.rdma_driver, args.rdma_port, 0};
  std::cout << "Initialize ibv attr" << std::endl;
  auto dev = transport::ibverbs::CreateDevice(ibv_attr);
  std::cout << "Initialize ibv dev" << std::endl;
  // Initialize the Gloo context
  auto rdzv_context = std::make_shared<rendezvous::Context>(args.rank, args.world_size);
  auto redis_store = std::make_shared<rendezvous::RedisStore>(args.redis_ip);
  rdzv_context->connectFullMesh(redis_store, dev);
  test_ctx->gloo_context = rdzv_context;
if (args.rank == 0) {
  	redis_store->flushall();
  }
  return 1;
}


void ctx_init_pingpong(TestContext *test_ctx, ParsedArgs args)  {
  // Create the MR
  std::cout << "Pingpong context initialized" << std::endl;
  auto cycle_buffer = sysconf(_SC_PAGESIZE);
  void *buf = memalign(cycle_buffer, args.buff_size);
  if (args.is_client) {
  	test_ctx->mr = test_ctx->gloo_context->getPair(args.peer_rank)->createSendBuffer(1, buf, args.buff_size);
  } else {
  	test_ctx->mr = test_ctx->gloo_context->getPair(args.peer_rank)->createRecvBuffer(1, buf, args.buff_size);
  }
  test_ctx->buf_start_addr = buf;

  void *ack_buf = memalign(cycle_buffer, 4);
  if (args.is_client) {
  	test_ctx->ack = test_ctx->gloo_context->getPair(args.peer_rank)->createRecvBuffer(2, buf, 4);
  } else {
  	test_ctx->ack = test_ctx->gloo_context->getPair(args.peer_rank)->createSendBuffer(2, buf, 4);
  }
  // Create the CQ
}

void ctx_init_ring(TestContext *test_ctx, ParsedArgs args) {
	std::cout << "Context initialized" << std::endl;
	auto cycle_buffer = sysconf(_SC_PAGESIZE);
	void *send_buf = memalign(cycle_buffer, args.buff_size);
	void *recv_buf = memalign(cycle_buffer, args.buff_size);
	test_ctx->send_mr = test_ctx->gloo_context->getPair(args.send_peer)->createSendBuffer(1, send_buf, args.buff_size);
	test_ctx->recv_mr = test_ctx->gloo_context->getPair(args.recv_peer)->createRecvBuffer(1, recv_buf, args.buff_size);
}

void set_up_connection() {
  return;
}

void ctx_set_send_wqes() {
  return;
}

int post_send_method(TestContext *test_ctx, int index, ParsedArgs args) {

}

#if 0
void run_iter_bw(TestContext *test_ctx, ParsedArgs args) {
	int totscnt = 0;
	int tot_iters = 0;
	int totccnt = 0;
	int sent = 0;
	int completed = 0;


	while (totscnt < tot_iters  || totccnt < tot_iters) {
		/* main loop to run over all the qps and post each time n messages */
		int index = 0;
		while (sent < tot_iters &&
				(sent + args.post_list) <= (args.tx_depth + completed)) {
			/* in multiple flow scenarios we will go to next cycle buffer address in the main buffer*/
			// From perftest_parameters->size
			int message_size = 65536;
			size_t offset  = sent * message_size;
			size_t remote_offset = offset;
			test_ctx->mr->send(offset, message_size, remote_offset);
			sent += args.post_list;
			totscnt += args.post_list;
			printf("post_send_method index %d scnt %lu\n",index,sent);

			/* ask for completion on this wr */
			// TODO: check if this is needed
			//ctx->wr[index].send_flags |= IBV_SEND_SIGNALED;
		}
		printf("End for loop. Current scnt is %d \n", sent);

		if (sent < tot_iters) {
			/* Make sure all completions from previous event were polled before waiting for another */
			if (args.use_event && ne == 0) {
				if (ctx_notify_events(ctx->send_channel)) {
					fprintf(stderr, "Couldn't request CQ notification\n");
					return_value = FAILURE;
					goto cleaning;
				}
			}
			ne = ibv_poll_cq(ctx->send_cq, CTX_POLL_BATCH, wc);
			printf("Polled %d new completions. Current totccnt is %d\n",ne, totccnt);
			if (ne > 0) {
				for (i = 0; i < ne; i++) {
					wc_id = (int)wc[i].wr_id;

					if (wc[i].status != IBV_WC_SUCCESS) {
						NOTIFY_COMP_ERROR_SEND(wc[i],totscnt,totccnt);
						return_value = FAILURE;
						goto cleaning;
					}
					int fill = args.cq_mod;
					if (args.fill_count && ctx->ccnt[wc_id] + args.cq_mod > args.iters) {
						fill = args.iters - ctx->ccnt[wc_id];
					}
					ctx->ccnt[wc_id] += fill;
					totccnt += fill;

					if (args.noPeak == OFF) {
						if (totccnt > tot_iters)
							args.tcompleted[args.iters*num_of_qps - 1] = get_cycles();
						else
							args.tcompleted[totccnt-1] = get_cycles();
					}

					if (args.test_type==DURATION && args.state == SAMPLE_STATE) {
						if (args.report_per_port) {
							args.iters_per_port[args.port_by_qp[wc_id]] += args.cq_mod;
						}
						args.iters += args.cq_mod;
					}
				}

			} else if (ne < 0) {
				fprintf(stderr, "poll CQ failed %d\n",ne);
				return_value = FAILURE;
				goto cleaning;
				}
		}
	}
	printf("Exit while loop\n");
}
#endif


void run_iter_pingpong_client (TestContext *test_ctx, ParsedArgs args) {
	int sent = 0;
	while(args.iter < 0? true : sent < args.iter) {
		int message_size = args.buff_size;
		size_t offset  = 0;//sent * message_size;
		size_t remote_offset = offset;
		//std::cout << "Start send " << sent << std::endl;
		test_ctx->mr->send(offset, message_size, remote_offset);
		//sent += args.post_list;
		//totscnt += args.post_list;
		//std::cout << "Sent " << sent << std::endl;
		sent += 1;
		test_ctx->mr->waitSend();
		if (!args.skip_ack) {
			test_ctx->ack->waitRecv();
		}
	}
}

void run_iter_pingpong_server (TestContext *test_ctx, ParsedArgs args) {
	int recv = 0;
	while(args.iter < 0? true : recv < args.iter) {
		//std::cout << "Start recv " << recv << std::endl;
		test_ctx->mr->waitRecv();
		recv += 1;
		//std::cout << "Recv" << recv << std::endl;
		if (!args.skip_ack) {
			test_ctx->ack->send();
			//sent += args.post_list;
			//totscnt += args.post_list;
			test_ctx->ack->waitSend();
		}
	}
}

void run_iter_ring (TestContext *test_ctx, ParsedArgs args) {
	int sent = 0;
	while(args.iter < 0? true : sent < args.iter) {
		test_ctx->send_mr->send();
		sent += 1;
		test_ctx->recv_mr->waitRecv();
		test_ctx->send_mr->waitSend();
	}
}

int main(int argc, char** argv) {
  TestContext *test_ctx = new TestContext();
  ParsedArgs args = parse_arguments(argc, argv);
  establish_connection(test_ctx, args);
  if (args.mode == RING) {
	ctx_init_ring(test_ctx, args);
  } else {
  	ctx_init_pingpong(test_ctx, args);
  }
  set_up_connection();
  ctx_set_send_wqes();
  //run_iter_bw(test_ctx, args);
  if (args.mode == RING) {
	std::cout << "Running ring test" << std::endl;
	run_iter_ring(test_ctx, args);
  } else if (args.is_client) {
  	std::cout << "Running pingpong test client" << std::endl;
	run_iter_pingpong_client(test_ctx, args);
  } else {
  	std::cout << "Running pingpong test server" << std::endl;
	run_iter_pingpong_server(test_ctx, args);
  }
  return 0;
}