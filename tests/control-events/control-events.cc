/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cublas_v2.h>
#include <cuda.h>
#include <realm.h>
#include <realm/cuda/cuda_module.h>
#include <realm/event.h>

#include "defer.h"
#include "processor.h"
#include "shape.h"
#include "store.h"
#include "tiled_array.h"

constexpr int64_t M = 8192;
constexpr int64_t N = 8192;
constexpr int64_t K = 8192;
constexpr cudaDataType_t kDataType = CUDA_R_32F;
constexpr cublasComputeType_t kComputeType = CUBLAS_COMPUTE_32F_FAST_TF32;
constexpr cublasGemmAlgo_t kAlgoType = CUBLAS_GEMM_DEFAULT;
constexpr cublasOperation_t kOperType = CUBLAS_OP_N;
constexpr int64_t kNumTasks = 10;
constexpr int64_t kCallsPerTask = 3;

struct RunCublas {
  void operator()(zuku::Stream* zs, cublasHandle_t handle, CUstream stream,
                  zuku::ArrayTile& c, const zuku::ArrayTile& a,
                  const zuku::ArrayTile& b) const {
    cublasSetStream(handle, stream);
    float alpha = 1.0;
    float beta = 0.5;
    for (int i = 0; i < kCallsPerTask; ++i) {
      cublasGemmEx(handle, kOperType, kOperType, M, N, K, &alpha, a.data(),
                   kDataType, M, b.data(), kDataType, K, &beta, c.data(),
                   kDataType, M, kComputeType, kAlgoType);
    }
    zs->Record(reinterpret_cast<zuku::EventStream*>(stream));
  }
};

auto run() {
  zuku::TileShape shapeA{
      .type = zuku::SupportedType::F32,
      .dims = {M, K},
  };
  zuku::TileShape shapeB{
      .type = zuku::SupportedType::F32,
      .dims = {K, N},
  };
  zuku::TileShape shapeC{.type = zuku::SupportedType::F32, .dims = {M, N}};

  auto proc = zuku::Processor::Default();

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_HOST);
  CUstream stream;
  cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);

  auto A = zuku::ArrayTile::CreateStore(shapeA, {.processor = proc});
  auto B = zuku::ArrayTile::CreateStore(shapeB, {.processor = proc});
  auto C = zuku::ArrayTile::CreateStore(shapeC, {.processor = proc});
  auto D = zuku::ArrayTile::CreateStore(shapeA, {.processor = proc});
  auto E = zuku::ArrayTile::CreateStore(shapeB, {.processor = proc});
  auto F = zuku::ArrayTile::CreateStore(shapeC, {.processor = proc});

  auto fillABC = on(proc).defer(
      [](zuku::ArrayTile& a, zuku::ArrayTile& b, zuku::ArrayTile& c) {
        cudaMemset(a.data(), a.byte_size(), 0);
        cudaMemset(b.data(), b.byte_size(), 0);
        cudaMemset(c.data(), c.byte_size(), 0);
      },
      A, B, C);

  auto fillDEF = on(proc).defer(
      [](zuku::ArrayTile& d, zuku::ArrayTile& e, zuku::ArrayTile& f) {
        cudaMemset(d.data(), d.byte_size(), 0);
        cudaMemset(e.data(), e.byte_size(), 0);
        cudaMemset(f.data(), f.byte_size(), 0);
      },
      D, E, F);

  Realm::UserEvent last_control_event = Realm::UserEvent::NO_USER_EVENT;
  Realm::Event last_exec_event = Realm::Event::NO_EVENT;
  for (int64_t i = 0; i < kNumTasks; ++i) {
    auto token = zuku::on(proc)
                     .region("cublas-abc")
                     .stream_ordered()
                     .after(last_control_event)
                     .defer(RunCublas{}, handle, stream, A, B, C);

    if (!token.HasControlEvent()) {
      std::cerr << "No control events on return token" << std::endl;
      abort();
    }
    last_control_event = token.ControlEvent();
    last_exec_event = token.Event();

    token = on(proc)
                .region("cublas-def")
                .stream_ordered()
                .after(last_control_event)
                .defer(RunCublas{}, handle, stream, D, E, F);

    last_control_event = token.ControlEvent();
    last_exec_event = token.Event();
  }

  C.Wait();
  F.Wait();
}

int main(int argc, char** argv) {
  return zuku::program(
      [] {
        std::cerr << "Hello world!" << std::endl;
        return run();
      },
      argc, argv,
      {.gpus = 1, .fbmem = 7 * M * N * sizeof(float) / 1e6, .network = false});
}
