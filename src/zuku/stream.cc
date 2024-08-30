/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "stream.h"

#include "realm.h"
#include "realm/cuda/cuda_module.h"
#include <dlfcn.h>

#ifdef zuku_HAS_CUDA
#include <cuda.h>
typedef CUresult (*cuEventCreateFxn)(CUevent* phEvent, unsigned int Flags);
typedef CUresult (*cuEventRecordFxn)(CUevent hEvent, CUstream hStream);
typedef CUresult (*cuStreamWaitEventFxn)(CUstream hStream, CUevent hEvent,
                                         unsigned int Flags);
#endif

namespace zuku {

void GpuStream::DoRecord(EventStream* stream) {
#ifdef zuku_HAS_CUDA
  static void* libcuda = dlopen("libcuda.so.1", RTLD_NOW);
  if (libcuda == nullptr) {
    throw std::runtime_error("failed to dlopen libcuda in Zuku");
  }

  static auto* cEventCreate = (cuEventCreateFxn)dlsym(libcuda, "cuEventCreate");
  static auto* cEventRecord = (cuEventRecordFxn)dlsym(libcuda, "cuEventRecord");
  static auto* cStreamWaitEvent =
      (cuStreamWaitEventFxn)dlsym(libcuda, "cuStreamWaitEvent");
  if (cEventCreate == nullptr || cEventRecord == nullptr ||
      cStreamWaitEvent == nullptr) {
    throw std::runtime_error(
        "failed to dlopen some CUDA driver functions in Zuku");
  }

  // unsafe pointer fun
  CUstream cuda_stream = reinterpret_cast<CUstream>(stream);
  Realm::Cuda::set_task_ctxsync_required(false);

  CUevent done_event;
  CUresult result;
  if ((result = (*cEventCreate)(&done_event, CU_EVENT_DISABLE_TIMING)) !=
      CUDA_SUCCESS) {
    throw std::runtime_error("failed to create CUevent in GpuStream::Record");
  }

  if ((result = (*cEventRecord)(done_event, cuda_stream)) != CUDA_SUCCESS) {
    throw std::runtime_error("failed to record CUevent in GpuStream::Record");
  }

  CUstream task_stream = Realm::Cuda::get_task_cuda_stream();

  if ((result = (*cStreamWaitEvent)(task_stream, done_event,
                                    CU_EVENT_WAIT_DEFAULT)) != CUDA_SUCCESS) {
    throw std::runtime_error("failed cuStreamWaitEvent in GpuStream::Record");
  }
#else
  throw std::runtime_error(
      "GpuStream::Record: cannot be called, not built with CUDA");
#endif
}

}  // namespace zuku
