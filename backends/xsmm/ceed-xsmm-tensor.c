// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <libxsmm.h>

#include "ceed-xsmm.h"

typedef struct {
  CeedInt              A, B, C, J, add;
  CeedTransposeMode    t_mode;
  libxsmm_gemmfunction kernel;
  bool                 kernel_is_valid;
} CeedTensorContract_Xsmm;

static int CeedTensorContractSetKernel_Xsmm(CeedTensorContract contract, CeedTensorContract_Xsmm *impl, CeedInt A, CeedInt B, CeedInt C,
                                            CeedInt J, CeedTransposeMode t_mode, CeedInt add) {
  if (impl->kernel_is_valid && impl->A == A && impl->B == B && impl->C == C && impl->J == J && impl->t_mode == t_mode && impl->add == add)
    return CEED_ERROR_SUCCESS;

  const int flags_t  = C == 1 ? LIBXSMM_GEMM_FLAGS(!t_mode ? 'T' : 'N', 'N') : LIBXSMM_GEMM_FLAGS('N', t_mode ? 'T' : 'N');
  const int flags_ab = !add ? LIBXSMM_GEMM_FLAG_BETA_0 : LIBXSMM_BASIC_GEMM_FLAG_NONE;
  const int m = C == 1 ? J : C, n = C == 1 ? A : J;
  const int lda = C == 1 ? (!t_mode ? B : J) : C, ldb = C == 1 ? B : (!t_mode ? B : J), ldc = C == 1 ? J : C;
  const libxsmm_datatype scalar_type = CEED_SCALAR_TYPE == CEED_SCALAR_FP64 ? LIBXSMM_DATATYPE_F64 : LIBXSMM_DATATYPE_F32;
  const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(m, n, B, lda, ldb, ldc, scalar_type, scalar_type, scalar_type, scalar_type);

  impl->kernel = libxsmm_dispatch_gemm(gemm_shape, (libxsmm_bitfield)(flags_t | flags_ab), (libxsmm_bitfield)LIBXSMM_GEMM_PREFETCH_NONE);
  CeedCheck(impl->kernel, CeedTensorContractReturnCeed(contract), CEED_ERROR_BACKEND, "LIBXSMM kernel failed to build.");
  impl->A               = A;
  impl->B               = B;
  impl->C               = C;
  impl->J               = J;
  impl->t_mode          = t_mode;
  impl->add             = add;
  impl->kernel_is_valid = true;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Apply
//------------------------------------------------------------------------------
static int CeedTensorContractApply_Xsmm(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t,
                                        CeedTransposeMode t_mode, const CeedInt add, const CeedScalar *restrict u, CeedScalar *restrict v) {
  CeedTensorContract_Xsmm *impl;
  libxsmm_gemm_param        gemm_param;

  CeedCallBackend(CeedTensorContractGetData(contract, &impl));
  CeedCallBackend(CeedTensorContractSetKernel_Xsmm(contract, impl, A, B, C, J, t_mode, add));
  if (C == 1) {
    // Run kernel
    gemm_param.a.primary = (CeedScalar *)&t[0];
    gemm_param.b.primary = (CeedScalar *)&u[0];
    gemm_param.c.primary = (CeedScalar *)&v[0];
    impl->kernel(&gemm_param);
  } else {
    // Run kernel
    gemm_param.b.primary = (CeedScalar *)&t[0];
    for (CeedInt a = 0; a < A; a++) {
      gemm_param.a.primary = (CeedScalar *)&u[a * B * C];
      gemm_param.c.primary = (CeedScalar *)&v[a * J * C];
      impl->kernel(&gemm_param);
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Destroy
//------------------------------------------------------------------------------
static int CeedTensorContractDestroy_Xsmm(CeedTensorContract contract) {
  CeedTensorContract_Xsmm *impl;

  CeedCallBackend(CeedTensorContractGetData(contract, &impl));
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Create
//------------------------------------------------------------------------------
int CeedTensorContractCreate_Xsmm(CeedTensorContract contract) {
  Ceed                     ceed = CeedTensorContractReturnCeed(contract);
  CeedTensorContract_Xsmm *impl;

  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedTensorContractSetData(contract, impl));
  CeedCallBackend(CeedSetBackendFunction(ceed, "TensorContract", contract, "Apply", CeedTensorContractApply_Xsmm));
  CeedCallBackend(CeedSetBackendFunction(ceed, "TensorContract", contract, "Destroy", CeedTensorContractDestroy_Xsmm));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
