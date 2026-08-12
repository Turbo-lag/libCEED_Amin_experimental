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

//------------------------------------------------------------------------------
// Tensor Contract Apply
//------------------------------------------------------------------------------
static int CeedTensorContractApply_Xsmm(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t,
                                        CeedTransposeMode t_mode, const CeedInt add, const CeedScalar *restrict u, CeedScalar *restrict v) {
  CeedTensorContract_Xsmm *impl;

  CeedCallBackend(CeedTensorContractGetData(contract, &impl));
  if (C == 1) {
    if (!impl->kernel_is_valid || impl->A != A || impl->B != B || impl->C != C || impl->J != J || impl->t_mode != t_mode || impl->add != add) {
      const int                flags_t    = LIBXSMM_GEMM_FLAGS(!t_mode ? 'T' : 'N', 'N');
      const int                flags_ab   = (!add) ? LIBXSMM_GEMM_FLAG_BETA_0 : LIBXSMM_BASIC_GEMM_FLAG_NONE;
      const int                flags      = (flags_t | flags_ab);
      const libxsmm_gemm_shape gemm_shape = (CEED_SCALAR_TYPE == CEED_SCALAR_FP64)
                                                ? libxsmm_create_gemm_shape(J, A, B, !t_mode ? B : J, B, J, LIBXSMM_DATATYPE_F64,
                                                                            LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64)
                                                : libxsmm_create_gemm_shape(J, A, B, !t_mode ? B : J, B, J, LIBXSMM_DATATYPE_F32,
                                                                            LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);

      impl->kernel          = libxsmm_dispatch_gemm(gemm_shape, (libxsmm_bitfield)(flags), (libxsmm_bitfield)LIBXSMM_GEMM_PREFETCH_NONE);
      impl->A               = A;
      impl->B               = B;
      impl->C               = C;
      impl->J               = J;
      impl->t_mode          = t_mode;
      impl->add             = add;
      impl->kernel_is_valid = true;
    }
    libxsmm_gemm_param gemm_param;

    CeedCheck(impl->kernel, CeedTensorContractReturnCeed(contract), CEED_ERROR_BACKEND, "LIBXSMM kernel failed to build.");

    // Run kernel
    gemm_param.a.primary = (CeedScalar *)&t[0];
    gemm_param.b.primary = (CeedScalar *)&u[0];
    gemm_param.c.primary = (CeedScalar *)&v[0];
    impl->kernel(&gemm_param);
  } else {
    if (!impl->kernel_is_valid || impl->A != A || impl->B != B || impl->C != C || impl->J != J || impl->t_mode != t_mode || impl->add != add) {
      const int                flags_t    = LIBXSMM_GEMM_FLAGS('N', t_mode ? 'T' : 'N');
      const int                flags_ab   = (!add) ? LIBXSMM_GEMM_FLAG_BETA_0 : LIBXSMM_BASIC_GEMM_FLAG_NONE;
      const int                flags      = (flags_t | flags_ab);
      const libxsmm_gemm_shape gemm_shape = (CEED_SCALAR_TYPE == CEED_SCALAR_FP64)
                                                ? libxsmm_create_gemm_shape(C, J, B, C, !t_mode ? B : J, C, LIBXSMM_DATATYPE_F64,
                                                                            LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64)
                                                : libxsmm_create_gemm_shape(C, J, B, C, !t_mode ? B : J, C, LIBXSMM_DATATYPE_F32,
                                                                            LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);

      impl->kernel          = libxsmm_dispatch_gemm(gemm_shape, (libxsmm_bitfield)(flags), (libxsmm_bitfield)LIBXSMM_GEMM_PREFETCH_NONE);
      impl->A               = A;
      impl->B               = B;
      impl->C               = C;
      impl->J               = J;
      impl->t_mode          = t_mode;
      impl->add             = add;
      impl->kernel_is_valid = true;
    }
    libxsmm_gemm_param gemm_param;

    CeedCheck(impl->kernel, CeedTensorContractReturnCeed(contract), CEED_ERROR_BACKEND, "LIBXSMM kernel failed to build.");

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
