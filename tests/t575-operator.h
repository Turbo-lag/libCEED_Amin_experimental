// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/types.h>

typedef struct {
  CeedScalar apply_scale;
} ScalarAssemblyContext;

CEED_QFUNCTION(scalar_apply)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const ScalarAssemblyContext *context = (const ScalarAssemblyContext *)ctx;
  const CeedScalar *u = in[0], *du = in[1], *q_data = in[2];
  CeedScalar       *v = out[0], *dv = out[1];
  const CeedScalar  apply_scale = context ? context->apply_scale : 1.;

  for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar x[3] = {u[i], du[i], du[Q + i]};

    v[i] = dv[i] = dv[Q + i] = 0.;
    for (CeedInt col = 0; col < 3; col++) {
      v[i] += apply_scale * q_data[(0 * 3 + col) * Q + i] * x[col];
      dv[i] += apply_scale * q_data[(1 * 3 + col) * Q + i] * x[col];
      dv[Q + i] += apply_scale * q_data[(2 * 3 + col) * Q + i] * x[col];
    }
  }
  return 0;
}

CEED_QFUNCTION(scalar_apply_fallback)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return scalar_apply(ctx, Q, in, out);
}

CEED_QFUNCTION(scalar_assembly)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *q_data = in[2];
  CeedScalar       *coeff  = out[0];

  for (CeedInt col = 0; col < 3; col++) {
    for (CeedInt row = 0; row < 3; row++) {
      for (CeedInt i = 0; i < Q; i++) coeff[(col * 3 + row) * Q + i] = q_data[(row * 3 + col) * Q + i];
    }
  }
  return 0;
}
