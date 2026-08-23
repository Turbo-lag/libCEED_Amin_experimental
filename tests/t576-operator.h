// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/types.h>

typedef struct {
  CeedInt dim, state_comp, target_comp;
  CeedInt grad_in_value_out_zero;
  CeedInt apply_calls;
} ResidualContext;

typedef struct {
  CeedInt dim, num_comp;
  CeedInt apply_calls, assembly_calls;
} TargetContext;

CEED_QFUNCTION(residual_apply)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  ResidualContext *context = (ResidualContext *)ctx;
  const CeedScalar *q = in[0], *grad_q = in[1];
  CeedScalar       *v = out[0], *grad_v = out[1], *jac_data = out[2];
  const CeedInt     dim = context->dim, state_comp = context->state_comp;
  const CeedInt     target_size = context->target_comp * (dim + 1);

  context->apply_calls++;
  for (CeedInt i = 0; i < Q; i++) {
    for (CeedInt c = 0; c < state_comp; c++) {
      v[c * Q + i] = (1.1 + 0.03 * c) * q[c * Q + i];
      for (CeedInt d = 0; d < dim; d++) {
        grad_v[(d * state_comp + c) * Q + i] = (0.7 + 0.04 * d - 0.01 * c) * grad_q[(d * state_comp + c) * Q + i];
      }
    }
    for (CeedInt row = 0; row < target_size; row++) {
      for (CeedInt col = 0; col < target_size; col++) {
        const CeedScalar state = q[(row % state_comp) * Q + i];
        const CeedScalar grad  = grad_q[((col % dim) * state_comp + (col % state_comp)) * Q + i];

        jac_data[(row * target_size + col) * Q + i] = context->grad_in_value_out_zero && row < context->target_comp && col >= context->target_comp
                                                           ? 0.
                                                           : (row == col ? 1.3 : 0.) + 0.02 * (row + 1) - 0.009 * (col + 1) + 0.015 * state +
                                                                 0.006 * grad;
      }
    }
  }
  return 0;
}

CEED_QFUNCTION(target_apply)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  TargetContext   *context = (TargetContext *)ctx;
  const CeedScalar *u = in[0], *du = in[1], *jac_data = in[2];
  CeedScalar       *v = out[0], *dv = out[1];
  const CeedInt     dim = context->dim, num_comp = context->num_comp, size = num_comp * (dim + 1);

  context->apply_calls++;
  for (CeedInt i = 0; i < Q; i++) {
    for (CeedInt row = 0; row < size; row++) {
      CeedScalar value = 0.;

      for (CeedInt col = 0; col < size; col++) {
        const CeedScalar x = col < num_comp ? u[col * Q + i] : du[(col - num_comp) * Q + i];

        value += jac_data[(row * size + col) * Q + i] * x;
      }
      if (row < num_comp) v[row * Q + i] = value;
      else dv[(row - num_comp) * Q + i] = value;
    }
  }
  return 0;
}

CEED_QFUNCTION(target_assembly)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  TargetContext    *context  = (TargetContext *)ctx;
  const CeedScalar *jac_data = in[2];
  CeedScalar       *coeff    = out[0];
  const CeedInt     size     = context->num_comp * (context->dim + 1);

  context->assembly_calls++;
  for (CeedInt col = 0; col < size; col++) {
    for (CeedInt row = 0; row < size; row++) {
      for (CeedInt i = 0; i < Q; i++) coeff[(col * size + row) * Q + i] = jac_data[(row * size + col) * Q + i];
    }
  }
  return 0;
}
