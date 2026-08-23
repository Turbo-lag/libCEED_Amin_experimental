// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Test scalar direct mapped assembly against the directional fallback
/// \test Test scalar direct mapped assembly against the directional fallback
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "t575-operator.h"

static int SetOperatorFields(Ceed ceed, CeedQFunction qf, CeedElemRestriction elem_restriction_u, CeedElemRestriction elem_restriction_q_data,
                             CeedBasis basis_u, CeedVector q_data, CeedOperator *op) {
  CeedOperatorCreate(ceed, qf, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, op);
  CeedOperatorSetField(*op, "u", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(*op, "du", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(*op, "q data", elem_restriction_q_data, CEED_BASIS_NONE, q_data);
  CeedOperatorSetField(*op, "v", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(*op, "dv", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  return 0;
}

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_u, elem_restriction_q_data;
  CeedBasis           basis_u;
  CeedQFunction       qf_direct, qf_fallback;
  CeedOperator        op_direct, op_fallback;
  CeedVector           q_data, assembled_mapped = NULL, assembled_fallback = NULL;
  CeedQFunctionContext qf_context;
  // On the opt backend, make the directional apply deliberately differ from
  // the direct callback. Equality with the independent fallback operator then
  // proves the mapped path used the callback, without relying on context-copy
  // side effects. Reference backends receive a consistent apply for fallback
  // coverage because they do not implement the opt mapped path.
  ScalarAssemblyContext context = {0};
  const CeedInt         dim = 2, p = 3, q = 4, num_elem = 1, num_dofs = p * p, num_qpts = q * q;
  CeedInt               offsets[9];
  int                   error_count = 0;

  if (argc < 2) return 1;
  context.apply_scale = strstr(argv[1], "/opt/") ? 2. : 1.;
  CeedInit(argv[1], &ceed);
  for (CeedInt i = 0; i < num_dofs; i++) offsets[i] = i;
  CeedElemRestrictionCreate(ceed, num_elem, num_dofs, 1, 1, num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, offsets, &elem_restriction_u);
  const CeedInt q_data_strides[3] = {1, num_qpts, 9 * num_qpts};
  CeedElemRestrictionCreateStrided(ceed, num_elem, num_qpts, 9, 9 * num_qpts, q_data_strides, &elem_restriction_q_data);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, p, q, CEED_GAUSS, &basis_u);

  CeedVectorCreate(ceed, 9 * num_qpts, &q_data);
  CeedVectorSetValue(q_data, 0.);
  {
    CeedScalar *q_data_array;

    CeedVectorGetArray(q_data, CEED_MEM_HOST, &q_data_array);
    for (CeedInt row = 0; row < 3; row++) {
      for (CeedInt col = 0; col < 3; col++) {
        for (CeedInt i = 0; i < num_qpts; i++) {
          q_data_array[(row * 3 + col) * num_qpts + i] = 0.2 * (1 + row) - 0.07 * col + 0.003 * i + (row == col ? 1. : 0.);
        }
      }
    }
    CeedVectorRestoreArray(q_data, &q_data_array);
  }

  CeedQFunctionCreateInterior(ceed, 1, scalar_apply, scalar_apply_loc, &qf_direct);
  CeedQFunctionAddInput(qf_direct, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_direct, "du", dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_direct, "q data", 9, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_direct, "v", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_direct, "dv", dim, CEED_EVAL_GRAD);
  CeedQFunctionSetAssemblyUserFunction(qf_direct, scalar_assembly);
  CeedQFunctionContextCreate(ceed, &qf_context);
  CeedQFunctionContextSetData(qf_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(context), &context);
  CeedQFunctionSetContext(qf_direct, qf_context);

  CeedQFunctionCreateInterior(ceed, 1, scalar_apply_fallback, scalar_apply_fallback_loc, &qf_fallback);
  CeedQFunctionAddInput(qf_fallback, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_fallback, "du", dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_fallback, "q data", 9, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_fallback, "v", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_fallback, "dv", dim, CEED_EVAL_GRAD);

  SetOperatorFields(ceed, qf_direct, elem_restriction_u, elem_restriction_q_data, basis_u, q_data, &op_direct);
  SetOperatorFields(ceed, qf_fallback, elem_restriction_u, elem_restriction_q_data, basis_u, q_data, &op_fallback);

  CeedSize num_entries_fallback;
  CeedInt *rows_fallback, *cols_fallback;

  CeedOperatorLinearAssembleSymbolic(op_fallback, &num_entries_fallback, &rows_fallback, &cols_fallback);
  if (num_entries_fallback != (CeedSize)num_dofs * num_dofs) {
    printf("Unexpected scalar block entry count: %zu != %" CeedInt_FMT "\n", num_entries_fallback, num_dofs * num_dofs);
    error_count++;
  } else {
    const CeedInt compact_size = 17;
    CeedInt       entry_map[81];
    CeedScalar    expected[17] = {0.};

    // Deliberately reduce many element entries into each compact destination;
    // this exercises the mapped path's += semantics, not merely identity copy.
    for (CeedInt i = 0; i < num_dofs * num_dofs; i++) entry_map[i] = (7 * i + 3 * (i / num_dofs)) % compact_size;
    CeedVectorCreate(ceed, compact_size, &assembled_mapped);
    CeedVectorCreate(ceed, num_entries_fallback, &assembled_fallback);
    CeedOperatorLinearAssembleMappedRepeatedBlock(op_direct, entry_map, assembled_mapped);
    CeedOperatorLinearAssemble(op_fallback, assembled_fallback);

    const CeedScalar *mapped_array, *fallback_array;

    CeedVectorGetArrayRead(assembled_mapped, CEED_MEM_HOST, &mapped_array);
    CeedVectorGetArrayRead(assembled_fallback, CEED_MEM_HOST, &fallback_array);
    for (CeedSize i = 0; i < num_entries_fallback; i++) expected[entry_map[i]] += fallback_array[i];
    for (CeedInt i = 0; i < compact_size; i++) {
      const CeedScalar scale = fmax(1., fmax(fabs(mapped_array[i]), fabs(expected[i])));

      if (fabs(mapped_array[i] - expected[i]) > 500. * CEED_EPSILON * scale) {
        printf("Scalar mapped assembly mismatch at compact entry %" CeedInt_FMT ": %.16e != %.16e\n", i, mapped_array[i], expected[i]);
        error_count++;
        break;
      }
    }
    CeedVectorRestoreArrayRead(assembled_mapped, &mapped_array);
    CeedVectorRestoreArrayRead(assembled_fallback, &fallback_array);
  }
  free(rows_fallback);
  free(cols_fallback);
  CeedVectorDestroy(&q_data);
  CeedVectorDestroy(&assembled_mapped);
  CeedVectorDestroy(&assembled_fallback);
  CeedElemRestrictionDestroy(&elem_restriction_u);
  CeedElemRestrictionDestroy(&elem_restriction_q_data);
  CeedBasisDestroy(&basis_u);
  CeedQFunctionContextDestroy(&qf_context);
  CeedQFunctionDestroy(&qf_direct);
  CeedQFunctionDestroy(&qf_fallback);
  CeedOperatorDestroy(&op_direct);
  CeedOperatorDestroy(&op_fallback);
  CeedDestroy(&ceed);
  return error_count ? 1 : 0;
}
