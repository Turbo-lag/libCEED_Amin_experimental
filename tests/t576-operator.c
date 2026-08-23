// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Test paired residual and mapped assembly on tensor-product and simplex volume/face operators
/// \test Test paired residual and mapped assembly on tensor-product and simplex volume/face operators
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "t576-operator.h"

typedef struct {
  CeedElemRestriction  residual_rstr, target_rstr, jac_data_rstr;
  CeedBasis            residual_basis, target_basis;
  CeedQFunction        residual_qf, target_qf;
  CeedQFunctionContext residual_context, target_context;
  CeedOperator         residual_op, target_op;
  CeedVector           jac_data;
  ResidualContext      residual_ctx;
  TargetContext        target_ctx;
  CeedSize             residual_l_size, target_l_size;
} OperatorPair;

static CeedInt IntPow(CeedInt base, CeedInt exponent) {
  CeedInt value = 1;

  for (CeedInt i = 0; i < exponent; i++) value *= base;
  return value;
}

static int CreateOperatorPairWithBases(Ceed ceed, CeedInt dim, CeedInt elem_size, CeedInt num_qpts, CeedInt target_comp,
                                       bool grad_in_value_out_zero, CeedInt num_elem, CeedInt num_nodes, const CeedInt node_offsets[],
                                       CeedBasis residual_basis, CeedBasis target_basis, OperatorPair *pair) {
  const CeedInt state_comp = target_comp + 1;
  const CeedInt target_qf_size = target_comp * (dim + 1), jac_size = target_qf_size * target_qf_size;
  CeedInt      *residual_offsets = malloc((CeedSize)num_elem * elem_size * sizeof(*residual_offsets));
  CeedInt      *target_offsets   = malloc((CeedSize)num_elem * elem_size * sizeof(*target_offsets));

  pair->residual_ctx    = (ResidualContext){.dim                    = dim,
                                            .state_comp             = state_comp,
                                            .target_comp            = target_comp,
                                            .grad_in_value_out_zero = grad_in_value_out_zero};
  pair->target_ctx      = (TargetContext){.dim = dim, .num_comp = target_comp};
  pair->residual_l_size = (CeedSize)num_nodes * state_comp;
  pair->target_l_size   = (CeedSize)num_nodes * target_comp;
  pair->residual_basis  = residual_basis;
  pair->target_basis    = target_basis;
  for (CeedInt i = 0; i < num_elem * elem_size; i++) {
    residual_offsets[i] = node_offsets[i] * state_comp;
    target_offsets[i]   = node_offsets[i] * target_comp;
  }
  CeedElemRestrictionCreate(ceed, num_elem, elem_size, state_comp, 1, pair->residual_l_size, CEED_MEM_HOST, CEED_COPY_VALUES, residual_offsets,
                            &pair->residual_rstr);
  CeedElemRestrictionCreate(ceed, num_elem, elem_size, target_comp, 1, pair->target_l_size, CEED_MEM_HOST, CEED_COPY_VALUES, target_offsets,
                            &pair->target_rstr);
  CeedElemRestrictionCreateStrided(ceed, num_elem, num_qpts, jac_size, (CeedSize)num_elem * num_qpts * jac_size, CEED_STRIDES_BACKEND,
                                   &pair->jac_data_rstr);
  free(residual_offsets);
  free(target_offsets);
  CeedVectorCreate(ceed, (CeedSize)num_elem * num_qpts * jac_size, &pair->jac_data);

  CeedQFunctionCreateInterior(ceed, 1, residual_apply, residual_apply_loc, &pair->residual_qf);
  CeedQFunctionAddInput(pair->residual_qf, "q", state_comp, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(pair->residual_qf, "Grad q", dim * state_comp, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(pair->residual_qf, "v", state_comp, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(pair->residual_qf, "Grad v", dim * state_comp, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(pair->residual_qf, "jac data", jac_size, CEED_EVAL_NONE);
  CeedQFunctionContextCreate(ceed, &pair->residual_context);
  CeedQFunctionContextSetData(pair->residual_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(pair->residual_ctx), &pair->residual_ctx);
  CeedQFunctionSetContext(pair->residual_qf, pair->residual_context);

  CeedQFunctionCreateInterior(ceed, 1, target_apply, target_apply_loc, &pair->target_qf);
  CeedQFunctionAddInput(pair->target_qf, "u", target_comp, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(pair->target_qf, "du", dim * target_comp, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(pair->target_qf, "jac data", jac_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(pair->target_qf, "v", target_comp, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(pair->target_qf, "dv", dim * target_comp, CEED_EVAL_GRAD);
  CeedQFunctionSetAssemblyUserFunction(pair->target_qf, target_assembly);
  CeedQFunctionSetAssemblyDataGradientInputToValueOutputZero(pair->target_qf, grad_in_value_out_zero);
  CeedQFunctionContextCreate(ceed, &pair->target_context);
  CeedQFunctionContextSetData(pair->target_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(pair->target_ctx), &pair->target_ctx);
  CeedQFunctionSetContext(pair->target_qf, pair->target_context);

  CeedOperatorCreate(ceed, pair->residual_qf, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &pair->residual_op);
  CeedOperatorSetField(pair->residual_op, "q", pair->residual_rstr, pair->residual_basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(pair->residual_op, "Grad q", pair->residual_rstr, pair->residual_basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(pair->residual_op, "v", pair->residual_rstr, pair->residual_basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(pair->residual_op, "Grad v", pair->residual_rstr, pair->residual_basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(pair->residual_op, "jac data", pair->jac_data_rstr, CEED_BASIS_NONE, pair->jac_data);

  CeedOperatorCreate(ceed, pair->target_qf, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &pair->target_op);
  CeedOperatorSetField(pair->target_op, "u", pair->target_rstr, pair->target_basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(pair->target_op, "du", pair->target_rstr, pair->target_basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(pair->target_op, "jac data", pair->jac_data_rstr, CEED_BASIS_NONE, pair->jac_data);
  CeedOperatorSetField(pair->target_op, "v", pair->target_rstr, pair->target_basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(pair->target_op, "dv", pair->target_rstr, pair->target_basis, CEED_VECTOR_ACTIVE);
  return 0;
}

static int CreateTensorOperatorPairWithQ(Ceed ceed, CeedInt dim, CeedInt p, CeedInt q, CeedInt target_comp, bool grad_in_value_out_zero,
                                         CeedInt num_elem, CeedInt num_nodes, const CeedInt node_offsets[], OperatorPair *pair) {
  const CeedInt state_comp = target_comp + 1, elem_size = IntPow(p, dim), num_qpts = IntPow(q, dim);
  CeedBasis     residual_basis, target_basis;

  CeedBasisCreateTensorH1Lagrange(ceed, dim, state_comp, p, q, CEED_GAUSS, &residual_basis);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, target_comp, p, q, CEED_GAUSS, &target_basis);
  return CreateOperatorPairWithBases(ceed, dim, elem_size, num_qpts, target_comp, grad_in_value_out_zero, num_elem, num_nodes, node_offsets,
                                     residual_basis, target_basis, pair);
}

static int CreateTensorOperatorPair(Ceed ceed, CeedInt dim, CeedInt p, CeedInt target_comp, bool grad_in_value_out_zero, CeedInt num_elem,
                                    CeedInt num_nodes, const CeedInt node_offsets[], OperatorPair *pair) {
  return CreateTensorOperatorPairWithQ(ceed, dim, p, p + 1, target_comp, grad_in_value_out_zero, num_elem, num_nodes, node_offsets, pair);
}

static void BuildTriangleBasis(CeedInt degree, CeedScalar q_ref[], CeedScalar q_weight[], CeedScalar interp[], CeedScalar grad[]) {
  const CeedInt elem_size = degree == 1 ? 3 : 6, num_qpts = 4;

  q_ref[0] = 0.2;
  q_ref[1] = 0.6;
  q_ref[2] = 1. / 3.;
  q_ref[3] = 0.2;
  q_ref[4] = 0.2;
  q_ref[5] = 0.2;
  q_ref[6] = 1. / 3.;
  q_ref[7] = 0.6;
  q_weight[0] = q_weight[1] = q_weight[3] = 25. / 96.;
  q_weight[2]                               = -27. / 96.;
  for (CeedInt q = 0; q < num_qpts; q++) {
    const CeedScalar x = q_ref[q], y = q_ref[num_qpts + q], l = 1. - x - y;

    if (degree == 1) {
      interp[q * elem_size + 0] = l;
      interp[q * elem_size + 1] = x;
      interp[q * elem_size + 2] = y;
      grad[(0 * num_qpts + q) * elem_size + 0] = -1.;
      grad[(0 * num_qpts + q) * elem_size + 1] = 1.;
      grad[(0 * num_qpts + q) * elem_size + 2] = 0.;
      grad[(1 * num_qpts + q) * elem_size + 0] = -1.;
      grad[(1 * num_qpts + q) * elem_size + 1] = 0.;
      grad[(1 * num_qpts + q) * elem_size + 2] = 1.;
    } else {
      interp[q * elem_size + 0] = l * (2. * l - 1.);
      interp[q * elem_size + 1] = 4. * l * x;
      interp[q * elem_size + 2] = x * (2. * x - 1.);
      interp[q * elem_size + 3] = 4. * l * y;
      interp[q * elem_size + 4] = 4. * x * y;
      interp[q * elem_size + 5] = y * (2. * y - 1.);
      grad[(0 * num_qpts + q) * elem_size + 0] = 1. - 4. * l;
      grad[(0 * num_qpts + q) * elem_size + 1] = 4. * (l - x);
      grad[(0 * num_qpts + q) * elem_size + 2] = 4. * x - 1.;
      grad[(0 * num_qpts + q) * elem_size + 3] = -4. * y;
      grad[(0 * num_qpts + q) * elem_size + 4] = 4. * y;
      grad[(0 * num_qpts + q) * elem_size + 5] = 0.;
      grad[(1 * num_qpts + q) * elem_size + 0] = 1. - 4. * l;
      grad[(1 * num_qpts + q) * elem_size + 1] = -4. * x;
      grad[(1 * num_qpts + q) * elem_size + 2] = 0.;
      grad[(1 * num_qpts + q) * elem_size + 3] = 4. * (l - y);
      grad[(1 * num_qpts + q) * elem_size + 4] = 4. * x;
      grad[(1 * num_qpts + q) * elem_size + 5] = 4. * y - 1.;
    }
  }
}

static void BuildLineBasis(CeedInt degree, CeedScalar q_ref[], CeedScalar q_weight[], CeedScalar interp[], CeedScalar grad[]) {
  const CeedInt elem_size = degree + 1, num_qpts = 4;
  const CeedScalar nodes[3] = {-1., degree == 1 ? 1. : 0., 1.};
  const CeedScalar points[4] = {-0.75, -0.25, 0.25, 0.75};

  for (CeedInt q = 0; q < num_qpts; q++) {
    q_ref[q]    = points[q];
    q_weight[q] = 0.5;
    for (CeedInt i = 0; i < elem_size; i++) {
      CeedScalar value = 1., derivative = 0.;

      for (CeedInt j = 0; j < elem_size; j++) {
        if (j != i) value *= (points[q] - nodes[j]) / (nodes[i] - nodes[j]);
      }
      for (CeedInt k = 0; k < elem_size; k++) {
        CeedScalar term = 1.;

        if (k == i) continue;
        term /= nodes[i] - nodes[k];
        for (CeedInt j = 0; j < elem_size; j++) {
          if (j != i && j != k) term *= (points[q] - nodes[j]) / (nodes[i] - nodes[j]);
        }
        derivative += term;
      }
      interp[q * elem_size + i] = value;
      grad[q * elem_size + i]   = derivative;
    }
  }
}

static int CreateSimplexOperatorPair(Ceed ceed, CeedInt dim, CeedInt degree, CeedInt target_comp, CeedInt num_elem, CeedInt num_nodes,
                                     const CeedInt node_offsets[], OperatorPair *pair) {
  const CeedInt elem_size = dim == 2 ? (degree == 1 ? 3 : 6) : degree + 1, num_qpts = 4;
  CeedScalar    q_ref[8], q_weight[4], interp[24], grad[48];
  CeedBasis     residual_basis, target_basis;

  if (dim == 2) BuildTriangleBasis(degree, q_ref, q_weight, interp, grad);
  else BuildLineBasis(degree, q_ref, q_weight, interp, grad);
  CeedBasisCreateH1(ceed, dim == 2 ? CEED_TOPOLOGY_TRIANGLE : CEED_TOPOLOGY_LINE, target_comp + 1, elem_size, num_qpts, interp, grad, q_ref,
                    q_weight, &residual_basis);
  CeedBasisCreateH1(ceed, dim == 2 ? CEED_TOPOLOGY_TRIANGLE : CEED_TOPOLOGY_LINE, target_comp, elem_size, num_qpts, interp, grad, q_ref, q_weight,
                    &target_basis);
  return CreateOperatorPairWithBases(ceed, dim, elem_size, num_qpts, target_comp, false, num_elem, num_nodes, node_offsets, residual_basis,
                                     target_basis, pair);
}

static int CreateOperatorsWithHandoffRestriction(const OperatorPair *pair, CeedElemRestriction handoff_rstr, CeedOperator *residual_op,
                                                  CeedOperator *target_op) {
  Ceed ceed;

  CeedOperatorGetCeed(pair->residual_op, &ceed);
  CeedOperatorCreate(ceed, pair->residual_qf, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, residual_op);
  CeedOperatorSetField(*residual_op, "q", pair->residual_rstr, pair->residual_basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(*residual_op, "Grad q", pair->residual_rstr, pair->residual_basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(*residual_op, "v", pair->residual_rstr, pair->residual_basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(*residual_op, "Grad v", pair->residual_rstr, pair->residual_basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(*residual_op, "jac data", handoff_rstr, CEED_BASIS_NONE, pair->jac_data);
  CeedOperatorCreate(ceed, pair->target_qf, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, target_op);
  CeedOperatorSetField(*target_op, "u", pair->target_rstr, pair->target_basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(*target_op, "du", pair->target_rstr, pair->target_basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(*target_op, "jac data", handoff_rstr, CEED_BASIS_NONE, pair->jac_data);
  CeedOperatorSetField(*target_op, "v", pair->target_rstr, pair->target_basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(*target_op, "dv", pair->target_rstr, pair->target_basis, CEED_VECTOR_ACTIVE);
  CeedDestroy(&ceed);
  return 0;
}

static int DestroyOperatorPair(OperatorPair *pair) {
  CeedOperatorDestroy(&pair->residual_op);
  CeedOperatorDestroy(&pair->target_op);
  CeedQFunctionDestroy(&pair->residual_qf);
  CeedQFunctionDestroy(&pair->target_qf);
  CeedQFunctionContextDestroy(&pair->residual_context);
  CeedQFunctionContextDestroy(&pair->target_context);
  CeedBasisDestroy(&pair->residual_basis);
  CeedBasisDestroy(&pair->target_basis);
  CeedElemRestrictionDestroy(&pair->residual_rstr);
  CeedElemRestrictionDestroy(&pair->target_rstr);
  CeedElemRestrictionDestroy(&pair->jac_data_rstr);
  CeedVectorDestroy(&pair->jac_data);
  return 0;
}

static int CompareVectors(const char *label, CeedVector actual, CeedVector expected) {
  CeedSize          actual_length, expected_length;
  const CeedScalar *actual_array, *expected_array;

  CeedVectorGetLength(actual, &actual_length);
  CeedVectorGetLength(expected, &expected_length);
  if (actual_length != expected_length) {
    printf("%s length mismatch: %zu != %zu\n", label, actual_length, expected_length);
    return 1;
  }
  CeedVectorGetArrayRead(actual, CEED_MEM_HOST, &actual_array);
  CeedVectorGetArrayRead(expected, CEED_MEM_HOST, &expected_array);
  for (CeedSize i = 0; i < actual_length; i++) {
    const CeedScalar scale = fmax(1., fmax(fabs(actual_array[i]), fabs(expected_array[i])));

    if (fabs(actual_array[i] - expected_array[i]) > 3000. * CEED_EPSILON * scale) {
      printf("%s mismatch at %zu: %.16e != %.16e\n", label, i, actual_array[i], expected_array[i]);
      CeedVectorRestoreArrayRead(actual, &actual_array);
      CeedVectorRestoreArrayRead(expected, &expected_array);
      return 1;
    }
  }
  CeedVectorRestoreArrayRead(actual, &actual_array);
  CeedVectorRestoreArrayRead(expected, &expected_array);
  return 0;
}

static int CheckPair(Ceed ceed, CeedOperator residual_op, CeedOperator target_op, CeedVector jac_data[], CeedInt num_jac_data,
                     CeedSize residual_l_size, const char *label, CeedInt map_seed, TargetContext *target_ctx,
                     bool expect_direct_assembly_callback, CeedOperator reference_target_op) {
  CeedSize   num_entries;
  CeedInt   *rows = NULL, *cols = NULL, *entry_map;
  CeedVector input, residual_sequential, residual_paired, values_sequential, values_paired;
  CeedVector *jac_expected = malloc(num_jac_data * sizeof(*jac_expected));
  int        error_count = 0;

  CeedOperatorLinearAssembleSymbolic(target_op, &num_entries, &rows, &cols);
  const CeedInt compact_size = 23 + map_seed;

  entry_map = malloc(num_entries * sizeof(*entry_map));
  for (CeedSize i = 0; i < num_entries; i++) {
    entry_map[i] = (CeedInt)((13 * i + 7 * (i / (map_seed + 1)) + map_seed) % compact_size);
  }
  free(rows);
  free(cols);
  CeedVectorCreate(ceed, residual_l_size, &input);
  CeedVectorCreate(ceed, residual_l_size, &residual_sequential);
  CeedVectorCreate(ceed, residual_l_size, &residual_paired);
  CeedVectorCreate(ceed, compact_size, &values_sequential);
  CeedVectorCreate(ceed, compact_size, &values_paired);
  {
    CeedScalar *input_array;

    CeedVectorGetArrayWrite(input, CEED_MEM_HOST, &input_array);
    for (CeedSize i = 0; i < residual_l_size; i++) input_array[i] = 0.2 + 0.003 * i - 0.00001 * i * i;
    CeedVectorRestoreArray(input, &input_array);
  }

  CeedOperatorApply(residual_op, input, residual_sequential, CEED_REQUEST_IMMEDIATE);
  CeedOperatorLinearAssembleMapped(reference_target_op ? reference_target_op : target_op, entry_map, values_sequential);
  for (CeedInt i = 0; i < num_jac_data; i++) {
    CeedSize          jac_length;
    const CeedScalar *jac_array;
    CeedScalar       *expected_array;

    CeedVectorGetLength(jac_data[i], &jac_length);
    CeedVectorCreate(ceed, jac_length, &jac_expected[i]);
    CeedVectorGetArrayRead(jac_data[i], CEED_MEM_HOST, &jac_array);
    CeedVectorGetArrayWrite(jac_expected[i], CEED_MEM_HOST, &expected_array);
    for (CeedSize j = 0; j < jac_length; j++) expected_array[j] = jac_array[j];
    CeedVectorRestoreArray(jac_expected[i], &expected_array);
    CeedVectorRestoreArrayRead(jac_data[i], &jac_array);
  }
  for (CeedInt i = 0; i < num_jac_data; i++) CeedVectorSetValue(jac_data[i], -91.25);
  const CeedInt target_apply_calls_before    = target_ctx ? target_ctx->apply_calls : 0;
  const CeedInt target_assembly_calls_before = target_ctx ? target_ctx->assembly_calls : 0;

  CeedOperatorApplyAndLinearAssembleMapped(residual_op, input, residual_paired, target_op, entry_map, values_paired);
  if (target_ctx && expect_direct_assembly_callback &&
      (target_ctx->assembly_calls == target_assembly_calls_before || target_ctx->apply_calls != target_apply_calls_before)) {
    printf("%s did not use the analytical assembly callback: apply delta %" CeedInt_FMT ", assembly delta %" CeedInt_FMT "\n", label,
           target_ctx->apply_calls - target_apply_calls_before, target_ctx->assembly_calls - target_assembly_calls_before);
    error_count++;
  }
  if (target_ctx && !expect_direct_assembly_callback &&
      (target_ctx->assembly_calls != target_assembly_calls_before || target_ctx->apply_calls == target_apply_calls_before)) {
    printf("%s did not use the directional fallback: apply delta %" CeedInt_FMT ", assembly delta %" CeedInt_FMT "\n", label,
           target_ctx->apply_calls - target_apply_calls_before, target_ctx->assembly_calls - target_assembly_calls_before);
    error_count++;
  }
  error_count += CompareVectors(label, residual_paired, residual_sequential);
  error_count += CompareVectors(label, values_paired, values_sequential);
  for (CeedInt i = 0; i < num_jac_data; i++) {
    error_count += CompareVectors(label, jac_data[i], jac_expected[i]);
  }

  // The fast path must leave all internal Q-vectors and shared passive vectors
  // reusable by the established paths. Exercise paired -> ordinary -> paired,
  // checking every public result and the materialized handoff each time.
  CeedVectorSetValue(residual_sequential, 73.);
  CeedVectorSetValue(values_sequential, 73.);
  CeedOperatorApply(residual_op, input, residual_sequential, CEED_REQUEST_IMMEDIATE);
  CeedOperatorLinearAssembleMapped(target_op, entry_map, values_sequential);
  error_count += CompareVectors(label, residual_sequential, residual_paired);
  error_count += CompareVectors(label, values_sequential, values_paired);
  for (CeedInt i = 0; i < num_jac_data; i++) error_count += CompareVectors(label, jac_data[i], jac_expected[i]);

  CeedVectorSetValue(residual_paired, -31.);
  CeedVectorSetValue(values_paired, -31.);
  for (CeedInt i = 0; i < num_jac_data; i++) CeedVectorSetValue(jac_data[i], -31.);
  CeedOperatorApplyAndLinearAssembleMapped(residual_op, input, residual_paired, target_op, entry_map, values_paired);
  error_count += CompareVectors(label, residual_paired, residual_sequential);
  error_count += CompareVectors(label, values_paired, values_sequential);
  for (CeedInt i = 0; i < num_jac_data; i++) error_count += CompareVectors(label, jac_data[i], jac_expected[i]);

  free(entry_map);
  CeedVectorDestroy(&input);
  CeedVectorDestroy(&residual_sequential);
  CeedVectorDestroy(&residual_paired);
  CeedVectorDestroy(&values_sequential);
  CeedVectorDestroy(&values_paired);
  for (CeedInt i = 0; i < num_jac_data; i++) CeedVectorDestroy(&jac_expected[i]);
  free(jac_expected);
  return error_count;
}

int main(int argc, char **argv) {
  Ceed ceed;
  int  error_count = 0;

  if (argc < 2) return 1;
  CeedInit(argv[1], &ceed);
  const bool expect_direct_assembly_callback = strstr(argv[1], "/opt/") || strstr(argv[1], "/xsmm/") || !strcmp(argv[1], "/cpu/self");
  for (CeedInt p = 2; p <= 3; p++) {
    for (CeedInt component_case = 0; component_case < 2; component_case++) {
      const CeedInt target_comp = component_case ? 1 : 5;
      const CeedInt nx = 2 * (p - 1) + 1, num_nodes = nx * p * p;
      const CeedInt volume_elem_size = p * p * p, face_elem_size = p * p;
      CeedInt      *volume_nodes = malloc(2 * volume_elem_size * sizeof(*volume_nodes));
      CeedInt      *face_nodes   = malloc(2 * face_elem_size * sizeof(*face_nodes));
      OperatorPair  volume = {0}, face = {0};
      CeedOperator  residual_composite, target_composite;

      for (CeedInt e = 0; e < 2; e++) {
        for (CeedInt k = 0; k < p; k++) {
          for (CeedInt j = 0; j < p; j++) {
            for (CeedInt i = 0; i < p; i++) {
              const CeedInt local = (k * p + j) * p + i;

              volume_nodes[e * volume_elem_size + local] = (k * p + j) * nx + e * (p - 1) + i;
            }
          }
        }
      }
      for (CeedInt e = 0; e < 2; e++) {
        const CeedInt x = e ? nx - 1 : 0;

        for (CeedInt j = 0; j < p; j++) {
          for (CeedInt i = 0; i < p; i++) face_nodes[e * face_elem_size + j * p + i] = (j * p + i) * nx + x;
        }
      }
      CreateTensorOperatorPair(ceed, 3, p, target_comp, false, 2, num_nodes, volume_nodes, &volume);
      CreateTensorOperatorPair(ceed, 2, p, target_comp, false, 2, num_nodes, face_nodes, &face);
      free(volume_nodes);
      free(face_nodes);

      {
        char       label[96];
        CeedVector volume_jac[] = {volume.jac_data}, face_jac[] = {face.jac_data};

        snprintf(label, sizeof(label), "hex P%" CeedInt_FMT " bs%" CeedInt_FMT, p - 1, target_comp);
        error_count += CheckPair(ceed, volume.residual_op, volume.target_op, volume_jac, 1, volume.residual_l_size, label, 3 + p + target_comp,
                                 &volume.target_ctx, expect_direct_assembly_callback, NULL);
        snprintf(label, sizeof(label), "quad face P%" CeedInt_FMT " bs%" CeedInt_FMT, p - 1, target_comp);
        error_count += CheckPair(ceed, face.residual_op, face.target_op, face_jac, 1, face.residual_l_size, label, 7 + p + target_comp,
                                 &face.target_ctx, expect_direct_assembly_callback, NULL);
      }
      {
        const CeedInt num_qpts = IntPow(p + 1, 3), target_qf_size = 4 * target_comp, jac_size = target_qf_size * target_qf_size;
        const CeedInt canonical_strides[3] = {1, num_qpts, num_qpts * jac_size};
        CeedElemRestriction canonical_jac_rstr;
        CeedOperator        canonical_residual_op, canonical_target_op;
        CeedVector          jac_data[] = {volume.jac_data};
        char                label[96];

        // Explicit canonical strides are as safe as backend strides when the
        // exact same restriction is shared by producer and consumer.
        CeedElemRestrictionCreateStrided(ceed, 2, num_qpts, jac_size, (CeedSize)2 * num_qpts * jac_size, canonical_strides,
                                         &canonical_jac_rstr);
        CreateOperatorsWithHandoffRestriction(&volume, canonical_jac_rstr, &canonical_residual_op, &canonical_target_op);
        snprintf(label, sizeof(label), "canonical handoff P%" CeedInt_FMT " bs%" CeedInt_FMT, p - 1, target_comp);
        error_count += CheckPair(ceed, canonical_residual_op, canonical_target_op, jac_data, 1, volume.residual_l_size, label,
                                 8 + p + target_comp, &volume.target_ctx, expect_direct_assembly_callback, NULL);
        CeedOperatorDestroy(&canonical_residual_op);
        CeedOperatorDestroy(&canonical_target_op);
        CeedElemRestrictionDestroy(&canonical_jac_rstr);
      }
      {
        const CeedInt num_qpts = IntPow(p + 1, 3), target_qf_size = 4 * target_comp, jac_size = target_qf_size * target_qf_size;
        CeedInt      *permuted_offsets = malloc(2 * num_qpts * sizeof(*permuted_offsets));
        CeedElemRestriction permuted_jac_rstr;
        CeedOperator        permuted_target_op;
        CeedVector          jac_data[] = {volume.jac_data};
        char                label[96];

        // Same passive vector and field sizes, but a distinct restriction that
        // reverses quadrature points within every element. Direct block-local
        // substitution would be wrong; the pair must take the exact global
        // materialize/restrict fallback.
        for (CeedInt e = 0; e < 2; e++) {
          for (CeedInt q = 0; q < num_qpts; q++) permuted_offsets[e * num_qpts + q] = e * jac_size * num_qpts + num_qpts - 1 - q;
        }
        CeedElemRestrictionCreate(ceed, 2, num_qpts, jac_size, num_qpts, (CeedSize)2 * num_qpts * jac_size, CEED_MEM_HOST, CEED_COPY_VALUES,
                                  permuted_offsets, &permuted_jac_rstr);
        free(permuted_offsets);
        CeedOperatorCreate(ceed, volume.target_qf, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &permuted_target_op);
        CeedOperatorSetField(permuted_target_op, "u", volume.target_rstr, volume.target_basis, CEED_VECTOR_ACTIVE);
        CeedOperatorSetField(permuted_target_op, "du", volume.target_rstr, volume.target_basis, CEED_VECTOR_ACTIVE);
        CeedOperatorSetField(permuted_target_op, "jac data", permuted_jac_rstr, CEED_BASIS_NONE, volume.jac_data);
        CeedOperatorSetField(permuted_target_op, "v", volume.target_rstr, volume.target_basis, CEED_VECTOR_ACTIVE);
        CeedOperatorSetField(permuted_target_op, "dv", volume.target_rstr, volume.target_basis, CEED_VECTOR_ACTIVE);
        snprintf(label, sizeof(label), "permuted handoff P%" CeedInt_FMT " bs%" CeedInt_FMT, p - 1, target_comp);
        error_count += CheckPair(ceed, volume.residual_op, permuted_target_op, jac_data, 1, volume.residual_l_size, label, 9 + p + target_comp,
                                 &volume.target_ctx, expect_direct_assembly_callback, NULL);
        CeedOperatorDestroy(&permuted_target_op);
        CeedElemRestrictionDestroy(&permuted_jac_rstr);
      }
      {
        const CeedInt num_qpts = IntPow(p + 1, 3), target_qf_size = 4 * target_comp, jac_size = target_qf_size * target_qf_size;
        CeedInt      *overlap_offsets = calloc((CeedSize)2 * num_qpts, sizeof(*overlap_offsets));
        CeedElemRestriction overlap_jac_rstr;
        CeedOperator        overlap_residual_op, overlap_target_op;
        CeedVector          jac_data[] = {volume.jac_data};
        char                label[96];

        // The exact same passive restriction is present on both operators,
        // but all quadrature points overlap in the L-vector. Residual scatter
        // must therefore finish globally before the target restriction. A
        // block-local handoff would be wrong, so this pair must fall back.
        CeedElemRestrictionCreate(ceed, 2, num_qpts, jac_size, 1, (CeedSize)2 * num_qpts * jac_size, CEED_MEM_HOST, CEED_COPY_VALUES,
                                  overlap_offsets, &overlap_jac_rstr);
        free(overlap_offsets);
        CreateOperatorsWithHandoffRestriction(&volume, overlap_jac_rstr, &overlap_residual_op, &overlap_target_op);
        snprintf(label, sizeof(label), "overlapping handoff P%" CeedInt_FMT " bs%" CeedInt_FMT, p - 1, target_comp);
        error_count += CheckPair(ceed, overlap_residual_op, overlap_target_op, jac_data, 1, volume.residual_l_size, label, 10 + p + target_comp,
                                 &volume.target_ctx, expect_direct_assembly_callback, NULL);
        CeedOperatorDestroy(&overlap_residual_op);
        CeedOperatorDestroy(&overlap_target_op);
        CeedElemRestrictionDestroy(&overlap_jac_rstr);
      }

      CeedOperatorCreateComposite(ceed, &residual_composite);
      CeedOperatorCompositeAddSub(residual_composite, volume.residual_op);
      CeedOperatorCompositeAddSub(residual_composite, face.residual_op);
      CeedOperatorCreateComposite(ceed, &target_composite);
      CeedOperatorCompositeAddSub(target_composite, volume.target_op);
      CeedOperatorCompositeAddSub(target_composite, face.target_op);
      {
        char       label[96];
        CeedVector jac_data[] = {volume.jac_data, face.jac_data};

        snprintf(label, sizeof(label), "hex+quad composite P%" CeedInt_FMT " bs%" CeedInt_FMT, p - 1, target_comp);
        error_count += CheckPair(ceed, residual_composite, target_composite, jac_data, 2, volume.residual_l_size, label, 11 + p + target_comp, NULL,
                                 expect_direct_assembly_callback, NULL);
      }
      {
        const CeedInt num_qpts = IntPow(p + 1, 2), target_qf_size = 3 * target_comp, jac_size = target_qf_size * target_qf_size;
        CeedInt      *permuted_offsets = malloc(2 * num_qpts * sizeof(*permuted_offsets));
        CeedElemRestriction permuted_face_jac_rstr;
        CeedOperator        permuted_face_target, mixed_target_composite;
        CeedVector          jac_data[] = {volume.jac_data, face.jac_data};
        char                label[96];

        // The volume child is eligible for pairing while this face child has
        // a distinct passive restriction and must use the additive child
        // fallback without clearing composite results accumulated so far.
        for (CeedInt e = 0; e < 2; e++) {
          for (CeedInt q = 0; q < num_qpts; q++) permuted_offsets[e * num_qpts + q] = e * jac_size * num_qpts + num_qpts - 1 - q;
        }
        CeedElemRestrictionCreate(ceed, 2, num_qpts, jac_size, num_qpts, (CeedSize)2 * num_qpts * jac_size, CEED_MEM_HOST, CEED_COPY_VALUES,
                                  permuted_offsets, &permuted_face_jac_rstr);
        free(permuted_offsets);
        CeedOperatorCreate(ceed, face.target_qf, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &permuted_face_target);
        CeedOperatorSetField(permuted_face_target, "u", face.target_rstr, face.target_basis, CEED_VECTOR_ACTIVE);
        CeedOperatorSetField(permuted_face_target, "du", face.target_rstr, face.target_basis, CEED_VECTOR_ACTIVE);
        CeedOperatorSetField(permuted_face_target, "jac data", permuted_face_jac_rstr, CEED_BASIS_NONE, face.jac_data);
        CeedOperatorSetField(permuted_face_target, "v", face.target_rstr, face.target_basis, CEED_VECTOR_ACTIVE);
        CeedOperatorSetField(permuted_face_target, "dv", face.target_rstr, face.target_basis, CEED_VECTOR_ACTIVE);
        CeedOperatorCreateComposite(ceed, &mixed_target_composite);
        CeedOperatorCompositeAddSub(mixed_target_composite, volume.target_op);
        CeedOperatorCompositeAddSub(mixed_target_composite, permuted_face_target);
        snprintf(label, sizeof(label), "mixed composite fallback P%" CeedInt_FMT " bs%" CeedInt_FMT, p - 1, target_comp);
        error_count += CheckPair(ceed, residual_composite, mixed_target_composite, jac_data, 2, volume.residual_l_size, label,
                                 13 + p + target_comp, NULL, expect_direct_assembly_callback, NULL);
        CeedOperatorDestroy(&mixed_target_composite);
        CeedOperatorDestroy(&permuted_face_target);
        CeedElemRestrictionDestroy(&permuted_face_jac_rstr);
      }
      {
        CeedOperator cross_target_composite;
        CeedVector   jac_data[] = {volume.jac_data, face.jac_data};
        char         label[96];

        // Reversing target children creates cross-child passive dependencies;
        // the whole operation must retain the exact two-pass fallback.
        CeedOperatorCreateComposite(ceed, &cross_target_composite);
        CeedOperatorCompositeAddSub(cross_target_composite, face.target_op);
        CeedOperatorCompositeAddSub(cross_target_composite, volume.target_op);
        snprintf(label, sizeof(label), "cross-dependent composite P%" CeedInt_FMT " bs%" CeedInt_FMT, p - 1, target_comp);
        error_count += CheckPair(ceed, residual_composite, cross_target_composite, jac_data, 2, volume.residual_l_size, label,
                                 15 + p + target_comp, NULL, expect_direct_assembly_callback, NULL);
        CeedOperatorDestroy(&cross_target_composite);
      }
      {
        CeedOperator shared_residual_composite, shared_target_composite;
        CeedVector   jac_data[] = {volume.jac_data};
        char         label[96];

        // Two residual children writing the same passive vector require the
        // whole residual to finish before either target child assembles.
        CeedOperatorCreateComposite(ceed, &shared_residual_composite);
        CeedOperatorCompositeAddSub(shared_residual_composite, volume.residual_op);
        CeedOperatorCompositeAddSub(shared_residual_composite, volume.residual_op);
        CeedOperatorCreateComposite(ceed, &shared_target_composite);
        CeedOperatorCompositeAddSub(shared_target_composite, volume.target_op);
        CeedOperatorCompositeAddSub(shared_target_composite, volume.target_op);
        snprintf(label, sizeof(label), "shared passive composite P%" CeedInt_FMT " bs%" CeedInt_FMT, p - 1, target_comp);
        error_count += CheckPair(ceed, shared_residual_composite, shared_target_composite, jac_data, 1, volume.residual_l_size, label,
                                 16 + p + target_comp, NULL, expect_direct_assembly_callback, NULL);
        CeedOperatorDestroy(&shared_residual_composite);
        CeedOperatorDestroy(&shared_target_composite);
      }
      {
        char       label[96];
        CeedVector jac_data[] = {face.jac_data};

        // A same-backend, non-composite pair with incompatible quadrature
        // block sizes must decline the fast path without partially updating
        // either public result, then execute the exact sequential fallback.
        CeedVectorSetValue(volume.jac_data, 0.37);
        snprintf(label, sizeof(label), "incompatible face-to-hex P%" CeedInt_FMT " bs%" CeedInt_FMT, p - 1, target_comp);
        error_count += CheckPair(ceed, face.residual_op, volume.target_op, jac_data, 1, face.residual_l_size, label, 17 + p + target_comp,
                                 &volume.target_ctx, expect_direct_assembly_callback, NULL);
      }
      CeedOperatorDestroy(&residual_composite);
      CeedOperatorDestroy(&target_composite);
      DestroyOperatorPair(&volume);
      DestroyOperatorPair(&face);
    }
  }

  // Production-shaped native 2D Q1 flow block: four components, 2x2
  // quadrature, and the declared zero gradient-input/value-output block. The
  // paired specialized contraction is compared against an independent target
  // QFunction that forces ordinary directional assembly.
  {
    const CeedInt target_comp = 4, num_nodes = 6;
    const CeedInt quad_nodes[8] = {0, 1, 3, 4, 1, 2, 4, 5};
    OperatorPair  quad = {0};
    CeedQFunction reference_qf;
    CeedOperator  reference_target_op;
    CeedVector    jac_data[] = {NULL};

    CreateTensorOperatorPairWithQ(ceed, 2, 2, 2, target_comp, true, 2, num_nodes, quad_nodes, &quad);
    CeedQFunctionCreateInterior(ceed, 1, target_apply, target_apply_loc, &reference_qf);
    CeedQFunctionAddInput(reference_qf, "u", target_comp, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(reference_qf, "du", 2 * target_comp, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(reference_qf, "jac data", 144, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(reference_qf, "v", target_comp, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(reference_qf, "dv", 2 * target_comp, CEED_EVAL_GRAD);
    CeedQFunctionSetContext(reference_qf, quad.target_context);
    CeedOperatorCreate(ceed, reference_qf, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &reference_target_op);
    CeedOperatorSetField(reference_target_op, "u", quad.target_rstr, quad.target_basis, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(reference_target_op, "du", quad.target_rstr, quad.target_basis, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(reference_target_op, "jac data", quad.jac_data_rstr, CEED_BASIS_NONE, quad.jac_data);
    CeedOperatorSetField(reference_target_op, "v", quad.target_rstr, quad.target_basis, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(reference_target_op, "dv", quad.target_rstr, quad.target_basis, CEED_VECTOR_ACTIVE);
    jac_data[0] = quad.jac_data;
    error_count += CheckPair(ceed, quad.residual_op, quad.target_op, jac_data, 1, quad.residual_l_size,
                             "native 2D Q1 flow specialized bs4", 31, &quad.target_ctx, expect_direct_assembly_callback, reference_target_op);
    CeedOperatorDestroy(&reference_target_op);
    CeedQFunctionDestroy(&reference_qf);
    DestroyOperatorPair(&quad);
  }

  // Native 2D unstructured airfoils use dense simplex bases rather than a
  // tensor contract. Exercise both affine P1 and quadratic P2 triangles, their
  // 1D segment boundaries, and the aligned volume+boundary composite.
  for (CeedInt degree = 1; degree <= 2; degree++) {
    for (CeedInt component_case = 0; component_case < 2; component_case++) {
      const CeedInt target_comp = component_case ? 1 : 5;
      const CeedInt triangle_elem_size = degree == 1 ? 3 : 6, line_elem_size = degree + 1;
      const CeedInt num_nodes = 2 * triangle_elem_size;
      CeedInt      *triangle_nodes = malloc((CeedSize)2 * triangle_elem_size * sizeof(*triangle_nodes));
      CeedInt      *line_nodes     = malloc((CeedSize)2 * line_elem_size * sizeof(*line_nodes));
      OperatorPair  triangle = {0}, line = {0};
      CeedOperator  residual_composite, target_composite;
      char          label[96];

      for (CeedInt e = 0; e < 2; e++) {
        for (CeedInt i = 0; i < triangle_elem_size; i++) triangle_nodes[e * triangle_elem_size + i] = e * triangle_elem_size + i;
        for (CeedInt i = 0; i < line_elem_size; i++) line_nodes[e * line_elem_size + i] = e * triangle_elem_size + i;
      }
      CreateSimplexOperatorPair(ceed, 2, degree, target_comp, 2, num_nodes, triangle_nodes, &triangle);
      CreateSimplexOperatorPair(ceed, 1, degree, target_comp, 2, num_nodes, line_nodes, &line);
      free(triangle_nodes);
      free(line_nodes);

      {
        CeedVector jac_data[] = {triangle.jac_data};

        snprintf(label, sizeof(label), "triangle P%" CeedInt_FMT " bs%" CeedInt_FMT, degree, target_comp);
        error_count += CheckPair(ceed, triangle.residual_op, triangle.target_op, jac_data, 1, triangle.residual_l_size, label,
                                 21 + degree + target_comp, &triangle.target_ctx, expect_direct_assembly_callback, NULL);
      }
      {
        CeedVector jac_data[] = {line.jac_data};

        snprintf(label, sizeof(label), "segment P%" CeedInt_FMT " bs%" CeedInt_FMT, degree, target_comp);
        error_count += CheckPair(ceed, line.residual_op, line.target_op, jac_data, 1, line.residual_l_size, label, 25 + degree + target_comp,
                                 &line.target_ctx, expect_direct_assembly_callback, NULL);
      }

      CeedOperatorCreateComposite(ceed, &residual_composite);
      CeedOperatorCompositeAddSub(residual_composite, triangle.residual_op);
      CeedOperatorCompositeAddSub(residual_composite, line.residual_op);
      CeedOperatorCreateComposite(ceed, &target_composite);
      CeedOperatorCompositeAddSub(target_composite, triangle.target_op);
      CeedOperatorCompositeAddSub(target_composite, line.target_op);
      {
        CeedVector jac_data[] = {triangle.jac_data, line.jac_data};

        snprintf(label, sizeof(label), "triangle+segment composite P%" CeedInt_FMT " bs%" CeedInt_FMT, degree, target_comp);
        error_count += CheckPair(ceed, residual_composite, target_composite, jac_data, 2, triangle.residual_l_size, label,
                                 29 + degree + target_comp, NULL, expect_direct_assembly_callback, NULL);
      }
      CeedOperatorDestroy(&residual_composite);
      CeedOperatorDestroy(&target_composite);
      DestroyOperatorPair(&triangle);
      DestroyOperatorPair(&line);
    }
  }

  CeedDestroy(&ceed);
  return error_count ? 1 : 0;
}
