// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ceed-opt.h"

static inline double CeedOperatorAssemblyTime_Opt(void) {
  struct timespec time;

  clock_gettime(CLOCK_MONOTONIC, &time);
  return (double)time.tv_sec + 1e-9 * time.tv_nsec;
}

//------------------------------------------------------------------------------
// Setup Input/Output Fields
//------------------------------------------------------------------------------
static int CeedOperatorSetupFields_Opt(CeedQFunction qf, CeedOperator op, bool is_input, bool *skip_rstr, bool *apply_add_basis,
                                       const CeedInt block_size, CeedElemRestriction *block_rstr, CeedVector *e_vecs_full, CeedVector *e_vecs,
                                       CeedVector *q_vecs, CeedInt start_e, CeedInt num_fields, CeedInt Q) {
  Ceed                ceed;
  CeedSize            e_size, q_size;
  CeedInt             num_comp, size, P;
  CeedQFunctionField *qf_fields;
  CeedOperatorField  *op_fields;

  {
    Ceed ceed_parent;

    CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
    CeedCallBackend(CeedGetParent(ceed, &ceed_parent));
    CeedCallBackend(CeedReferenceCopy(ceed_parent, &ceed));
    CeedCallBackend(CeedDestroy(&ceed_parent));
  }
  if (is_input) {
    CeedCallBackend(CeedOperatorGetFields(op, NULL, &op_fields, NULL, NULL));
    CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_fields, NULL, NULL));
  } else {
    CeedCallBackend(CeedOperatorGetFields(op, NULL, NULL, NULL, &op_fields));
    CeedCallBackend(CeedQFunctionGetFields(qf, NULL, NULL, NULL, &qf_fields));
  }

  // Loop over fields
  for (CeedInt i = 0; i < num_fields; i++) {
    CeedEvalMode eval_mode;
    CeedBasis    basis;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
    if (eval_mode != CEED_EVAL_WEIGHT) {
      Ceed                ceed_rstr;
      CeedSize            l_size;
      CeedInt             num_elem, elem_size, comp_stride;
      CeedRestrictionType rstr_type;
      CeedElemRestriction rstr;

      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[i], &rstr));
      CeedCallBackend(CeedElemRestrictionGetCeed(rstr, &ceed_rstr));
      CeedCallBackend(CeedElemRestrictionGetNumElements(rstr, &num_elem));
      CeedCallBackend(CeedElemRestrictionGetElementSize(rstr, &elem_size));
      CeedCallBackend(CeedElemRestrictionGetLVectorSize(rstr, &l_size));
      CeedCallBackend(CeedElemRestrictionGetNumComponents(rstr, &num_comp));
      CeedCallBackend(CeedElemRestrictionGetCompStride(rstr, &comp_stride));

      CeedCallBackend(CeedElemRestrictionGetType(rstr, &rstr_type));
      switch (rstr_type) {
        case CEED_RESTRICTION_STANDARD: {
          const CeedInt *offsets = NULL;

          CeedCallBackend(CeedElemRestrictionGetOffsets(rstr, CEED_MEM_HOST, &offsets));
          CeedCallBackend(CeedElemRestrictionCreateBlocked(ceed_rstr, num_elem, elem_size, block_size, num_comp, comp_stride, l_size, CEED_MEM_HOST,
                                                           CEED_COPY_VALUES, offsets, &block_rstr[i + start_e]));
          CeedCallBackend(CeedElemRestrictionRestoreOffsets(rstr, &offsets));
        } break;
        case CEED_RESTRICTION_ORIENTED: {
          const bool    *orients = NULL;
          const CeedInt *offsets = NULL;

          CeedCallBackend(CeedElemRestrictionGetOffsets(rstr, CEED_MEM_HOST, &offsets));
          CeedCallBackend(CeedElemRestrictionGetOrientations(rstr, CEED_MEM_HOST, &orients));
          CeedCallBackend(CeedElemRestrictionCreateBlockedOriented(ceed_rstr, num_elem, elem_size, block_size, num_comp, comp_stride, l_size,
                                                                   CEED_MEM_HOST, CEED_COPY_VALUES, offsets, orients, &block_rstr[i + start_e]));
          CeedCallBackend(CeedElemRestrictionRestoreOffsets(rstr, &offsets));
          CeedCallBackend(CeedElemRestrictionRestoreOrientations(rstr, &orients));
        } break;
        case CEED_RESTRICTION_CURL_ORIENTED: {
          const CeedInt8 *curl_orients = NULL;
          const CeedInt  *offsets      = NULL;

          CeedCallBackend(CeedElemRestrictionGetOffsets(rstr, CEED_MEM_HOST, &offsets));
          CeedCallBackend(CeedElemRestrictionGetCurlOrientations(rstr, CEED_MEM_HOST, &curl_orients));
          CeedCallBackend(CeedElemRestrictionCreateBlockedCurlOriented(ceed_rstr, num_elem, elem_size, block_size, num_comp, comp_stride, l_size,
                                                                       CEED_MEM_HOST, CEED_COPY_VALUES, offsets, curl_orients,
                                                                       &block_rstr[i + start_e]));
          CeedCallBackend(CeedElemRestrictionRestoreOffsets(rstr, &offsets));
          CeedCallBackend(CeedElemRestrictionRestoreCurlOrientations(rstr, &curl_orients));
        } break;
        case CEED_RESTRICTION_STRIDED: {
          CeedInt strides[3];

          CeedCallBackend(CeedElemRestrictionGetStrides(rstr, strides));
          CeedCallBackend(CeedElemRestrictionCreateBlockedStrided(ceed_rstr, num_elem, elem_size, block_size, num_comp, l_size, strides,
                                                                  &block_rstr[i + start_e]));
        } break;
        // LCOV_EXCL_START
        case CEED_RESTRICTION_POINTS:
          // Empty case - won't occur
          break;
          // LCOV_EXCL_STOP
      }
      CeedCallBackend(CeedDestroy(&ceed_rstr));
      CeedCallBackend(CeedElemRestrictionDestroy(&rstr));
      CeedCallBackend(CeedElemRestrictionCreateVector(block_rstr[i + start_e], NULL, &e_vecs_full[i + start_e]));
    }

    switch (eval_mode) {
      case CEED_EVAL_NONE:
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_fields[i], &size));
        e_size = (CeedSize)Q * size * block_size;
        CeedCallBackend(CeedVectorCreate(ceed, e_size, &e_vecs[i]));
        q_size = (CeedSize)Q * size * block_size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        break;
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis));
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_fields[i], &size));
        CeedCallBackend(CeedBasisGetNumNodes(basis, &P));
        CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
        CeedCallBackend(CeedBasisDestroy(&basis));
        e_size = (CeedSize)P * num_comp * block_size;
        CeedCallBackend(CeedVectorCreate(ceed, e_size, &e_vecs[i]));
        q_size = (CeedSize)Q * size * block_size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        break;
      case CEED_EVAL_WEIGHT:  // Only on input fields
        CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis));
        q_size = (CeedSize)Q * block_size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        CeedCallBackend(CeedBasisApply(basis, block_size, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT, CEED_VECTOR_NONE, q_vecs[i]));
        CeedCallBackend(CeedBasisDestroy(&basis));
        break;
    }
    // Initialize E-vec arrays
    if (e_vecs[i]) CeedCallBackend(CeedVectorSetValue(e_vecs[i], 0.0));
  }
  // Drop duplicate restrictions
  if (is_input) {
    for (CeedInt i = 0; i < num_fields; i++) {
      CeedVector          vec_i;
      CeedElemRestriction rstr_i;

      CeedCallBackend(CeedOperatorFieldGetVector(op_fields[i], &vec_i));
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[i], &rstr_i));
      for (CeedInt j = i + 1; j < num_fields; j++) {
        CeedVector          vec_j;
        CeedElemRestriction rstr_j;

        CeedCallBackend(CeedOperatorFieldGetVector(op_fields[j], &vec_j));
        CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[j], &rstr_j));
        if (vec_i == vec_j && rstr_i == rstr_j) {
          CeedCallBackend(CeedVectorReferenceCopy(e_vecs[i], &e_vecs[j]));
          CeedCallBackend(CeedVectorReferenceCopy(e_vecs_full[i + start_e], &e_vecs_full[j + start_e]));
          skip_rstr[j] = true;
        }
        CeedCallBackend(CeedVectorDestroy(&vec_j));
        CeedCallBackend(CeedElemRestrictionDestroy(&rstr_j));
      }
      CeedCallBackend(CeedVectorDestroy(&vec_i));
      CeedCallBackend(CeedElemRestrictionDestroy(&rstr_i));
    }
  } else {
    for (CeedInt i = num_fields - 1; i >= 0; i--) {
      CeedVector          vec_i;
      CeedElemRestriction rstr_i;

      CeedCallBackend(CeedOperatorFieldGetVector(op_fields[i], &vec_i));
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[i], &rstr_i));
      for (CeedInt j = i - 1; j >= 0; j--) {
        CeedVector          vec_j;
        CeedElemRestriction rstr_j;

        CeedCallBackend(CeedOperatorFieldGetVector(op_fields[j], &vec_j));
        CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[j], &rstr_j));
        if (vec_i == vec_j && rstr_i == rstr_j) {
          CeedCallBackend(CeedVectorReferenceCopy(e_vecs[i], &e_vecs[j]));
          CeedCallBackend(CeedVectorReferenceCopy(e_vecs_full[i + start_e], &e_vecs_full[j + start_e]));
          skip_rstr[j]       = true;
          apply_add_basis[i] = true;
        }
        CeedCallBackend(CeedVectorDestroy(&vec_j));
        CeedCallBackend(CeedElemRestrictionDestroy(&rstr_j));
      }
      CeedCallBackend(CeedVectorDestroy(&vec_i));
      CeedCallBackend(CeedElemRestrictionDestroy(&rstr_i));
    }
  }
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup Operator
//------------------------------------------------------------------------------
static int CeedOperatorSetup_Opt(CeedOperator op) {
  bool                is_setup_done;
  Ceed                ceed;
  Ceed_Opt           *ceed_impl;
  CeedInt             Q, num_input_fields, num_output_fields;
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Opt   *impl;

  CeedCallBackend(CeedOperatorIsSetupDone(op, &is_setup_done));
  if (is_setup_done) return CEED_ERROR_SUCCESS;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedGetData(ceed, &ceed_impl));
  CeedCallBackend(CeedDestroy(&ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedQFunctionIsIdentity(qf, &impl->is_identity_qf));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));
  const CeedInt block_size = ceed_impl->block_size;

  // Allocate
  CeedCallBackend(CeedCalloc(num_input_fields + num_output_fields, &impl->block_rstr));
  CeedCallBackend(CeedCalloc(num_input_fields + num_output_fields, &impl->e_vecs_full));

  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->skip_rstr_in));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->skip_rstr_out));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->apply_add_basis_out));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->input_states));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->e_vecs_in));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->e_vecs_out));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->q_vecs_in));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->q_vecs_out));

  impl->num_inputs  = num_input_fields;
  impl->num_outputs = num_output_fields;

  // Set up infield and outfield pointer arrays
  // Infields
  CeedCallBackend(CeedOperatorSetupFields_Opt(qf, op, true, impl->skip_rstr_in, NULL, block_size, impl->block_rstr, impl->e_vecs_full,
                                              impl->e_vecs_in, impl->q_vecs_in, 0, num_input_fields, Q));
  // Outfields
  CeedCallBackend(CeedOperatorSetupFields_Opt(qf, op, false, impl->skip_rstr_out, impl->apply_add_basis_out, block_size, impl->block_rstr,
                                              impl->e_vecs_full, impl->e_vecs_out, impl->q_vecs_out, num_input_fields, num_output_fields, Q));

  // Identity QFunctions
  if (impl->is_identity_qf) {
    CeedEvalMode        in_mode, out_mode;
    CeedQFunctionField *in_fields, *out_fields;

    CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &in_fields, NULL, &out_fields));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(in_fields[0], &in_mode));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(out_fields[0], &out_mode));

    if (in_mode == CEED_EVAL_NONE && out_mode == CEED_EVAL_NONE) {
      impl->is_identity_rstr_op = true;
    } else {
      CeedCallBackend(CeedVectorReferenceCopy(impl->q_vecs_in[0], &impl->q_vecs_out[0]));
    }
  }

  CeedCallBackend(CeedOperatorSetSetupDone(op));
  CeedCallBackend(CeedQFunctionDestroy(&qf));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup Input Fields
//------------------------------------------------------------------------------
static inline int CeedOperatorSetupInputs_Opt(CeedInt num_input_fields, CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
                                              CeedVector in_vec, CeedScalar *e_data[2 * CEED_FIELD_MAX], CeedOperator_Opt *impl,
                                              CeedRequest *request) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedEvalMode eval_mode;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      uint64_t   state;
      CeedVector vec;

      // Get input vector
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec != CEED_VECTOR_ACTIVE) {
        // Restrict
        CeedCallBackend(CeedVectorGetState(vec, &state));
        if (state != impl->input_states[i] && impl->block_rstr[i] && !impl->skip_rstr_in[i]) {
          CeedCallBackend(CeedElemRestrictionApply(impl->block_rstr[i], CEED_NOTRANSPOSE, vec, impl->e_vecs_full[i], request));
        }
        impl->input_states[i] = state;
        // Get evec
        CeedCallBackend(CeedVectorGetArrayRead(impl->e_vecs_full[i], CEED_MEM_HOST, (const CeedScalar **)&e_data[i]));
      } else {
        // Set Qvec for CEED_EVAL_NONE
        if (eval_mode == CEED_EVAL_NONE) {
          CeedCallBackend(CeedVectorGetArrayRead(impl->e_vecs_in[i], CEED_MEM_HOST, (const CeedScalar **)&e_data[i]));
          CeedCallBackend(CeedVectorSetArray(impl->q_vecs_in[i], CEED_MEM_HOST, CEED_USE_POINTER, e_data[i]));
          CeedCallBackend(CeedVectorRestoreArrayRead(impl->e_vecs_in[i], (const CeedScalar **)&e_data[i]));
        }
      }
      CeedCallBackend(CeedVectorDestroy(&vec));
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Input Basis Action
//------------------------------------------------------------------------------
static inline int CeedOperatorInputBasis_Opt(CeedInt e, CeedInt Q, CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
                                             CeedInt num_input_fields, CeedInt block_size, CeedVector in_vec, bool skip_active,
                                             CeedScalar *e_data[2 * CEED_FIELD_MAX], CeedOperator_Opt *impl, CeedRequest *request) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    bool                is_active;
    CeedInt             elem_size, size, num_comp;
    CeedEvalMode        eval_mode;
    CeedVector          vec;
    CeedElemRestriction elem_rstr;
    CeedBasis           basis;

    // Skip active input
    CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
    is_active = vec == CEED_VECTOR_ACTIVE;
    CeedCallBackend(CeedVectorDestroy(&vec));
    if (skip_active && is_active) continue;

    // Get elem_size, eval_mode, size
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &elem_rstr));
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
    CeedCallBackend(CeedElemRestrictionDestroy(&elem_rstr));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    CeedCallBackend(CeedQFunctionFieldGetSize(qf_input_fields[i], &size));
    // Restrict block active input
    if (is_active && impl->block_rstr[i]) {
      CeedCallBackend(CeedElemRestrictionApplyBlock(impl->block_rstr[i], e / block_size, CEED_NOTRANSPOSE, in_vec, impl->e_vecs_in[i], request));
    }
    // Basis action
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        if (!is_active) {
          CeedCallBackend(CeedVectorSetArray(impl->q_vecs_in[i], CEED_MEM_HOST, CEED_USE_POINTER, &e_data[i][(CeedSize)e * Q * size]));
        }
        break;
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
        if (!is_active) {
          CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
          CeedCallBackend(CeedVectorSetArray(impl->e_vecs_in[i], CEED_MEM_HOST, CEED_USE_POINTER, &e_data[i][(CeedSize)e * elem_size * num_comp]));
        }
        CeedCallBackend(CeedBasisApply(basis, block_size, CEED_NOTRANSPOSE, eval_mode, impl->e_vecs_in[i], impl->q_vecs_in[i]));
        CeedCallBackend(CeedBasisDestroy(&basis));
        break;
      case CEED_EVAL_WEIGHT:
        break;  // No action
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Output Basis Action
//------------------------------------------------------------------------------
static inline int CeedOperatorOutputBasis_Opt(CeedInt e, CeedInt Q, CeedQFunctionField *qf_output_fields, CeedOperatorField *op_output_fields,
                                              CeedInt block_size, CeedInt num_input_fields, CeedInt num_output_fields, bool *apply_add_basis,
                                              bool *skip_rstr, CeedOperator op, CeedVector out_vec, CeedOperator_Opt *impl, CeedRequest *request) {
  for (CeedInt i = 0; i < num_output_fields; i++) {
    bool         is_active;
    CeedEvalMode eval_mode;
    CeedVector   vec;
    CeedBasis    basis;

    // Get eval_mode
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    // Basis action
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        break;  // No action
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
        if (apply_add_basis[i]) {
          CeedCallBackend(CeedBasisApplyAdd(basis, block_size, CEED_TRANSPOSE, eval_mode, impl->q_vecs_out[i], impl->e_vecs_out[i]));
        } else {
          CeedCallBackend(CeedBasisApply(basis, block_size, CEED_TRANSPOSE, eval_mode, impl->q_vecs_out[i], impl->e_vecs_out[i]));
        }
        CeedCallBackend(CeedBasisDestroy(&basis));
        break;
      // LCOV_EXCL_START
      case CEED_EVAL_WEIGHT: {
        return CeedError(CeedOperatorReturnCeed(op), CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
        // LCOV_EXCL_STOP
      }
    }
    // Restrict output block
    if (skip_rstr[i]) continue;
    // Get output vector
    CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
    is_active = vec == CEED_VECTOR_ACTIVE;
    if (is_active) vec = out_vec;
    // Restrict
    CeedCallBackend(CeedElemRestrictionApplyBlock(impl->block_rstr[i + impl->num_inputs], e / block_size, CEED_TRANSPOSE, impl->e_vecs_out[i], vec,
                                                  request));
    if (!is_active) CeedCallBackend(CeedVectorDestroy(&vec));
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Restore Input Vectors
//------------------------------------------------------------------------------
static inline int CeedOperatorRestoreInputs_Opt(CeedInt num_input_fields, CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
                                                CeedScalar *e_data[2 * CEED_FIELD_MAX], CeedOperator_Opt *impl) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedEvalMode eval_mode;
    CeedVector   vec;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
    if (eval_mode != CEED_EVAL_WEIGHT && vec != CEED_VECTOR_ACTIVE) {
      CeedCallBackend(CeedVectorRestoreArrayRead(impl->e_vecs_full[i], (const CeedScalar **)&e_data[i]));
    }
    CeedCallBackend(CeedVectorDestroy(&vec));
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Apply
//------------------------------------------------------------------------------
static int CeedOperatorApplyAdd_Opt(CeedOperator op, CeedVector in_vec, CeedVector out_vec, CeedRequest *request) {
  Ceed                ceed;
  Ceed_Opt           *ceed_impl;
  CeedInt             Q, num_input_fields, num_output_fields, num_elem;
  CeedEvalMode        eval_mode;
  CeedScalar         *e_data[2 * CEED_FIELD_MAX] = {0};
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Opt   *impl;

  // Setup
  CeedCallBackend(CeedOperatorSetup_Opt(op));

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedGetData(ceed, &ceed_impl));
  CeedCallBackend(CeedDestroy(&ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  const CeedInt block_size = ceed_impl->block_size;
  const CeedInt num_blocks = (num_elem / block_size) + !!(num_elem % block_size);

  // Restriction only operator
  if (impl->is_identity_rstr_op) {
    for (CeedInt b = 0; b < num_blocks; b++) {
      CeedCallBackend(CeedElemRestrictionApplyBlock(impl->block_rstr[0], b, CEED_NOTRANSPOSE, in_vec, impl->e_vecs_in[0], request));
      CeedCallBackend(CeedElemRestrictionApplyBlock(impl->block_rstr[1], b, CEED_TRANSPOSE, impl->e_vecs_in[0], out_vec, request));
    }
    return CEED_ERROR_SUCCESS;
  }

  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

  // Input Evecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Opt(num_input_fields, qf_input_fields, op_input_fields, in_vec, e_data, impl, request));

  // Output Lvecs, Evecs, and Qvecs
  for (CeedInt i = 0; i < num_output_fields; i++) {
    // Set Qvec if needed
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_NONE) {
      // Set qvec to single block evec
      CeedCallBackend(CeedVectorGetArrayWrite(impl->e_vecs_out[i], CEED_MEM_HOST, &e_data[i + num_input_fields]));
      CeedCallBackend(CeedVectorSetArray(impl->q_vecs_out[i], CEED_MEM_HOST, CEED_USE_POINTER, e_data[i + num_input_fields]));
      CeedCallBackend(CeedVectorRestoreArray(impl->e_vecs_out[i], &e_data[i + num_input_fields]));
    }
  }

  // Loop through elements
  for (CeedInt e = 0; e < num_blocks * block_size; e += block_size) {
    // Input basis apply
    CeedCallBackend(CeedOperatorInputBasis_Opt(e, Q, qf_input_fields, op_input_fields, num_input_fields, block_size, in_vec, false, e_data, impl,
                                               request));

    // Q function
    if (!impl->is_identity_qf) {
      CeedCallBackend(CeedQFunctionApply(qf, Q * block_size, impl->q_vecs_in, impl->q_vecs_out));
    }

    // Output basis apply and restriction
    CeedCallBackend(CeedOperatorOutputBasis_Opt(e, Q, qf_output_fields, op_output_fields, block_size, num_input_fields, num_output_fields,
                                                impl->apply_add_basis_out, impl->skip_rstr_out, op, out_vec, impl, request));
  }

  // Restore input arrays
  CeedCallBackend(CeedOperatorRestoreInputs_Opt(num_input_fields, qf_input_fields, op_input_fields, e_data, impl));
  CeedCallBackend(CeedQFunctionDestroy(&qf));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Core code for linear QFunction assembly
//------------------------------------------------------------------------------
static inline int CeedOperatorLinearAssembleQFunctionCore_Opt(CeedOperator op, bool build_objects, CeedVector *assembled, CeedElemRestriction *rstr,
                                                              CeedRequest *request) {
  Ceed                ceed;
  Ceed_Opt           *ceed_impl;
  CeedInt             qf_size_in, qf_size_out, Q, num_input_fields, num_output_fields, num_elem;
  CeedScalar         *l_vec_array, *e_data[2 * CEED_FIELD_MAX] = {0};
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Opt   *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedGetData(ceed, &ceed_impl));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  qf_size_in  = impl->qf_size_in;
  qf_size_out = impl->qf_size_out;

  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));
  const CeedInt       block_size = ceed_impl->block_size;
  const CeedInt       num_blocks = (num_elem / block_size) + !!(num_elem % block_size);
  CeedVector          l_vec      = impl->qf_l_vec;
  CeedElemRestriction block_rstr = impl->qf_block_rstr;

  // Setup
  CeedCallBackend(CeedOperatorSetup_Opt(op));

  // Check for restriction only operator
  CeedCheck(!impl->is_identity_rstr_op, ceed, CEED_ERROR_BACKEND, "Assembling restriction only operators is not supported");

  // Input Evecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Opt(num_input_fields, qf_input_fields, op_input_fields, NULL, e_data, impl, request));

  // Count number of active input fields
  if (qf_size_in == 0) {
    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedInt    field_size;
      CeedVector vec;

      // Check if active input
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) {
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_input_fields[i], &field_size));
        qf_size_in += field_size;
      }
      CeedCallBackend(CeedVectorDestroy(&vec));
    }
    CeedCheck(qf_size_in > 0, ceed, CEED_ERROR_BACKEND, "Cannot assemble QFunction without active inputs and outputs");
    impl->qf_size_in = qf_size_in;
  }

  // Count number of active output fields
  if (qf_size_out == 0) {
    for (CeedInt i = 0; i < num_output_fields; i++) {
      CeedInt    field_size;
      CeedVector vec;

      // Check if active output
      CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) {
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[i], &field_size));
        qf_size_out += field_size;
      }
      CeedCallBackend(CeedVectorDestroy(&vec));
    }
    CeedCheck(qf_size_out > 0, ceed, CEED_ERROR_BACKEND, "Cannot assemble QFunction without active inputs and outputs");
    impl->qf_size_out = qf_size_out;
  }

  // Setup l_vec
  if (!l_vec) {
    const CeedSize l_size = (CeedSize)block_size * Q * qf_size_in * qf_size_out;

    CeedCallBackend(CeedVectorCreate(ceed, l_size, &l_vec));
    CeedCallBackend(CeedVectorSetValue(l_vec, 0.0));
    impl->qf_l_vec = l_vec;
  }

  // Output blocked restriction
  if (!block_rstr) {
    CeedInt strides[3] = {1, Q, qf_size_in * qf_size_out * Q};

    CeedCallBackend(CeedElemRestrictionCreateBlockedStrided(ceed, num_elem, Q, block_size, qf_size_in * qf_size_out,
                                                            qf_size_in * qf_size_out * num_elem * Q, strides, &block_rstr));
    impl->qf_block_rstr = block_rstr;
  }

  // Build objects if needed
  if (build_objects) {
    const CeedSize l_size     = (CeedSize)num_elem * Q * qf_size_in * qf_size_out;
    CeedInt        strides[3] = {1, Q, qf_size_in * qf_size_out * Q};

    // Create output restriction
    CeedCallBackend(CeedElemRestrictionCreateStrided(ceed, num_elem, Q, qf_size_in * qf_size_out,
                                                     (CeedSize)qf_size_in * (CeedSize)qf_size_out * (CeedSize)num_elem * (CeedSize)Q, strides, rstr));
    // Create assembled vector
    CeedCallBackend(CeedVectorCreate(ceed, l_size, assembled));
  }

  // Clear input QFunction buffers
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedVector vec;

    // Clear if active input
    CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) CeedCallBackend(CeedVectorSetValue(impl->q_vecs_in[i], 0.0));
    CeedCallBackend(CeedVectorDestroy(&vec));
  }

  // Clear output vector
  CeedCallBackend(CeedVectorSetValue(*assembled, 0.0));

  // Loop through elements
  for (CeedInt e = 0; e < num_blocks * block_size; e += block_size) {
    CeedCallBackend(CeedVectorGetArray(l_vec, CEED_MEM_HOST, &l_vec_array));

    // Input basis apply
    CeedCallBackend(CeedOperatorInputBasis_Opt(e, Q, qf_input_fields, op_input_fields, num_input_fields, block_size, NULL, true, e_data, impl,
                                               request));

    // Assemble QFunction
    for (CeedInt i = 0; i < num_input_fields; i++) {
      bool       is_active;
      CeedInt    field_size;
      CeedVector vec;

      // Check if active input
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      is_active = vec == CEED_VECTOR_ACTIVE;
      CeedCallBackend(CeedVectorDestroy(&vec));
      if (!is_active) continue;
      CeedCallBackend(CeedQFunctionFieldGetSize(qf_input_fields[i], &field_size));
      for (CeedInt field = 0; field < field_size; field++) {
        // Set current portion of input to 1.0
        {
          CeedScalar *array;

          CeedCallBackend(CeedVectorGetArray(impl->q_vecs_in[i], CEED_MEM_HOST, &array));
          for (CeedInt j = 0; j < Q * block_size; j++) array[field * Q * block_size + j] = 1.0;
          CeedCallBackend(CeedVectorRestoreArray(impl->q_vecs_in[i], &array));
        }

        if (!impl->is_identity_qf) {
          // Set Outputs
          for (CeedInt out = 0; out < num_output_fields; out++) {
            CeedVector vec;

            // Check if active output
            CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &vec));
            if (vec == CEED_VECTOR_ACTIVE) {
              CeedInt field_size;

              CeedCallBackend(CeedVectorSetArray(impl->q_vecs_out[out], CEED_MEM_HOST, CEED_USE_POINTER, l_vec_array));
              CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[out], &field_size));
              l_vec_array += field_size * Q * block_size;  // Advance the pointer by the size of the output
            }
            CeedCallBackend(CeedVectorDestroy(&vec));
          }
          // Apply QFunction
          CeedCallBackend(CeedQFunctionApply(qf, Q * block_size, impl->q_vecs_in, impl->q_vecs_out));
        } else {
          CeedInt           field_size;
          const CeedScalar *array;

          // Copy Identity Outputs
          CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[0], &field_size));
          CeedCallBackend(CeedVectorGetArrayRead(impl->q_vecs_out[0], CEED_MEM_HOST, &array));
          for (CeedInt j = 0; j < field_size * Q * block_size; j++) l_vec_array[j] = array[j];
          CeedCallBackend(CeedVectorRestoreArrayRead(impl->q_vecs_out[0], &array));
          l_vec_array += field_size * Q * block_size;
        }
        // Reset input to 0.0
        {
          CeedScalar *array;

          CeedCallBackend(CeedVectorGetArray(impl->q_vecs_in[i], CEED_MEM_HOST, &array));
          for (CeedInt j = 0; j < Q * block_size; j++) array[field * Q * block_size + j] = 0.0;
          CeedCallBackend(CeedVectorRestoreArray(impl->q_vecs_in[i], &array));
        }
      }
    }

    // Un-set output Qvecs to prevent accidental overwrite of Assembled
    if (!impl->is_identity_qf) {
      for (CeedInt out = 0; out < num_output_fields; out++) {
        CeedVector vec;

        // Check if active output
        CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &vec));
        if (vec == CEED_VECTOR_ACTIVE && num_elem > 0) {
          CeedCallBackend(CeedVectorTakeArray(impl->q_vecs_out[out], CEED_MEM_HOST, NULL));
        }
        CeedCallBackend(CeedVectorDestroy(&vec));
      }
    }

    // Assemble into assembled vector
    CeedCallBackend(CeedVectorRestoreArray(l_vec, &l_vec_array));
    CeedCallBackend(CeedElemRestrictionApplyBlock(block_rstr, e / block_size, CEED_TRANSPOSE, l_vec, *assembled, request));
  }

  // Reset output Qvecs
  for (CeedInt out = 0; out < num_output_fields; out++) {
    CeedVector vec;

    // Initialize array if active output
    CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &vec));
    if (vec == CEED_VECTOR_ACTIVE) CeedCallBackend(CeedVectorSetValue(impl->q_vecs_out[out], 0.0));
    CeedCallBackend(CeedVectorDestroy(&vec));
  }

  // Restore input arrays
  CeedCallBackend(CeedOperatorRestoreInputs_Opt(num_input_fields, qf_input_fields, op_input_fields, e_data, impl));
  CeedCallBackend(CeedDestroy(&ceed));
  CeedCallBackend(CeedQFunctionDestroy(&qf));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Fused mapped linear assembly
//------------------------------------------------------------------------------
static int CeedOperatorAssembleSingleBlockMapped_Opt(CeedOperator op, CeedSize offset, CeedInt active_input, CeedInt active_output, CeedVector values,
                                                     const CeedInt entry_map[], bool *is_supported) {
  Ceed                     ceed;
  Ceed_Opt                *ceed_impl;
  CeedOperator_Opt        *impl;
  CeedInt                  Q, num_elem, num_input_fields, num_output_fields;
  CeedScalar              *e_data[2 * CEED_FIELD_MAX] = {0}, *qf_block = NULL, *vals = NULL;
  CeedQFunction            qf;
  CeedQFunctionUser        assembly_qf = NULL;
  bool                     assembly_grad_in_value_out_zero;
  CeedQFunctionField      *qf_input_fields, *qf_output_fields;
  CeedOperatorField       *op_input_fields, *op_output_fields;
  CeedOperatorAssemblyData data;
  CeedInt                  num_active_bases_in, *num_eval_modes_in, num_active_bases_out, *num_eval_modes_out;
  CeedSize                 num_output_components, **eval_mode_offsets_in, **eval_mode_offsets_out;
  CeedBasis               *active_bases_in, *active_bases_out;
  CeedElemRestriction     *active_rstrs_in, *active_rstrs_out;
  const CeedScalar       **B_mats_in, **B_mats_out;

  *is_supported = false;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedGetData(ceed, &ceed_impl));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  const CeedInt block_size = ceed_impl->block_size;
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedQFunctionGetAssemblyUserFunction(qf, &assembly_qf));
  CeedCallBackend(CeedQFunctionGetAssemblyDataGradientInputToValueOutputZero(qf, &assembly_grad_in_value_out_zero));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

  CeedCallBackend(CeedOperatorGetOperatorAssemblyData(op, &data));
  CeedCallBackend(CeedOperatorAssemblyDataGetEvalModes(data, &num_active_bases_in, &num_eval_modes_in, NULL, &eval_mode_offsets_in,
                                                       &num_active_bases_out, &num_eval_modes_out, NULL, &eval_mode_offsets_out,
                                                       &num_output_components));
  CeedCallBackend(CeedOperatorAssemblyDataGetBases(data, NULL, &active_bases_in, &B_mats_in, NULL, &active_bases_out, &B_mats_out));
  CeedCallBackend(CeedOperatorAssemblyDataGetElemRestrictions(data, NULL, &active_rstrs_in, NULL, &active_rstrs_out));
  if (num_active_bases_in != 1 || num_active_bases_out != 1) {
    CeedCallBackend(CeedDestroy(&ceed));
    CeedCallBackend(CeedQFunctionDestroy(&qf));
    return CEED_ERROR_SUCCESS;
  }
  CeedCheck(active_input < num_active_bases_in && active_output < num_active_bases_out, ceed, CEED_ERROR_BACKEND, "Invalid active basis block");

  CeedElemRestriction elem_rstr_in = active_rstrs_in[active_input], elem_rstr_out = active_rstrs_out[active_output];
  CeedRestrictionType rstr_type_in, rstr_type_out;
  CeedCallBackend(CeedElemRestrictionGetType(elem_rstr_in, &rstr_type_in));
  CeedCallBackend(CeedElemRestrictionGetType(elem_rstr_out, &rstr_type_out));
  if (rstr_type_in != CEED_RESTRICTION_STANDARD || rstr_type_out != CEED_RESTRICTION_STANDARD) {
    CeedCallBackend(CeedDestroy(&ceed));
    CeedCallBackend(CeedQFunctionDestroy(&qf));
    return CEED_ERROR_SUCCESS;
  }
  *is_supported = true;

  const char *timing_env     = getenv("CEED_OPT_MAPPED_ASSEMBLY_TIMING");
  const bool  timing_enabled = timing_env && strcmp(timing_env, "0");
  double total_start = 0.0, phase_start = 0.0, qfunction_start = 0.0, setup_time = 0.0, input_basis_time = 0.0, qfunction_time = 0.0,
         qf_seed_time = 0.0, qf_bind_time = 0.0, qf_apply_time = 0.0, qf_unbind_time = 0.0, qf_reset_time = 0.0, btd_time = 0.0,
         contract_time = 0.0, scatter_time = 0.0;
  if (timing_enabled) total_start = CeedOperatorAssemblyTime_Opt();

  CeedCallBackend(CeedOperatorSetup_Opt(op));
  CeedCheck(!impl->is_identity_rstr_op, ceed, CEED_ERROR_BACKEND, "Assembling restriction only operators is not supported");
  CeedCallBackend(CeedOperatorSetupInputs_Opt(num_input_fields, qf_input_fields, op_input_fields, NULL, e_data, impl, CEED_REQUEST_IMMEDIATE));

  CeedInt qf_size_in = impl->qf_size_in, qf_size_out = impl->qf_size_out;
  if (!qf_size_in) {
    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedInt    field_size;
      CeedVector vec;

      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) {
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_input_fields[i], &field_size));
        qf_size_in += field_size;
      }
      CeedCallBackend(CeedVectorDestroy(&vec));
    }
    impl->qf_size_in = qf_size_in;
  }
  if (!qf_size_out) {
    for (CeedInt i = 0; i < num_output_fields; i++) {
      CeedInt    field_size;
      CeedVector vec;

      CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) {
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[i], &field_size));
        qf_size_out += field_size;
      }
      CeedCallBackend(CeedVectorDestroy(&vec));
    }
    impl->qf_size_out = qf_size_out;
  }
  CeedCheck(qf_size_in > 0 && qf_size_out > 0, ceed, CEED_ERROR_BACKEND, "Cannot assemble QFunction without active inputs and outputs");

  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedVector vec;

    CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) CeedCallBackend(CeedVectorSetValue(impl->q_vecs_in[i], 0.0));
    CeedCallBackend(CeedVectorDestroy(&vec));
  }

  CeedInt           num_elem_in, elem_size_in, num_comp_in, num_elem_out, elem_size_out, num_comp_out;
  CeedBasis         basis_in = active_bases_in[active_input];
  const CeedScalar *B_mat_in = B_mats_in[active_input], *B_mat_out = B_mats_out[active_output];

  CeedCallBackend(CeedElemRestrictionGetNumElements(elem_rstr_in, &num_elem_in));
  CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr_in, &elem_size_in));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(elem_rstr_in, &num_comp_in));
  CeedCallBackend(CeedElemRestrictionGetNumElements(elem_rstr_out, &num_elem_out));
  CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr_out, &elem_size_out));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(elem_rstr_out, &num_comp_out));
  if (num_comp_in == 1 && num_comp_out == 1) {
    *is_supported = false;
    CeedCallBackend(CeedOperatorRestoreInputs_Opt(num_input_fields, qf_input_fields, op_input_fields, e_data, impl));
    CeedCallBackend(CeedDestroy(&ceed));
    CeedCallBackend(CeedQFunctionDestroy(&qf));
    return CEED_ERROR_SUCCESS;
  }
  CeedCheck(num_elem_in == num_elem_out && num_elem_in == num_elem, ceed, CEED_ERROR_BACKEND, "Incompatible active input and output restrictions");

  CeedTensorContract contract;
  CeedScalar        *BTD_mat = NULL, *BTD_sum = NULL, *elem_mat = NULL;
  const CeedInt      num_component_blocks = num_comp_in * num_comp_out;
  const CeedSize     btd_stride           = Q * num_eval_modes_in[active_input];
  const CeedSize     qf_block_size        = (CeedSize)qf_size_in * qf_size_out * Q * block_size;
  const char         *native_p1_env        = getenv("CEED_OPT_NATIVE_P1_DIRECT");

  CeedCallBackend(CeedBasisGetTensorContract(basis_in, &contract));
  const bool native_p1_direct = assembly_qf && assembly_grad_in_value_out_zero && contract && num_comp_in == 4 && num_comp_out == 4 &&
                                elem_size_in == 4 && elem_size_out == 4 && Q == 4 &&
                                qf_size_in == 12 && qf_size_out == 12 && num_eval_modes_in[active_input] == 3 &&
                                num_eval_modes_out[active_output] == 3 && eval_mode_offsets_in[active_input][0] == 0 &&
                                eval_mode_offsets_in[active_input][1] == 4 && eval_mode_offsets_in[active_input][2] == 8 &&
                                eval_mode_offsets_out[active_output][0] == 0 && eval_mode_offsets_out[active_output][1] == 4 &&
                                eval_mode_offsets_out[active_output][2] == 8 && (!native_p1_env || strcmp(native_p1_env, "0"));
  CeedScalar native_p1_basis_products[16][4][9];

  if (native_p1_direct) {
    for (CeedInt node_out = 0; node_out < 4; node_out++) {
      for (CeedInt node_in = 0; node_in < 4; node_in++) {
        const CeedInt node_pair = node_out * 4 + node_in;

        for (CeedInt q = 0; q < 4; q++) {
          for (CeedInt eval_out = 0; eval_out < 3; eval_out++) {
            const CeedScalar b_out = B_mat_out[(q * 3 + eval_out) * 4 + node_out];

            for (CeedInt eval_in = 0; eval_in < 3; eval_in++) {
              native_p1_basis_products[node_pair][q][eval_out * 3 + eval_in] =
                  b_out * B_mat_in[(q * 3 + eval_in) * 4 + node_in];
            }
          }
        }
      }
    }
  }
  if (!impl->mapped_btd)
    CeedCallBackend(CeedCalloc((CeedSize)num_component_blocks * elem_size_out * btd_stride, &impl->mapped_btd));
  if (!contract && !impl->mapped_btd_sum) CeedCallBackend(CeedCalloc(elem_size_out, &impl->mapped_btd_sum));
  if (!impl->mapped_elem_mat)
    CeedCallBackend(CeedCalloc((CeedSize)num_component_blocks * elem_size_out * elem_size_in, &impl->mapped_elem_mat));
  if (!impl->mapped_qf_block) CeedCallBackend(CeedCalloc(qf_block_size, &impl->mapped_qf_block));
  BTD_mat  = impl->mapped_btd;
  BTD_sum  = impl->mapped_btd_sum;
  elem_mat = impl->mapped_elem_mat;
  qf_block = impl->mapped_qf_block;
  CeedCallBackend(CeedVectorGetArray(values, CEED_MEM_HOST, &vals));
  if (timing_enabled) setup_time = CeedOperatorAssemblyTime_Opt() - total_start;

  CeedSize count = 0;
  for (CeedInt e_block = 0; e_block < num_elem; e_block += block_size) {
    CeedScalar *qf_write = qf_block;

    if (timing_enabled) phase_start = CeedOperatorAssemblyTime_Opt();
    CeedCallBackend(CeedOperatorInputBasis_Opt(e_block, Q, qf_input_fields, op_input_fields, num_input_fields, block_size, NULL, true, e_data, impl,
                                               CEED_REQUEST_IMMEDIATE));
    if (timing_enabled) {
      const double phase_end = CeedOperatorAssemblyTime_Opt();

      input_basis_time += phase_end - phase_start;
      phase_start = phase_end;
    }
    if (timing_enabled) qfunction_start = CeedOperatorAssemblyTime_Opt();
    if (assembly_qf) {
      const CeedScalar *assembly_inputs[CEED_FIELD_MAX] = {NULL};
      CeedScalar       *assembly_outputs[1]             = {qf_block};
      void             *ctx_data                        = NULL;

      if (timing_enabled) phase_start = CeedOperatorAssemblyTime_Opt();
      for (CeedInt i = 0; i < num_input_fields; i++) {
        CeedCallBackend(CeedVectorGetArrayRead(impl->q_vecs_in[i], CEED_MEM_HOST, &assembly_inputs[i]));
      }
      CeedCallBackend(CeedQFunctionGetContextData(qf, CEED_MEM_HOST, &ctx_data));
      CeedCallBackend(assembly_qf(ctx_data, Q * block_size, assembly_inputs, assembly_outputs));
      CeedCallBackend(CeedQFunctionRestoreContextData(qf, &ctx_data));
      for (CeedInt i = 0; i < num_input_fields; i++) {
        CeedCallBackend(CeedVectorRestoreArrayRead(impl->q_vecs_in[i], &assembly_inputs[i]));
      }
      qf_write = qf_block + qf_block_size;
      if (timing_enabled) qf_apply_time += CeedOperatorAssemblyTime_Opt() - phase_start;
    } else {
      for (CeedInt i = 0; i < num_input_fields; i++) {
        CeedInt    field_size;
        CeedVector vec;

        CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
        const bool is_active = vec == CEED_VECTOR_ACTIVE;
        CeedCallBackend(CeedVectorDestroy(&vec));
        if (!is_active) continue;
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_input_fields[i], &field_size));
        for (CeedInt field = 0; field < field_size; field++) {
          CeedScalar *array;

          if (timing_enabled) phase_start = CeedOperatorAssemblyTime_Opt();
          CeedCallBackend(CeedVectorGetArray(impl->q_vecs_in[i], CEED_MEM_HOST, &array));
          for (CeedInt j = 0; j < Q * block_size; j++) array[field * Q * block_size + j] = 1.0;
          CeedCallBackend(CeedVectorRestoreArray(impl->q_vecs_in[i], &array));
          if (timing_enabled) qf_seed_time += CeedOperatorAssemblyTime_Opt() - phase_start;
          if (!impl->is_identity_qf) {
            if (timing_enabled) phase_start = CeedOperatorAssemblyTime_Opt();
            for (CeedInt out = 0; out < num_output_fields; out++) {
              CeedInt    out_size;
              CeedVector out_vec;

              CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &out_vec));
              if (out_vec == CEED_VECTOR_ACTIVE) {
                CeedCallBackend(CeedVectorSetArray(impl->q_vecs_out[out], CEED_MEM_HOST, CEED_USE_POINTER, qf_write));
                CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[out], &out_size));
                qf_write += out_size * Q * block_size;
              }
              CeedCallBackend(CeedVectorDestroy(&out_vec));
            }
            if (timing_enabled) {
              qf_bind_time += CeedOperatorAssemblyTime_Opt() - phase_start;
              phase_start = CeedOperatorAssemblyTime_Opt();
            }
            CeedCallBackend(CeedQFunctionApply(qf, Q * block_size, impl->q_vecs_in, impl->q_vecs_out));
            if (timing_enabled) {
              qf_apply_time += CeedOperatorAssemblyTime_Opt() - phase_start;
              phase_start = CeedOperatorAssemblyTime_Opt();
            }
            for (CeedInt out = 0; out < num_output_fields; out++) {
              CeedVector out_vec;

              CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &out_vec));
              if (out_vec == CEED_VECTOR_ACTIVE) CeedCallBackend(CeedVectorTakeArray(impl->q_vecs_out[out], CEED_MEM_HOST, NULL));
              CeedCallBackend(CeedVectorDestroy(&out_vec));
            }
            if (timing_enabled) qf_unbind_time += CeedOperatorAssemblyTime_Opt() - phase_start;
          } else {
            CeedInt           out_size;
            const CeedScalar *array_read;

            CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[0], &out_size));
            CeedCallBackend(CeedVectorGetArrayRead(impl->q_vecs_out[0], CEED_MEM_HOST, &array_read));
            memcpy(qf_write, array_read, out_size * Q * block_size * sizeof(*qf_write));
            CeedCallBackend(CeedVectorRestoreArrayRead(impl->q_vecs_out[0], &array_read));
            qf_write += out_size * Q * block_size;
          }
          if (timing_enabled) phase_start = CeedOperatorAssemblyTime_Opt();
          CeedCallBackend(CeedVectorGetArray(impl->q_vecs_in[i], CEED_MEM_HOST, &array));
          for (CeedInt j = 0; j < Q * block_size; j++) array[field * Q * block_size + j] = 0.0;
          CeedCallBackend(CeedVectorRestoreArray(impl->q_vecs_in[i], &array));
          if (timing_enabled) qf_reset_time += CeedOperatorAssemblyTime_Opt() - phase_start;
        }
      }
    }
    CeedCheck(qf_write == qf_block + qf_block_size, ceed, CEED_ERROR_BACKEND, "Incorrect fused QFunction assembly size");
    if (timing_enabled) qfunction_time += CeedOperatorAssemblyTime_Opt() - qfunction_start;
    bool native_p1_value_output_zero = native_p1_direct;
    if (native_p1_direct) {
      // With a separately assembled lumped pseudo-time mass, the complete
      // value-output block is exactly zero. Detect that property rather than
      // assuming it, preserving correctness for consistent-mass operators.
      for (CeedInt col = 0; col < 4 && native_p1_value_output_zero; col++) {
        for (CeedInt row = 0; row < 4 && native_p1_value_output_zero; row++) {
          for (CeedInt point = 0; point < Q * block_size; point++) {
            if (qf_block[((CeedSize)col * 12 + row) * Q * block_size + point] != 0.) {
              native_p1_value_output_zero = false;
              break;
            }
          }
        }
      }
    }

    const CeedInt block_end = CeedIntMin(e_block + block_size, num_elem);
    for (CeedInt e = e_block; e < block_end; e++) {
      const CeedInt lane = e - e_block;

      if (timing_enabled) phase_start = CeedOperatorAssemblyTime_Opt();
      const CeedInt num_eval_in = num_eval_modes_in[active_input], num_eval_out = num_eval_modes_out[active_output];
      if (native_p1_direct) {
        // Native 2D P1 flow block: contract the complete 3x3 value/gradient
        // coefficient block at each quadrature point directly into each 4x4
        // nodal component block, then reduce immediately into the persistent
        // compact destination. This removes BTD staging, the generic tensor
        // contraction, and the temporary element-matrix stream.
        for (CeedInt comp_in = 0; comp_in < 4; comp_in++) {
          for (CeedInt comp_out = 0; comp_out < 4; comp_out++) {
            CeedScalar entries[16] = {0.};
            const CeedSize col_0 = comp_in, col_1 = 4 + comp_in, col_2 = 8 + comp_in;
            const CeedSize row_0 = comp_out, row_1 = 4 + comp_out, row_2 = 8 + comp_out;

            for (CeedInt q = 0; q < 4; q++) {
              const CeedSize   point_index = (CeedSize)q * block_size + lane;
              const CeedScalar coeff_00    = native_p1_value_output_zero
                                                 ? 0.
                                                 : qf_block[(col_0 * 12 + row_0) * Q * block_size + point_index];
              const CeedScalar coeff_10 = qf_block[(col_0 * 12 + row_1) * Q * block_size + point_index];
              const CeedScalar coeff_11 = qf_block[(col_1 * 12 + row_1) * Q * block_size + point_index];
              const CeedScalar coeff_12 = qf_block[(col_2 * 12 + row_1) * Q * block_size + point_index];
              const CeedScalar coeff_20 = qf_block[(col_0 * 12 + row_2) * Q * block_size + point_index];
              const CeedScalar coeff_21 = qf_block[(col_1 * 12 + row_2) * Q * block_size + point_index];
              const CeedScalar coeff_22 = qf_block[(col_2 * 12 + row_2) * Q * block_size + point_index];

              CeedPragmaSIMD for (CeedInt node_pair = 0; node_pair < 16; node_pair++) {
                const CeedScalar *basis_product = native_p1_basis_products[node_pair][q];

                entries[node_pair] += basis_product[0] * coeff_00 + basis_product[3] * coeff_10 + basis_product[4] * coeff_11 +
                                      basis_product[5] * coeff_12 + basis_product[6] * coeff_20 + basis_product[7] * coeff_21 +
                                      basis_product[8] * coeff_22;
              }
            }
            for (CeedInt node_pair = 0; node_pair < 16; node_pair++) {
              vals[entry_map[offset + count++]] += entries[node_pair];
            }
          }
        }
      } else {
        for (CeedInt comp_in = 0; comp_in < num_comp_in; comp_in++) {
          for (CeedInt comp_out = 0; comp_out < num_comp_out; comp_out++) {
            const CeedInt component_block = comp_in * num_comp_out + comp_out;
            CeedScalar   *BTD_component = &BTD_mat[(CeedSize)component_block * elem_size_out * btd_stride];

            for (CeedInt q = 0; q < Q; q++) {
              for (CeedInt e_in = 0; e_in < num_eval_in; e_in++) {
                const CeedSize row_offset = eval_mode_offsets_in[active_input][e_in] + comp_in;
                CeedScalar    *btd_values = contract ? &BTD_mat[((CeedSize)q * num_eval_in + e_in) * elem_size_out * num_component_blocks +
                                                                (CeedSize)component_block * elem_size_out]
                                                     : BTD_sum;

                for (CeedInt e_out = 0; e_out < num_eval_out; e_out++) {
                  const CeedSize eval_mode_index = row_offset * num_output_components + eval_mode_offsets_out[active_output][e_out] + comp_out;
                  const CeedSize point_index     = (CeedSize)q * block_size + lane;
                  const CeedScalar qf_value      = qf_block[eval_mode_index * Q * block_size + point_index];
                  const CeedScalar *b_out        = &B_mat_out[(q * num_eval_out + e_out) * elem_size_out];

                  if (!e_out) {
                    CeedPragmaSIMD for (CeedInt n = 0; n < elem_size_out; n++) btd_values[n] = b_out[n] * qf_value;
                  } else {
                    CeedPragmaSIMD for (CeedInt n = 0; n < elem_size_out; n++) btd_values[n] += b_out[n] * qf_value;
                  }
                }
                if (!contract) {
                  for (CeedInt n = 0; n < elem_size_out; n++) BTD_component[n * btd_stride + q * num_eval_in + e_in] = BTD_sum[n];
                }
              }
            }
            if (!contract)
              CeedCallBackend(CeedMatrixMatrixMultiply(ceed, BTD_component, B_mat_in,
                                                       &elem_mat[(CeedSize)component_block * elem_size_out * elem_size_in], elem_size_out, elem_size_in,
                                                       btd_stride));
          }
        }
      }
      if (timing_enabled) {
        const double phase_end = CeedOperatorAssemblyTime_Opt();

        btd_time += phase_end - phase_start;
        phase_start = phase_end;
      }
      if (contract && !native_p1_direct)
        CeedCallBackend(CeedTensorContractApply(contract, 1, btd_stride, elem_size_in, elem_size_out * num_component_blocks, BTD_mat, CEED_TRANSPOSE,
                                                false, B_mat_in, elem_mat));
      if (timing_enabled) {
        const double phase_end = CeedOperatorAssemblyTime_Opt();

        contract_time += phase_end - phase_start;
        phase_start = phase_end;
      }
      if (!native_p1_direct) {
        for (CeedInt component_block = 0; component_block < num_component_blocks; component_block++) {
          const CeedScalar *elem_component = &elem_mat[(CeedSize)component_block * elem_size_out * elem_size_in];

          for (CeedInt i = 0; i < elem_size_out; i++) {
            for (CeedInt j = 0; j < elem_size_in; j++) vals[entry_map[offset + count++]] += elem_component[i * elem_size_in + j];
          }
        }
      }
      if (timing_enabled) scatter_time += CeedOperatorAssemblyTime_Opt() - phase_start;
    }
  }

  const CeedSize local_num_entries = (CeedSize)elem_size_out * num_comp_out * elem_size_in * num_comp_in * num_elem;
  CeedCheck(count == local_num_entries, ceed, CEED_ERROR_BACKEND, "Error computing fused mapped entries");
  CeedCallBackend(CeedVectorRestoreArray(values, &vals));
  for (CeedInt out = 0; out < num_output_fields; out++) {
    CeedVector vec;

    CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &vec));
    if (vec == CEED_VECTOR_ACTIVE) CeedCallBackend(CeedVectorSetValue(impl->q_vecs_out[out], 0.0));
    CeedCallBackend(CeedVectorDestroy(&vec));
  }
  CeedCallBackend(CeedOperatorRestoreInputs_Opt(num_input_fields, qf_input_fields, op_input_fields, e_data, impl));
  CeedCallBackend(CeedDestroy(&ceed));
  CeedCallBackend(CeedQFunctionDestroy(&qf));
  if (timing_enabled) {
    static unsigned long call_id      = 0;
    const char          *rank_env     = getenv("OMPI_COMM_WORLD_RANK");
    const int            rank         = rank_env ? atoi(rank_env) : -1;
    const double         total_time   = CeedOperatorAssemblyTime_Opt() - total_start;
    const double         cleanup_time = total_time - setup_time - input_basis_time - qfunction_time - btd_time - contract_time - scatter_time;

    fprintf(stderr,
            "libCEED mapped assembly timing: rank=%d call=%lu elements=%" CeedInt_FMT " components=%" CeedInt_FMT
            " setup=%.9f input_basis=%.9f qfunction=%.9f qf_seed=%.9f qf_bind=%.9f qf_apply=%.9f qf_unbind=%.9f qf_reset=%.9f"
            " btd=%.9f contract=%.9f scatter=%.9f cleanup=%.9f total=%.9f\n",
            rank, call_id++, num_elem, num_component_blocks, setup_time, input_basis_time, qfunction_time, qf_seed_time, qf_bind_time, qf_apply_time,
            qf_unbind_time, qf_reset_time, btd_time, contract_time, scatter_time, cleanup_time, total_time);
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunction_Opt(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Opt(op, true, assembled, rstr, request);
}

//------------------------------------------------------------------------------
// Update Assembled Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunctionUpdate_Opt(CeedOperator op, CeedVector assembled, CeedElemRestriction rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Opt(op, false, &assembled, &rstr, request);
}

//------------------------------------------------------------------------------
// Operator Destroy
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Opt(CeedOperator op) {
  CeedOperator_Opt *impl;

  CeedCallBackend(CeedOperatorGetData(op, &impl));
  for (CeedInt i = 0; i < impl->num_inputs + impl->num_outputs; i++) {
    CeedCallBackend(CeedElemRestrictionDestroy(&impl->block_rstr[i]));
    CeedCallBackend(CeedVectorDestroy(&impl->e_vecs_full[i]));
  }
  CeedCallBackend(CeedFree(&impl->block_rstr));
  CeedCallBackend(CeedFree(&impl->e_vecs_full));
  CeedCallBackend(CeedFree(&impl->input_states));
  CeedCallBackend(CeedFree(&impl->skip_rstr_in));
  CeedCallBackend(CeedFree(&impl->skip_rstr_out));
  CeedCallBackend(CeedFree(&impl->apply_add_basis_out));

  for (CeedInt i = 0; i < impl->num_inputs; i++) {
    CeedCallBackend(CeedVectorDestroy(&impl->e_vecs_in[i]));
    CeedCallBackend(CeedVectorDestroy(&impl->q_vecs_in[i]));
  }
  CeedCallBackend(CeedFree(&impl->e_vecs_in));
  CeedCallBackend(CeedFree(&impl->q_vecs_in));

  for (CeedInt i = 0; i < impl->num_outputs; i++) {
    CeedCallBackend(CeedVectorDestroy(&impl->e_vecs_out[i]));
    CeedCallBackend(CeedVectorDestroy(&impl->q_vecs_out[i]));
  }
  CeedCallBackend(CeedFree(&impl->e_vecs_out));
  CeedCallBackend(CeedFree(&impl->q_vecs_out));

  // QFunction assembly data
  CeedCallBackend(CeedVectorDestroy(&impl->qf_l_vec));
  CeedCallBackend(CeedElemRestrictionDestroy(&impl->qf_block_rstr));
  CeedCallBackend(CeedFree(&impl->mapped_qf_block));
  CeedCallBackend(CeedFree(&impl->mapped_btd));
  CeedCallBackend(CeedFree(&impl->mapped_btd_sum));
  CeedCallBackend(CeedFree(&impl->mapped_elem_mat));

  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Create
//------------------------------------------------------------------------------
int CeedOperatorCreate_Opt(CeedOperator op) {
  Ceed              ceed;
  Ceed_Opt         *ceed_impl;
  CeedOperator_Opt *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedGetData(ceed, &ceed_impl));
  const CeedInt block_size = ceed_impl->block_size;

  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedOperatorSetData(op, impl));

  CeedCheck(block_size == 1 || block_size == 8, ceed, CEED_ERROR_BACKEND, "Opt backend cannot use blocksize: %" CeedInt_FMT, block_size);

  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction", CeedOperatorLinearAssembleQFunction_Opt));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunctionUpdate", CeedOperatorLinearAssembleQFunctionUpdate_Opt));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleSingleBlockMapped", CeedOperatorAssembleSingleBlockMapped_Opt));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Opt));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Opt));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
