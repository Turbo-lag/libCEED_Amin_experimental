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
#include <string.h>

#include "ceed-opt.h"

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
                                              CeedInt skip_field, CeedVector in_vec, CeedScalar *e_data[2 * CEED_FIELD_MAX],
                                              CeedOperator_Opt *impl, CeedRequest *request) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedEvalMode eval_mode;

    if (i == skip_field) continue;
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
                                             CeedInt skip_field, CeedScalar *e_data[2 * CEED_FIELD_MAX], CeedOperator_Opt *impl,
                                             CeedRequest *request) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    bool                is_active;
    CeedInt             elem_size, size, num_comp;
    CeedEvalMode        eval_mode;
    CeedVector          vec;
    CeedElemRestriction elem_rstr;
    CeedBasis           basis;

    if (i == skip_field) continue;
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
                                                CeedInt skip_field, CeedScalar *e_data[2 * CEED_FIELD_MAX], CeedOperator_Opt *impl) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedEvalMode eval_mode;
    CeedVector   vec;

    if (i == skip_field) continue;
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
  CeedCallBackend(CeedOperatorSetupInputs_Opt(num_input_fields, qf_input_fields, op_input_fields, -1, in_vec, e_data, impl, request));

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
    CeedCallBackend(CeedOperatorInputBasis_Opt(e, Q, qf_input_fields, op_input_fields, num_input_fields, block_size, in_vec, false, -1, e_data, impl,
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
  CeedCallBackend(CeedOperatorRestoreInputs_Opt(num_input_fields, qf_input_fields, op_input_fields, -1, e_data, impl));
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
  CeedCallBackend(CeedOperatorSetupInputs_Opt(num_input_fields, qf_input_fields, op_input_fields, -1, NULL, e_data, impl, request));

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
    CeedCallBackend(CeedOperatorInputBasis_Opt(e, Q, qf_input_fields, op_input_fields, num_input_fields, block_size, NULL, true, -1, e_data, impl,
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
  CeedCallBackend(CeedOperatorRestoreInputs_Opt(num_input_fields, qf_input_fields, op_input_fields, -1, e_data, impl));
  CeedCallBackend(CeedDestroy(&ceed));
  CeedCallBackend(CeedQFunctionDestroy(&qf));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Fused mapped linear assembly
//------------------------------------------------------------------------------
static bool CeedOperatorCanUseNativeP1Assembly_Opt(CeedQFunctionUser assembly_qf, bool grad_in_value_out_zero, CeedTensorContract contract,
                                                   CeedInt num_comp_in, CeedInt num_comp_out, CeedInt elem_size_in, CeedInt elem_size_out,
                                                   CeedInt Q, CeedInt qf_size_in, CeedInt qf_size_out, CeedInt num_eval_modes_in,
                                                   CeedInt num_eval_modes_out, const CeedSize eval_mode_offsets_in[],
                                                   const CeedSize eval_mode_offsets_out[]) {
  return assembly_qf && grad_in_value_out_zero && contract && num_comp_in == 4 && num_comp_out == 4 && elem_size_in == 4 && elem_size_out == 4 &&
         Q == 4 && qf_size_in == 12 && qf_size_out == 12 && num_eval_modes_in == 3 && num_eval_modes_out == 3 && eval_mode_offsets_in[0] == 0 &&
         eval_mode_offsets_in[1] == 4 && eval_mode_offsets_in[2] == 8 && eval_mode_offsets_out[0] == 0 && eval_mode_offsets_out[1] == 4 &&
         eval_mode_offsets_out[2] == 8;
}

static void CeedOperatorBuildNativeP1BasisProducts_Opt(const CeedScalar B_mat_in[], const CeedScalar B_mat_out[],
                                                       CeedScalar basis_products[16][4][9]) {
  for (CeedInt node_out = 0; node_out < 4; node_out++) {
    for (CeedInt node_in = 0; node_in < 4; node_in++) {
      const CeedInt node_pair = node_out * 4 + node_in;

      for (CeedInt q = 0; q < 4; q++) {
        for (CeedInt eval_out = 0; eval_out < 3; eval_out++) {
          const CeedScalar b_out = B_mat_out[(q * 3 + eval_out) * 4 + node_out];

          for (CeedInt eval_in = 0; eval_in < 3; eval_in++) {
            basis_products[node_pair][q][eval_out * 3 + eval_in] = b_out * B_mat_in[(q * 3 + eval_in) * 4 + node_in];
          }
        }
      }
    }
  }
}

static bool CeedOperatorNativeP1ValueOutputIsZero_Opt(const CeedScalar qf_block[], CeedInt Q, CeedInt block_size) {
  for (CeedInt col = 0; col < 4; col++) {
    for (CeedInt row = 0; row < 4; row++) {
      for (CeedInt point = 0; point < Q * block_size; point++) {
        if (qf_block[((CeedSize)col * 12 + row) * Q * block_size + point] != 0.) return false;
      }
    }
  }
  return true;
}

static void CeedOperatorAssembleNativeP1Mapped_Opt(const CeedScalar qf_block[], CeedInt Q, CeedInt block_size, CeedInt lane,
                                                   bool value_output_zero, const CeedScalar basis_products[16][4][9], CeedSize entry_offset,
                                                   const CeedInt entry_map[], CeedSize *count, CeedScalar vals[]) {
  for (CeedInt comp_in = 0; comp_in < 4; comp_in++) {
    for (CeedInt comp_out = 0; comp_out < 4; comp_out++) {
      CeedScalar entries[16] = {0.};
      const CeedSize col_0 = comp_in, col_1 = 4 + comp_in, col_2 = 8 + comp_in;
      const CeedSize row_0 = comp_out, row_1 = 4 + comp_out, row_2 = 8 + comp_out;

      for (CeedInt q = 0; q < 4; q++) {
        const CeedSize   point_index = (CeedSize)q * block_size + lane;
        const CeedScalar coeff_00 = value_output_zero ? 0. : qf_block[(col_0 * 12 + row_0) * Q * block_size + point_index];
        const CeedScalar coeff_10 = qf_block[(col_0 * 12 + row_1) * Q * block_size + point_index];
        const CeedScalar coeff_11 = qf_block[(col_1 * 12 + row_1) * Q * block_size + point_index];
        const CeedScalar coeff_12 = qf_block[(col_2 * 12 + row_1) * Q * block_size + point_index];
        const CeedScalar coeff_20 = qf_block[(col_0 * 12 + row_2) * Q * block_size + point_index];
        const CeedScalar coeff_21 = qf_block[(col_1 * 12 + row_2) * Q * block_size + point_index];
        const CeedScalar coeff_22 = qf_block[(col_2 * 12 + row_2) * Q * block_size + point_index];

        CeedPragmaSIMD for (CeedInt node_pair = 0; node_pair < 16; node_pair++) {
          const CeedScalar *basis_product = basis_products[node_pair][q];

          entries[node_pair] += basis_product[0] * coeff_00 + basis_product[3] * coeff_10 + basis_product[4] * coeff_11 +
                                basis_product[5] * coeff_12 + basis_product[6] * coeff_20 + basis_product[7] * coeff_21 +
                                basis_product[8] * coeff_22;
        }
      }
      for (CeedInt node_pair = 0; node_pair < 16; node_pair++) vals[entry_map[entry_offset + (*count)++]] += entries[node_pair];
    }
  }
}

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

  CeedCallBackend(CeedOperatorSetup_Opt(op));
  CeedCheck(!impl->is_identity_rstr_op, ceed, CEED_ERROR_BACKEND, "Assembling restriction only operators is not supported");
  CeedCallBackend(
      CeedOperatorSetupInputs_Opt(num_input_fields, qf_input_fields, op_input_fields, -1, NULL, e_data, impl, CEED_REQUEST_IMMEDIATE));

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
  // Preserve the established scalar fallback unless the QFunction supplies a
  // complete pointwise Jacobian. In that case the same mapped contraction used
  // by block operators is valid for the single 1x1 component block and avoids
  // repeated directional QFunction applications.
  if (num_comp_in == 1 && num_comp_out == 1 && !assembly_qf) {
    *is_supported = false;
    CeedCallBackend(CeedOperatorRestoreInputs_Opt(num_input_fields, qf_input_fields, op_input_fields, -1, e_data, impl));
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
  CeedCallBackend(CeedBasisGetTensorContract(basis_in, &contract));
  const bool native_p1_direct = CeedOperatorCanUseNativeP1Assembly_Opt(
      assembly_qf, assembly_grad_in_value_out_zero, contract, num_comp_in, num_comp_out, elem_size_in, elem_size_out, Q, qf_size_in, qf_size_out,
      num_eval_modes_in[active_input], num_eval_modes_out[active_output], eval_mode_offsets_in[active_input], eval_mode_offsets_out[active_output]);
  CeedScalar native_p1_basis_products[16][4][9];

  if (native_p1_direct) CeedOperatorBuildNativeP1BasisProducts_Opt(B_mat_in, B_mat_out, native_p1_basis_products);
  if (!native_p1_direct && !impl->mapped_btd)
    CeedCallBackend(CeedCalloc((CeedSize)num_component_blocks * elem_size_out * btd_stride, &impl->mapped_btd));
  if (!native_p1_direct && !contract && !impl->mapped_btd_sum) CeedCallBackend(CeedCalloc(elem_size_out, &impl->mapped_btd_sum));
  if (!native_p1_direct && !impl->mapped_elem_mat)
    CeedCallBackend(CeedCalloc((CeedSize)num_component_blocks * elem_size_out * elem_size_in, &impl->mapped_elem_mat));
  if (!impl->mapped_qf_block) CeedCallBackend(CeedCalloc(qf_block_size, &impl->mapped_qf_block));
  BTD_mat  = impl->mapped_btd;
  BTD_sum  = impl->mapped_btd_sum;
  elem_mat = impl->mapped_elem_mat;
  qf_block = impl->mapped_qf_block;
  CeedCallBackend(CeedVectorGetArray(values, CEED_MEM_HOST, &vals));

  CeedSize count = 0;
  for (CeedInt e_block = 0; e_block < num_elem; e_block += block_size) {
    CeedScalar *qf_write = qf_block;

    CeedCallBackend(CeedOperatorInputBasis_Opt(e_block, Q, qf_input_fields, op_input_fields, num_input_fields, block_size, NULL, true, -1, e_data,
                                               impl, CEED_REQUEST_IMMEDIATE));
    if (assembly_qf) {
      const CeedScalar *assembly_inputs[CEED_FIELD_MAX] = {NULL};
      CeedScalar       *assembly_outputs[1]             = {qf_block};
      void             *ctx_data                        = NULL;

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

          CeedCallBackend(CeedVectorGetArray(impl->q_vecs_in[i], CEED_MEM_HOST, &array));
          for (CeedInt j = 0; j < Q * block_size; j++) array[field * Q * block_size + j] = 1.0;
          CeedCallBackend(CeedVectorRestoreArray(impl->q_vecs_in[i], &array));
          if (!impl->is_identity_qf) {
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
            CeedCallBackend(CeedQFunctionApply(qf, Q * block_size, impl->q_vecs_in, impl->q_vecs_out));
            for (CeedInt out = 0; out < num_output_fields; out++) {
              CeedVector out_vec;

              CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &out_vec));
              if (out_vec == CEED_VECTOR_ACTIVE) CeedCallBackend(CeedVectorTakeArray(impl->q_vecs_out[out], CEED_MEM_HOST, NULL));
              CeedCallBackend(CeedVectorDestroy(&out_vec));
            }
          } else {
            CeedInt           out_size;
            const CeedScalar *array_read;

            CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[0], &out_size));
            CeedCallBackend(CeedVectorGetArrayRead(impl->q_vecs_out[0], CEED_MEM_HOST, &array_read));
            memcpy(qf_write, array_read, out_size * Q * block_size * sizeof(*qf_write));
            CeedCallBackend(CeedVectorRestoreArrayRead(impl->q_vecs_out[0], &array_read));
            qf_write += out_size * Q * block_size;
          }
          CeedCallBackend(CeedVectorGetArray(impl->q_vecs_in[i], CEED_MEM_HOST, &array));
          for (CeedInt j = 0; j < Q * block_size; j++) array[field * Q * block_size + j] = 0.0;
          CeedCallBackend(CeedVectorRestoreArray(impl->q_vecs_in[i], &array));
        }
      }
    }
    CeedCheck(qf_write == qf_block + qf_block_size, ceed, CEED_ERROR_BACKEND, "Incorrect fused QFunction assembly size");
    // With separately assembled lumped pseudo-time mass the value-output block
    // is zero. Detect that property so consistent-mass operators remain valid.
    const bool native_p1_value_output_zero = native_p1_direct && CeedOperatorNativeP1ValueOutputIsZero_Opt(qf_block, Q, block_size);

    const CeedInt block_end = CeedIntMin(e_block + block_size, num_elem);
    for (CeedInt e = e_block; e < block_end; e++) {
      const CeedInt lane = e - e_block;

      const CeedInt num_eval_in = num_eval_modes_in[active_input], num_eval_out = num_eval_modes_out[active_output];
      if (native_p1_direct) {
        CeedOperatorAssembleNativeP1Mapped_Opt(qf_block, Q, block_size, lane, native_p1_value_output_zero, native_p1_basis_products, offset,
                                               entry_map, &count, vals);
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
      if (contract && !native_p1_direct)
        CeedCallBackend(CeedTensorContractApply(contract, 1, btd_stride, elem_size_in, elem_size_out * num_component_blocks, BTD_mat, CEED_TRANSPOSE,
                                                false, B_mat_in, elem_mat));
      if (!native_p1_direct) {
        for (CeedInt component_block = 0; component_block < num_component_blocks; component_block++) {
          const CeedScalar *elem_component = &elem_mat[(CeedSize)component_block * elem_size_out * elem_size_in];

          for (CeedInt i = 0; i < elem_size_out; i++) {
            for (CeedInt j = 0; j < elem_size_in; j++) vals[entry_map[offset + count++]] += elem_component[i * elem_size_in + j];
          }
        }
      }
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
  CeedCallBackend(CeedOperatorRestoreInputs_Opt(num_input_fields, qf_input_fields, op_input_fields, -1, e_data, impl));
  CeedCallBackend(CeedDestroy(&ceed));
  CeedCallBackend(CeedQFunctionDestroy(&qf));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Paired operator apply and mapped linear assembly
//------------------------------------------------------------------------------
static int CeedOperatorApplyAndLinearAssembleMapped_Opt(CeedOperator residual_op, CeedVector in_vec, CeedVector out_vec,
                                                        CeedOperator jacobian_op, const CeedInt entry_map[], CeedVector values,
                                                        bool apply_add, bool *is_supported) {
  Ceed                     residual_ceed, jacobian_ceed;
  Ceed_Opt                *residual_ceed_impl, *jacobian_ceed_impl;
  CeedOperator_Opt        *residual_impl, *jacobian_impl;
  CeedInt                  Q, jacobian_Q, num_elem, jacobian_num_elem;
  CeedInt                  num_residual_inputs, num_residual_outputs, num_jacobian_inputs, num_jacobian_outputs;
  CeedScalar              *residual_e_data[2 * CEED_FIELD_MAX] = {0}, *jacobian_e_data[2 * CEED_FIELD_MAX] = {0};
  CeedQFunction            residual_qf, jacobian_qf;
  CeedQFunctionUser        assembly_qf = NULL;
  bool                     assembly_grad_in_value_out_zero;
  CeedQFunctionField      *residual_qf_inputs, *residual_qf_outputs, *jacobian_qf_inputs, *jacobian_qf_outputs;
  CeedOperatorField       *residual_inputs, *residual_outputs, *jacobian_inputs, *jacobian_outputs;
  CeedOperatorAssemblyData data;
  CeedInt                  num_active_bases_in, *num_eval_modes_in, num_active_bases_out, *num_eval_modes_out;
  CeedSize                 num_output_components, **eval_mode_offsets_in, **eval_mode_offsets_out;
  CeedBasis               *active_bases_in, *active_bases_out;
  CeedElemRestriction     *active_rstrs_in, *active_rstrs_out;
  const CeedScalar       **B_mats_in, **B_mats_out;
  CeedInt                  residual_handoff = -1, jacobian_handoff = -1;

  *is_supported = false;
  CeedCallBackend(CeedOperatorGetCeed(residual_op, &residual_ceed));
  CeedCallBackend(CeedOperatorGetCeed(jacobian_op, &jacobian_ceed));
  CeedCallBackend(CeedGetData(residual_ceed, &residual_ceed_impl));
  // A pair is only backend-local when both operators belong to the same Ceed
  // instance. This also makes the following backend data access safe.
  if (jacobian_ceed != residual_ceed) {
    CeedCallBackend(CeedDestroy(&residual_ceed));
    CeedCallBackend(CeedDestroy(&jacobian_ceed));
    return CEED_ERROR_SUCCESS;
  }
  CeedCallBackend(CeedGetData(jacobian_ceed, &jacobian_ceed_impl));
  if (residual_ceed_impl->block_size != jacobian_ceed_impl->block_size) {
    CeedCallBackend(CeedDestroy(&residual_ceed));
    CeedCallBackend(CeedDestroy(&jacobian_ceed));
    return CEED_ERROR_SUCCESS;
  }
  CeedCallBackend(CeedOperatorGetData(residual_op, &residual_impl));
  CeedCallBackend(CeedOperatorGetData(jacobian_op, &jacobian_impl));
  CeedCallBackend(CeedOperatorGetNumElements(residual_op, &num_elem));
  CeedCallBackend(CeedOperatorGetNumElements(jacobian_op, &jacobian_num_elem));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(residual_op, &Q));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(jacobian_op, &jacobian_Q));
  if (num_elem != jacobian_num_elem || Q != jacobian_Q) {
    CeedCallBackend(CeedDestroy(&residual_ceed));
    CeedCallBackend(CeedDestroy(&jacobian_ceed));
    return CEED_ERROR_SUCCESS;
  }
  CeedCallBackend(CeedOperatorGetQFunction(residual_op, &residual_qf));
  CeedCallBackend(CeedOperatorGetQFunction(jacobian_op, &jacobian_qf));
  CeedCallBackend(CeedQFunctionGetAssemblyUserFunction(jacobian_qf, &assembly_qf));
  CeedCallBackend(CeedQFunctionGetAssemblyDataGradientInputToValueOutputZero(jacobian_qf, &assembly_grad_in_value_out_zero));
  if (!assembly_qf) goto unsupported;
  CeedCallBackend(CeedOperatorGetFields(residual_op, &num_residual_inputs, &residual_inputs, &num_residual_outputs, &residual_outputs));
  CeedCallBackend(CeedOperatorGetFields(jacobian_op, &num_jacobian_inputs, &jacobian_inputs, &num_jacobian_outputs, &jacobian_outputs));
  CeedCallBackend(CeedQFunctionGetFields(residual_qf, NULL, &residual_qf_inputs, NULL, &residual_qf_outputs));
  CeedCallBackend(CeedQFunctionGetFields(jacobian_qf, NULL, &jacobian_qf_inputs, NULL, &jacobian_qf_outputs));

  // Locate one passive EVAL_NONE vector produced by the residual QFunction and
  // consumed by the target assembly QFunction. Equality of the CeedVector
  // handle is the explicit data-dependency contract between the two operators.
  for (CeedInt i = 0; i < num_residual_outputs; i++) {
    CeedEvalMode residual_mode;
    CeedVector   residual_vec;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(residual_qf_outputs[i], &residual_mode));
    CeedCallBackend(CeedOperatorFieldGetVector(residual_outputs[i], &residual_vec));
    if (residual_mode == CEED_EVAL_NONE && residual_vec != CEED_VECTOR_ACTIVE && residual_vec != CEED_VECTOR_NONE) {
      for (CeedInt j = 0; j < num_jacobian_inputs; j++) {
        CeedEvalMode        jacobian_mode;
        CeedVector          jacobian_vec;
        CeedElemRestriction residual_rstr, jacobian_rstr;
        CeedInt             residual_size, jacobian_size;
        bool                handoff_layout_safe = false;

        CeedCallBackend(CeedQFunctionFieldGetEvalMode(jacobian_qf_inputs[j], &jacobian_mode));
        CeedCallBackend(CeedOperatorFieldGetVector(jacobian_inputs[j], &jacobian_vec));
        CeedCallBackend(CeedOperatorFieldGetElemRestriction(residual_outputs[i], &residual_rstr));
        CeedCallBackend(CeedOperatorFieldGetElemRestriction(jacobian_inputs[j], &jacobian_rstr));
        CeedCallBackend(CeedQFunctionFieldGetSize(residual_qf_outputs[i], &residual_size));
        CeedCallBackend(CeedQFunctionFieldGetSize(jacobian_qf_inputs[j], &jacobian_size));
        if (residual_rstr == jacobian_rstr) {
          CeedRestrictionType handoff_rstr_type;
          CeedInt             handoff_num_elem, handoff_elem_size, handoff_num_comp, handoff_strides[3];
          bool                has_backend_strides;

          CeedCallBackend(CeedElemRestrictionGetType(residual_rstr, &handoff_rstr_type));
          if (handoff_rstr_type == CEED_RESTRICTION_STRIDED) {
            CeedCallBackend(CeedElemRestrictionGetNumElements(residual_rstr, &handoff_num_elem));
            CeedCallBackend(CeedElemRestrictionGetElementSize(residual_rstr, &handoff_elem_size));
            CeedCallBackend(CeedElemRestrictionGetNumComponents(residual_rstr, &handoff_num_comp));
            CeedCallBackend(CeedElemRestrictionHasBackendStrides(residual_rstr, &has_backend_strides));
            CeedCallBackend(CeedElemRestrictionGetStrides(residual_rstr, handoff_strides));
            handoff_layout_safe = handoff_num_elem == num_elem && handoff_elem_size == Q && handoff_num_comp == residual_size &&
                                  (has_backend_strides || (handoff_strides[0] == 1 && handoff_strides[1] == Q &&
                                                           handoff_strides[2] == Q * residual_size));
          }
        }
        if (jacobian_mode == CEED_EVAL_NONE && residual_vec == jacobian_vec && residual_size == jacobian_size && handoff_layout_safe) {
          if (residual_handoff >= 0) {
            CeedCallBackend(CeedElemRestrictionDestroy(&residual_rstr));
            CeedCallBackend(CeedElemRestrictionDestroy(&jacobian_rstr));
            CeedCallBackend(CeedVectorDestroy(&jacobian_vec));
            CeedCallBackend(CeedVectorDestroy(&residual_vec));
            goto unsupported;
          }
          residual_handoff = i;
          jacobian_handoff = j;
        }
        CeedCallBackend(CeedElemRestrictionDestroy(&residual_rstr));
        CeedCallBackend(CeedElemRestrictionDestroy(&jacobian_rstr));
        CeedCallBackend(CeedVectorDestroy(&jacobian_vec));
      }
    }
    CeedCallBackend(CeedVectorDestroy(&residual_vec));
  }
  if (residual_handoff < 0) goto unsupported;

  CeedCallBackend(CeedOperatorGetOperatorAssemblyData(jacobian_op, &data));
  CeedCallBackend(CeedOperatorAssemblyDataGetEvalModes(data, &num_active_bases_in, &num_eval_modes_in, NULL, &eval_mode_offsets_in,
                                                       &num_active_bases_out, &num_eval_modes_out, NULL, &eval_mode_offsets_out,
                                                       &num_output_components));
  CeedCallBackend(CeedOperatorAssemblyDataGetBases(data, NULL, &active_bases_in, &B_mats_in, NULL, &active_bases_out, &B_mats_out));
  CeedCallBackend(CeedOperatorAssemblyDataGetElemRestrictions(data, NULL, &active_rstrs_in, NULL, &active_rstrs_out));
  if (num_active_bases_in != 1 || num_active_bases_out != 1) goto unsupported;

  CeedElemRestriction elem_rstr_in = active_rstrs_in[0], elem_rstr_out = active_rstrs_out[0];
  CeedRestrictionType rstr_type_in, rstr_type_out;
  CeedInt             num_elem_in, elem_size_in, num_comp_in, num_elem_out, elem_size_out, num_comp_out;
  CeedTensorContract  contract;

  CeedCallBackend(CeedElemRestrictionGetType(elem_rstr_in, &rstr_type_in));
  CeedCallBackend(CeedElemRestrictionGetType(elem_rstr_out, &rstr_type_out));
  if (rstr_type_in != CEED_RESTRICTION_STANDARD || rstr_type_out != CEED_RESTRICTION_STANDARD) goto unsupported;
  CeedCallBackend(CeedElemRestrictionGetNumElements(elem_rstr_in, &num_elem_in));
  CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr_in, &elem_size_in));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(elem_rstr_in, &num_comp_in));
  CeedCallBackend(CeedElemRestrictionGetNumElements(elem_rstr_out, &num_elem_out));
  CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr_out, &elem_size_out));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(elem_rstr_out, &num_comp_out));
  if (num_elem_in != num_elem_out || num_elem_in != num_elem) goto unsupported;
  CeedCallBackend(CeedBasisGetTensorContract(active_bases_in[0], &contract));

  CeedInt qf_size_in = 0, qf_size_out = 0;

  for (CeedInt i = 0; i < num_jacobian_inputs; i++) {
    CeedInt    field_size;
    CeedVector vec;

    CeedCallBackend(CeedOperatorFieldGetVector(jacobian_inputs[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedCallBackend(CeedQFunctionFieldGetSize(jacobian_qf_inputs[i], &field_size));
      qf_size_in += field_size;
    }
    CeedCallBackend(CeedVectorDestroy(&vec));
  }
  for (CeedInt i = 0; i < num_jacobian_outputs; i++) {
    CeedInt    field_size;
    CeedVector vec;

    CeedCallBackend(CeedOperatorFieldGetVector(jacobian_outputs[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedCallBackend(CeedQFunctionFieldGetSize(jacobian_qf_outputs[i], &field_size));
      qf_size_out += field_size;
    }
    CeedCallBackend(CeedVectorDestroy(&vec));
  }
  if (!qf_size_in || !qf_size_out || out_vec == CEED_VECTOR_NONE || in_vec == CEED_VECTOR_NONE) goto unsupported;

  const CeedInt     block_size           = residual_ceed_impl->block_size;
  const CeedInt     num_component_blocks = num_comp_in * num_comp_out;
  const CeedSize    btd_stride            = Q * num_eval_modes_in[0];
  const CeedSize    qf_block_size         = (CeedSize)qf_size_in * qf_size_out * Q * block_size;
  const CeedScalar *B_mat_in              = B_mats_in[0], *B_mat_out = B_mats_out[0];
  CeedScalar       *qf_block, *BTD_mat, *BTD_sum = NULL, *elem_mat, *vals = NULL;
  bool              residual_skip_out[CEED_FIELD_MAX] = {false};
  const bool native_p1_direct = CeedOperatorCanUseNativeP1Assembly_Opt(
      assembly_qf, assembly_grad_in_value_out_zero, contract, num_comp_in, num_comp_out, elem_size_in, elem_size_out, Q, qf_size_in, qf_size_out,
      num_eval_modes_in[0], num_eval_modes_out[0], eval_mode_offsets_in[0], eval_mode_offsets_out[0]);
  CeedScalar native_p1_basis_products[16][4][9];

  if (native_p1_direct) CeedOperatorBuildNativeP1BasisProducts_Opt(B_mat_in, B_mat_out, native_p1_basis_products);

  *is_supported = true;
  CeedDebug(residual_ceed, "\nUsing optimized paired residual and mapped assembly path\n");
  if (native_p1_direct) CeedDebug(residual_ceed, "\nUsing optimized paired native 2D P1 contraction\n");
  // Do not mutate public vectors while probing support. A standard pair clears
  // its destinations here; an aligned composite clears them once before
  // calling supported children with additive semantics.
  if (!apply_add) {
    CeedCallBackend(CeedVectorSetValue(out_vec, 0.0));
    CeedCallBackend(CeedVectorSetValue(values, 0.0));
  }
  CeedCallBackend(CeedOperatorSetup_Opt(residual_op));
  CeedCallBackend(CeedOperatorSetup_Opt(jacobian_op));
  CeedCallBackend(CeedOperatorSetupInputs_Opt(num_residual_inputs, residual_qf_inputs, residual_inputs, -1, in_vec, residual_e_data, residual_impl,
                                              CEED_REQUEST_IMMEDIATE));
  // The shared passive input comes directly from the residual's block-local
  // Q-vector. Do not restrict stale global data or rebind the target Q-vector;
  // this also leaves its input-state cache valid for the next ordinary call.
  CeedCallBackend(CeedOperatorSetupInputs_Opt(num_jacobian_inputs, jacobian_qf_inputs, jacobian_inputs, jacobian_handoff, NULL, jacobian_e_data,
                                              jacobian_impl, CEED_REQUEST_IMMEDIATE));
  for (CeedInt i = 0; i < num_residual_outputs; i++) {
    CeedEvalMode eval_mode;

    residual_skip_out[i] = residual_impl->skip_rstr_out[i];
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(residual_qf_outputs[i], &eval_mode));
    if (eval_mode == CEED_EVAL_NONE) {
      CeedScalar *array;

      CeedCallBackend(CeedVectorGetArrayWrite(residual_impl->e_vecs_out[i], CEED_MEM_HOST, &array));
      CeedCallBackend(CeedVectorSetArray(residual_impl->q_vecs_out[i], CEED_MEM_HOST, CEED_USE_POINTER, array));
      CeedCallBackend(CeedVectorRestoreArray(residual_impl->e_vecs_out[i], &array));
    }
  }
  if (!apply_add) {
    // Output restriction is additive across blocks. Match Apply semantics by
    // clearing every passive output, including the shared handoff, once before
    // the traversal. Composite callers already cleared all passive outputs.
    for (CeedInt i = 0; i < num_residual_outputs; i++) {
      CeedVector vec;

      CeedCallBackend(CeedOperatorFieldGetVector(residual_outputs[i], &vec));
      if (vec != CEED_VECTOR_ACTIVE && vec != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorSetValue(vec, 0.0));
      CeedCallBackend(CeedVectorDestroy(&vec));
    }
  }
  for (CeedInt i = 0; i < num_jacobian_inputs; i++) {
    CeedVector vec;

    CeedCallBackend(CeedOperatorFieldGetVector(jacobian_inputs[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) CeedCallBackend(CeedVectorSetValue(jacobian_impl->q_vecs_in[i], 0.0));
    CeedCallBackend(CeedVectorDestroy(&vec));
  }
  // These workspaces have the same immutable shape as the target operator's
  // direct mapped-assembly workspaces. Reuse them across calls instead of
  // allocating and clearing three arrays on every nonlinear stage.
  if (!jacobian_impl->mapped_qf_block) CeedCallBackend(CeedMalloc(qf_block_size, &jacobian_impl->mapped_qf_block));
  if (!native_p1_direct && !jacobian_impl->mapped_btd)
    CeedCallBackend(CeedMalloc((CeedSize)num_component_blocks * elem_size_out * btd_stride, &jacobian_impl->mapped_btd));
  if (!native_p1_direct && !contract && !jacobian_impl->mapped_btd_sum) CeedCallBackend(CeedMalloc(elem_size_out, &jacobian_impl->mapped_btd_sum));
  if (!native_p1_direct && !jacobian_impl->mapped_elem_mat)
    CeedCallBackend(CeedMalloc((CeedSize)num_component_blocks * elem_size_out * elem_size_in, &jacobian_impl->mapped_elem_mat));
  qf_block = jacobian_impl->mapped_qf_block;
  BTD_mat  = jacobian_impl->mapped_btd;
  BTD_sum  = jacobian_impl->mapped_btd_sum;
  elem_mat = jacobian_impl->mapped_elem_mat;
  CeedCallBackend(CeedVectorGetArray(values, CEED_MEM_HOST, &vals));

  CeedSize count = 0;
  for (CeedInt e_block = 0; e_block < num_elem; e_block += block_size) {
    const CeedScalar *handoff_array = NULL;
    const CeedScalar *assembly_inputs[CEED_FIELD_MAX] = {NULL};
    CeedScalar       *assembly_outputs[1]             = {qf_block};
    void             *ctx_data                        = NULL;

    CeedCallBackend(CeedOperatorInputBasis_Opt(e_block, Q, residual_qf_inputs, residual_inputs, num_residual_inputs, block_size, in_vec, false, -1,
                                               residual_e_data, residual_impl, CEED_REQUEST_IMMEDIATE));
    CeedCallBackend(CeedQFunctionApply(residual_qf, Q * block_size, residual_impl->q_vecs_in, residual_impl->q_vecs_out));
    CeedCallBackend(CeedOperatorOutputBasis_Opt(e_block, Q, residual_qf_outputs, residual_outputs, block_size, num_residual_inputs,
                                                num_residual_outputs, residual_impl->apply_add_basis_out, residual_skip_out, residual_op, out_vec,
                                                residual_impl, CEED_REQUEST_IMMEDIATE));

    CeedCallBackend(CeedOperatorInputBasis_Opt(e_block, Q, jacobian_qf_inputs, jacobian_inputs, num_jacobian_inputs, block_size, NULL, true,
                                               jacobian_handoff, jacobian_e_data, jacobian_impl, CEED_REQUEST_IMMEDIATE));
    CeedCallBackend(CeedVectorGetArrayRead(residual_impl->q_vecs_out[residual_handoff], CEED_MEM_HOST, &handoff_array));
    for (CeedInt i = 0; i < num_jacobian_inputs; i++) {
      if (i == jacobian_handoff) assembly_inputs[i] = handoff_array;
      else CeedCallBackend(CeedVectorGetArrayRead(jacobian_impl->q_vecs_in[i], CEED_MEM_HOST, &assembly_inputs[i]));
    }
    CeedCallBackend(CeedQFunctionGetContextData(jacobian_qf, CEED_MEM_HOST, &ctx_data));
    CeedCallBackend(assembly_qf(ctx_data, Q * block_size, assembly_inputs, assembly_outputs));
    CeedCallBackend(CeedQFunctionRestoreContextData(jacobian_qf, &ctx_data));
    for (CeedInt i = 0; i < num_jacobian_inputs; i++) {
      if (i != jacobian_handoff) CeedCallBackend(CeedVectorRestoreArrayRead(jacobian_impl->q_vecs_in[i], &assembly_inputs[i]));
    }
    CeedCallBackend(CeedVectorRestoreArrayRead(residual_impl->q_vecs_out[residual_handoff], &handoff_array));

    const bool native_p1_value_output_zero = native_p1_direct && CeedOperatorNativeP1ValueOutputIsZero_Opt(qf_block, Q, block_size);

    const CeedInt block_end = CeedIntMin(e_block + block_size, num_elem);
    for (CeedInt e = e_block; e < block_end; e++) {
      const CeedInt lane         = e - e_block;
      const CeedInt num_eval_in  = num_eval_modes_in[0];
      const CeedInt num_eval_out = num_eval_modes_out[0];

      if (native_p1_direct) {
        CeedOperatorAssembleNativeP1Mapped_Opt(qf_block, Q, block_size, lane, native_p1_value_output_zero, native_p1_basis_products, 0, entry_map,
                                               &count, vals);
      } else {
        for (CeedInt comp_in = 0; comp_in < num_comp_in; comp_in++) {
          for (CeedInt comp_out = 0; comp_out < num_comp_out; comp_out++) {
            const CeedInt component_block = comp_in * num_comp_out + comp_out;
            CeedScalar   *BTD_component = &BTD_mat[(CeedSize)component_block * elem_size_out * btd_stride];

            for (CeedInt q = 0; q < Q; q++) {
              for (CeedInt e_in = 0; e_in < num_eval_in; e_in++) {
                const CeedSize row_offset = eval_mode_offsets_in[0][e_in] + comp_in;
                CeedScalar    *btd_values = contract ? &BTD_mat[((CeedSize)q * num_eval_in + e_in) * elem_size_out * num_component_blocks +
                                                                 (CeedSize)component_block * elem_size_out]
                                                      : BTD_sum;

                for (CeedInt e_out = 0; e_out < num_eval_out; e_out++) {
                  const CeedSize eval_mode_index = row_offset * num_output_components + eval_mode_offsets_out[0][e_out] + comp_out;
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
              CeedCallBackend(CeedMatrixMatrixMultiply(residual_ceed, BTD_component, B_mat_in,
                                                       &elem_mat[(CeedSize)component_block * elem_size_out * elem_size_in], elem_size_out,
                                                       elem_size_in, btd_stride));
          }
        }
        if (contract)
          CeedCallBackend(CeedTensorContractApply(contract, 1, btd_stride, elem_size_in, elem_size_out * num_component_blocks, BTD_mat,
                                                  CEED_TRANSPOSE, false, B_mat_in, elem_mat));
        for (CeedInt component_block = 0; component_block < num_component_blocks; component_block++) {
          const CeedScalar *elem_component = &elem_mat[(CeedSize)component_block * elem_size_out * elem_size_in];

          for (CeedInt i = 0; i < elem_size_out; i++) {
            for (CeedInt j = 0; j < elem_size_in; j++) vals[entry_map[count++]] += elem_component[i * elem_size_in + j];
          }
        }
      }
    }
  }
  CeedCheck(count == (CeedSize)elem_size_out * num_comp_out * elem_size_in * num_comp_in * num_elem, residual_ceed, CEED_ERROR_BACKEND,
            "Error computing paired mapped entries");
  CeedCallBackend(CeedVectorRestoreArray(values, &vals));
  CeedCallBackend(CeedOperatorRestoreInputs_Opt(num_residual_inputs, residual_qf_inputs, residual_inputs, -1, residual_e_data, residual_impl));
  CeedCallBackend(CeedOperatorRestoreInputs_Opt(num_jacobian_inputs, jacobian_qf_inputs, jacobian_inputs, jacobian_handoff, jacobian_e_data,
                                                jacobian_impl));
  CeedCallBackend(CeedDestroy(&residual_ceed));
  CeedCallBackend(CeedDestroy(&jacobian_ceed));
  CeedCallBackend(CeedQFunctionDestroy(&residual_qf));
  CeedCallBackend(CeedQFunctionDestroy(&jacobian_qf));
  return CEED_ERROR_SUCCESS;

unsupported:
  CeedCallBackend(CeedDestroy(&residual_ceed));
  CeedCallBackend(CeedDestroy(&jacobian_ceed));
  CeedCallBackend(CeedQFunctionDestroy(&residual_qf));
  CeedCallBackend(CeedQFunctionDestroy(&jacobian_qf));
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
  CeedCallBackend(
      CeedSetBackendFunction(ceed, "Operator", op, "ApplyAndLinearAssembleMapped", CeedOperatorApplyAndLinearAssembleMapped_Opt));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Opt));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Opt));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
