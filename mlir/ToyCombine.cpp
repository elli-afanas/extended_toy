//===- ToyCombine.cpp - Toy High Level Optimizer --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a set of simple combiners for optimizing operations in
// the Toy dialect.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "toy/Dialect.h"
#include <numeric>
using namespace mlir;
using namespace toy;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "ToyCombine.inc"
} // namespace

/// This is an example of a c++ rewrite pattern for the TransposeOp. It
/// optimizes the following scenario: transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose.
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

    // Input defined by another transpose? If not, no match.
    if (!transposeInputOp)
      return failure();

    // Otherwise, we have a redundant transpose. Use the rewriter.
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SimplifyMatAddAndTranspose : public mlir::OpRewritePattern<MatAddOp> {
  /// We register this pattern to match every toy.mat_add in the IR.
  SimplifyMatAddAndTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<MatAddOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it.
  mlir::LogicalResult matchAndRewrite(MatAddOp op, mlir::PatternRewriter &rewriter) const override {
    TransposeOp lhs_def_op = op.getLhs().getDefiningOp<TransposeOp>();
    TransposeOp rhs_def_op = op.getRhs().getDefiningOp<TransposeOp>();

    if(!rhs_def_op || !lhs_def_op)
      return failure();

    auto new_mul = rewriter.create<MatAddOp>(op.getLoc(), op.getType(),
                                           op.getLhs().getDefiningOp<TransposeOp>().getInput(),
                                           op.getRhs().getDefiningOp<TransposeOp>().getInput());

    rewriter.replaceOpWithNewOp<TransposeOp>(op, op.getType(), new_mul);

    return success();
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}

void MatAddOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<SimplifyMatAddAndTranspose>(context);
}

/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonicalization framework.
void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
              FoldConstantReshapeOptPattern>(context);
}

