#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace {
/// This pass illustrates the IR nesting through printing.
struct DebugPrintPass
: public mlir::PassWrapper<DebugPrintPass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DebugPrintPass)

  mlir::StringRef getArgument() const final { return "test-print-nesting"; }
  mlir::StringRef getDescription() const final { return "Test various printing."; }

  // Entry point for the pass.
  void runOnOperation() override {
    mlir::Operation *op = getOperation();
    resetIndent();
    printOperation(op);
  }

  /// The three methods below are mutually recursive and follow the nesting of
  /// the IR: operation->region->block->operation->...

  void printOperation(mlir::Operation *op) {
    // Print the operation itself and some of its properties
    printIndent() << "visiting op: '" << op->getName() << "' with "
                  << op->getNumOperands() << " operands and "
                  << op->getNumResults() << " results\n";
    // Print the operation attributes
    if (!op->getAttrs().empty()) {
      printIndent() << op->getAttrs().size() << " attributes:\n";
      for (mlir::NamedAttribute attr : op->getAttrs())
        printIndent() << " - '" << attr.getName().getValue() << "' : '"
                      << attr.getValue() << "'\n";
    }

    // Recurse into each of the regions attached to the operation.
    printIndent() << " " << op->getNumRegions() << " nested regions:\n";
    auto indent = pushIndent();
    for (mlir::Region &region : op->getRegions())
      printRegion(region);
  }

  void printRegion(mlir::Region &region) {
    // A region does not hold anything by itself other than a list of blocks.
    printIndent() << "Region with " << region.getBlocks().size()
                  << " blocks:\n";
    auto indent = pushIndent();
    for (mlir::Block &block : region.getBlocks())
      printBlock(block);
  }

  void printBlock(mlir::Block &block) {
    // Print the block intrinsics properties (basically: argument list)
    printIndent()
        << "Block with " << block.getNumArguments() << " arguments, "
        << block.getNumSuccessors()
        << " successors, and "
        // Note, this `.size()` is traversing a linked-list and is O(n).
        << block.getOperations().size() << " operations\n";

    // Block main role is to hold a list of Operations: let's recurse.
    auto indent = pushIndent();
    for (mlir::Operation &op : block.getOperations())
      printOperation(&op);
  }

  /// Manages the indentation as we traverse the IR nesting.
  int indent;
  struct IdentRAII {
    int &indent;
    IdentRAII(int &indent) : indent(indent) {}
    ~IdentRAII() { --indent; }
  };
  void resetIndent() { indent = 0; }
  IdentRAII pushIndent() { return IdentRAII(++indent); }

  llvm::raw_ostream &printIndent() {
    for (int i = 0; i < indent; ++i)
      llvm::outs() << "  ";
    return llvm::outs();
  }
};
} // namespace


std::unique_ptr<mlir::Pass> mlir::toy::createDebugPrintPass() {
  return std::make_unique<DebugPrintPass>();
}
