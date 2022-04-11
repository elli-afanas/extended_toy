#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

void printOperation(mlir::Operation *op) {
  // Print the operation itself and some of its properties
  llvm::printIndent() << "visiting op: '" << op->getName() << "' with "
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
  /*auto indent = pushIndent();
  for (Region &region : op->getRegions())
    printRegion(region);*/
}