#include <iostream>

#include "mlir/Pass/Pass.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"
#include "toy/ShapeInferenceInterface.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <iomanip>
#include <memory>

using namespace mlir;
using namespace toy;

namespace {
/// Here we utilize the CRTP `PassWrapper` utility class to provide some
/// necessary utility hooks. This is only necessary for passes defined directly
/// in C++. Passes defined declaratively use a cleaner mechanism for providing
/// these utilities.
struct TestPass : public PassWrapper<TestPass, OperationPass<toy::MatAddOp>> {
    void runOnOperation() override {
        std::cout << "Testing pass" << std::endl;
    }
  };
} // namespace

/// Create a Test pass.
std::unique_ptr<mlir::Pass> mlir::toy::createTestPass() {
  return std::make_unique<TestPass>();
}

/*
/// Register this pass so that it can be built via from a textual pass pipeline.
/// (Pass registration is discussed more below)
void registerTestPass() {
    PassRegistration<TestPass>();
}*/
