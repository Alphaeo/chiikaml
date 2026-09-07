#pragma once

namespace chiikaml {

// This is just to define the kernel types in a separate header file, so that they can be used in other parts of the codebase without including the entire SVMClassifier class definition.
// Kernel functions supported by the SVM classifiers.
enum class SVMKernel {
    Linear,
    Polynomial,
    RBF
};

} // namespace chiikaml