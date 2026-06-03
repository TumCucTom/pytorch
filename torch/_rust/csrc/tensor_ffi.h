#pragma once

#include <ATen/core/Tensor.h>
#include <rust/cxx.h>
#include <cstdint>

namespace torch::rust {

using Tensor = at::Tensor;

// Returns a borrowed pointer to the at::Tensor owned by the THPVariable, or
// nullptr if py_obj is not a torch.Tensor. Lifetime is tied to the Python
// object — the caller must keep the Python reference alive while using it.
const Tensor* tensor_from_pyobject(std::uintptr_t py_obj) noexcept;

int64_t tensor_dim(const Tensor& t);
int64_t tensor_numel(const Tensor& t);
int64_t tensor_size_at(const Tensor& t, int64_t dim);
int64_t tensor_stride_at(const Tensor& t, int64_t dim);

::rust::Slice<const int64_t> tensor_sizes(const Tensor& t);
::rust::Slice<const int64_t> tensor_strides(const Tensor& t);

bool tensor_is_contiguous(const Tensor& t);
bool tensor_is_cpu(const Tensor& t);
bool tensor_is_cuda(const Tensor& t);
bool tensor_defined(const Tensor& t);
bool tensor_requires_grad(const Tensor& t);

} // namespace torch::rust
