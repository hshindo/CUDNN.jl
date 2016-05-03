export TensorDescriptor

type TensorDescriptor
  ptr
end

function TensorDescriptor(a::CudaArray)
  csize = Cint[size(a,i) for i=ndims(a):-1:1]
  cstrides = Cint[stride(a,i) for i=ndims(a):-1:1]
  p = cudnnTensorDescriptor_t[0]
  cudnnCreateTensorDescriptor(p)
  td = TensorDescriptor(p[1])
  finalizer(td, cudnnDestroyTensorDescriptor)
  cudnnSetTensorNdDescriptor(td, datatype(a), ndims(a), csize, cstrides)
  td
end

Base.unsafe_convert(::Type{cudnnTensorDescriptor_t}, td::TensorDescriptor) = td.ptr

function transformtensor()
end

# y = alpha*x + beta*y
function addtensor{T}(alpha, x::CudaArray{T}, beta, y::CudaArray{T})
  handle = gethandle(x.dev)
  xdesc = TensorDescriptor(x)
  ydesc = TensorDescriptor(y)
  cudnnAddTensor(handle, T[alpha], xdesc, x, T[beta], ydesc, y)
end

# C = op ( alpha1[0] * A, alpha2[0] * B ) + beta[0] * C
function optensor()
end

function settensor{T}(y::CudaArray{T}, value)
  handle = gethandle(y.dev)
  ydesc = TensorDescriptor(y)
  cudnnSetTensor(handle, ydesc, y, T[value])
end

# y = alpha * y
function scaletensor{T}(y::CudaArray{T}, alpha)
  handle = gethandle(y.dev)
  ydesc = TensorDescriptor(y)
  cudnnScaleTensor(handle, ydesc, y, T[alpha])
end
