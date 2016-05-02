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
