function create_tensor_descriptor(a::AbstractCudaArray)
  csize = Cint[size(a,i) for i=ndims(a):-1:1]
  cstrides = Cint[stride(a,i) for i=ndims(a):-1:1]
  p = cudnnTensorDescriptor_t[0]
  cudnnCreateTensorDescriptor(p)
  desc = p[1]
  cudnnSetTensorNdDescriptor(desc, datatype(a), ndims(a), csize, cstrides)
  desc
end

#Base.unsafe_convert(::Type{cudnnTensorDescriptor_t}, td::TensorDescriptor) = td.ptr
