type FilterDescriptor
  ptr
end

function FilterDescriptor(a::CudaArray, format=CUDNN_TENSOR_NCHW)
  csize = Cint[size(a,i) for i=ndims(a):-1:1]
  p = cudnnFilterDescriptor_t[0]
  cudnnCreateFilterDescriptor(p)
  cudnnSetFilterNdDescriptor_v4(p[1], datatype(a), format, ndims(a), csize)
  fd = FilterDescriptor(p[1])
  finalizer(fd, cudnnDestroyFilterDescriptor)
  fd
end

Base.unsafe_convert(::Type{cudnnFilterDescriptor_t}, fd::FilterDescriptor) = fd.ptr
