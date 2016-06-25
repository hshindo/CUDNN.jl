function filter_desc(a::CuArray{T}, format=CUDNN_TENSOR_NCHW)
  csize = Cint[size(a,i) for i=ndims(a):-1:1]
  #csize = Cint[size(a,i) for i=1:ndims(a)]
  p = Ptr{Void}[0]
  cudnnCreateFilterDescriptor(p)
  cudnnSetFilterNdDescriptor(p[1], datatype(T), format, ndims(a), csize)
  fd = new(p[1])
  finalizer(fd, cudnnDestroyFilterDescriptor)
  fd
end
