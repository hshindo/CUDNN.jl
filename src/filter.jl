type FD
  ptr
end

function FD(a::CudaArray, format=CUDNN_TENSOR_NCHW)
  csize = Cint[size(a,i) for i=ndims(a):-1:1]
  p = cudnnFilterDescriptor_t[0]
  cudnnCreateFilterDescriptor(p)
  p = p[1]
  cudnnSetFilterNdDescriptor_v4(p, datatype(a), format, ndims(a), csize)
  fd = FD(p)
  finalizer(fd, cudnnDestroyFilterDescriptor)
  fd
end

Base.unsafe_convert(::Type{cudnnFilterDescriptor_t}, fd::FD) = fd.ptr

function filter_forward!()
  csize = Cint[size(a,i) for i=ndims(a):-1:1]
  p = cudnnFilterDescriptor_t[0]
  cudnnCreateFilterDescriptor(p)
  fdesc = p[1]
  cudnnSetFilterNdDescriptor_v4(p, datatype(a), format, ndims(a), csize)


end
