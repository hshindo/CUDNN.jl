type FilterDesc
  ptr::Ptr{Void}

  function FilterDesc{T,N}(a::CuArray{T,N}, format=CUDNN_TENSOR_NCHW)
    csize = Cint[size(a,i) for i=ndims(a):-1:1]
    p = Ptr{Void}[0]
    cudnnCreateFilterDescriptor(p)
    cudnnSetFilterNdDescriptor(p[1], datatype(a), format, ndims(a), csize)
    fd = new(p[1])
    finalizer(fd, cudnnDestroyFilterDescriptor)
    fd
  end
end

Base.unsafe_convert(::Type{Ptr{Void}}, fd::FilterDesc) = fd.ptr
