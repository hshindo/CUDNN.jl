type CD
  ptr
end

function CD(nd, padding, stride, upscale, mode, xtype)
  p = cudnnConvolutionDescriptor_t[0]
  cudnnCreateConvolutionDescriptor(p)
  #cudnnSetConvolutionNdDescriptor(cd[1],nd,cdsize(padding,nd),cdsize(stride,nd),cdsize(upscale,nd),
  #                                mode,cudnnDataType(xtype))
  cd = CD(p)
  finalizer(cd, cudnnDestroyConvolutionDescriptor)
  cd
end

Base.unsafe_convert(::Type{cudnnConvolutionDescriptor_t}, cd::CD) = cd.ptr

function convolution_forward!()
end
