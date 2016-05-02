export ConvolutionDescriptor

type ConvolutionDescriptor
  ptr
end

function ConvolutionDescriptor(nd, padding, stride, upscale, mode, xtype)
  p = cudnnConvolutionDescriptor_t[0]
  cudnnCreateConvolutionDescriptor(p)
  #cudnnSetConvolutionNdDescriptor(cd[1],nd,cdsize(padding,nd),cdsize(stride,nd),cdsize(upscale,nd),
  #                                mode,cudnnDataType(xtype))
  cd = ConvolutionDescriptor(p)
  finalizer(cd, cudnnDestroyConvolutionDescriptor)
  cd
end

Base.unsafe_convert(::Type{cudnnConvolutionDescriptor_t}, cd::ConvolutionDescriptor) = cd.ptr

function convolution_forward!()
end

function convolution_backward!()
end
