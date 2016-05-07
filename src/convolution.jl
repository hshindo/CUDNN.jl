export ConvolutionDescriptor

type ConvolutionDescriptor
  ptr
end

function ConvolutionDescriptor(x::CudaArray, nd::Int, pads::Vector{Int}, strides::Vector{Int};
  upscales::Vector{Int}=Int[], mode=CUDNN_CONVOLUTION)
  upscales = fill(1, nd)
  p = cudnnConvolutionDescriptor_t[0]
  cudnnCreateConvolutionDescriptor(p)
  cudnnSetConvolutionNdDescriptor(p[1], nd, reverse(pads), reverse(strides), reverse(upscales), mode, datatype(x))
  cd = ConvolutionDescriptor(p[1])
  finalizer(cd, cudnnDestroyConvolutionDescriptor)
  cd
end

Base.unsafe_convert(::Type{cudnnConvolutionDescriptor_t}, cd::ConvolutionDescriptor) = cd.ptr

function cudnnConvolutionForward(src, fd, dest;
                                 handle=cudnnHandle, alpha=1.0, beta=0.0,
                                 algorithm=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                 workSpace=C_NULL, workSpaceSizeInBytes=0,
                                 cd=nothing, o...)
                               end

function conovolution_forward!{T}(alpha, x::CudaArray{T}, fd, cd, algo)
  cudnnConvolutionForward

  handle = gethandle(x.dev)


  cudnnSetActivationDescriptor(ad, mode, relu_nanopt, relu_ceiling)
  xdesc = TensorDescriptor(x)
  ydesc = TensorDescriptor(y)
  cudnnActivationForward(handle, ad, T[alpha], xdesc, x, T[beta], ydesc, y)
  ad
end

function conovolution_backward!()
end
