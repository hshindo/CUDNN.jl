export CUDNN_CONVOLUTION, CUDNN_CROSS_CORRELATION

type ConvolutionDesc
  ptr::Ptr{Void}
end

function ConvolutionDesc{T,N}(::Type{T}, pad::NTuple{N,Int}, stride::NTuple{N,Int}, mode)
  p = Ptr{Void}[0]
  cudnnCreateConvolutionDescriptor(p)
  cpad = Cint[pad[i] for i=N:-1:1]
  cstride = Cint[stride[i] for i=N:-1:1]
  cupscale = fill(Cint(1), N)
  cudnnSetConvolutionNdDescriptor(p[1], N, cpad, cstride, cupscale, mode, datatype(T))
  cd = ConvolutionDesc(p[1], pad, stride)
  finalizer(cd, cudnnDestroyConvolutionDescriptor)
  cd
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::ConvolutionDesc) = desc.ptr

function convolution!{T,N}(x::CuArray{T}, w::CuArray{T}, pad::NTuple{N,Int}, stride::NTuple{N,Int}, y::CuArray{T};
  alpha=1.0, beta=0.0, mode=CUDNN_CROSS_CORRELATION)

  h = gethandle(x.dev)
  xdesc = TensorDesc(x)
  wdesc = FilterDesc(w)
  convdesc = ConvolutionDesc(T, pad, stride)
  ydesc = TensorDesc(y)

  algo_p = cudnnConvolutionFwdAlgo_t[0]
  cudnnGetConvolutionForwardAlgorithm(h, xdesc, wdesc, convdesc, ydesc,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algo_p)
  algo = algo_p[1]

  worksize_p = Cint[0]
  cudnnGetConvolutionForwardWorkspaceSize(h, xdesc, wdesc, convdesc, ydesc, algo, worksize_p)
  worksize = worksize_p[1]
  workspace = CuArray(Int8, Int(worksize))

  cudnnConvolutionForward(h, T[alpha], xdesc, x, wdesc, w, convdesc, algo, workspace, worksize, T[beta], ydesc, y)
  y
end

function ∇convolution!{T}(x::CuArray{T})

end
