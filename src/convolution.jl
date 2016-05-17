export conv, ∇conv
export CUDNN_CONVOLUTION, CUDNN_CROSS_CORRELATION

type ConvDesc
  ptr::Ptr{Void}

  function ConvDesv{T,N}(::Type{T}, pads::NTuple{N,Int}, strides::NTuple{N,Int};
    mode=CUDNN_CONVOLUTION)

    p = Ptr{Void}[0]
    cudnnCreateConvolutionDescriptor(p)
    pads = Cint[pads[i] for i=N:-1:1]
    strides = Cint[strides[i] for i=N:-1:1]
    upscales = fill(1, N)
    cudnnSetConvolutionNdDescriptor(p[1], N, pads, strides, upscales, mode, datatype(T))
    cd = new(p[1])
    finalizer(cd, cudnnDestroyConvolutionDescriptor)
    cd
  end
end

Base.unsafe_convert(::Type{Ptr{Void}}, cd::ConvDesc) = cd.ptr

function conv{T,N}(x::CuArray{T,N}, w::CuArray{T}, convdesc::ConvDesc, ysize;
  alpha=1.0)

  h = gethandle(x.dev)
  xdesc = TensorDesc(x)
  wdesc = FilterDesc(w)
  y = CuArray(T, ysize)
  ydesc = TensorDesc(y)

  algo_p = cudnnConvolutionFwdAlgo_t[0]
  cudnnGetConvolutionForwardAlgorithm(h, xdesc, wdesc, convdesc, ydesc,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algo_p)
  algo = algo_p[1]

  worksize_p = Csize_t[0]
  cudnnGetConvolutionForwardWorkspaceSize(h, xdesc, wdesc, convdesc, ydesc, algo, size_p)
  worksize = Int(worksize_p[1])

  cudnnConvolutionForward(h, T[alpha], xdesc, x, wdesc, w, convdesc, algo, worksize, T[0], ydesc, y)
  y
end

function ∇conv{T}(x::CuArray{T})

end
