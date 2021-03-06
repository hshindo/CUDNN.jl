export convolution, convolution!, ∇convolution!
export CUDNN_CONVOLUTION, CUDNN_CROSS_CORRELATION

#=
type ConvolutionDesc{N}
  ptr::Ptr{Void}
  padding::NTuple{N,Int}
  stride::NTuple{N,Int}
end

function ConvolutionDesc{T,N}(::Type{T}, padding::NTuple{N,Int}, stride::NTuple{N,Int};
  mode=CUDNN_CROSS_CORRELATION)
  p = Ptr{Void}[0]
  cudnnCreateConvolutionDescriptor(p)
  cpadding = Cint[padding[i] for i=N:-1:1]
  cstride = Cint[stride[i] for i=N:-1:1]
  cupscale = fill(Cint(1), N)
  cudnnSetConvolutionNdDescriptor(p[1], N, cpadding, cstride, cupscale, mode, datatype(T))
  desc = ConvolutionDesc(p[1], padding, stride)
  finalizer(desc, cudnnDestroyConvolutionDescriptor)
  desc
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::ConvolutionDesc) = desc.ptr
=#

function convolution_desc(padding, stride)
  N = length(padding)
  p = Ptr{Void}[0]
  cudnnCreateConvolutionDescriptor(p)
  cpadding = Cint[padding[i] for i=N:-1:1]
  cstride = Cint[stride[i] for i=N:-1:1]
  cupscale = fill(Cint(1), N)
  cudnnSetConvolutionNdDescriptor(p[1], N, cpadding, cstride, cupscale, mode, datatype(T))
  p[1]
end

function convolution!{T}(x::CuArray{T}, w::CuArray{T}, padding, stride, y::CuArray{T};
  mode=CUDNN_CROSS_CORRELATION, alpha=1.0, beta=0.0)

  h = gethandle(x.dev)
  xdesc = tensor_desc(x)
  wdesc = filter_desc(w)
  convdesc = convolution_desc(padding, stride)
  ydesc = tensor_desc(y)

  algo_p = cudnnConvolutionFwdAlgo_t[0]
  cudnnGetConvolutionForwardAlgorithm(h, xdesc, wdesc, convdesc, ydesc,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algo_p)
  algo = algo_p[1]

  worksize_p = Cint[0]
  cudnnGetConvolutionForwardWorkspaceSize(h, xdesc, wdesc, desc, ydesc, algo, worksize_p)
  worksize = worksize_p[1]
  workspace = CuArray(Int8, Int(worksize))

  cudnnConvolutionForward(h, T[alpha], xdesc, x, wdesc, w, convdesc,
    algo, workspace, worksize, T[beta], ydesc, y)

  cudnnDestroyTensorDescriptor(xdesc)
  cudnnDestroyFilterDescriptor(wdesc)
  cudnnDestroyConvolutionDescriptor(convdesc)
  cudnnDestroyTensorDescriptor(ydesc)
  y
end

function convolution(x::CuArray, w::CuArray, padding, stride;
  mode=CUDNN_CROSS_CORRELATION, alpha=1.0)

  N = length(padding)
  outdims = Array(Int, N)
  for i = 1:N
    outdims[i] = (size(x,i) + 2*padding[i] - size(w,i)) ÷ stride[i] + 1
  end
  y = similar(x, outdims..., size(w,N+2), size(x,N+2))
  convolution!(x, w, padding, stride, y, mode=mode, alpha=alpha)
end

function ∇convolution_bias!{T}(dy::CuArray{T}, db::CuArray{T}; alpha=1.0, beta=0.0)
  h = gethandle(x.dev)
  dydesc = tensor_desc(dy)
  dbdesc = tensor_desc(db)
  cudnnConvolutionBackwardBias(h, T[alpha], dydesc, dy, T[beta], dbdesc, db)

  cudnnDestroyTensorDescriptor(dydesc)
  cudnnDestroyTensorDescriptor(dbdesc)
end

function ∇convolution_filter!{T}(x::CuArray{T}, dy::CuArray{T}, convdesc, dw::CuArray{T};
  alpha=1.0, beta=0.0)
  h = gethandle(x.dev)
  xdesc = tensor_desc(x)
  dydesc = tensor_desc(dy)
  dwdesc = filter_desc(dw)

  algo_p = cudnnConvolutionBwdAlgo_t[0]
  cudnnFindConvolutionBackwardFilterAlgorithm(h, xdesc, dydesc, convdesc, dwdesc,
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, algo_p)
  algo = algo_p[1]

  worksize_p = Cint[0]
  cudnnGetConvolutionBackwardFilterWorkspaceSize(h, xdesc, dydesc, convdesc, dwdesc, algo, worksize_p)
  worksize = worksize_p[1]
  workspace = CuArray(Int8, Int(worksize))

  cudnnConvolutionBackwardFilter(h, T[alpha], xdesc, x, dydesc, dy, convdesc,
    algo, workspace, worksize, T[beta], dwdesc, dw)


end

function ∇convolution_data!{T}(w::CuArray{T}, dy::CuArray{T}, convdesc, dx::CuArray{T};
  alpha=1.0, beta=0.0)
  wdesc = FilterDesc(w)
  dydesc = TensorDesc(dy)
  dxdesc = TensorDesc(dx)

  algo_p = cudnnConvolutionBwdDataAlgo_t[0]
  cudnnGetConvolutionBackwardDataAlgorithm(h, wdesc, dydesc, convdesc, dxdesc,
    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, algo_p)
  algo = algo_p[1]

  worksize_p = Cint[0]
  cudnnGetConvolutionBackwardDataWorkspaceSize(h, wdesc, dydesc, convdesc,
    dxdesc, algo, worksize_p)
  worksize = worksize_p[1]
  workspace = CuArray(Int8, Int(worksize))

  cudnnConvolutionBackwardData(h, T[alpha], wdesc, w, dydesc, dy, convdesc,
    algo, workspace, T[beta], dxdesc, dx)
end
