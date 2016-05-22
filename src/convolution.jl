export CUDNN_CONVOLUTION, CUDNN_CROSS_CORRELATION

type ConvolutionDesc{N}
  ptr::Ptr{Void}
  pads::NTuple{N,Int}
  strides::NTuple{N,Int}
end

function ConvolutionDesc{T,N}(::Type{T}, pads::NTuple{N,Int}, strides::NTuple{N,Int};
  mode=CUDNN_CROSS_CORRELATION)

  p = Ptr{Void}[0]
  cudnnCreateConvolutionDescriptor(p)
  cpads = Cint[pads[i] for i=N:-1:1]
  cstrides = Cint[strides[i] for i=N:-1:1]
  cupscales = fill(Cint(1), N)
  cudnnSetConvolutionNdDescriptor(p[1], N, cpads, cstrides, cupscales, mode, datatype(T))
  cd = ConvolutionDesc(p[1], pads, strides)
  finalizer(cd, cudnnDestroyConvolutionDescriptor)
  cd
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::ConvolutionDesc) = desc.ptr

function convolution!{T}(x::CuArray{T}, w::CuArray{T}, pads, strides, y::CuArray{T};
  alpha=1.0, beta=0.0)

  h = gethandle(x.dev)
  xdesc = TensorDesc(x)
  wdesc = FilterDesc(w)
  convdesc = ConvolutionDesc(T, pads, strides)
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

function convolution{N}(x::CuArray, w::CuArray, desc::ConvolutionDesc{N})
  @assert ndims(x) == ndims(w) == N+2
  dims = Int[]
  for i = 1:N
    d = (size(x,i) + 2*desc.pads[i] - size(w,i)) ÷ desc.strides[i] + 1
    push!(dims, d)
  end
  push!(dims, size(w,N+2), size(x,N+2))
  convolution!(x, w, desc, similar(x, dims...))
end

function ∇convolution!{T}(x::CuArray{T})

end
