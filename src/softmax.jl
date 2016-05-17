export softmax!, ∇softmax!
export CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_LOG # algorithm
export CUDNN_SOFTMAX_MODE_INSTANCE, CUDNN_SOFTMAX_MODE_CHANNEL # mode

function softmax!{T}(algo, mode, x::CuArray{T}, y::CuArray{T};
  alpha=1.0, beta=0.0)

  h = gethandle(x.dev)
  xdesc = TensorDesc(x)
  ydesc = TensorDesc(y)
  cudnnSoftmaxForward(h, algo, mode, T[alpha], xdesc, x, T[beta], ydesc, y)
  y
end

function ∇softmax!{T}(algo, mode, y::CuArray{T}, dy::CuArray{T}, dx::CuArray{T};
  alpha=1.0, beta=0.0)

  h = gethandle(x.dev)
  ydesc = TensorDesc(y)
  dydesc = TensorDesc(dy)
  dxdesc = TensorDesc(dx)
  cudnnSoftmaxBackward(h, algo, mode, T[alpha], ydesc, y, dydesc, dy, T[beta], dxdesc, dx)
  dx
end
