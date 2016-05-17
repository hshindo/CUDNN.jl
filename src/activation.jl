export ActivationDesc
export activation!, ∇activation!
export CUDNN_ACTIVATION_SIGMOID, CUDNN_ACTIVATION_RELU, CUDNN_ACTIVATION_TANH, CUDNN_ACTIVATION_CLIPPED_RELU
export CUDNN_NOT_PROPAGATE_NAN, CUDNN_PROPAGATE_NAN

type ActivationDesc
  ptr::Ptr{Void}

  function ActivationDesc(mode; relu_nanopt=CUDNN_NOT_PROPAGATE_NAN, relu_ceiling=1.0)
    p = Ptr{Void}[0]
    cudnnCreateActivationDescriptor(p)
    ad = new(p[1])
    finalizer(ad, cudnnDestroyActivationDescriptor)
    cudnnSetActivationDescriptor(ad, mode, relu_nanopt, relu_ceiling)
    ad
  end
end

Base.unsafe_convert(::Type{Ptr{Void}}, ad::ActivationDesc) = ad.ptr

"""
reluNanOpt: whether propagates NaN or not
reluCeiling: floating point number to specify the clipping threashod
when the activation mode is set to CUDNN_ACTIVATION_CLIPPED_RELU.
"""
function activation!{T}(desc, x::CuArray{T}, y::CuArray{T}; alpha=1.0, beta=0.0)
  h = gethandle(x.dev)
  xdesc = TensorDesc(x)
  ydesc = TensorDesc(y)
  cudnnActivationForward(h, desc, T[alpha], xdesc, x, T[beta], ydesc, y)
  y
end

function ∇activation!{T}(desc, y::CuArray{T}, dy::CuArray{T}, x::CuArray{T}, dx::CuArray{T};
  alpha=1.0, beta=0.0)

  h = gethandle(x.dev)
  ydesc = TensorDesc(y)
  dydesc = TensorDesc(y)
  xdesc = TensorDesc(x)
  dxdesc = TensorDesc(dx)
  cudnnActivationBackward(h, desc, T[alpha], ydesc, y, dydesc, dy, xdesc, x, T[beta], dxdesc, dx)
  dx
end
