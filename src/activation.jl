export ActivationDesc
export CUDNN_ACTIVATION_SIGMOID, CUDNN_ACTIVATION_RELU, CUDNN_ACTIVATION_TANH, CUDNN_ACTIVATION_CLIPPED_RELU
export CUDNN_NOT_PROPAGATE_NAN, CUDNN_PROPAGATE_NAN

type ActivationDesc
  ptr::Ptr{Void}
end

function ActivationDesc(mode; relu_nanopt=CUDNN_NOT_PROPAGATE_NAN, relu_ceiling=1.0)
  p = Ptr{Void}[0]
  cudnnCreateActivationDescriptor(p)
  desc = new(p[1])
  finalizer(ad, cudnnDestroyActivationDescriptor)
  cudnnSetActivationDescriptor(desc, mode, relu_nanopt, relu_ceiling)
  desc
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::ActivationDesc) = desc.ptr

"""
reluNanOpt: whether propagates NaN or not
reluCeiling: floating point number to specify the clipping threashod
when the activation mode is set to CUDNN_ACTIVATION_CLIPPED_RELU.
"""
function activation!{T}(desc, x::CuArray{T}, y::CuArray{T};
  alpha=1.0, beta=0.0)

  h = gethandle(x.dev)
  xdesc = TensorDesc(x)
  ydesc = TensorDesc(y)
  cudnnActivationForward(h, desc, T[alpha], xdesc, x, T[beta], ydesc, y)
  y
end
activation(desc, x::CuArray; alpha=1.0) = activation!(desc, x, similar(x), alpha=alpha)

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
