export ActivationDescriptor
export activation_forward!, activation_backward!
export CUDNN_ACTIVATION_SIGMOID, CUDNN_ACTIVATION_RELU, CUDNN_ACTIVATION_TANH, CUDNN_ACTIVATION_CLIPPED_RELU
export CUDNN_NOT_PROPAGATE_NAN, CUDNN_PROPAGATE_NAN

type ActivationDescriptor
  ptr
end

function ActivationDescriptor()
  p = cudnnActivationDescriptor_t[0]
  cudnnCreateActivationDescriptor(p)
  ad = ActivationDescriptor(p[1])
  finalizer(ad, cudnnDestroyActivationDescriptor)
  ad
end

Base.unsafe_convert(::Type{cudnnActivationDescriptor_t}, ad::ActivationDescriptor) = ad.ptr

"""
mode: sigmoid | relu | tanh | clipped_relu
reluNanOpt: whether propagates NaN or not
reluCeiling: floating point number to specify the clipping threashod
when the activation mode is set to CUDNN_ACTIVATION_CLIPPED_RELU.
"""
function activation_forward!{T}(mode, alpha, x::CudaArray{T}, beta, y::CudaArray{T};
  relu_nanopt=CUDNN_NOT_PROPAGATE_NAN, relu_ceiling=1.0)
  
  handle = gethandle(x.dev)
  ad = ActivationDescriptor()
  cudnnSetActivationDescriptor(ad, mode, relu_nanopt, relu_ceiling)
  xdesc = TensorDescriptor(x)
  ydesc = TensorDescriptor(y)
  cudnnActivationForward_v4(handle, ad, T[alpha], xdesc, x, T[beta], ydesc, y)
  ad
end

function activation_backward!{T}(ad, alpha, y::CudaArray{T}, dy::CudaArray{T}, x::CudaArray{T}, beta, dx::CudaArray{T})
  handle = gethandle(x.dev)
  xdesc = TensorDescriptor(x)
  dxdesc = TensorDescriptor(dx)
  ydesc = TensorDescriptor(y)
  dydesc = TensorDescriptor(y)
  cudnnActivationBackward_v4(handle, ad, T[alpha], ydesc, y, dydesc, dy, xdesc, x, T[beta], dxdesc, dx)
  dx
end
