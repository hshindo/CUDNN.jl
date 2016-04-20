export AD
export activation_forward!, activation_backward!
export CUDNN_ACTIVATION_SIGMOID, CUDNN_ACTIVATION_RELU, CUDNN_ACTIVATION_TANH, CUDNN_ACTIVATION_CLIPPED_RELU
export CUDNN_NOT_PROPAGATE_NAN, CUDNN_PROPAGATE_NAN

# Activation Descriptor
type AD
  ptr
end

function AD()
  p = cudnnActivationDescriptor_t[0]
  cudnnCreateActivationDescriptor(p)
  ad = AD(p[1])
  finalizer(ad, cudnnDestroyActivationDescriptor)
  ad
end

Base.unsafe_convert(::Type{cudnnActivationDescriptor_t}, ad::AD) = ad.ptr

"""
mode: sigmoid | relu | tanh | clipped_relu
reluNanOpt: whether propagates NaN or not
reluCeiling: floating point number to specify the clipping threashod
when the activation mode is set to CUDNN_ACTIVATION_CLIPPED_RELU.
"""
function activation_forward!{T}(mode, alpha, x::CudaArray{T}, beta, y::CudaArray{T};
                                relu_nanopt=CUDNN_NOT_PROPAGATE_NAN, relu_ceiling=1.0)
  handle = gethandle(x.dev)
  ad = AD()
  cudnnSetActivationDescriptor(ad, mode, relu_nanopt, relu_ceiling)
  cudnnActivationForward_v4(handle, ad, T[alpha], TD(x), x, T[beta], TD(y), y)
  ad
end

function activation_backward!{T}(ad::AD, alpha, y::CudaArray{T}, dy::CudaArray{T}, x::CudaArray{T}, beta, dx::CudaArray{T})
  handle = gethandle(x.dev)
  cudnnActivationBackward_v4(handle, ad, T[alpha], TD(y), y, TD(dy), dy, TD(x), x, T[beta], TD(dx), dx)
  dx
end
