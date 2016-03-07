export activation_forward!, activation_backward!
export CUDNN_ACTIVATION_SIGMOID, CUDNN_ACTIVATION_RELU, CUDNN_ACTIVATION_TANH, CUDNN_ACTIVATION_CLIPPED_RELU
export CUDNN_NOT_PROPAGATE_NAN, CUDNN_PROPAGATE_NAN

#type ActivationDescriptor
#  ptr
#end

#Base.unsafe_convert(::Type{cudnnActivationDescriptor_t}, ad::ActivationDescriptor) = ad.ptr

# mode: sigmoid | relu | tanh | clipped_relu
# reluNanOpt: whether propagates NaN or not
# reluCeiling: floating point number to specify the clipping threashod when the activation mode is set to CUDNN_ACTIVATION_CLIPPED_RELU.
function activation_forward!{T}(mode, alpha::Float64, x::AbstractCudaArray{T}, beta::Float64, y::AbstractCudaArray{T};
                             relu_nanopt=CUDNN_NOT_PROPAGATE_NAN, relu_ceiling=1.0)
  handle = gethandle(x.dev)

  p = cudnnActivationDescriptor_t[0]
  cudnnCreateActivationDescriptor(p)
  adesc = p[1]
  cudnnSetActivationDescriptor(adesc, mode, relu_nanopt, relu_ceiling)
  xdesc = create_tensor_descriptor(x)
  ydesc = create_tensor_descriptor(y)

  cudnnActivationForward_v4(handle, adesc, T[alpha], xdesc, x, T[beta], ydesc, y)

  cudnnDestroyActivationDescriptor(adesc)
  cudnnDestroyTensorDescriptor(xdesc)
  cudnnDestroyTensorDescriptor(ydesc)
  nothing
end

#function activation_backward!{T}(ad::ActivationDescriptor, alpha::Float64, y::AbstractCudaArray{T}, dy, x::AbstractCudaArray{T}, beta::Float64, dx)
#  handle = gethandle(x.dev)
#  cudnnActivationBackward_v4(handle, ad, T[alpha], TD(y), y, TD(dy), dy, TD(x), x, T[beta], TD(dx), dx)
#  dx
#end
