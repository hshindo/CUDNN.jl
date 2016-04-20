workspace()
using CUDArt
using CUDNN

x = randn(10,5,1,1) |> CudaArray
to_host(x)
activation_forward!(CUDNN_ACTIVATION_RELU, 1.0, x, 0.0, x)
to_host(x)

function bench()
  x = rand(Float32,100,100,10,10) |> CudaArray
  for i = 1:1000
    td = CUDNN.TensorDescriptor(x)
  end
end

@time bench()
