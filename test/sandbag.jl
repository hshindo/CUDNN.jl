workspace()
using CUDA
using CUDNN

x = randn(10,5,1,1) |> CuArray
y = randn(10,5,1,1) |> CuArray
y = softmax(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, x)
yy = Array(y)


function bench()
  x = rand(Float32,100,100,10,10) |> CudaArray
  for i = 1:1000
    td = CUDNN.TensorDescriptor(x)
  end
end

@time bench()
