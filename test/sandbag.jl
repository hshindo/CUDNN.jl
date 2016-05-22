workspace()
using CUDA
using CUDNN

x = rand(Float32,5,4,3,2) |> CuArray
Array(x)
w = CuArray(rand(Float32,2,2,3,4))
Array(w)
desc = ConvolutionDesc(Float32, (0,0), (1,1))
y = CUDNN.convolution(x, w, desc)
Array(y)

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
