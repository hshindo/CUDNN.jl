@testset begin
  x = randn(10,5,1,1) |> CudaArray
  activation_forward!(CUDNN_ACTIVATION_RELU, 1.0, x, 0.0, x)
end
