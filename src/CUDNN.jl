module CUDNN

using CUDA

include("libcudnn_h.jl")
include("libcudnn.jl")

version() = Int(cudnnGetVersion())

@windows? (
  begin
    const libcudnn = Libdl.find_library(["cudnn64_5"])
  end : begin
    const libcudnn = Libdl.find_library(["libcudnn"])
  end)
if isempty(libcudnn)
  throw("CUDNN library cannot be found.")
else
  println("CUDNN version $(version()) is loaded.")
end

function checkstatus(status)
  if status != CUDNN_STATUS_SUCCESS
    Base.show_backtrace(STDOUT, backtrace())
    throw(bytestring(cudnnGetErrorString(status)))
  end
end

datatype(::Type{Float32}) = CUDNN_DATA_FLOAT
datatype(::Type{Float64}) = CUDNN_DATA_DOUBLE
datatype(::Type{Float16}) = CUDNN_DATA_HALF

include("handle.jl")
include("tensor.jl")
include("activation.jl")
include("convolution.jl")
include("filter.jl")
include("softmax.jl")

end
