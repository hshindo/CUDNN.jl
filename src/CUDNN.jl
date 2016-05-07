module CUDNN

using CUDA

include("libcudnn_h.jl")
include("libcudnn.jl")

@windows? (
  begin
    const libcudnn = Libdl.find_library(["cudnn64_5"])
  end : begin
    const libcudnn = Libdl.find_library(["libcudnn"])
  end)
isempty(libcudnn) && throw("CUDNN library cannot be found.")

function checkstatus(status)
  if status != CUDNN_STATUS_SUCCESS
      Base.show_backtrace(STDOUT, backtrace())
      throw(cudnnGetErrorString(status))
  end
end

datatype(a::CudaArray{Float32}) = CUDNN_DATA_FLOAT
datatype(a::CudaArray{Float64}) = CUDNN_DATA_DOUBLE
datatype(a::CudaArray{Float16}) = CUDNN_DATA_HALF

########## Handle ##########

const handles = Dict{Int, cudnnHandle_t}()
atexit(() -> for h in handles destroy(h) end)

function gethandle(dev::Int)
  if !haskey(handles, dev)
    h = cudnnHandle_t[0]
    cudnnCreate(h)
    handles[dev] = h[1]
    h[1]
  else
    handles[dev]
  end
end

include("tensor.jl")
include("activation.jl")
include("convolution.jl")
include("filter.jl")
include("pooling.jl")

########## Misc ##########

""" y = alpha * x + beta * y """
function add{T}(x::CudaArray{T}, y::CudaArray{T}; alpha=1.0, beta=0.0)
  handle = gethandle(bias.dev)
  xdesc = descriptor(x)
  ydesc = descriptor(y)
end

end
