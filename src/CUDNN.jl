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
    throw(bytestring(cudnnGetErrorString(status)))
  end
end

datatype(::Type{Float32}) = CUDNN_DATA_FLOAT
datatype(::Type{Float64}) = CUDNN_DATA_DOUBLE
datatype(::Type{Float16}) = CUDNN_DATA_HALF

########## Handle ##########

const handles = Dict{Int,cudnnHandle_t}()
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
include("softmax.jl")

end
