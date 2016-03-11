module CUDNN

using CUDArt

include("types.jl")
include("libcudnn.jl")

@windows? (
  begin
    const libcudnn = Libdl.find_library(["cudnn64_4"])
  end : begin
    const libcudnn = Libdl.find_library(["libcudnn"])
  end)
isempty(libcudnn) && throw("CUDNN library cannot be found.")

function checkstatus(status)
  if status != CUDNN_STATUS_SUCCESS
      Base.show_backtrace(STDOUT, backtrace())
      println()
      throw(cudnnGetErrorString(status))
  end
end

datatype(a::AbstractCudaArray{Float32}) = CUDNN_DATA_FLOAT
datatype(a::AbstractCudaArray{Float64}) = CUDNN_DATA_DOUBLE
datatype(a::AbstractCudaArray{Float16}) = CUDNN_DATA_HALF

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

function create_tensor_descriptor(a::AbstractCudaArray)
  csize = Cint[size(a,i) for i=ndims(a):-1:1]
  cstrides = Cint[stride(a,i) for i=ndims(a):-1:1]
  p = cudnnTensorDescriptor_t[0]
  cudnnCreateTensorDescriptor(p)
  desc = p[1]
  cudnnSetTensorNdDescriptor(desc, datatype(a), ndims(a), csize, cstrides)
  desc
end

include("filter.jl")
include("convolution.jl")
include("pooling.jl")

########## Misc ##########

""" y = alpha * x + beta * y """
function add{T}(x::AbstractCudaArray{T}, y::AbstractCudaArray{T}; alpha=1.0, beta=0.0)
  handle = gethandle(bias.dev)
  xdesc = descriptor(x)
  ydesc = descriptor(y)
end

end
