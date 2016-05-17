export TensorDesc

type TensorDesc
  ptr::Ptr{Void}

  function TensorDesc{T}(a::CuArray{T})
    csize = Cint[size(a,i) for i=ndims(a):-1:1]
    cstrides = Cint[stride(a,i) for i=ndims(a):-1:1]
    p = Ptr{Void}[0]
    cudnnCreateTensorDescriptor(p)
    td = new(p[1])
    finalizer(td, cudnnDestroyTensorDescriptor)
    cudnnSetTensorNdDescriptor(td, datatype(T), ndims(a), csize, cstrides)
    td
  end
end

Base.unsafe_convert(::Type{Ptr{Void}}, td::TensorDesc) = td.ptr

# y = alpha*x + beta*y
function add{T}(alpha, x::CuArray{T}, beta, y::CuArray{T})
  h = gethandle(x.dev)
  xdesc, ydesc = TensorDesc(x), TensorDesc(y)
  cudnnAddTensor(h, T[alpha], xdesc, x, T[beta], ydesc, y)
end

# C = op ( alpha1[0] * A, alpha2[0] * B ) + beta[0] * C
function optensor()
end

function set{T}(y::CuArray{T}, value)
  h = gethandle(y.dev)
  ydesc = TensorDesc(y)
  cudnnSetTensor(h, ydesc, y, T[value])
end

# y = alpha * y
function scale{T}(y::CuArray{T}, alpha)
  h = gethandle(y.dev)
  ydesc = TensorDesc(y)
  cudnnScaleTensor(h, ydesc, y, T[alpha])
end
