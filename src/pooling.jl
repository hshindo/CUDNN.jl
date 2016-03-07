type PD
  ptr
end

function PD()
end

Base.unsafe_convert(::Type{cudnnPoolingDescriptor_t}, pd::PD) = pd.ptr
