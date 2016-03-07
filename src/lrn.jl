type LRNDescriptor
  ptr
end

Base.unsafe_convert(::Type{cudnnLRNDescriptor_t}, ld::LRNDescriptor) = ld.ptr
