abstract type AbstractLoading end

struct Triangular{T<:AbstractFloat} <: AbstractLoading
    v::T
    t_max::T
end

