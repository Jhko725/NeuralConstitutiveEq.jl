abstract type AbstractLoading end

struct Triangular <: AbstractLoading
    v::Float64
    t_max::Float64
end