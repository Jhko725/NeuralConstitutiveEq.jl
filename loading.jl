abstract type AbstractLoading end

struct Triangular <: AbstractLoading
    v::Float32
    t_max::Float32
end