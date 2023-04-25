abstract type AbstractLoading end

struct Triangular{T<:AbstractFloat} <: AbstractLoading
    v::T
    t_max::T
end

function sample(loading::Triangular{T}, fₛ::T) where {T<:AbstractFloat}
    tₛ = one(T) / fₛ
    t = zero(T):tₛ:2*loading.t_max

end