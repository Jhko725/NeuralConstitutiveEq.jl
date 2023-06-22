abstract type Indentation end

struct Triangular{T<:AbstractFloat} <: Indentation
    v::T
    t_max::T
end

(max_indent_time(indent::Triangular{T})::T) where {T} = indent.t_max

function sample(loading::Triangular{T}, fₛ::T) where {T<:AbstractFloat}
    tₛ = one(T) / fₛ
    t = zero(T):tₛ:2*loading.t_max

end