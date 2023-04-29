abstract type TipGeometry end

struct Spherical{T<:AbstractFloat} <: TipGeometry
    R::T
end

struct Conical{T<:AbstractFloat} <: TipGeometry
    θ::T
end

function α(tip::Spherical{T})::T where {T<:AbstractFloat}
    return T(16 / 9) * sqrt(tip.R)
end

function α(tip::Conical{T})::T where {T<:AbstractFloat}
    return T(8.0 / (3.0π)) * tan(tip.θ)
end

function β(tip::Spherical{T})::T where {T<:AbstractFloat}
    return T(3 / 2)
end

function β(tip::Conical{T})::T where {T<:AbstractFloat}
    return T(2.0)
end