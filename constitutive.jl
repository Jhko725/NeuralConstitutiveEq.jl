abstract type AbstractConstitutiveEqn end

struct PowerLawRheology{T<:AbstractFloat} <: AbstractConstitutiveEqn
    E₀::T
    t₀::T
    γ::T
end

struct KWW <: AbstractConstitutiveEqn
    τ::Float32
    β::Float32
end

struct KV{T<:AbstractFloat} <: AbstractConstitutiveEqn
    E::T
    η::T
end
