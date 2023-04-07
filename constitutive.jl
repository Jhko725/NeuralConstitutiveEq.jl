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
