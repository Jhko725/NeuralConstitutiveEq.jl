abstract type ConstitutiveEqn end

struct PowerLawRheology{T<:AbstractFloat} <: ConstitutiveEqn
    E₀::T
    t₀::T
    γ::T
end

struct KelvinVoigt{T<:AbstractFloat} <: ConstitutiveEqn
    E::T
    η::T
end

struct Maxwell{T<:AbstractFloat} <: ConstitutiveEqn
    E::T
    η::T
end

struct StandardLinearSolid{T<:AbstractFloat} <: ConstitutiveEqn
    E₀::T
    E₁::T
    η₁::T
end

struct KWW <: ConstitutiveEqn
    τ::Float32
    β::Float32
end

(relaxation_time(constit::KelvinVoigt{T})::T) where {T} = constit.η / constit.E