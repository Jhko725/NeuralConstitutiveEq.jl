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

struct SLS{T<:AbstractFloat} <: ConstitutiveEqn
    E₁::T
    E₂::T
    η::T
end

struct KWW <: ConstitutiveEqn
    τ::Float32
    β::Float32
end

(relaxation_time(constit::KelvinVoigt{T})::T) where {T} = constit.η / constit.E

function stress_relaxation(t, constit::PowerLawRheology{T}) where {T}
    return constit.E₀ * (t / constit.t₀)^(-constit.γ)
end
