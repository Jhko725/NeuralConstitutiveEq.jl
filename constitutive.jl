abstract type AbstractConstitutiveEqn end

struct PowerLawRheology{T<:AbstractFloat} <: AbstractConstitutiveEqn
    E₀::T
    t₀::T
    γ::T
end
