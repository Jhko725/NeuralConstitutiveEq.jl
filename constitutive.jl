abstract type AbstractConstitutiveEqn end

struct PowerLawRheology <: AbstractConstitutiveEqn
    E₀::Float32
    t₀::Float32
    γ::Float32
end
