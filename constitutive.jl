abstract type AbstractConstitutiveEqn end

struct PowerLawRheology <: AbstractConstitutiveEqn
    E₀::Float32
    t₀::Float32
    γ::Float32
end
<<<<<<< HEAD

struct KWW <: AbstractConstitutiveEqn
    τ::Float32
    β::Float32
end
=======
>>>>>>> 72c38ffd9eb1467068d4effd627cc297be0216a6
