import SpecialFunctions
include("constitutive.jl")
include("indentation.jl")
include("tipgeometry.jl")

function force(t, constit::ConstitutiveEqn, indent::Indentation, tip::TipGeometry)
    t_max = max_indent_time(indent)
    F(t) = _force(t, constit, indent, tip)
    return t <= t_max ? F(t) : F(t₁(t, constit, indent))
end

function force_kv(t, constit::ConstitutiveEqn, indent::Indentation, tip::TipGeometry)
    t_max = max_indent_time(indent)
    F1(t) = kv_force_app(t, constit, indent, tip)
    F2(t) = kv_force_ret(t, constit, indent, tip)
    return t <= t_max ? F1(t) : F2(t₁(t, constit, indent))
end

function t₁(t, constit::PowerLawRheology{T}, indent::Triangular{T}) where {T}
    γ, t_max = constit.γ, max_indent_time(indent)
    coeff = T(2.0^(1.0 / (1.0 - γ)))
    return clamp(t - coeff * (t - t_max), zero(T), Inf)
end

function t₁(t, constit::KelvinVoigt{T}, indent::Triangular{T}) where {T}
    τ = relaxation_time(constit)
    v, t_max = indent.v, max_indent_time(indent)
    return v * clamp(2t_max - t - τ, zero(T), Inf)
end

function _force(t, constit::PowerLawRheology{T}, indent::Triangular{T}, tip::TipGeometry) where {T}
    E₀, t₀, γ = constit.E₀, constit.t₀, constit.γ
    v = indent.v
    a, b = α(tip), β(tip)
    coeff = E₀ * t₀^γ * a * b * v^b * SpecialFunctions.beta(b, 1 - γ)
    return coeff * t^(b - γ)
end

function kv_force_app(t, constit::KelvinVoigt{T}, indent::Triangular{T}, tip::TipGeometry) where {T}
    E, η = constit.E, constit.η
    v = indent.v
    a, b = α(tip), β(tip)
    coeff = a*v^b
    return coeff*t*(6*η+E*t)
end

function kv_force_ret(t, constit::KelvinVoigt{T}, indent::Triangular{T}, tip::TipGeometry) where {T}
    E = constit.E
    v = indent.v
    a, b = α(tip), β(tip)
    coeff = a*v^b
    return coeff*E*t^b
end
    