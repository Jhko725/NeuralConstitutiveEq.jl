##
import SpecialFunctions
import FFTW
using Plots
include("./constitutive.jl")
include("./loading.jl")
include("./tipgeometry.jl")

function analytic_t₁(constit_eqn::PowerLawRheology{T}, loading::Triangular{T})::Function where {T<:AbstractFloat}
    γ = constit_eqn.γ
    t_max = loading.t_max
    coeff = T(2.0^(1.0 / (1.0 - γ)))

    t₁(t::T)::T = clamp(t - coeff * (t - t_max), zero(T), Inf)
    return t₁
end

function analytic_force(constit_eqn::PowerLawRheology{T}, loading::Triangular{T}, tip::AbstractTipGeometry)::Function where {T<:AbstractFloat}
    E₀, t₀, γ = constit_eqn.E₀, constit_eqn.t₀, constit_eqn.γ
    v, t_max = loading.v, loading.t_max
    a, b = α(tip), β(tip)
    coeff = E₀ * t₀^γ * a * b * v^b * SpecialFunctions.beta(b, 1 - γ)
    t₁ = analytic_t₁(constit_eqn, loading)

    F(t::T)::T = t <= t_max ? coeff * t^(b - γ) : coeff * t₁(t)^(b - γ)
    return F
end

function stress_relaxation_modulus(constit_eqn::PowerLawRheology{T}) where {T<:AbstractFloat}
    E₀, t₀, γ = constit_eqn.E₀, constit_eqn.t₀, constit_eqn.γ
    ϕ(t::T)::T = E₀ * (t / t₀)^(-γ)
    return ϕ
end

function memory_function_Z(constit_eqn::PowerLawRheology{T}, Δt::T) where {T<:AbstractFloat}
    E₀, t₀, γ = constit_eqn.E₀, constit_eqn.t₀, constit_eqn.γ
    Q(z::Complex{T})::Complex{T} = E₀ * (t₀ / Δt)^γ * SpecialFunctions.gamma(one(T) - γ) * (one(T) - one(T) / z)^γ
    return Q
end
##
function memory_function_Z2(constit_eqn::PowerLawRheology{T}, Δt::T) where {T<:AbstractFloat}
    E₀, t₀, γ = constit_eqn.E₀, constit_eqn.t₀, constit_eqn.γ
    coeff = E₀ * (2t₀ / Δt)^γ * SpecialFunctions.gamma(one(T) - γ)
    Q(z::Complex{T})::Complex{T} = coeff * ((z - one(T)) / (z + one(T)))^γ
    return Q
end
##
function memory_function_Z3(constit_eqn::PowerLawRheology{T}, Δt::T) where {T<:AbstractFloat}
    E₀, t₀, γ = constit_eqn.E₀, constit_eqn.t₀, constit_eqn.γ
    coeff = E₀ * (t₀ / Δt)^γ * SpecialFunctions.gamma(one(T) - γ)
    Q(z::Complex{T})::Complex{T} = coeff * (log(z))^γ
    return Q
end
##
loading = Triangular(10.0, 0.2) # 10um/s, 0.2
plr = PowerLawRheology(0.572, 1.0, 0.42) # 572Pa -> 0.572
tip = Conical(π / 10.0)
t₁ = analytic_t₁(plr, loading)
F = analytic_force(plr, loading, tip)
##
let
    t_s = 2e-3
    t_data = 0.0:t_s:2*loading.t_max
    f_data = F.(t_data)
    plot(t_data, f_data)
end
##
let
    t_s = 2e-3
    t_data = 0.0:t_s:loading.t_max
    h_data = loading.v * t_data
    f_data = F.(t_data)
    plot(h_data, f_data)
end
##
function weight_powerlaw_decay(X::AbstractVector{T}, r::T) where {T<:AbstractFloat}
    exponent = zero(T):(length(X)-one(T))
    return @. X * (r^-exponent)
end

function MDFT(X::AbstractVector{T}, r::T) where {T<:AbstractFloat}
    X_weighted = weight_powerlaw_decay(X, r)
    X_fft = FFTW.fft(X_weighted)
    return X_fft
end
##
let
    r = 1.005
    t_s = 2e-3
    t_data = 0.0:t_s:loading.t_max
    h_data = loading.v * t_data
    f_data = F.(t_data)
    f_mdft = MDFT(f_data, r) |> FFTW.fftshift
    freqs = FFTW.fftfreq(length(t_data)) |> FFTW.fftshift

    plot_real = plot(freqs, real.(f_mdft))
    plot_imag = plot(freqs, imag.(f_mdft))
    plot(plot_real, plot_imag, layout=(1, 2))
    #freqs
end
##
##
let
    r = 1.01
    t_s = 2e-3
    t_data = 0.0:t_s:loading.t_max
    h_data = loading.v * t_data
    f_data = F.(t_data)
    f_mdft = MDFT(f_data, r) |> FFTW.fftshift
    h_mdft = MDFT(h_data .^ β(tip), r) |> FFTW.fftshift
    freqs = FFTW.fftfreq(length(t_data)) |> FFTW.fftshift
    ratio = f_mdft ./ (α(tip) * h_mdft)
    plot_real = plot(freqs, real.(ratio))
    plot_imag = plot(freqs, imag.(ratio))
    plot(plot_real, plot_imag, layout=(1, 2))
end
##
let
    r = 1.5
    t_s = 2e-4
    t_data = 0.0:t_s:loading.t_max
    h_data = loading.v * t_data
    f_data = F.(t_data)
    f_mdft = MDFT(f_data, r)
    print(typeof(f_mdft))
    freqs = FFTW.fftfreq(length(t_data)) |> FFTW.fftshift
    z = @. r * exp(im * freqs * 2π)
    print(typeof(z))
    FFTW.fftshift(f_mdft)
    freqs
end
##
let
    r = 1.05
    t_s = 2e-4
    t_data = 0.0:t_s:loading.t_max
    h_data = loading.v * t_data
    f_data = F.(t_data)
    f_mdft = MDFT(f_data, r) |> FFTW.fftshift
    h_mdft = MDFT(h_data .^ β(tip), r) |> FFTW.fftshift
    freqs = FFTW.fftfreq(length(t_data)) |> FFTW.fftshift
    ratio = f_mdft ./ (α(tip) * h_mdft)

    plot_real = plot(freqs, real.(ratio))
    plot_imag = plot(freqs, imag.(ratio))

    Q = memory_function_Z(plr, t_s)
    z = @. r * exp(im * freqs * 2π)
    q_true = Q.(z)
    plot!(plot_real, freqs, real.(q_true))
    plot!(plot_imag, freqs, imag.(q_true))


    Q = memory_function_Z2(plr, t_s)
    z = @. r * exp(im * freqs * 2π)
    q_true = Q.(z)
    plot!(plot_real, freqs, real.(q_true))
    plot!(plot_imag, freqs, imag.(q_true))

    Q = memory_function_Z3(plr, t_s)
    z = @. r * exp(im * freqs * 2π)
    q_true = Q.(z)
    plot!(plot_real, freqs, real.(q_true))
    plot!(plot_imag, freqs, imag.(q_true))

    plot(plot_real, plot_imag, layout=(1, 2))
    #plot(imag.(ratio))
end
##
let

end
##

