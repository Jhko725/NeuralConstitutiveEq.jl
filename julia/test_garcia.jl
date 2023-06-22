##
using Plots
using Integrals
include("./constitutive.jl")
include("./indentation.jl")
include("./tipgeometry.jl")
include("force.jl")

function force_numerical(t, constit::PowerLawRheology, indent::Triangular, tip::TipGeometry)
    ϕ(t) = stress_relaxation(t, constit)
    a, b = α(tip), β(tip)
    v, t_max = indent.v, max_indent_time(indent)
    t_upper = t > t_max ? t₁(t, constit, indent) : t

    if t_upper == zero(t)
        return zero(t)
    else
        f(t_, p) = @. ϕ(p - t_) * t_^(b - 1)
        prob = IntegralProblem(f, zero(t), t_upper, [t])
        sol = solve(prob, QuadGKJL())[1]
    end

    return a * b * v^b * sol
end
##
loading = Triangular(10.0, 0.2) # 10um/s, 0.2
plr = PowerLawRheology(0.572, 1.0, 0.2) # 572Pa -> 0.572
tip = Conical(π / 10.0)
##
let
    t_s = 2e-3
    t_data = 0.0:t_s:2*max_indent_time(loading)
    f_true = map(x -> force_numerical(x, plr, loading, tip), t_data)
    ax = plot(t_data, f_true, label="Numerical", linewidth=2.0, alpha=0.7)

    F(t) = force(t, plr, loading, tip)
    f_garcia = F.(t_data)
    plot!(ax, t_data, f_garcia, label="Garcia", lineiwdth=1.0)

    f_me = map(x -> force2(x, plr, loading, tip), t_data)
    plot!(ax, t_data, f_me, label="Correction", linewidth=1.0)
    xlabel!(ax, "Time")
    ylabel!(ax, "Force")
end


##
