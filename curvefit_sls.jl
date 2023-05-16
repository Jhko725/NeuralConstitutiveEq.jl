##
using LsqFit
using Plots

include("./constitutive.jl")
include("./indentation.jl")
include("./tipgeometry.jl")
include("force.jl")
##
loading = Triangular(10.0, 0.2) # 10um/s, 0.2
sls = SLS(0.1, 0.01, 0.25)
tip = Conical(π / 10.0)
F_sls(t) = force_sls(t, sls, loading, tip)
##
let
    t_s = 2e-3
    t_data = 0.0:t_s:2*loading.t_max
    f_true = F_sls.(t_data)
    p = plot(t_data, f_true, xlabel = "Time(s)", ylabel="Force(nN)", label="SLS_Analytic")
    noise = randn(Float64, size(f_true)) * 0.04
    scatter!(t_data, f_true+noise, label="SLS_Simulation")
end
##
let
    import Statistics: mean
    t_s = 2e-3
    noise_amp = 0.08
    t_data = 0.0:t_s:2*loading.t_max
    f_true = F_sls.(t_data)
    noise = randn(Float64, size(f_true)) * noise_amp
    f_data = f_true + noise

    function f_fit(t, params)
        _force(t) = force_sls(t, SLS(params[1], params[2], params[3]), loading, tip)
        return _force.(t)
    end

    fit = curve_fit(f_fit, t_data, f_data, [0.1, 0.5, 0.3], lower=[0.0, 0.0, 0.0], upper=[Inf, Inf, Inf])
    print(fit.param)
    print(standard_errors(fit))
    print(fit.resid)
    mse = mean(fit.resid.^2)
    print(mse)
end
##
let
    t_s = 2e-3
    noise_amp = 0.08
    t_data = 0.0:t_s:2*loading.t_max
    f_true = F_sls.(t_data)
    noise = randn(Float64, size(f_true)) * noise_amp
    f_data = f_true + noise

    function f_fit_sls(t, params)
        _force(t) = force_sls(t, SLS(params[1], params[2], params[3]), loading, tip)
        return _force.(t)
    end

    fit = curve_fit(f_fit_sls, t_data, f_data, [0.1, 0.5, 0.3], lower=[0.0, 0.0, 0.0], upper=[Inf, Inf, Inf])
    print(fit.param)
    print(standard_errors(fit))
end
##
function run_experiment_sls(E₁, E₂, η, noise_amp, t_s, app_only=true)
    tip = Conical(π / 10.0)
    loading = Triangular(10.0, 0.2) # 10um/s, 0.2
    sls = SLS(E₁, E₂, η)
    F(t) = force_sls(t, sls, loading, tip)

    if app_only
        t_end = t_s
    else
        t_end = 2t_s
    end
    t_data = 0.0:t_s:2*loading.t_max
    f_true = F.(t_data)
    noise = randn(Float64, size(f_true)) * noise_amp
    f_data = f_true + noise

    function f_fit(t, params)
        _force(t) = force_sls(t, SLS(params[1], params[2], params[3]), loading, tip)
        return _force.(t)
    end

    fit = curve_fit(f_fit, t_data, f_data, [0.1, 0.5, 0.3], lower=[0.0, 0.0, 0.0], upper=[Inf, Inf, Inf])
    param = fit.param
    se = standard_errors(fit)
    return cat(param, se, dims=2)
end

##
let
E₁_true, E₂_true, η_true = 0.1, 0.01, 0.25
    f_s = 2e-3
    noise_amps = 0:0.002:0.04
    result_app = map(x -> run_experiment_sls(E₁_true, E₂_true, η_true, x, f_s, true), noise_amps) |> x -> cat(x..., dims=3)
    result_total = map(x -> run_experiment_sls(E₁_true, E₂_true, η_true, x, f_s, false), noise_amps) |> x -> cat(x..., dims=3)
    plot_E₁ = scatter(noise_amps, result_app[1, 1, :], yerror=result_app[1, 2, :], label="Approach only", ylabel="E₁")
    scatter!(plot_E₁, noise_amps, result_total[1, 1, :], yerror=result_total[1, 2, :], label="Approach & Retract")
    hline!(plot_E₁, [E₁_true], label="Ground truth")

    plot_E₂ = scatter(noise_amps, result_app[1, 1, :], yerror=result_app[1, 2, :], label="Approach only", ylabel="E₂")
    scatter!(plot_E₂, noise_amps, result_total[1, 1, :], yerror=result_total[1, 2, :], label="Approach & Retract")
    hline!(plot_E₂, [E₂_true], label="Ground truth")
    
    plot_η = scatter(noise_amps, result_app[2, 1, :], yerror=result_app[2, 2, :], label="Approach only", ylabel="η")
    scatter!(plot_η, noise_amps, result_total[2, 1, :], yerror=result_total[2, 2, :], label="Approach & Retract")
    hline!(plot_η, [η_true], label="Ground truth")
    plot(plot_E₁, plot_E₂, plot_η, layout=(1, 3), xlabel="Noise amplitude", )
end

##
let
    E₁_true,E₂_true, noise_amp = 0.1, 0.01, 0.02
    f_s = 2e-3
    η_trues = 0.0:0.005:0.1
    result_app = map(x -> run_experiment_sls(E₁_true, E₂_true, x, noise_amp, f_s, true), η_trues) |> x -> cat(x..., dims=3)
    result_total = map(x -> run_experiment_sls(E₁_true, E₂_true, x, noise_amp, f_s, false), η_trues) |> x -> cat(x..., dims=3)
    plot_E₁ = scatter(η_trues, result_app[1, 1, :], yerror=result_app[1, 2, :], label="Approach only", ylabel="E₁_true")
    scatter!(plot_E₁, η_trues, result_total[1, 1, :], yerror=result_total[1, 2, :], label="Approach & Retract")
    hline!(plot_E₁, [E₁_true], label="Ground truth")

    plot_E₂ = scatter(η_trues, result_app[1, 1, :], yerror=result_app[1, 2, :], label="Approach only", ylabel="E₁_true")
    scatter!(plot_E₂, η_trues, result_total[1, 1, :], yerror=result_total[1, 2, :], label="Approach & Retract")
    hline!(plot_E₂, [E₂_true], label="Ground truth")



    plot_η = scatter(η_trues, result_app[2, 1, :] - η_trues, yerror=result_app[2, 2, :], label="Approach only", ylabel="η_fit - η_true")
    scatter!(plot_η, η_trues, result_total[2, 1, :] - η_trues, yerror=result_total[2, 2, :], label="Approach & Retract")
    print(result_app[2, 1, :])
    hline!(plot_η, [0.0], label="Ground truth")
    plot(plot_E₁, plot_E₂, plot_η, layout=(1, 3), xlabel="η")
end