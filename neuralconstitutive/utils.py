from scipy.interpolate import make_smoothing_spline

from neuralconstitutive.indentation import Indentation


def smooth_data(indentation: Indentation) -> Indentation:
    t = indentation.time
    spl = make_smoothing_spline(t, indentation.depth)
    return Indentation(t, spl(t))


def normalize_indentations(approach: Indentation, retract: Indentation):
    t_m, h_m = approach.time[-1], approach.depth[-1]
    t_app, t_ret = approach.time / t_m, retract.time / t_m
    h_app, h_ret = approach.depth / h_m, retract.depth / h_m
    app_norm = Indentation(t_app, h_app)
    ret_norm = Indentation(t_ret, h_ret)
    return (app_norm, ret_norm), (t_m, h_m)


def normalize_forces(force_app, force_ret):
    f_m = force_app[-1]
    return (force_app / f_m, force_ret / f_m), f_m
