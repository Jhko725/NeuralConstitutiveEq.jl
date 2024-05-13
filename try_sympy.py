# %%
import numpy as np
import sympy
from sympy.abc import t, s
from more_itertools import pairwise

knot_names = ["k0", "k1", "k2", "k3"]
knots = sympy.symbols(knot_names)
knots


# %%
def M_spline(x, knots):
    return [
        (sympy.Heaviside(x - k_i) - sympy.Heaviside(x - k_ip1)) / (k_ip1 - k_i)
        for k_i, k_ip1 in pairwise(knots)
    ]


m_splines = M_spline(t, knots)
m_splines[0]
# %%

# %%
L_splines = [sympy.laplace_transform(spl, t, s, noconds=True) for spl in m_splines]
# %%
m_spl1 = (
    2
    * ((t - knots[0]) * m_splines[0] + (knots[2] - t) * m_splines[1])
    / (knots[2] - knots[0])
)
m_spl12 = (
    2
    * ((t - knots[1]) * m_splines[1] + (knots[3] - t) * m_splines[2])
    / (knots[3] - knots[1])
)
# %%
out1 = sympy.laplace_transform(sympy.simplify(m_spl1), t, s, noconds=True)
out2 = sympy.laplace_transform(sympy.simplify(m_spl12), t, s, noconds=True)
# %%
m_spl1_ = sympy.simplify(m_spl1)
m_spl2_ = sympy.simplify(m_spl12)
m_spl2 = (
    (3 / 2)
    * ((t - knots[0]) * m_spl1_ + (knots[3] - t) * m_spl2_)
    / (knots[3] - knots[0])
)
# %%
sympy.laplace_transform(
    (t - knots[1]) * sympy.Heaviside(t - knots[0]), t, s, noconds=True
)
# %%
# sympy.simplify(out)
# %%
out2 = sympy.laplace_transform(sympy.simplify(m_spl2), t, s, noconds=True)
out2
# out2 = sympy.simplify(out2)
# %%
m_spl2_fn = sympy.lambdify((t, knots), m_spl2)
# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
knot_vals = np.arange(0.0, 12.0, 0.01)
taus = np.linspace(0.0, 10.0, 1000)  # 10 ** np.linspace(-6, 3, 100)
for i in np.arange(len(knot_vals) - 4):
    m_spl2_val = m_spl2_fn(taus, knot_vals[i : i + 4])
    ax.plot(taus, m_spl2_val)

    # ax.set_xscale("log", base = 10)
    # ax.set_yscale("log", base = 10)
# %%
Lm_spl2_fn = sympy.lambdify((s, knots), out2, "numpy")
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
knot_vals = np.arange(0.0, 12.0, 0.1)
ts = np.linspace(0.0, 1.0, 100)
for i in np.arange(len(knot_vals) - 4):
    Lm_spl2_val = Lm_spl2_fn(ts[1:], knot_vals[i : i + 4])
    ax.plot(ts[1:], Lm_spl2_val, label=f"{knot_vals[i]}")
# ax.legend()
ax.set_ylim(-0.1, 2.0)

# %%
K = sympy.symbols("K")
out2_ = sympy.simplify(out2)
out2__ = (
    out2_.subs(knots[1], knots[0] + K)
    .subs(knots[2], knots[0] + 2 * K)
    .subs(knots[3], knots[0] + 3 * K)
)
# %%
sympy.simplify(out2__)


# %%
def Lmspline(s, k0, dK):
    c = np.expm1(dK * s) / (dK * s)
    return np.exp(-s * (3 * dK + k0)) * (c**3)


Lm_spl2_fn = sympy.lambdify((s, knots), out2, "numpy")
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
knot_vals = np.arange(0.0, 12.0, 0.01)
dK = knot_vals[1] - knot_vals[0]
ts = np.linspace(0.0, 1.0, 100)
for i in np.arange(len(knot_vals) - 4):
    Lm_spl2_val = Lmspline(ts[1:], knot_vals[i], dK)
    ax.plot(ts[1:], Lm_spl2_val, label=f"{knot_vals[i]}")
# ax.legend()
ax.set_ylim(-0.1, 2.0)
# %%
