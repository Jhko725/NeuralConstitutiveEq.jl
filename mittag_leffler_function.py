#%%
import jax.numpy as jnp
from jax.scipy.special import gamma
from jax._src.numpy.util import promote_args_inexact, promote_dtypes_inexact
from jax._src.lax.lax import _const as _lax_const
from jax import Array
from quadax import quadgk
import matplotlib.pyplot as plt
#%%
def MLF1(alpha, beta, z, rho=1e-5) :
    val = 0
    a = jnp.ceil((1-beta)/alpha)
    b = jnp.ceil(jnp.log(rho*(1-jnp.abs(z)/jnp.log(jnp.abs(z)))))
    k0 = (a+b+jnp.abs(a-b))/2
    for k in jnp.arange(k0+1):
        val += z**k/gamma(beta+alpha*k)
    return val

def MLF2(alpha, beta, z, rho=1e-5) :
    k0 = jnp.floor(-jnp.log(rho)/jnp.log(jnp.abs(z)))
    val = z**((1-beta)/alpha)*jnp.exp(1)**(z**(1/alpha))/alpha
    for k in jnp.arange(1, k0+1):
        val -= z**(-k)/gamma(beta-alpha*k)
    return val

def K(alpha, beta, chi, z):
    return chi**((1-beta)/alpha)/(alpha*jnp.pi)*jnp.exp(-chi**(1/alpha))*(chi*jnp.sin(jnp.pi*(1-beta))-z*jnp.sin(jnp.pi*(1-beta+alpha)))

def P(alpha, beta, eps, phi, z):
    w = phi*(1+(1-beta)/alpha)+eps**(1/alpha)*jnp.sin(phi/alpha)
    return eps**(1+(1-beta)/alpha)/(2*alpha*jnp.pi)*jnp.exp(eps**(1/alpha)*jnp.cos(phi/alpha))*((complex(jnp.cos(w), jnp.sin(w)))/((eps*jnp.exp(complex(0, phi)))-z))

def MLF3(alpha, beta, z, rho=1e-5):
    chi0 = max((1, 2*jnp.abs(z), (-jnp.log(jnp.pi*rho/6))**alpha))
    val = quadgk(K, [0, chi0], args=(alpha, beta, z))[0] + z**((1-beta)/alpha)*jnp.exp(z**(1/alpha))/alpha
    return val

def MLF4(alpha, beta, z, rho=1e-5):
    chi0 = max((1, 2*jnp.abs(z), (-jnp.log(jnp.pi*rho/6))**alpha))
    
    val = quadgk(K, [jnp.abs(z)/2, chi0], args=(alpha, beta, z))[0] + quadgk(P, [-alpha*jnp.pi,alpha*jnp.pi], args=(alpha, beta, z))
#%%    
def MLF(alpha, beta, z) -> Array:
    alpha, beta, z = promote_args_inexact("MLF", alpha, beta, z)
    _c = _lax_const
    zero = _c(z, 0)
    one = _c(z, 1)
    conds = [
    (alpha<=0) | (alpha>1),
    (0<alpha<=1) & (z==0),
    (0<alpha<=1) & (jnp.abs(z)<1),
    (0<alpha<=1) & (jnp.abs(z)>jnp.floor(10+5*alpha)),
    (0<alpha<=1) & (0<=beta<=1),
    ]
    vals = [
    jnp.nan,
    gamma(beta),
    MLF1(alpha, beta, z),
    MLF2(alpha, beta, z),
    MLF3(alpha, beta, z),
    ]
    ret = jnp.piecewise(z, conds, vals)
    return ret
# %%
a = [MLF(0.25, 0.45, i) for i in jnp.arange(0,1,1e-2)]
# %%

# %%
# condition : (0<alpha<=1) & (z==0), arbitraty beta


# %%
# condition : (0<alpha<=1) & (jnp.abs(z)<1)
t = jnp.arange(0,1,1e-2)
a1 = [MLF(0.25, 0.5, -i) for i in t]
a2 = [MLF(0.5, 0.5, -i) for i in t]
a3 = [MLF(0.75, 0.5, -i) for i in t]
a4 = [MLF(1.0, 0.5, -i) for i in t]

fig, ax = plt.subplots(1, 1, figsize=(7,5))
ax.plot(t, a1, label="alpha = 0.25")
ax.plot(t, a2, label="alpha = 0.5")
ax.plot(t, a3, label="alpha = 0.75")
ax.plot(t, a4, label="alpha = 1.0")
# %%
