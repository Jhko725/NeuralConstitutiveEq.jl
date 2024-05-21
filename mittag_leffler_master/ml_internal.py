#pythran export LTInversion(float64, float64, float64, float64, float64, float64)

import jax.numpy as jnp

def LTInversion(t,lamda,alpha,beta,gama,log_epsilon):
    # Evaluation of the relevant poles
    theta = jnp.angle(lamda)
    kmin = jnp.ceil(-alpha/2. - theta/2./jnp.pi)
    kmax = jnp.floor(alpha/2. - theta/2./jnp.pi)
    k_vett = jnp.arange(kmin, kmax+1)
    s_star = jnp.abs(lamda)**(1./alpha) * jnp.exp(1j*(theta+2*k_vett*jnp.pi)/alpha)

    # Evaluation of phi(s_star) for each pole
    phi_s_star = (jnp.real(s_star)+jnp.abs(s_star))/2

    # Sorting of the poles according to the value of phi(s_star)
    index_s_star = jnp.argsort(phi_s_star)
    phi_s_star = phi_s_star.take(index_s_star)
    s_star = s_star.take(index_s_star)

    # Deleting possible poles with phi_s_star=0
    index_save = phi_s_star > 1.0e-15
    s_star = s_star.repeat(index_save)
    phi_s_star = phi_s_star.repeat(index_save)

    # Inserting the origin in the set of the singularities
    s_star = jnp.hstack([[0], s_star])
    phi_s_star = jnp.hstack([[0], phi_s_star])
    J1 = len(s_star)
    J = J1 - 1

    # Strength of the singularities
    p = gama*jnp.ones((J1,), float)
    p = p.at[0].set(max(0,-2*(alpha*gama-beta+1)))
    q = gama*jnp.ones((J1,), float)
    q = q.at[-1].set(jnp.inf)
    phi_s_star = jnp.hstack([phi_s_star, [jnp.inf]])

    # Looking for the admissible regions with respect to round-off errors
    admissible_regions = \
       jnp.nonzero(jnp.bitwise_and(
           (phi_s_star[:-1] < (log_epsilon - jnp.log(jnp.finfo(jnp.float64).eps))/t),
           (phi_s_star[:-1] < phi_s_star[1:])))[0]
    # Initializing vectors for optimal parameters
    JJ1 = admissible_regions[-1]
    mu_vett = jnp.ones((JJ1+1,), float)*jnp.inf
    N_vett = jnp.ones((JJ1+1,), float)*jnp.inf
    h_vett = jnp.ones((JJ1+1,), float)*jnp.inf

    # Evaluation of parameters for inversion of LT in each admissible region
    find_region = False
    while not find_region:
        for j1 in admissible_regions:
            if j1 < J1-1:
                muj, hj, Nj = OptimalParam_RB(t, phi_s_star[j1], phi_s_star[j1+1], p[j1], q[j1], log_epsilon)
            else:
                muj, hj, Nj = OptimalParam_RU(t, phi_s_star[j1], p[j1], log_epsilon)
            mu_vett = mu_vett.at[j1].set(muj)
            h_vett = h_vett.at[j1].set(j1)
            N_vett = N_vett.at[j1].set(Nj)
            
        if N_vett.min() > 200:
            log_epsilon = log_epsilon + jnp.log(10)
        else:
            find_region = True

    # Selection of the admissible region for integration which
    # involves the minimum number of nodes
    iN = jnp.argmin(N_vett)
    N = N_vett[iN]
    mu = mu_vett[iN]
    h = h_vett[iN]

    # Evaluation of the inverse Laplace transform
    k = jnp.arange(-N, N+1)
    u = h*k
    z = mu*(1j*u+1.)**2
    zd = -2.*mu*u + 2j*mu
    zexp = jnp.exp(z*t)
    F = z**(alpha*gama-beta)/(z**alpha - lamda)**gama*zd
    S = zexp*F ;
    Integral = h*jnp.sum(S)/2./jnp.pi/1j

    # Evaluation of residues
    ss_star = s_star[iN+1:]
    Residues = jnp.sum(1./alpha*(ss_star)**(1-beta)*jnp.exp(t*ss_star))

    # Evaluation of the ML function
    E = Integral + Residues
    if jnp.imag(lamda) == 0.:
        E = jnp.real(E)
    return E

# =========================================================================
# Finding optimal parameters in a right-bounded region
# =========================================================================
def OptimalParam_RB(t, phi_s_star_j, phi_s_star_j1, pj, qj, log_epsilon):
    # Definition of some constants
    log_eps = -36.043653389117154 # log(eps)
    fac = 1.01
    conservative_error_analysis = False

    # Maximum value of fbar as the ration between tolerance and round-off unit
    f_max = jnp.exp(log_epsilon - log_eps)

    # Evaluation of the starting values for sq_phi_star_j and sq_phi_star_j1
    sq_phi_star_j = jnp.sqrt(phi_s_star_j)
    threshold = 2.*jnp.sqrt((log_epsilon - log_eps)/t)
    sq_phi_star_j1 = min(jnp.sqrt(phi_s_star_j1), threshold - sq_phi_star_j)

    # Zero or negative values of pj and qj
    if pj < 1.0e-14 and qj < 1.0e-14:
        sq_phibar_star_j = sq_phi_star_j
        sq_phibar_star_j1 = sq_phi_star_j1
        adm_region = 1

    # Zero or negative values of just pj
    if pj < 1.0e-14 and qj >= 1.0e-14:
        sq_phibar_star_j = sq_phi_star_j
        if sq_phi_star_j > 0:
            f_min = fac*(sq_phi_star_j/(sq_phi_star_j1-sq_phi_star_j))**qj
        else:
            f_min = fac
        if f_min < f_max:
            f_bar = f_min + f_min/f_max*(f_max-f_min)
            fq = f_bar**(-1/qj)
            sq_phibar_star_j1 = (2*sq_phi_star_j1-fq*sq_phi_star_j)/(2+fq)
            adm_region = True
        else:
            adm_region = False

    # Zero or negative values of just qj
    if pj >= 1.0e-14 and qj < 1.0e-14:
        sq_phibar_star_j1 = sq_phi_star_j1
        f_min = fac*(sq_phi_star_j1/(sq_phi_star_j1-sq_phi_star_j))**pj
        if f_min < f_max:
            f_bar = f_min + f_min/f_max*(f_max-f_min)
            fp = f_bar**(-1./pj)
            sq_phibar_star_j = (2.*sq_phi_star_j+fp*sq_phi_star_j1)/(2-fp)
            adm_region = True
        else:
            adm_region = False

    # Positive values of both pj and qj
    if pj >= 1.0e-14 and qj >= 1.0e-14:
        f_min = fac*(sq_phi_star_j+sq_phi_star_j1) / \
                (sq_phi_star_j1-sq_phi_star_j)**max(pj, qj)
        if f_min < f_max:
            f_min = max(f_min,1.5)
            f_bar = f_min + f_min/f_max*(f_max-f_min)
            fp = f_bar**(-1/pj)
            fq = f_bar**(-1/qj)
            if ~conservative_error_analysis:
                w = -phi_s_star_j1*t/log_epsilon
            else:
                w = -2.*phi_s_star_j1*t/(log_epsilon-phi_s_star_j1*t)
            den = 2+w - (1+w)*fp + fq
            sq_phibar_star_j = ((2+w+fq)*sq_phi_star_j + fp*sq_phi_star_j1)/den
            sq_phibar_star_j1 = (-(1.+w)*fq*sq_phi_star_j + (2.+w-(1.+w)*fp)*sq_phi_star_j1)/den
            adm_region = True
        else:
            adm_region = False

    if adm_region:
        log_epsilon = log_epsilon  - jnp.log(f_bar)
        if not conservative_error_analysis:
            w = -sq_phibar_star_j1**2*t/log_epsilon
        else:
            w = -2.*sq_phibar_star_j1**2*t/(log_epsilon-sq_phibar_star_j1**2*t)
        muj = (((1.+w)*sq_phibar_star_j + sq_phibar_star_j1)/(2.+w))**2
        hj = -2.*jnp.pi/log_epsilon*(sq_phibar_star_j1-sq_phibar_star_j) \
             / ((1.+w)*sq_phibar_star_j + sq_phibar_star_j1)
        Nj = jnp.ceil(jnp.sqrt(1-log_epsilon/t/muj)/hj)
    else:
        muj = 0.
        hj = 0.
        Nj = jnp.inf

    return muj, hj, Nj


# =========================================================================
# Finding optimal parameters in a right-unbounded region
# =========================================================================
def OptimalParam_RU(t, phi_s_star_j, pj, log_epsilon):
    # Evaluation of the starting values for sq_phi_star_j
    sq_phi_s_star_j = jnp.sqrt(phi_s_star_j)
    if phi_s_star_j > 0:
        phibar_star_j = phi_s_star_j*1.01
    else:
        phibar_star_j = 0.01
    sq_phibar_star_j = jnp.sqrt(phibar_star_j)

    # Definition of some constants
    f_min = 1
    f_max = 10
    f_tar = 5

    # Iterative process to look for fbar in [f_min,f_max]
    while True:
        phi_t = phibar_star_j*t
        log_eps_phi_t = log_epsilon/phi_t
        Nj = jnp.ceil(phi_t/jnp.pi*(1. - 3*log_eps_phi_t/2 + jnp.sqrt(1-2*log_eps_phi_t)))
        A = jnp.pi*Nj/phi_t
        sq_muj = sq_phibar_star_j*jnp.abs(4-A)/jnp.abs(7-jnp.sqrt(1+12*A))
        fbar = ((sq_phibar_star_j-sq_phi_s_star_j)/sq_muj)**(-pj)
        if (pj < 1.0e-14) or (f_min < fbar and fbar < f_max):
            break
        sq_phibar_star_j = f_tar**(-1./pj)*sq_muj + sq_phi_s_star_j
        phibar_star_j = sq_phibar_star_j**2
    muj = sq_muj**2
    hj = (-3*A - 2 + 2*jnp.sqrt(1+12*A))/(4-A)/Nj
    
    # Adjusting integration parameters to keep round-off errors under control
    log_eps = jnp.log(jnp.finfo(jnp.float64).eps)
    threshold = (log_epsilon - log_eps)/t
    if muj > threshold:
        if abs(pj) < 1.0e-14:
            Q = 0
        else:
            Q = f_tar**(-1/pj)*jnp.sqrt(muj)
        phibar_star_j = (Q + jnp.sqrt(phi_s_star_j))**2
        if phibar_star_j < threshold:
            w = jnp.sqrt(log_eps/(log_eps-log_epsilon))
            u = jnp.sqrt(-phibar_star_j*t/log_eps)
            muj = threshold
            Nj = jnp.ceil(w*log_epsilon/2/jnp.pi/(u*w-1))
            hj = jnp.sqrt(log_eps/(log_eps - log_epsilon))/Nj
        else:
            Nj = jnp.inf
            hj = 0

    return muj, hj, Nj
