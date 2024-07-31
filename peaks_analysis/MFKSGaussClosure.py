import numpy as np
from scipy.special import erf
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

def gauss(x, s=1):
    return 1 / np.sqrt(2 * np.pi) / s * np.exp(-0.5 * x**2 / s**2)

def erfc(x):
    return 0.5 * (1 - erf(x / np.sqrt(2)))

def g_avg(mu, sig, kappa, k, eps, lbd_spl, r):
    mu_tilde = mu - k * sig**2
    return kappa * np.exp(-k * mu + k**2 * sig**2 / 2) * erfc(-(mu - k * sig**2) / sig) / erfc(-mu / sig) + eps * mu + eps * sig**2 * gauss(mu, sig) / erfc(- mu / sig)
    
def lbd_g_avg(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return (kappa * (mu - k * sig**2) * np.exp(-k * mu + 0.5 * k**2 * sig**2) * erfc(-(mu - k * sig**2) / sig) / erfc(-mu / sig) +
            kappa * sig**2 * gauss(mu, sig) +
            eps * (sig**2 + mu**2) +
            eps * mu * sig**2 * gauss(mu, sig) / erfc(-mu / sig)
            )

def rate_avg(mu, sig, kappa, k, eps, lbd_spl, r):
	if lbd_spl == np.inf:
		return 0.0 * mu # stupid syntax to keep shape
	else:
		return r * sig**2 * gauss(lbd_spl - mu, sig) / erfc(- mu / sig) - r * (lbd_spl - mu) * erfc((lbd_spl - mu) / sig) / erfc(- mu / sig)
        
def lbd2_rate_avg(mu, sig, kappa, k, eps, lbd_spl, r):
	if lbd_spl == np.inf:
		return 0.0 * mu # stupid syntax to keep shape
	else:
		return ((mu**2 + sig**2) * rate_avg(mu, sig, kappa, k, eps, lbd_spl, r) + 
            2 * r * mu * sig**2 * erfc((lbd_spl - mu) / sig) / erfc(- mu / sig) + 
            r * sig**4 * gauss(lbd_spl - mu, sig) / erfc(- mu / sig) )

def mu_prime(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return - mu * (gauss(mu, sig) * (g_avg(*args) - kappa) + 0.5 * rate_avg(*args))

def sig_prime(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    res = g_avg(*args) * (mu + sig**2 * gauss(mu, sig) / erfc(- mu / sig)) - lbd_g_avg(*args) - 1/8 * lbd2_rate_avg(*args) + 0.5 * (mu**2 - sig**2) * (gauss(mu, sig) * (g_avg(*args) - kappa) + 0.5 * rate_avg(*args))
    return res / sig

def mu_sig_prime(mu_sig, kappa, k, eps, lbd_spl, r):
    args = (kappa, k, eps, lbd_spl, r)
    mu, sig = mu_sig
    return [mu_prime(mu, sig, *args), sig_prime(mu, sig, *args)]
    
def rhs(t, xy, *args):
    mu, sig = xy
    return [mu_prime(mu, sig, *args), sig_prime(mu, sig, *args)]

def get_traj(mu_ini, sig_ini, kappa, k, eps, lbd_spl, r, t_final=1e7):
	t_span = (0, t_final)
	t_array = np.linspace(t_span[0], t_span[1], 1000)
	xy0 = [mu_ini, sig_ini]
	
	sol = solve_ivp(rhs, t_span, xy0, args=(kappa, k, eps, lbd_spl, r), t_eval=t_array)
	
	return sol.y

def f1(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return gauss(mu, sig) * (g_avg(*args) - kappa) + 0.5 * rate_avg(*args)

def f2(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return g_avg(*args) * (mu + sig**2 * gauss(mu, sig) / erfc(- mu / sig)) - lbd_g_avg(*args) - 1/8 * lbd2_rate_avg(*args)

def F1(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return -mu * f1(*args)

def F2(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return f2(*args) / sig + (mu**2 - sig **2) / 2 / sig * f1(*args)


def gauss_spl(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return gauss(mu - lbd_spl, sig)

def erfc_spl(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return erfc(-(mu - lbd_spl) / sig)

def erfc_til(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return erfc(-(mu - k * sig**2) / sig)


def dmu_gauss(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return -mu / sig**2 * gauss(mu, sig)

def dsig_gauss(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return (-1 / sig + mu**2 / sig**3) * gauss(mu, sig)

def dmu_gauss_spl(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return -(mu - lbd_spl) / sig**2 * gauss(mu - lbd_spl, sig)

def dsig_gauss_spl(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return (-1 / sig + (mu - lbd_spl)**2 / sig**3) * gauss(mu - lbd_spl, sig)
    
def dmu_erfc(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return gauss(mu, sig)

def dsig_erfc(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return -mu / sig * gauss(mu, sig)

def dmu_erfc_spl(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return gauss(mu - lbd_spl, sig)

def dsig_erfc_spl(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return (lbd_spl - mu) / sig * gauss(mu - lbd_spl, sig)

def dmu_erfc_til(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return gauss(mu - k * sig**2, sig)

def dsig_erfc_til(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return (- mu / sig - k * sig) * gauss(mu - k * sig**2, sig)

def dmu_g_avg(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return (-k * kappa * np.exp(-k * mu + k**2 * sig**2 / 2) * erfc(-(mu - k * sig**2) / sig) / erfc(-mu / sig) + 
            dmu_erfc_til(*args) * kappa * np.exp(-k * mu + k**2 * sig**2 / 2) / erfc(-mu / sig) + 
            -dmu_erfc(*args) / erfc(-mu / sig)**2 * kappa * np.exp(-k * mu + k**2 * sig**2 / 2) * erfc(-(mu - k * sig**2) / sig) + 
            eps +
            dmu_gauss(*args) * eps * sig**2 / erfc(-mu / sig) +
            -dmu_erfc(*args) / erfc(- mu / sig)**2 * eps * sig**2 * gauss(mu, sig)
           )

def dsig_g_avg(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return (k**2 * sig * kappa * np.exp(-k * mu + k**2 * sig**2 / 2) * erfc(-(mu - k * sig**2) / sig) / erfc(-mu / sig) + 
            dsig_erfc_til(*args) * kappa * np.exp(-k * mu + k**2 * sig**2 / 2) / erfc(-mu / sig) + 
            -dsig_erfc(*args) / erfc(-mu / sig)**2 * kappa * np.exp(-k * mu + k**2 * sig**2 / 2) * erfc(-(mu - k * sig**2) / sig) + 
            2 * sig * eps * gauss(mu, sig) / erfc(-mu / sig) +
            dsig_gauss(*args) * eps * sig**2 / erfc(-mu / sig) +
            -dsig_erfc(*args) / erfc(- mu / sig)**2 * eps * sig**2 * gauss(mu, sig)
           )

def dmu_rate_avg(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return (dmu_gauss_spl(*args) * r * sig**2 / erfc(- mu / sig) +
            -dmu_erfc(*args) / erfc(-mu / sig)**2 * r * sig**2 * gauss(lbd_spl - mu, sig) + 
            r * erfc((lbd_spl - mu) / sig) / erfc(- mu / sig) + 
            dmu_erfc_spl(*args) * r * (mu - lbd_spl) / erfc(- mu / sig) +
            -dmu_erfc(*args) / erfc(- mu / sig)**2 * r * (mu - lbd_spl) * erfc((lbd_spl - mu) / sig)
           )

def dsig_rate_avg(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return (2 * sig * r * gauss(lbd_spl - mu, sig) / erfc(- mu / sig) +
            dsig_gauss_spl(*args) * r * sig**2 / erfc(- mu / sig) +
            -dsig_erfc(*args) / erfc(-mu / sig)**2 * r * sig**2 * gauss(lbd_spl - mu, sig) + 
            dsig_erfc_spl(*args) * r * (mu - lbd_spl) / erfc(- mu / sig) +
            -dsig_erfc(*args) / erfc(- mu / sig)**2 * r * (mu - lbd_spl) * erfc((lbd_spl - mu) / sig)
           )

def dmu_f1(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return (dmu_gauss(*args) * (g_avg(*args) - kappa) +
            gauss(mu, sig) * dmu_g_avg(*args) + 
            0.5 * dmu_rate_avg(*args)
           )
    
def dsig_f1(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return (dsig_gauss(*args) * (g_avg(*args) - kappa) +
            gauss(mu, sig) * dsig_g_avg(*args) + 
            0.5 * dsig_rate_avg(*args)
           )

def dmu_lbd_g_avg(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return (kappa * 1                   * np.exp(-k * mu + 0.5 * k**2 * sig**2) * erfc(-(mu - k * sig**2) / sig) / erfc(-mu / sig) +
            kappa * (mu - k*sig**2)*(-k)* np.exp(-k * mu + 0.5 * k**2 * sig**2) * erfc(-(mu - k * sig**2) / sig) / erfc(-mu / sig) +
            kappa * (mu - k * sig**2)   * np.exp(-k * mu + 0.5 * k**2 * sig**2) * dmu_erfc_til(*args)            / erfc(-mu / sig) +
            kappa * (mu - k * sig**2)   * np.exp(-k * mu + 0.5 * k**2 * sig**2) * erfc(-(mu - k * sig**2) / sig) * (-dmu_erfc(*args)) / erfc(-mu / sig)**2 +
            dmu_gauss(*args) * kappa * sig**2 +
            2 * mu * eps +
            eps * sig**2 * gauss(mu, sig) / erfc(-mu / sig) +
            dmu_gauss(*args) * eps * mu * sig**2 / erfc(-mu / sig) +
            -dmu_erfc(*args) / erfc(-mu / sig)**2 * eps * mu * sig**2 * gauss(mu, sig)
            )

def dsig_lbd_g_avg(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return (kappa * (-2 * k * sig)    * np.exp(-k * mu + 0.5 * k**2 * sig**2) * erfc(-(mu - k * sig**2) / sig) / erfc(-mu / sig) +
            kappa * (mu - k * sig**2) * (k**2 * sig) *  np.exp(-k * mu + 0.5 * k**2 * sig**2) * erfc(-(mu - k * sig**2) / sig) / erfc(-mu / sig) +
            kappa * (mu - k * sig**2) * np.exp(-k * mu + 0.5 * k**2 * sig**2) * dsig_erfc_til(*args)           / erfc(-mu / sig) +
            kappa * (mu - k * sig**2) * np.exp(-k * mu + 0.5 * k**2 * sig**2) * erfc(-(mu - k * sig**2) / sig) * (-dsig_erfc(*args)) / erfc(-mu / sig)**2 +
            2 * sig * kappa * gauss(mu, sig) +
            dsig_gauss(*args) * kappa * sig**2 +
            2 * sig * eps +
            2 * sig * eps * mu * gauss(mu, sig) / erfc(-mu / sig) +
            dsig_gauss(*args) * eps * mu * sig**2 / erfc(-mu / sig) +
            -dsig_erfc(*args) / erfc(-mu / sig)**2 * eps * mu * sig**2 * gauss(mu, sig)
            )

def dmu_lbd2_rate_avg(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return (2 * mu * rate_avg(mu, sig, kappa, k, eps, lbd_spl, r) + 
            dmu_rate_avg(*args) * (mu**2 + sig**2) + 
            2 * r * sig**2 * erfc((lbd_spl - mu) / sig) / erfc(- mu / sig) + 
            dmu_erfc_spl(*args) * 2 * r * mu * sig**2 / erfc(- mu / sig) + 
            -dmu_erfc(*args) / erfc(- mu / sig)**2 * 2 * r * mu * sig**2 * erfc((lbd_spl - mu) / sig) + 
            dmu_gauss_spl(*args) * r * sig**4 / erfc(- mu / sig) +
            -dmu_erfc(*args) / erfc(- mu / sig)**2 * r * sig**4 * gauss(lbd_spl - mu, sig)
            )

def dsig_lbd2_rate_avg(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return (2 * sig * rate_avg(mu, sig, kappa, k, eps, lbd_spl, r) + 
            dsig_rate_avg(*args) * (mu**2 + sig**2) + 
            2 * sig * 2 * r * mu * erfc((lbd_spl - mu) / sig) / erfc(- mu / sig) + 
            dsig_erfc_spl(*args) * 2 * r * mu * sig**2 / erfc(- mu / sig) + 
            -dsig_erfc(*args) / erfc(- mu / sig)**2 * 2 * r * mu * sig**2 * erfc((lbd_spl - mu) / sig) + 
            4 * sig**3 * r * gauss(lbd_spl - mu, sig) / erfc(- mu / sig) +
            dsig_gauss_spl(*args) * r * sig**4 / erfc(- mu / sig) +
            -dsig_erfc(*args) / erfc(- mu / sig)**2 * r * sig**4 * gauss(lbd_spl - mu, sig)
            )


def dmu_f2(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return ((1 + 
                dmu_gauss(*args) * sig**2 / erfc(- mu / sig) +
                -dmu_erfc(*args) / erfc(- mu / sig)**2 * sig**2 * gauss(mu, sig)) * g_avg(*args) +
            dmu_g_avg(*args) * (mu + sig**2 * gauss(mu, sig) / erfc(- mu / sig)) +
            -dmu_lbd_g_avg(*args) +
            -1/8 * dmu_lbd2_rate_avg(*args)
           )

def dsig_f2(mu, sig, kappa, k, eps, lbd_spl, r):
    args = (mu, sig, kappa, k, eps, lbd_spl, r)
    return ((2 * sig * gauss(mu, sig) / erfc(- mu / sig) +
             dsig_gauss(*args) * sig**2 / erfc(- mu / sig) +
             -dsig_erfc(*args) / erfc(- mu / sig)**2 * sig**2 * gauss(mu, sig)) * g_avg(*args) +
            dsig_g_avg(*args) * (mu + sig**2 * gauss(mu, sig) / erfc(- mu / sig)) +
            -dsig_lbd_g_avg(*args) +
            -1/8 * dsig_lbd2_rate_avg(*args)
           )
