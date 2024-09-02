# steady state
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import Akima1DInterpolator
from scipy.optimize import bisect
from scipy.special import spence

def rate(lbd, lbd_spl, r):
    return np.maximum(0, r * (lbd - lbd_spl))

def g(lbd, kappa, k, eps):
	if eps == 0.0:
		return kappa * np.exp(-k * lbd)
	else: 
		return kappa * np.exp(-k * lbd) + eps * lbd

def v_Mss(lbd, g_avg, kappa, k, eps):
    return g_avg - g(lbd, kappa, k, eps)

def get_lbd_inf(g_avg, kappa, k, eps):
    if eps == 0.:
        return np.inf
    else:
        xleft = -1 / k * np.log(eps / kappa / k) # location of maximum of v_Mss
        assert v_Mss(xleft, g_avg, kappa, k, eps) > 0, f"Assumed v_M(.) > 0 but instead is {v_Mss(xleft, g_avg, kappa, k, eps)}"
        xright = g_avg / eps + 1 # zero if k = inf. add 1 for stability
        return bisect(v_Mss, xleft, xright, args=(g_avg, kappa, k, eps)) - 3e-12 # ensure the root found is always < real root
        
def get_lbd_0(g_avg, kappa, k, eps):
    if eps == 0.:
        return - 1 / k * np.log(g_avg / kappa)
    else:
        xright = -1 / k * np.log(eps / kappa / k) # location of maximum of v_Mss
        assert v_Mss(xright, g_avg, kappa, k, eps) > 0,  f"Assumed v_M(.) > 0 but instead is {v_Mss(xright, g_avg, kappa, k, eps)}"
        xleft= -1 / k * np.log(g_avg / kappa) # zero if eps = 0
        return bisect(v_Mss, xleft, xright, args=(g_avg, kappa, k, eps)) + 3e-12 # ensure the root found is always > real root

def cumulative_trapezoid_reversed(y, x, final_value):
    return cumulative_trapezoid(y[::-1], x[::-1], initial=0)[::-1] + final_value

def parameter_control(kappa, k, eps, lbd_spl):
    g_avg_min = g(lbd_spl, kappa, k, eps)
    g_avg_max = g(lbd_spl/2, kappa, k, eps)
    if g_avg_min < g_avg_max:
        return True
    else:
        print("Parameters do not produce a steady state solution."  )
        return False

def h(lbd, lbd_inf, g_avg, kappa, k, eps, lbd_spl, r):
    
    if eps == 0.0: # analytic solution
        x = k * (lbd - lbd_spl)
        u0= kappa / g_avg * np.exp(-k * lbd_spl)
        u = u0 * np.exp(-x)
        res = np.exp(- 0.5 * r / g_avg / k**2 * (
            0.5 * np.log(u/u0)**2 -
            np.log(1 - u) * np.log(u/u0) -
            spence(1 - u) + 
            spence(1 - u0)))
        return np.where(lbd <= lbd_spl, 1, res)

    else: # numerically solve int rate / v_M
        lbd_upper = lbd_inf 
        av_int = solve_ivp(lambda t, y: rate(t, lbd_spl, r) / v_Mss(t, g_avg, kappa, k, eps), t_span=(lbd_spl, lbd_upper), y0=[0.0], dense_output=True)
        res = np.where((lbd > lbd_spl) & (lbd < lbd_inf), np.exp(-0.5*av_int.sol(lbd)), np.nan)
        return np.where(lbd <= lbd_spl, 1, np.where(lbd >= lbd_inf, 0, res))[0]

def get_particular(lbd_array_current, lbd_array_full, P_array_full, lbd_inf, g_avg, kappa, k, eps, lbd_spl, r):
    
    # interpolate P on the known interval
    P_full_intepr = Akima1DInterpolator(lbd_array_full, P_array_full)

    # evaluate particular_prime on lbd_array_current
    rate_P_2 = 2 * rate(2 * lbd_array_current, lbd_spl, r) * P_full_intepr(2 * lbd_array_current)
    p_hom_array = h(lbd_array_current, lbd_inf, g_avg, kappa, k, eps, lbd_spl, r)
    with np.errstate(divide='ignore', invalid='ignore'): # safe division
        particular_prime = np.true_divide(rate_P_2, p_hom_array)
        particular_prime = np.nan_to_num(particular_prime, nan=0.0)

    # boundary condition (safe)
    p_hom_0 = h(lbd_array_full[0], lbd_inf, g_avg, kappa, k, eps, lbd_spl, r)
    final = 0 if p_hom_0 == 0 else P_array_full[0] * v_Mss(lbd_array_full[0], g_avg, kappa, k, eps) / p_hom_0
        
    # integrate from above
    return cumulative_trapezoid_reversed(particular_prime, lbd_array_current, final_value=final) 


def P_solve(lbd_array_current, lbd_array_full, P_array_full, lbd_inf, g_avg, kappa, k, eps, lbd_spl, r):
    # given P on some array, estimate it on another array 
    particular = get_particular(lbd_array_current, lbd_array_full, P_array_full, lbd_inf, g_avg, kappa, k, eps, lbd_spl, r)
    p_hom_array = h(lbd_array_current, lbd_inf, g_avg, kappa, k, eps, lbd_spl, r)

    # safe division
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.true_divide(p_hom_array, v_Mss(lbd_array_current, g_avg, kappa, k, eps))
        res = np.nan_to_num(res, nan=0.0)

    return res * particular


def cascade_solve_P(lbd_max, lbd_inf, g_avg, kappa, k, eps, lbd_spl, r, dx, lbd_lim_left, verbose):    

    # if lbd_inf is outside the interval, inform the user of the approximation
    if lbd_max < lbd_inf and verbose: print(f"We have lbd_max < lbd_inf, homogeneous approximation will be made. ")

    # on initial interval, the solution is homogeneous 
    lbd_max = min(lbd_max, lbd_inf)
    lbd_min, lbd_max = 0.5*lbd_max, lbd_max
    lbd_array_full = np.arange(lbd_min, lbd_max, dx)
    P_array_full = h(lbd_array_full, lbd_inf, g_avg, kappa, k, eps, lbd_spl, r) / v_Mss(lbd_array_full, g_avg, kappa, k, eps)

    # update to new interval
    lbd_min, lbd_max = 0.5*lbd_min, lbd_min

    # loop through intervals to the left
    while lbd_min > lbd_lim_left:
        
        # solve on current interval
        x_array_current = np.arange(lbd_min, lbd_max, dx)
        P_array_current = P_solve(x_array_current, lbd_array_full, P_array_full, lbd_inf, g_avg, kappa, k, eps, lbd_spl, r)
        
        if verbose: print(f'Dealing with {lbd_min, lbd_max}. Maximum of P here is {np.max(P_array_current)}. ')

        # update solutions
        lbd_array_full = np.concatenate((x_array_current, lbd_array_full))
        P_array_full = np.concatenate((P_array_current, P_array_full))
        
        # update interval
        lbd_min, lbd_max = 0.5*lbd_min, lbd_min

    return lbd_array_full, P_array_full


def find_leftmost_root(x_array, f_array):
    # Ensure that the input arrays are numpy arrays
    x_array = np.array(x_array)
    f_array = np.array(f_array)
    
    # Check for sign change and perform linear interpolation
    for i in range(len(f_array) - 1):
        if f_array[i] * f_array[i + 1] < 0:
            # Sign change detected between f_array[i] and f_array[i + 1]
            x1, x2 = x_array[i], x_array[i + 1]
            f1, f2 = f_array[i], f_array[i + 1]
            # Linear interpolation formula to find the root
            x_root = x1 - f1 * (x2 - x1) / (f2 - f1)
            return x_root
    
    # if no root found 
    raise ValueError('The array has no root.')

def get_P_ss(lbd_max, kappa, k, eps, lbd_spl, r, dx=1e-3, lbd_lim_left=1e-1, verbose=False, max_count=20, tol=1e-8):

    # assert parameter_control(kappa, k, eps, lbd_spl), "Parameters do not produce a steady state solution."  

    # find bracket for g_avg and define the initial guess
    g_avg_min = g(lbd_spl, kappa, k, eps)
    g_avg_max = g(lbd_spl/2, kappa, k, eps)
    g_avg_best = 0.5 * (g_avg_min + g_avg_max)
    if verbose: print(f"Bracket for g_avg ({g_avg_min}, {g_avg_max})")

    # initialize loop
    g_avg_prev = np.inf
    _counter = 0
    while np.abs(g_avg_best - g_avg_prev) > tol and _counter < max_count:
        lbd_0 = get_lbd_0(g_avg_best, kappa, k, eps)
        lbd_inf = get_lbd_inf(g_avg_best, kappa, k, eps)
        if verbose: print(f'Trying with {g_avg_best}, giving interval lbd_0 = {lbd_0},  lbd_inf = {lbd_inf}')

        # solve for P
        lbd_array, P_array = cascade_solve_P(lbd_max, lbd_inf, g_avg_best, kappa, k, eps, lbd_spl, r, dx, lbd_lim_left, verbose)

        # find root of v_M P
        vP_array = v_Mss(lbd_array, g_avg_best, kappa, k, eps) * P_array
        root = find_leftmost_root(lbd_array, vP_array)

        # set new g_avg_best
        g_avg_prev = g_avg_best
        g_avg_best = g(root, kappa, k, eps)
        # make sure we stay within the physical bracket
        assert g_avg_best > g_avg_min, f"g_avg_best too small, {g_avg_best = }, {g_avg_min = }"
        assert g_avg_best < g_avg_max, f"g_avg_best too big, {g_avg_best = }, {g_avg_max = }"

        _counter += 1
        
    # normalize 
    P_array /= np.trapz(P_array, lbd_array)
    if verbose: print(f"Numerically set g_avg: {g_avg_best:.2e}, emerging g_avg: {np.trapz(P_array * g(lbd_array, kappa, k, eps), lbd_array):.2e}. ")
    return lbd_array, P_array
