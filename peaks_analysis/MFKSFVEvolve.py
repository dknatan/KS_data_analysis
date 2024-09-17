import numpy as np
from tqdm import tqdm
import sscfw.general
from scipy.integrate import cumtrapz
from utils import g

# finite volume approach. Label cell i from x_i to x_i+1. We have P = P_i on [x_i, x_i+1)
def prepare_grid(lbd_spl, N, factor):
    # linear with dx below lbd_spl
    # linear with 2 dx on lbd_spl, 2 lbd_spl
    # ...         4 dx   2         4
    cell_boundaries = np.linspace(0, lbd_spl, 2*N, endpoint=False)
    for i in range(factor):
        cell_boundaries = np.concatenate((cell_boundaries, np.linspace(2**i * lbd_spl, 2**(i+1) * lbd_spl, N, endpoint=False)))
    # add an extra point in the end
    cell_boundaries = np.concatenate((cell_boundaries, [2*cell_boundaries[-1] - cell_boundaries[-2]]))
    
    dx_array = cell_boundaries[1:] - cell_boundaries[:-1]
    cell_centers = 0.5 * (cell_boundaries[1:] + cell_boundaries[:-1])
    return cell_boundaries, dx_array, cell_centers

def g_int(cell_boundaries, dx_array, cell_centers, kappa, k, eps):
    return kappa / k * np.exp(-k * cell_boundaries[:-1]) * (1. - np.exp(-k * dx_array)) + eps * dx_array * cell_centers

def g_mean(P, g_int_array, dx_array, kappa, k, eps):
    return np.sum(P * g_int_array) / np.sum(P * dx_array)

def v_M(x, P, g_int_array, dx_array, kappa, k, eps):
    return g_mean(P, g_int_array, dx_array, kappa, k, eps) - g(x, kappa, k, eps)
    # return 0.5 -  1 / (1 + np.exp(x - 6.32)) - eps * x
    
def rate_matrix(cell_boundaries, dx_array, cell_centers, lbd_spl, r, N):
    dx_array_theta = dx_array.copy()
    dx_array_theta[cell_boundaries[1:] <= lbd_spl+1e-10] = 0.
    rvec_out = -0.5 * dx_array_theta * (cell_centers - lbd_spl)
    rvec_in = dx_array_theta[N:] * (cell_centers[N:] - lbd_spl)
    return r * (np.diag(rvec_out) + np.diag(rvec_in, k = N))

def rhs(P, cell_boundaries, g_int_array, dx_array, kappa, k, eps, rmat):
    v_M_array = v_M(cell_boundaries, P, g_int_array, dx_array, kappa, k, eps)
    mask = v_M_array > 0.
    # v_M_array = np.ones_like(cell_boundaries)
    v_M_array_pos = np.where(mask, v_M_array, 0)
    v_M_array_neg = np.where(~mask, v_M_array, 0)

        
    mat = (-np.diag(v_M_array_neg[1:-1], k=1) +
           np.diag(v_M_array_neg[:-1]) -
           np.diag(v_M_array_pos[1:]) + 
           np.diag(v_M_array_pos[1:-1], k=-1) + 
           rmat)
    
    center_inds = np.where(mask[:-1] != mask[1:])[0]
    if center_inds.size > 0:
        center_ind = center_inds[0]
        mat[center_ind, center_ind + 1] = -v_M_array_neg[1 + center_ind]/2
        mat[center_ind, center_ind - 1] = v_M_array_pos[1 + center_ind]/2
    # if len(center_inds) == 2:
    #     center_ind = center_inds[1]
    #     mat[center_ind, center_ind + 1] = -v_M_array_neg[1 + center_ind]/2
    #     mat[center_ind, center_ind - 1] = v_M_array_pos[1 + center_ind]/2
    # elif len(center_inds) > 2:
    #     raise ValueError
    #     mat = np.diag(v_M_array_0) - np.diag(v_M_array_1[:-1], k=1) + rmat
    return (mat.T / dx_array).T
        
def step(P, dt, cell_boundaries, g_int_array, dx_array, kappa, k, eps, rmat):
    P_new = P + dt * rhs(P, cell_boundaries, g_int_array, dx_array, kappa, k, eps, rmat) @ P
    # P_new[-1] = P_new[-2] # fix bc
    return P_new

def evolve_P(n_steps, stride, P0, dt, cell_boundaries, dx_array, cell_centers, g_int_array, kappa, k, eps, rmat, normalize=True, tqdm_bool=True, description='Evolving P '):
    P = P0.copy()
    stats = np.zeros((3, n_steps//stride))
    _range = tqdm(range(n_steps), desc=description) if tqdm_bool else range(n_steps)
    for i in _range:
        if i % stride == 0:
            rhs_mat = rhs(P, cell_boundaries, g_int_array, dx_array, kappa, k, eps, rmat)
            _norm = np.sum(P * dx_array)
            _mean = np.sum(cell_centers * P * dx_array) / _norm
            stats[0, i // stride] = _norm
            stats[1, i // stride] = _mean
            stats[2, i // stride] = np.sqrt(np.sum((cell_centers - _mean)**2 * P * dx_array) / _norm)
        P += dt * rhs_mat @ P
    if normalize:
        P /= np.trapz(P, x=cell_centers)
    return P, stats

@sscfw.general.cache_numpys(num_of_returns=2)
def evolve_P_from_gauss(n_steps, stride, dt, n_grid, fact, kappa, k, eps, lbd_spl, r, mu, sig, normalize=True, n_precondition=2000):
    
    cell_boundaries, dx_array, cell_centers = prepare_grid(lbd_spl, n_grid, fact)
    rmat = rate_matrix(cell_boundaries, dx_array, cell_centers, lbd_spl, r, n_grid)    
    g_int_array= g_int(cell_boundaries, dx_array, cell_centers, kappa, k, eps)
    
    P_ini = np.exp(- 0.5 * (cell_centers - mu)**2 / sig**2) / np.sqrt(2 * np.pi * sig**2)

    # preconditioning steps
    P_preconditioned, stats0 = evolve_P(n_precondition, 1, P_ini, dt, cell_boundaries, dx_array, cell_centers, g_int_array, kappa, k, eps, rmat, normalize, description='Preconditioning: ')

    P_final, stats1 = evolve_P(n_steps, stride, P_preconditioned, dt, cell_boundaries, dx_array, cell_centers, g_int_array, kappa, k, eps, rmat, normalize)

    return P_final, np.concatenate((stats0, stats1), axis=1)


def evolve_P_from_gauss_stop_condition(n_steps, stride, dt, n_grid, fact, kappa, k, eps, lbd_spl, r, mu, sig, check_fact=10, normalize=True, n_precondition=2000):
    cell_boundaries, dx_array, cell_centers = prepare_grid(lbd_spl, n_grid, fact)
    rmat = rate_matrix(cell_boundaries, dx_array, cell_centers, lbd_spl, r, n_grid)    
    g_int_array= g_int(cell_boundaries, dx_array, cell_centers, kappa, k, eps)
    
    P_ini = np.exp(- 0.5 * (cell_centers - mu)**2 / sig**2) / np.sqrt(2 * np.pi * sig**2)

    # preconditioning steps
    P_preconditioned, stats0 = evolve_P(n_precondition, 1, P_ini, dt, cell_boundaries, dx_array, cell_centers, g_int_array, kappa, k, eps, rmat, normalize, tqdm_bool=False)

    n_steps_batch = n_steps // check_fact
    P_final_current = P_preconditioned
    for i in range(check_fact):
        P_final_current, stats1 = evolve_P(n_steps_batch, stride, P_final_current, dt, cell_boundaries, dx_array, cell_centers, g_int_array, kappa, k, eps, rmat, normalize, tqdm_bool=False)
        stats0 = np.concatenate((stats0, stats1), axis=1)

        if i > 1 and (stop_condition(stats0)):  
            print('Realization skipped, going to trivial eq. ')
            break

    return P_final_current, stats0

def check_loop(x_points, y_points, dt=1):
    dxdt = np.gradient(x_points, dt)
    dydt = np.gradient(y_points, dt)
    tnorm = np.sqrt(dxdt**2 + dydt**2)
    Tx = dxdt / tnorm
    Ty = dydt / tnorm 
    angles = cumtrapz(np.mod(np.diff(np.arctan2(Ty, Tx)) + np.pi, 2*np.pi) - np.pi)
    if np.any(np.abs(angles) > 2 * np.pi):
        return True 
    else:
        return False

def check_fall(x_points, y_points, dt=1, y_lo=1.0):
    dxdt = np.gradient(x_points, dt)
    dydt = np.gradient(y_points, dt)
    tnorm = np.sqrt(dxdt**2 + dydt**2)
    Tx = dxdt / tnorm
    Ty = dydt / tnorm
    if y_points[-1] < y_lo and np.all(np.abs(np.arctan2(Ty, Tx) + np.pi/2)[-10:] < np.pi/4):
        return True 
    
    dTxds = np.gradient(Tx, dt) / tnorm
    dTyds = np.gradient(Ty, dt) / tnorm
    dTnorm = np.sqrt(dTxds**2 + dTyds**2)

    if np.mean(dTnorm[-10:]) < 1e-2 and np.all(np.abs(np.arctan2(Ty, Tx) + np.pi/2)[-10:] < np.pi/6):
        return True
    else:
        return False
