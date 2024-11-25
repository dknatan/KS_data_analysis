import numpy as np
from utils import reparameterize
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.integrate import solve_ivp
from MFKSSteadyState import get_P_ss
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from bisect import bisect_left


# functions
def shuffle_nonzero_elements(arr, seed=0):
    mask = arr > 0
    arr_masked = arr[mask]
    np.random.seed(seed)
    np.random.shuffle(arr_masked)
    arr[mask] = arr_masked
    return arr.copy()

def get_MF_samples(num_samples, Drho, Dc, T, epsilon, r, lbd_spl, dx=1e-4, lbd_max=30, tol=1e-12, seed=0):

    kappa, k, eps, _, r = reparameterize(Drho, Dc, T, epsilon, r)

    # get steady state. Expected usage makes this computationally relatively inexpensive, so use low tolerance
    x_range, P_range = get_P_ss(lbd_max, kappa, k, eps, lbd_spl, r, dx, tol, verbose=False)
    # kill upper part
    P_range[x_range >= lbd_spl] = 0.0
    # renormalize
    P_range /= np.trapz(P_range, x_range)

    # get cdf
    cdf_array = cumulative_trapezoid(P_range, x_range)

    # interpolate to inverse cdf
    inv_cdf_array = interp1d(cdf_array, x_range[:-1], kind='linear', bounds_error=False, fill_value=(x_range[0], x_range[-1]))

    # sample from uniform and transform
    np.random.seed(seed)
    uniform_samples = np.random.uniform(0, 1, num_samples)
    samples = inv_cdf_array(uniform_samples)  # Interpolated inverse CDF

    return np.array(samples)

def get_initials(Nmax, L_full, Drho, Dc, T, epsilon, r, lbd_spl, mu, sig, ini_dist_type='chaos', seed=0):

    if ini_dist_type == 'gauss':
        np.random.seed(seed)
        lbd_vect = mu + sig * np.random.randn(Nmax)
    elif ini_dist_type == 'chaos':
        lbd_vect = get_MF_samples(Nmax, Drho, Dc, T, epsilon, r, lbd_spl, seed=seed)
       
    too_big_inds = np.where(np.cumsum(lbd_vect) > L_full)[0]
    lbd_vect[too_big_inds] = 0.0
    lbd_vect[too_big_inds[0]] = L_full - np.sum(lbd_vect)

    adj_matr = np.zeros((Nmax, Nmax), dtype=int)

    # prepare initials
    for i in range(Nmax):
        if lbd_vect[i] != 0.0:
            if i == 0:
                adj_matr[i, i+1] = 1
                adj_matr[i, i] = -1
            else:
                adj_matr[i, i+1] = 1
                adj_matr[i, i] = -2
                adj_matr[i, i-1] = 1
        
        else:
            adj_matr[i-1, i-1] += 1
            adj_matr[i-1, i] += -1
            break

    lbd_vect_ini, adj_matr_ini = lbd_vect, csr_matrix(adj_matr)

    # presplit
    while np.any(lbd_vect_ini > lbd_spl):
        lbd_vect_ini, adj_matr_ini = split(np.where(lbd_vect_ini > lbd_spl)[0][0], lbd_vect_ini, adj_matr_ini)
    # make sure all positive
    while np.any(lbd_vect_ini < 0):
        lbd_vect_ini, adj_matr_ini = fix_negative(lbd_vect_ini, adj_matr_ini)

    return shuffle_nonzero_elements(lbd_vect_ini, seed=seed), adj_matr_ini

def split(ind_split, lbd_vect, adj_matr):
    # single plateau
    if np.all(adj_matr.diagonal() == 0.0):
        # get another empty index
        ind_split2 = (ind_split + 1) % len(lbd_vect)

        # set lbd_vect
        split_val = 0.5 * lbd_vect[ind_split]
        for ind in [ind_split, ind_split2]:
            lbd_vect[ind] = split_val
        
        # set adj_matr
        adj_matr[ind_split, ind_split2] = adj_matr[ind_split2, ind_split] = 1
        
        adj_matr[ind_split, ind_split] = adj_matr[ind_split2, ind_split2] = -1
    else:
        # get another empty index
        ind_split2 = np.where(adj_matr.diagonal() == 0)[0][0]

        # set lbd_vect
        split_val = 0.5 * lbd_vect[ind_split]
        for ind in [ind_split, ind_split2]:
            lbd_vect[ind] = split_val
        
        # set adj_matr
        ind_neighbor = np.argmax(adj_matr[ind_split, :]) # here is the choice of insert on the left or on the right
        
        adj_matr[ind_split, ind_neighbor] = adj_matr[ind_neighbor, ind_split] = 0

        adj_matr[ind_neighbor, ind_split2] = adj_matr[ind_split2, ind_neighbor] = adj_matr[ind_split, ind_split2] = adj_matr[ind_split2, ind_split] = 1
        
        adj_matr[ind_split2, ind_split2] = -2
    return lbd_vect, csr_matrix(adj_matr.toarray())

def sparse_find_neighbors(sparse_adj_matr, node):
    return sparse_adj_matr[node].indices[sparse_adj_matr[node].data == 1]

def fix_negative(lbd_vect, adj_matr):
    while np.any(lbd_vect < 0): # here can put some constant to make it faster
        ind_negative = np.argmin(lbd_vect)
        inds_neighbors = sparse_find_neighbors(adj_matr, ind_negative) 
        if len(inds_neighbors) == 2:
            ind_neighbor1 = inds_neighbors[0]
            ind_neighbor2 = inds_neighbors[1]
            
            adj_matr[ind_neighbor1, ind_neighbor2] = 1
            adj_matr[ind_neighbor2, ind_neighbor1] = 1
            
            adj_matr[ind_neighbor1, ind_negative] = 0
            adj_matr[ind_negative, ind_neighbor1] = 0
            
            adj_matr[ind_negative, ind_neighbor2] = 0
            adj_matr[ind_neighbor2, ind_negative] = 0

            adj_matr[ind_negative, ind_negative] = 0

            lbd_vect[ind_neighbor1] += 0.5 * lbd_vect[ind_negative]
            lbd_vect[ind_neighbor2] += 0.5 * lbd_vect[ind_negative]

        elif len(inds_neighbors) == 1:
            ind_neighbor1 = inds_neighbors[0]
            
            adj_matr[ind_neighbor1, ind_negative] = 0
            adj_matr[ind_negative, ind_neighbor1] = 0
            
            adj_matr[ind_negative, ind_negative] = 0
            adj_matr[ind_neighbor1, ind_neighbor1] = -1

            lbd_vect[ind_neighbor1] += lbd_vect[ind_negative]

        else:
            raise ValueError(f"Adjacency matrix has {len(inds_neighbors)} neighbors. ")

        # set lbd_vect
        lbd_vect[ind_negative] = 0
        adj_matr[ind_negative, ind_negative] = 0

    return lbd_vect, adj_matr

def g(lbd, kappa, k, eps):
    return 0.5 * (kappa * np.exp(-k * lbd) + eps * lbd)

class PrecompiledRandomGenerator:
    def __init__(self, n, seed):
        self.n = n
        np.random.seed(seed)
        self.prepared_numbers = np.random.uniform(0, 1, n)
        self.index = 0

    def get_next(self):
        self.index += 1
        if self.index >= self.n:
            print('Reached the last prepared random number, looping back.') 
        self.index %= self.n
        return self.prepared_numbers[self.index]    

# functions - adaptive step size
def dydt(t, y, adj_matr, kappa, k, eps, lbd_spl): # number of additional arguments to conform to the solve_ivp
    return adj_matr @ g(y, kappa, k, eps)

def g_prime(lbd, kappa, k, eps):
    return 0.5 * (-k * kappa * np.exp(-k * lbd) + eps)

def jac_dydt(t, y, adj_matr, kappa, k, eps, lbd_spl):
    return adj_matr @ np.diag(g_prime(y, kappa, k, eps))

def event_zero(t, y, adj_matr, kappa, k, eps, lbd_spl):
    return np.min(y[y != 0]) - 1e-1

event_zero.terminal = True
event_zero.direction = -1

def get_event_split(included_inds, adj_matr, kappa, k, eps, lbd_spl):
    def event_split(t, y, adj_matr, kappa, k, eps, lbd_spl):
        return np.max(y[included_inds] - lbd_spl)
    
    event_split.terminal = True
    event_split.direction = 1

    return event_split

def get_event_split_single(ind, adj_matr, kappa, k, eps, lbd_spl):
    def event_split(t, y, adj_matr, kappa, k, eps, lbd_spl):
        return y[ind] - lbd_spl
    
    event_split.terminal = True
    event_split.direction = 1

    return event_split

def fix_negative_adaptive(lbd_vect, adj_matr):
    while np.any(np.isclose(lbd_vect[lbd_vect != 0.0], 1e-1)):
        ind_negative = np.where(np.logical_and(np.isclose(lbd_vect, 1e-1), lbd_vect != 0.0))[0][0]

        inds_neighbors = sparse_find_neighbors(adj_matr, ind_negative) 
        if len(inds_neighbors) == 2:
            ind_neighbor1 = inds_neighbors[0]
            ind_neighbor2 = inds_neighbors[1]
            
            adj_matr[ind_neighbor1, ind_neighbor2] = adj_matr[ind_neighbor2, ind_neighbor1] = 1
            
            adj_matr[ind_neighbor1, ind_negative] = adj_matr[ind_negative, ind_neighbor1] = adj_matr[ind_negative, ind_neighbor2] = adj_matr[ind_neighbor2, ind_negative] = 0

            adj_matr[ind_negative, ind_negative] = 0

            lbd_vect[ind_neighbor1] += 0.5 * lbd_vect[ind_negative]
            lbd_vect[ind_neighbor2] += 0.5 * lbd_vect[ind_negative]

        elif len(inds_neighbors) == 1:
            ind_neighbor1 = inds_neighbors[0]
            
            adj_matr[ind_neighbor1, ind_negative] = adj_matr[ind_negative, ind_neighbor1] = 0
            
            adj_matr[ind_negative, ind_negative] = 0
            adj_matr[ind_neighbor1, ind_neighbor1] += 1

            lbd_vect[ind_neighbor1] += lbd_vect[ind_negative]

        else:
            raise ValueError(f"Nonzero element has {len(inds_neighbors)} neighbors. ")

        # set lbd_vect
        lbd_vect[ind_negative] = 0
        adj_matr[ind_negative, ind_negative] = 0

    return lbd_vect, csr_matrix(adj_matr.toarray())


def insert_sorted(arr, element):
    # Find the position where the element should be inserted
    position = bisect_left(arr, element)
    
    # Insert the element at the found position
    return np.insert(arr, position, element)
    
def insert_sorted2(arr, element, arr2, element2):
    # Find the position where the element should be inserted
    position = bisect_left(arr, element)
    
    # Insert the element at the found position
    arr.insert(position, element)
    arr2.insert(position, element2)
    
    return arr, arr2

def evolve_adapt_timestep(Nmax, L_full, Drho, Dc, T, epsilon, r, lbd_spl, n_steps, dt, ini_dist_type, mu, sig, verbose=False, tqdm_bool=True, maxiter=1000, seed=0):
    
    kappa, k, eps, _, _ = reparameterize(Drho, Dc, T, epsilon, r)
    # initial conditions
    lbd_vect_ini, adj_matr_ini = get_initials(Nmax, L_full, Drho, Dc, T, epsilon, r, lbd_spl, mu, sig, ini_dist_type, seed=seed)

    # initialize variables
    t_range = np.arange(0, dt * n_steps, dt)
    lbd_vect_t = np.zeros((Nmax, n_steps))
    adj_matr_t = np.zeros((n_steps), dtype=object)
    lbd_vect_t[:, 0] = lbd_vect_ini.copy()
    adj_matr_t[0] = adj_matr_ini.copy()

    lbd_vect, adj_matr = lbd_vect_ini.copy(), adj_matr_ini.copy()
    random_uniform = PrecompiledRandomGenerator(100_000, seed=seed)

    # initialize loop variables
    time_current = 0.0
    ind_current_lo = 0
    ind_current_hi = n_steps
        
    split_times_queue = [dt * n_steps]
    split_index_queue = [None]

    included_inds = np.arange(stop=len(lbd_vect), dtype=int)
    finished_bool = False
    
    _iter = tqdm(range(maxiter), desc='Worst case timer: ') if tqdm_bool else range(maxiter)
    for _counter in _iter:
        # check if finished
        if finished_bool:
            break

        if _counter == 0:
            t_eval_current = t_range
        else:
            # insert current time to evaluation array to ensure potential event occurs on the evaluated interval
            t_eval_current = np.insert(t_range[ind_current_lo:ind_current_hi], 0, time_current)
            if split_index_queue[0] != None:
                # insert last split time to know the initial condition for next steps
                t_eval_current = np.append(t_eval_current, split_times_queue[0])

        if _counter > 11000: verbose = True

        if verbose: print(f'# active: {np.sum(lbd_vect > 0)}')
        
        # evolve and (maybe) be interrupted 
        sol = solve_ivp(dydt, 
                        t_span=(time_current, split_times_queue[0]), 
                        y0=lbd_vect, 
                        method='BDF',
                        t_eval=t_eval_current, 
                        events=[event_zero, get_event_split(included_inds, adj_matr, kappa, k, eps, lbd_spl)], 
                        args=(adj_matr, kappa, k, eps, lbd_spl),
                        jac=jac_dydt,
                        rtol=1e-6) # also can help: max_step (parameter)
        
        if len(sol.y_events[0] > 0): # merging event
            # save
            lbd_vect_t[:, ind_current_lo:int(sol.t_events[0][0] / dt) + 1] = sol.y[:, min(_counter, 1):]
            adj_matr_t[ind_current_lo:int(sol.t_events[0][0] / dt) + 1] = adj_matr.copy()

            # update timing
            time_current = sol.t_events[0][0]
            ind_current_lo = int(time_current / dt) + 1

            if verbose: print(f'Interrupted by negative at {time_current = :.1f}')
            
            # perform merge
            lbd_vect = sol.y_events[0][0]
            lbd_vect, adj_matr = fix_negative_adaptive(lbd_vect, adj_matr)

        elif len(sol.y_events[1] > 0): # split threshold reached
            # save
            lbd_vect_t[:, ind_current_lo:int(sol.t_events[1][0] / dt) + 1] = sol.y[:, min(_counter, 1):]
            adj_matr_t[ind_current_lo:int(sol.t_events[1][0] / dt) + 1] = adj_matr.copy()
            
            # update timing
            time_current = sol.t_events[1][0]
            ind_current_lo = int(time_current / dt) + 1

            # determine where split happened
            lbd_vect = sol.y_events[1][0]
            split_index = np.argmin(np.abs(lbd_vect - lbd_spl))
            
            # exclude the split index from split monitoring
            included_inds = np.delete(included_inds, np.where(included_inds == split_index)[0][0])
                
            # sample time-to-split: inv_P(uniform) is a sampled time-to-split (approx)
            dLambdadt = dydt(0.0, lbd_vect, adj_matr, kappa, k, eps, lbd_spl)[split_index] 
            split_time = time_current + np.sqrt(-2 / r / dLambdadt * np.log(1 - random_uniform.get_next()))
            
            # add time and split index to split queues
            split_times_queue, split_index_queue = insert_sorted2(split_times_queue, split_time, split_index_queue, split_index)
            if verbose: print(f'Interrupted by split at {time_current = :.1f}, {split_index = }, sampled new split time: {split_time:.2f}')
            
            # update first expected time of split
            ind_current_hi = int(split_times_queue[0] / dt) + 1
        
        else:
            if split_index_queue[0] != None: # splitting time reached
                if verbose: print('Integrated until split time')
                # save
                try:
                    lbd_vect_t[:, ind_current_lo:ind_current_hi] = sol.y[:, min(_counter, 1):-1]
                except:
                    print(sol)
                    print(adj_matr)
                adj_matr_t[ind_current_lo:ind_current_hi] = adj_matr.copy()

                # perform split
                split_index = split_index_queue.pop(0)
                lbd_vect, adj_matr = split(split_index, sol.y[:, -1], adj_matr)
                adj_matr = csr_matrix(adj_matr.toarray())
                
                # include the split index to split monitoring
                included_inds = insert_sorted(included_inds, split_index)

                # remove time from split queue, update times and indices
                time_current = split_times_queue.pop(0)
                ind_current_lo = int(time_current / dt) + 1
                ind_current_hi = int(split_times_queue[0] / dt) + 1

            else: # final time reached
                if verbose: print('Finishing up.')
                # save
                try:
                    lbd_vect_t[:, ind_current_lo:ind_current_hi] = sol.y[:, min(_counter, 1):]
                except:
                    print(sol)
                    print(adj_matr)
                adj_matr_t[ind_current_lo:ind_current_hi] = adj_matr.copy()

                finished_bool = True

        _counter += 1
        if verbose: print(f'excluded indices: {[item for item in np.arange(stop=len(lbd_vect), dtype=int) if item not in included_inds]}')
    
    return t_range, lbd_vect_t, adj_matr_t

# 3D hard splitting functions
def get_event_split_hard(adj_matr, kappa, k, eps, lbd_spl):
    def event_split(t, y, adj_matr, kappa, k, eps, lbd_spl):
        return np.max(y - 4/3*lbd_spl)
    
    event_split.terminal = True
    event_split.direction = 1

    return event_split

def evolve_adapt_timestep_hard_split(Nmax, L_full, Drho, Dc, T, epsilon, r, lbd_spl, n_steps, dt, ini_dist_type, mu, sig, verbose=False, tqdm_bool=True, maxiter=1000, seed=0):
    # differs from evolve_adapt_timestep by 1. r=inf, 2. saves points at splitting
    kappa, k, eps, _, _ = reparameterize(Drho, Dc, T, epsilon, r)
    # initial conditions
    lbd_vect_ini, adj_matr_ini = get_initials(Nmax, L_full, Drho, Dc, T, epsilon, r, lbd_spl, mu, sig, ini_dist_type, seed=seed)
    # presplit
    if verbose: print('Presplitting ...')
    while np.any(lbd_vect_ini > lbd_spl):
        lbd_vect_ini, adj_matr_ini = split(np.where(lbd_vect_ini > lbd_spl)[0][0], lbd_vect_ini, adj_matr_ini)
    # make sure all positive
    while np.any(lbd_vect_ini < 0):
        lbd_vect_ini, adj_matr_ini = fix_negative(lbd_vect_ini, adj_matr_ini)

    Nmax = len(lbd_vect_ini)

    # initialize variables
    t_range = np.arange(0, dt * n_steps, dt)
    lbd_vect_t = np.zeros((Nmax, n_steps))
    adj_matr_t = np.zeros((n_steps), dtype=object)
    lbd_vect_t[:, 0] = lbd_vect_ini
    adj_matr_t[0] = adj_matr_ini

    lbd_vect, adj_matr = lbd_vect_ini, adj_matr_ini
    t_range_split, lbd_vect_split, adj_matr_split = [], [], []

    # initialize loop variables
    time_current = 0.0
    ind_current_lo = 0
        
    finished_bool = False
    
    _iter = tqdm(range(maxiter), desc='Worst case timer: ') if tqdm_bool else range(maxiter)
    for _counter in _iter:
        # check if finished
        if finished_bool:
            break

        if _counter == 0:
            t_eval_current = t_range
        else:
            # insert current time to evaluation array to ensure potential event occurs on the evaluated interval
            t_eval_current = np.insert(t_range[ind_current_lo:n_steps], 0, time_current)

        if verbose: print(f'# active: {np.sum(lbd_vect > 0)}')
        
        # evolve and (maybe) be interrupted 
        sol = solve_ivp(dydt, 
                        t_span=(time_current, dt * n_steps), 
                        y0=lbd_vect, 
                        method='BDF',
                        t_eval=t_eval_current, 
                        events=[event_zero, get_event_split_hard(adj_matr, kappa, k, eps, lbd_spl)], 
                        args=(adj_matr, kappa, k, eps, lbd_spl),
                        jac=jac_dydt,
                        rtol=1e-6) # also can help: max_step (parameter)
        
        if len(sol.y_events[0] > 0): # merging event
            # save
            lbd_vect_t[:, ind_current_lo:int(sol.t_events[0][0] / dt) + 1] = sol.y[:, min(_counter, 1):]
            adj_matr_t[ind_current_lo:int(sol.t_events[0][0] / dt) + 1] = adj_matr.copy()

            # update timing
            time_current = sol.t_events[0][0]
            ind_current_lo = int(time_current / dt) + 1

            if verbose: print(f'Interrupted by negative at {time_current = :.1f}')
            
            # perform merge
            lbd_vect, adj_matr = fix_negative_adaptive(sol.y_events[0][0], adj_matr)
            adj_matr = csr_matrix(adj_matr.toarray())

        elif len(sol.y_events[1] > 0): # split threshold reached
            # save
            lbd_vect_t[:, ind_current_lo:int(sol.t_events[1][0] / dt) + 1] = sol.y[:, min(_counter, 1):]
            adj_matr_t[ind_current_lo:int(sol.t_events[1][0] / dt) + 1] = adj_matr.copy()
            t_range_split.append(sol.t_events[1][0])
            lbd_vect_split.append(sol.y_events[1][0])
            adj_matr_split.append(adj_matr.copy())

            # update timing
            time_current = sol.t_events[1][0]
            ind_current_lo = int(time_current / dt) + 1

            # determine where split happened & split
            lbd_vect = sol.y_events[1][0]
            split_index = np.argmin(np.abs(lbd_vect - lbd_spl))
            lbd_vect, adj_matr = split(split_index, sol.y_events[1][0], adj_matr)
            adj_matr = csr_matrix(adj_matr.toarray())
                        
            # add time and split index to split queues
            if verbose: print(f'Interrupted by split at {time_current = :.1f}, {split_index = }')
                        
            ind_current_lo = int(time_current / dt) + 1

        else:
            # final time reached
            if verbose: print('Finishing up.')
            # save
            # lbd_vect_t[:, ind_current_lo:n_steps] = sol.y[:, min(_counter, 1):]
            # adj_matr_t[ind_current_lo:n_steps] = adj_matr.copy()

            finished_bool = True

        _counter += 1
    
    return t_range, lbd_vect_t, adj_matr_t, np.array(t_range_split), np.array(lbd_vect_split).T, np.array(adj_matr_split)

# equilibriation time functions
def check_finished(lbd_vect, adj_matr, kappa, k, eps, lbd_spl):
    lbd_max = -1/k * np.log(eps/k/kappa)
    in_interval_bool = np.all(np.logical_and(lbd_vect > lbd_max, lbd_vect < lbd_spl), where=lbd_vect>0)
    contracting_bool = np.sum((lbd_vect - np.mean(lbd_vect, where=lbd_vect>0)) * dydt(0.0, lbd_vect, adj_matr, kappa, k, eps, lbd_spl)) < 0
    return in_interval_bool and contracting_bool

def evolve_adapt_timestep_equi_time(Nmax, L_full, Drho, Dc, T, epsilon, r, lbd_spl, n_steps, dt, ini_dist_type, mu, sig, verbose=False, tqdm_bool=True, maxiter=1000, seed=0):
    kappa, k, eps, _, _ = reparameterize(Drho, Dc, T, epsilon, r)
    # initial conditions
    lbd_vect_ini, adj_matr_ini = get_initials(Nmax, L_full, Drho, Dc, T, epsilon, r, lbd_spl, mu, sig, ini_dist_type, seed=seed)

    # initialize variables

    lbd_vect, adj_matr = lbd_vect_ini.copy(), adj_matr_ini.copy()
    random_uniform = PrecompiledRandomGenerator(100_000, seed=seed)

    # initialize loop variables
    time_current = 0.0
        
    split_times_queue = [n_steps * dt]
    split_index_queue = [None]

    included_inds = np.arange(stop=len(lbd_vect), dtype=int)
    finished_bool = False
    
    _iter = tqdm(range(maxiter), desc='Worst case timer: ') if tqdm_bool else range(maxiter)
    for _counter in _iter:
        # check if finished
        if finished_bool:
            break

        if verbose: print(f'# active: {np.sum(lbd_vect > 0)}')
        
        # evolve and (maybe) be interrupted 
        sol = solve_ivp(dydt, 
                        t_span=(time_current, split_times_queue[0]), 
                        y0=lbd_vect, 
                        method='BDF',
                        events=[event_zero, get_event_split(included_inds, adj_matr, kappa, k, eps, lbd_spl)], 
                        args=(adj_matr, kappa, k, eps, lbd_spl),
                        jac=jac_dydt,
                        rtol=1e-6) # also can help: max_step, first_step (parameters)
        
        if len(sol.y_events[0] > 0): # merging event
            # update timing
            time_current = sol.t_events[0][0]

            if verbose: print(f'Interrupted by negative at {time_current = :.1f}')
            
            # perform merge
            lbd_vect, adj_matr = fix_negative_adaptive(sol.y_events[0][0], adj_matr)
            adj_matr = csr_matrix(adj_matr.toarray())

            if check_finished(lbd_vect, adj_matr, kappa, k, eps, lbd_spl):
                return time_current, True, np.sum(lbd_vect > 0)

        elif len(sol.y_events[1] > 0): # split threshold reached
            # update timing
            time_current = sol.t_events[1][0]

            # determine where split happened
            lbd_vect = sol.y_events[1][0]
            split_index = np.argmin(np.abs(lbd_vect - lbd_spl))
            
            # exclude the split index from split monitoring
            included_inds = np.delete(included_inds, np.where(included_inds == split_index)[0][0])
                
            # sample time-to-split: inv_P(uniform) is a sampled time-to-split (approx)
            dLambdadt = dydt(0.0, lbd_vect, adj_matr, kappa, k, eps, lbd_spl)[split_index] 
            split_time = time_current +  np.sqrt(-2 / r / dLambdadt * np.log(1 - random_uniform.get_next()))
            
            # add time and split index to split queues
            split_times_queue, split_index_queue = insert_sorted2(split_times_queue, split_time, split_index_queue, split_index)
            if verbose: print(f'Interrupted by split at {time_current = :.1f}, {split_index = }, sampled new split time: {split_time:.2f}')
            
        else:
            if split_index_queue[0] != None: # splitting time reached
                if verbose: print('Integrated until split time')
                # perform split
                split_index = split_index_queue.pop(0)
                lbd_vect, adj_matr = split(split_index, sol.y[:, -1], adj_matr)
                adj_matr = csr_matrix(adj_matr.toarray())
                
                # include the split index to split monitoring
                included_inds = insert_sorted(included_inds, split_index)

                # remove time from split queue, update times and indices
                time_current = split_times_queue.pop(0)
                if check_finished(lbd_vect, adj_matr, kappa, k, eps, lbd_spl):
                    return time_current, True, np.sum(lbd_vect > 0)

            else: # final time reached
                if verbose: print('Finishing up.')
                finished_bool = True

        _counter += 1
        if verbose: print(f'excluded indices: {[item for item in np.arange(stop=len(lbd_vect), dtype=int) if item not in included_inds]}')
        
    if check_finished(sol.y[:,-1], adj_matr, kappa, k, eps, lbd_spl):
        return time_current, True, np.sum(lbd_vect > 0)
    else:
        return time_current, False, np.sum(lbd_vect > 0)

def evolve_adapt_timestep_equi_time_hardmax(Nmax, L_full, Drho, Dc, T, epsilon, r, lbd_spl, n_steps, dt, ini_dist_type, mu, sig, verbose=False, tqdm_bool=True, maxiter=1000, seed=0):
    kappa, k, eps, _, _ = reparameterize(Drho, Dc, T, epsilon, r)
    # initial conditions
    lbd_vect_ini, adj_matr_ini = get_initials(Nmax, L_full, Drho, Dc, T, epsilon, r, lbd_spl, mu, sig, ini_dist_type, seed=seed)

    # initialize variables

    lbd_vect, adj_matr = lbd_vect_ini.copy(), adj_matr_ini.copy()
    random_uniform = PrecompiledRandomGenerator(100_000, seed=seed)

    # initialize loop variables
    time_current = 0.0
        
    split_times_queue = [n_steps * dt]
    split_index_queue = [None]

    included_inds = np.arange(stop=len(lbd_vect), dtype=int)
    finished_bool = False
    
    _iter = tqdm(range(maxiter), desc='Worst case timer: ') if tqdm_bool else range(maxiter)
    for _counter in _iter:
        # check if finished
        if finished_bool:
            break

        if verbose: print(f'# active: {np.sum(lbd_vect > 0)}')
        
        # evolve and (maybe) be interrupted 
        sol = solve_ivp(dydt, 
                        t_span=(time_current, split_times_queue[0]), 
                        y0=lbd_vect, 
                        method='BDF',
                        events=[event_zero, get_event_split(included_inds, adj_matr, kappa, k, eps, lbd_spl), get_event_split_hard(adj_matr, kappa, k, eps, 4/3*lbd_spl)], 
                        args=(adj_matr, kappa, k, eps, lbd_spl),
                        jac=jac_dydt,
                        rtol=1e-6) # also can help: max_step, first_step (parameters)
        
        if len(sol.y_events[0] > 0): # merging event
            # update timing
            time_current = sol.t_events[0][0]

            if verbose: print(f'Interrupted by negative at {time_current = :.1f}')
            
            # perform merge
            lbd_vect, adj_matr = fix_negative_adaptive(sol.y_events[0][0], adj_matr)
            adj_matr = csr_matrix(adj_matr.toarray())
            if check_finished(lbd_vect, adj_matr, kappa, k, eps, lbd_spl):
                return time_current, True, np.sum(lbd_vect > 0)

        elif len(sol.y_events[1] > 0): # split threshold reached
            # update timing
            time_current = sol.t_events[1][0]

            # determine where split happened
            lbd_vect = sol.y_events[1][0]
            split_index = np.argmin(np.abs(lbd_vect - lbd_spl))
            
            # exclude the split index from split monitoring
            included_inds = np.delete(included_inds, np.where(included_inds == split_index)[0][0])
                
            # sample time-to-split: inv_P(uniform) is a sampled time-to-split (approx)
            dLambdadt = dydt(0.0, lbd_vect, adj_matr, kappa, k, eps, lbd_spl)[split_index] 
            split_time = time_current +  np.sqrt(-2 / r / dLambdadt * np.log(1 - random_uniform.get_next()))
            
            # add time and split index to split queues
            split_times_queue, split_index_queue = insert_sorted2(split_times_queue, split_time, split_index_queue, split_index)
            if verbose: print(f'Interrupted by split at {time_current = :.1f}, {split_index = }, sampled new split time: {split_time:.2f}')
            
        elif len(sol.y_events[2] > 0): # hard split threshold reached
            # update timing
            time_current = sol.t_events[2][0]

            # determine where hard split happened & split
            lbd_vect = sol.y_events[2][0]
            split_index = np.argmin(np.abs(lbd_vect - 4/3*lbd_spl))
            lbd_vect, adj_matr = split(split_index, sol.y_events[2][0], adj_matr)
            adj_matr = csr_matrix(adj_matr.toarray())
            
            # find the split index in the splitting queue
            ind_ind = split_index_queue.index(split_index)
            # exclude it from the queue
            split_index_queue.pop(ind_ind)
            # exclude the previously sampled split time 
            split_times_queue.pop(ind_ind)

            # include the split index to split monitoring
            included_inds = insert_sorted(included_inds, split_index)

            if check_finished(lbd_vect, adj_matr, kappa, k, eps, lbd_spl):
                return time_current, True, np.sum(lbd_vect > 0)

        else:
            if split_index_queue[0] != None: # splitting time reached
                if verbose: print('Integrated until split time')
                # perform split
                split_index = split_index_queue.pop(0)
                lbd_vect, adj_matr = split(split_index, sol.y[:, -1], adj_matr)
                adj_matr = csr_matrix(adj_matr.toarray())
                
                # include the split index to split monitoring
                included_inds = insert_sorted(included_inds, split_index)

                # remove time from split queue, update times and indices
                time_current = split_times_queue.pop(0)
                if check_finished(lbd_vect, adj_matr, kappa, k, eps, lbd_spl):
                    return time_current, True, np.sum(lbd_vect > 0)

            else: # final time reached
                if verbose: print('Finishing up.')
                finished_bool = True

        _counter += 1
        if verbose: print(f'excluded indices: {[item for item in np.arange(stop=len(lbd_vect), dtype=int) if item not in included_inds]}')
        
    if check_finished(sol.y[:,-1], adj_matr, kappa, k, eps, lbd_spl):
        return time_current, True, np.sum(lbd_vect > 0)
    else:
        return time_current, False, np.sum(lbd_vect > 0)


def evolve_adapt_timestep_smart(Nmax, L_full, Drho, Dc, T, epsilon, r, lbd_spl, n_steps, dt, ini_dist_type, mu, sig, save_bool=False, verbose=False, tqdm_bool=True, maxiter=1000, seed=0):
    
    kappa, k, eps, _, _ = reparameterize(Drho, Dc, T, epsilon, r)
    # initial conditions
    lbd_vect_ini, adj_matr_ini = get_initials(Nmax, L_full, Drho, Dc, T, epsilon, r, lbd_spl, mu, sig, ini_dist_type, seed=seed)

    # initialize save variables
    if save_bool:
        t_range = np.arange(0, dt * n_steps, dt)
        lbd_vect_t = np.zeros((Nmax, n_steps))
        adj_matr_t = np.zeros((n_steps), dtype=object)
        lbd_vect_t[:, 0] = lbd_vect_ini.copy()
        adj_matr_t[0] = adj_matr_ini.copy()

    lbd_vect, adj_matr = lbd_vect_ini.copy(), adj_matr_ini.copy()
    random_uniform = PrecompiledRandomGenerator(100_000, seed=seed)

    # initialize loop variables
    time_current = 0.0
    ind_current_lo = 0
    ind_current_hi = n_steps
        
    split_times_queue = [n_steps * dt]
    split_index_queue = [None]

    included_inds = [ind for ind in np.arange(stop=len(lbd_vect), dtype=int) if lbd_vect[ind] > 0.0]
    finished_bool = False
    
    _iter = tqdm(range(maxiter), desc='Worst case timer: ') if tqdm_bool else range(maxiter)
    for _counter in _iter:
        # check if finished
        if finished_bool:
            break

        if save_bool:
            if _counter == 0:
                t_eval_current = t_range
            else:
                # insert current time to evaluation array to ensure potential event occurs on the evaluated interval
                t_eval_current = np.insert(t_range[ind_current_lo:ind_current_hi], 0, time_current)
                if split_index_queue[0] != None:
                    # insert last split time to know the initial condition for next steps
                    t_eval_current = np.append(t_eval_current, split_times_queue[0])

        if verbose: print(f'# active: {np.sum(lbd_vect > 0)}')
        
        # evolve and (maybe) be interrupted 
        sol = solve_ivp(dydt, 
                        t_span=(time_current, split_times_queue[0]), 
                        y0=lbd_vect, 
                        method='BDF',
                        t_eval=t_eval_current if save_bool else None, 
                        events=[event_zero] + [get_event_split(np.where(lbd_vect > lbd_spl)[0], adj_matr, kappa, k, eps, lbd_spl)] + [get_event_split_single(ind, adj_matr, kappa, k, eps, lbd_spl) for ind in included_inds if ind not in split_index_queue], 
                        args=(adj_matr, kappa, k, eps, lbd_spl),
                        jac=jac_dydt,
                        rtol=1e-6) # also can help: max_step (parameter)
        
        if sol.status == -1:
            print(f'Solver failed. Message: {sol.message}.\n{sol}')

        elif sol.status == 1:
            # determine which event happened
            event_ind = [ind for ind in np.arange(1 + 1 + len(included_inds) - (len(split_index_queue) - 1), dtype=int) if len(sol.y_events[ind]) > 0][0]

            if save_bool:
                lbd_vect_t[:, ind_current_lo:int(sol.t_events[event_ind][0] / dt) + 1] = sol.y[:, min(_counter, 1):]
                adj_matr_t[ind_current_lo:int(sol.t_events[event_ind][0] / dt) + 1] = adj_matr.copy()
                
            # update timing
            time_current = sol.t_events[event_ind][0]
            ind_current_lo = int(time_current / dt) + 1
            lbd_vect = sol.y_events[event_ind][0]

            if event_ind == 0: # merging event
                if verbose: print(f'Interrupted by merge at {time_current = :.1f}')
                
                # perform merge
                lbd_vect, adj_matr = fix_negative_adaptive(lbd_vect, adj_matr)

            elif event_ind == 1: # hard split event
                if verbose: print(f'Interrupted by hard split at {time_current = :.1f}')
                print(f'{split_index_queue = }, {lbd_vect }')
                # determine where split happened
                split_index = np.argmin(np.abs(lbd_vect - 4/3*lbd_spl))
                lbd_vect, adj_matr = split(split_index, lbd_vect, adj_matr)
                
                # find the split index in the splitting queue
                ind_ind = split_index_queue.index(split_index)
                # exclude it from the queue
                split_index_queue.pop(ind_ind)
                # exclude the previously sampled split time 
                split_times_queue.pop(ind_ind)

                # possibly fix ind_current_hi
                ind_current_hi = int(split_times_queue[0] / dt) + 1

                # after splitting there are more plateaus
                included_inds = [ind for ind in np.arange(stop=len(lbd_vect), dtype=int) if lbd_vect[ind] > 0.0]

            else: # split threshold reached
                # determine where split happened
                split_index = [ind for ind in included_inds if ind not in split_index_queue][event_ind - 2]
                
                # exclude the split index from split monitoring
                # included_inds = np.delete(included_inds, np.where(included_inds == split_index)[0][0])
                    
                # sample time-to-split: inv_P(uniform) is a sampled time-to-split (approx)
                dLambdadt = dydt(0.0, lbd_vect, adj_matr, kappa, k, eps, lbd_spl)[split_index] 
                split_time = time_current + np.sqrt(-2 / r / dLambdadt * np.log(1 - random_uniform.get_next()))
                
                # add time and split index to split queues
                split_times_queue, split_index_queue = insert_sorted2(split_times_queue, split_time, split_index_queue, split_index)
                if verbose: print(f'Interrupted by split at {time_current = :.1f}, {split_index = }, sampled new split time: {split_time:.2f}')
                
                # update first expected time of split
                ind_current_hi = int(split_times_queue[0] / dt) + 1
            
            if not save_bool and check_finished(lbd_vect, adj_matr, kappa, k, eps, lbd_spl):
                return time_current, True, np.sum(lbd_vect > 0)

        elif sol.status == 0:
            finished_bool = split_index_queue[0] == None # if no more splitting left we are done

            if save_bool:
                lbd_vect_t[:, ind_current_lo:ind_current_hi] = sol.y[:, min(_counter, 1):] if finished_bool else sol.y[:, min(_counter, 1):-1]
                adj_matr_t[ind_current_lo:ind_current_hi] = adj_matr.copy()

            if not finished_bool: # splitting time reached
                if verbose: print('Integrated until split time')
                # perform split
                split_index = split_index_queue.pop(0)
                lbd_vect, adj_matr = split(split_index, sol.y[:, -1], adj_matr)
                
                # remove time from split queue, update times and indices
                time_current = split_times_queue.pop(0)
                ind_current_lo = int(time_current / dt) + 1
                ind_current_hi = int(split_times_queue[0] / dt) + 1

                # after splitting there are more plateaus
                included_inds = [ind for ind in np.arange(stop=len(lbd_vect), dtype=int) if lbd_vect[ind] > 0.0]

            if not save_bool and check_finished(lbd_vect, adj_matr, kappa, k, eps, lbd_spl):
                return time_current, True, np.sum(lbd_vect > 0)
            
        _counter += 1
        if verbose: print(f'excluded indices: {[item for item in np.arange(stop=len(lbd_vect), dtype=int) if item not in included_inds]}')
    
    if save_bool:
        return t_range, lbd_vect_t, adj_matr_t
    else:
        if check_finished(sol.y[:,-1], adj_matr, kappa, k, eps, lbd_spl):
            return time_current, True, np.sum(lbd_vect > 0)
        else:
            return time_current, False, np.sum(lbd_vect > 0)


# document

# end