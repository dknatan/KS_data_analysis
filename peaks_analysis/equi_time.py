# standalone file to compute and save equilibration times for the stochastic particle solver
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.integrate import solve_ivp
from bisect import bisect_left

# functions
def _lbd_spl_epsilon(epsilon):
    if epsilon == 0.0:
        return np.inf 
    else:
        return np.exp(1.13734914) * epsilon**(-0.3579829)
lbd_spl_epsilon = np.vectorize(_lbd_spl_epsilon)

def reparameterize(Drho, Dc, T, epsilon, r):
    M = np.sqrt(12 * Drho * Dc / T)
    kappa = 1 / (1 / 4 / np.sqrt(Dc) + 1 / M / T)
    k = 1 / np.sqrt(Dc) 
    eps = 4 * epsilon * np.exp(-T * M / 2 / Drho / np.sqrt(Dc))
    lbd_spl = lbd_spl_epsilon(epsilon)
    return kappa, k, eps, lbd_spl, r

def get_initials(L_full, lbd_spl, mu_fact, sig, Nmax_fact=2):
    Nmax = int(Nmax_fact * L_full / lbd_spl)
    n_ini = int(L_full / lbd_spl / mu_fact) + 1

    lbd_vect = np.zeros((Nmax))
    adj_matr = np.zeros((Nmax, Nmax), dtype=int)

    # prepare initials
    for i in range(Nmax):
        if np.sum(lbd_vect) < L_full:
            lbd_vect[i] = L_full / n_ini + sig * np.random.randn()
            adj_matr[i, i+1] = 1
            adj_matr[i, i] += -1
            if i > 0:
                adj_matr[i, i-1] = 1
                adj_matr[i, i] += -1
        
        else:
            lbd_vect[i-1] = 0
            lbd_vect[i-1] = L_full - np.sum(lbd_vect)
            adj_matr[i-1, i] = 0
            adj_matr[i-1, i-1] += 1
            break
    return lbd_vect, csr_matrix(adj_matr)

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
    return lbd_vect, adj_matr

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

def rhs(lbd_vect, adj_matr, kappa, k, eps):
    return adj_matr @ g(lbd_vect, kappa, k, eps)

def step(lbd_vect, adj_matr, random_uniform, dt, kappa, k, eps, lbd_spl, r):
    # handle splitting
    potential_split_inds = np.where(lbd_vect > lbd_spl)[0]
    for i, potential_split_ind in enumerate(potential_split_inds):
        if random_uniform.get_next() < dt * r * (lbd_vect[potential_split_ind] - lbd_spl):
            lbd_vect, adj_matr = split(potential_split_ind, lbd_vect, adj_matr)
    # handle dynamics
    lbd_vect_tilde = lbd_vect + dt * rhs(lbd_vect, adj_matr, kappa, k, eps)   
    return fix_negative(lbd_vect_tilde, adj_matr)

class PrecompiledRandomGenerator:
    def __init__(self, n):
        self.n = n
        self.prepared_numbers = np.random.uniform(0, 1, n)
        self.index = 0

    def get_next(self):
        self.index += 1
        if self.index >= self.n:
            print('Reached the last prepared random number, looping back.') 
        self.index %= self.n
        return self.prepared_numbers[self.index]    

def dydt(t, y, adj_matr, kappa, k, eps, lbd_spl): 
    return adj_matr @ g(y, kappa, k, eps)

def g_prime(lbd, kappa, k, eps):
    return 0.5 * (-k * kappa * np.exp(-k * lbd) + eps)

def jac_dydt(t, y, adj_matr, kappa, k, eps, lbd_spl):
    return adj_matr @ np.diag(g_prime(y, kappa, k, eps))

def event_zero(t, y, adj_matr, kappa, k, eps, lbd_spl):
    return np.min(y[y != 0])

def get_event_split(included_inds, adj_matr, kappa, k, eps, lbd_spl):
    def event_split(t, y, adj_matr, kappa, k, eps, lbd_spl):
        return np.max(y[included_inds] - lbd_spl)
    
    event_split.terminal = True
    event_split.direction = 1

    return event_split

def fix_negative_adaptive(lbd_vect, adj_matr):
    while np.any(np.isclose(lbd_vect[lbd_vect != 0.0], 0.0)):
        ind_negative = np.where(np.logical_and(np.isclose(lbd_vect, 0.0), lbd_vect != 0.0))[0][0]

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

        elif len(inds_neighbors) == 0:            
            adj_matr[ind_negative, ind_negative] = 0

        else:
            raise ValueError(f"Adjacency matrix has {len(inds_neighbors)} neighbors. ")

        # set lbd_vect
        lbd_vect[ind_negative] = 0
        adj_matr[ind_negative, ind_negative] = 0

    return lbd_vect, adj_matr

event_zero.terminal = True
event_zero.direction = -1

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

def check_finished(lbd_vect, adj_matr, kappa, k, eps, lbd_spl):
    lbd_max = -1/k * np.log(eps/k/kappa)
    in_interval_bool = np.all(np.logical_and(lbd_vect > lbd_max, lbd_vect < lbd_spl), where=lbd_vect>0)
    contracting_bool = np.sum((lbd_vect - np.mean(lbd_vect, where=lbd_vect>0)) * dydt(0.0, lbd_vect, adj_matr, kappa, k, eps, lbd_spl)) < 0
    return in_interval_bool and contracting_bool

def evolve_adapt_timestep_equi_time(L_full, Drho, Dc, T, epsilon, r, lbd_spl, mu_fact, sig, n_steps, dt, verbose=False, tqdm_bool=True, maxiter=1000, Nmax_fact=2):
    kappa, k, eps, _, _ = reparameterize(Drho, Dc, T, epsilon, r)
    # initial conditions
    lbd_vect_ini, adj_matr_ini = get_initials(L_full, lbd_spl, mu_fact, sig, Nmax_fact=Nmax_fact)
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

    lbd_vect, adj_matr = lbd_vect_ini, adj_matr_ini
    random_uniform = PrecompiledRandomGenerator(100_000)

    # initialize loop variables
    time_current = 0.0
    ind_current_lo = 0
    ind_current_hi = n_steps
        
    split_times_queue = [n_steps * dt]
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
            # update timing
            time_current = sol.t_events[0][0]
            ind_current_lo = int(time_current / dt) + 1

            if verbose: print(f'Interrupted by negative at {time_current = :.1f}')
            
            # perform merge
            lbd_vect, adj_matr = fix_negative_adaptive(sol.y_events[0][0], adj_matr)
            adj_matr = csr_matrix(adj_matr.toarray())
            if check_finished(lbd_vect, adj_matr, kappa, k, eps, lbd_spl):
                return time_current, True

        elif len(sol.y_events[1] > 0): # split threshold reached
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
            split_time = time_current +  np.sqrt(-2 / r / dLambdadt * np.log(1 - random_uniform.get_next()))
            
            # add time and split index to split queues
            split_times_queue, split_index_queue = insert_sorted2(split_times_queue, split_time, split_index_queue, split_index)
            if verbose: print(f'Interrupted by split at {time_current = :.1f}, {split_index = }, sampled new split time: {split_time:.2f}')
            
            # update first expected time of split
            ind_current_hi = int(split_times_queue[0] / dt) + 1
            # make sure we will cross it
            lbd_vect[split_index] += 1e-6

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
                ind_current_lo = int(time_current / dt) + 1
                ind_current_hi = int(split_times_queue[0] / dt) + 1
                if check_finished(lbd_vect, adj_matr, kappa, k, eps, lbd_spl):
                    return time_current, True

            else: # final time reached
                if verbose: print('Finishing up.')
                finished_bool = True

        _counter += 1
        if verbose: print(f'excluded indices: {[item for item in np.arange(stop=len(lbd_vect), dtype=int) if item not in included_inds]}')
        
    if check_finished(sol.y[:,-1], adj_matr, kappa, k, eps, lbd_spl):
        return time_current, True
    else:
        return time_current, False

# system parameters
# L_full = 100   # system size
Drho = 0.1      # fixed
Dc = 1.0        # fixed 
T = 5.0         # fixed
epsilon = 1e-1  # important
r = 1e-2        # important
lbd_spl = 20.0  # important

# initial conditions
mu_fact = 0.75  # part of lbd_spl to start at
sig = 5.0       # spread

# evolution parameters
n_steps = 10_000
dt = 1e4        # true dt is now adaptive anyway

kappa, k, eps, _, _ = reparameterize(Drho, Dc, T, epsilon, r)
lower_stab = -1/k * np.log(eps / k / kappa)

L_full_range = np.logspace(3.6, 1.7, 10, endpoint=True)

params_list = [(L_full, Drho, Dc, T, epsilon, r, lbd_spl, mu_fact * lbd_spl, sig, n_steps, dt) for L_full in L_full_range]

L_fulls, last_times, time_successs = [],[],[]

num_repeat = 1_000

for i, params in enumerate(params_list):
    for j in range(num_repeat):
        L_full, Drho, Dc, T, epsilon, r, lbd_spl, mu_fact, sig, n_steps, dt = params
        np.random.seed(4111*i + 121*j + 21)
        try:
            last_time, time_success = evolve_adapt_timestep_equi_time(*params, tqdm_bool=False, maxiter=5, Nmax_fact=3)
        except:
            last_time = dt * n_steps
            time_success = False
        
        L_fulls.append(L_full)
        last_times.append(last_time)
        time_successs.append(time_success)


np.save("L_fulls.npy", np.array(L_fulls))
np.save("last_times.npy", np.array(last_times))
np.save("time_successs.npy", np.array(time_successs))