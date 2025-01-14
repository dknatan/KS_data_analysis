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

def get_MF_samples(num_samples, KS_params, r, lbd_spl, dx=1e-4, lbd_max=30, tol=1e-12, seed=0):

    kappa, k, eps = reparameterize(*KS_params)

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

def get_adj_matr_ini(Nmax, n_active, boundary_condition='periodic'):
    adj_matr = np.zeros((Nmax, Nmax), dtype=int)
    if boundary_condition == 'periodic':
        for i in range(n_active):
            adj_matr[i, i] = -2
            adj_matr[(i+1)%n_active, i] = 1
            adj_matr[i, (i+1)%n_active] = 1
    elif boundary_condition == 'neumann':
        adj_matr[0,0] = adj_matr[n_active-1, n_active-1] = -1
        adj_matr[0,1] = adj_matr[1,0] = 1
        for i in range(1,n_active-1):
            adj_matr[i, i] = -2
            adj_matr[i+1, i] = 1
            adj_matr[i, i+1] = 1
    else:
         raise ValueError(f'Unimplemented boundary conditions: {boundary_condition}')
    return csr_matrix(adj_matr)

def get_initial_state_chaos(Nmax, L_full, KS_params, r, lbd_spl, merge_threshold=1e-1, boundary_condition='periodic', seed=0):
    
    lbd_vect_ini = get_MF_samples(Nmax, KS_params, r, lbd_spl, seed=seed)
       
    too_big_inds = np.where(np.cumsum(lbd_vect_ini) > L_full)[0]
    lbd_vect_ini[too_big_inds] = 0.0
    lbd_vect_ini[too_big_inds[0]] = L_full - np.sum(lbd_vect_ini)

    adj_matr_ini = get_adj_matr_ini(Nmax, np.sum(lbd_vect_ini > 0), boundary_condition)

    # presplit
    while np.any(lbd_vect_ini > lbd_spl):
        lbd_vect_ini, adj_matr_ini = split(np.where(lbd_vect_ini > lbd_spl)[0][0], lbd_vect_ini, adj_matr_ini)
    # make sure all positive
    while np.any(lbd_vect_ini < 0):
        lbd_vect_ini, adj_matr_ini = merge(lbd_vect_ini, adj_matr_ini, merge_threshold)

    return shuffle_nonzero_elements(lbd_vect_ini, seed=seed), adj_matr_ini

def get_initial_state_gauss(Nmax, L_full, lbd_spl, mu, sig, merge_threshold=1e-1, boundary_condition='periodic', seed=0):

    rng = np.random.default_rng(seed)
    lbd_vect_ini = mu + sig * rng.standard_normal(Nmax)
       
    too_big_inds = np.where(np.cumsum(lbd_vect_ini) > L_full)[0]
    lbd_vect_ini[too_big_inds] = 0.0
    lbd_vect_ini[too_big_inds[0]] = L_full - np.sum(lbd_vect_ini)

    adj_matr_ini = get_adj_matr_ini(Nmax, np.sum(lbd_vect_ini > 0), boundary_condition)

    # presplit
    while np.any(lbd_vect_ini > lbd_spl):
        lbd_vect_ini, adj_matr_ini = split(np.where(lbd_vect_ini > lbd_spl)[0][0], lbd_vect_ini, adj_matr_ini)
    # make sure all positive
    while np.any(lbd_vect_ini < 0):
        lbd_vect_ini, adj_matr_ini = merge(lbd_vect_ini, adj_matr_ini, merge_threshold)

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

# def merge(lbd_vect, adj_matr):
#     while np.any(lbd_vect < 0): # here can put some constant to make it faster
#         ind_negative = np.argmin(lbd_vect)
#         inds_neighbors = sparse_find_neighbors(adj_matr, ind_negative) 
#         if len(inds_neighbors) == 2:
#             ind_neighbor1 = inds_neighbors[0]
#             ind_neighbor2 = inds_neighbors[1]
            
#             adj_matr[ind_neighbor1, ind_neighbor2] = 1
#             adj_matr[ind_neighbor2, ind_neighbor1] = 1
            
#             adj_matr[ind_neighbor1, ind_negative] = 0
#             adj_matr[ind_negative, ind_neighbor1] = 0
            
#             adj_matr[ind_negative, ind_neighbor2] = 0
#             adj_matr[ind_neighbor2, ind_negative] = 0

#             adj_matr[ind_negative, ind_negative] = 0

#             lbd_vect[ind_neighbor1] += 0.5 * lbd_vect[ind_negative]
#             lbd_vect[ind_neighbor2] += 0.5 * lbd_vect[ind_negative]

#         elif len(inds_neighbors) == 1:
#             ind_neighbor1 = inds_neighbors[0]
            
#             adj_matr[ind_neighbor1, ind_negative] = 0
#             adj_matr[ind_negative, ind_neighbor1] = 0
            
#             adj_matr[ind_negative, ind_negative] = 0
#             adj_matr[ind_neighbor1, ind_neighbor1] = -1

#             lbd_vect[ind_neighbor1] += lbd_vect[ind_negative]

#         else:
#             raise ValueError(f"Adjacency matrix has {len(inds_neighbors)} neighbors. ")

#         # set lbd_vect
#         lbd_vect[ind_negative] = 0
#         adj_matr[ind_negative, ind_negative] = 0

#     return lbd_vect, adj_matr


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
def get_event_merge(merge_threshold):

    def event_merge(t, y, adj_matr):
        return np.min(y[y != 0]) - merge_threshold
    
    event_merge.terminal = True
    event_merge.direction = -1

    return event_merge

def get_event_split(included_inds, lbd_spl):

    if len(included_inds) == 0:
        def event_split(t, y, adj_matr):
            return -1
    else:
        def event_split(t, y, adj_matr):
            return np.max(y[included_inds] - lbd_spl)
    
    event_split.terminal = True
    event_split.direction = 1

    return event_split

def get_event_split_single(ind, lbd_spl):
    def event_split(t, y, adj_matr):
        return y[ind] - lbd_spl
    
    event_split.terminal = True
    event_split.direction = 1

    return event_split

def merge(lbd_vect, adj_matr, merge_threshold):
    while np.any(np.isclose(lbd_vect, merge_threshold)):
        ind_negative = np.where(np.logical_and(np.isclose(lbd_vect, merge_threshold), lbd_vect != 0.0))[0][0]

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

def evolve_adapt_timestep(
        lbd_vect_ini, 
        adj_matr_ini,
        KS_params,
        r, 
        lbd_spl, 
        n_steps, 
        dt, 
        lbd_spl_hard, 
        merge_threshold=1e-1,
        save_states='grid', # can be 'grid', 'event', or 'end'
        verbose=False, 
        tqdm_bool=True, 
        maxiter=1_000, 
        seed=0
    ):

    kappa, k, eps = reparameterize(*KS_params)

    def dydt(t, y, adj_matr): # number of additional arguments to conform to the solve_ivp
        return 0.5 * adj_matr @ (kappa * np.exp(-k * y) + eps * y)

    def jac_dydt(t, y, adj_matr):
        return 0.5 * adj_matr @ np.diag((-k * kappa * np.exp(-k * y) + eps))

    # initialize save variables
    if save_states == 'grid':
        t_range, lbd_vect_t, adj_matr_t = np.arange(0, dt * n_steps, dt), np.zeros((len(lbd_vect_ini), n_steps)), np.zeros((n_steps), dtype=object)
        lbd_vect_t[:, 0] = lbd_vect_ini.copy()
        adj_matr_t[0] = adj_matr_ini.copy()
    elif save_states == 'event':
        t_range, lbd_vect_t, adj_matr_t = [0.0], [lbd_vect_ini.copy()], [adj_matr_ini.copy()] # np.arange(0, dt * n_steps, dt), np.zeros((Nmax, n_steps)), np.zeros((n_steps), dtype=object)
    else:
        t_range, lbd_vect_t, adj_matr_t = [0.0], [lbd_vect_ini.copy()], [adj_matr_ini.copy()] # np.arange(0, dt * n_steps, dt), np.zeros((Nmax, n_steps)), np.zeros((n_steps), dtype=object)

    random_uniform = PrecompiledRandomGenerator(100_000, seed=seed)

    # initialize loop variables
    lbd_vect, adj_matr = lbd_vect_ini.copy(), adj_matr_ini.copy()
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

        if save_states == 'grid':
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
                        t_eval=t_eval_current if save_states == 'grid' else None, 
                        events=[get_event_merge(merge_threshold)] + [get_event_split(np.where(lbd_vect > lbd_spl)[0], lbd_spl_hard)] + [get_event_split_single(ind, lbd_spl) for ind in included_inds if ind not in split_index_queue], 
                        args=[adj_matr],
                        jac=jac_dydt,
                        rtol=1e-3, 
                        atol=1e-5) # also can help: max_step (parameter)
        
        if sol.status == -1:
            if verbose: print(f'Solver failed. Message: {sol.message}.\n{sol}')
            raise ValueError(f'Solver failed. Message: {sol.message}.')

        elif sol.status == 1:
            # determine which event happened
            event_ind = [ind for ind in np.arange(1 + 1 + len(included_inds) - len(split_index_queue) + 1, dtype=int) if len(sol.y_events[ind]) > 0][0]

            if save_states == 'grid':
                lbd_vect_t[:, ind_current_lo:int(sol.t_events[event_ind][0] / dt) + 1] = sol.y[:, min(_counter, 1):]
                adj_matr_t[ind_current_lo:int(sol.t_events[event_ind][0] / dt) + 1] = adj_matr.copy()
            
            elif save_states == 'event':
                t_range.append(sol.t_events[event_ind][0])
                lbd_vect_t.append(sol.y_events[event_ind][0])
                adj_matr_t.append(adj_matr.copy())
                
            # update timing
            time_current = sol.t_events[event_ind][0]
            ind_current_lo = int(time_current / dt) + 1
            lbd_vect = sol.y_events[event_ind][0]

            if event_ind == 0: # merging event
                if verbose: print(f'Interrupted by merge at {time_current = :.1f}')
                
                # perform merge
                lbd_vect, adj_matr = merge(lbd_vect, adj_matr, merge_threshold)

                # when merging happens at Lambda_int > 0, a Lambda can jump the split threshold
                new_split_inds = set(np.where(lbd_vect > lbd_spl)[0]) - set(split_index_queue)
                for split_index in new_split_inds:
                    # sample time-to-split: inv_P(uniform) is a sampled time-to-split (approx)
                    dLambdadt = max(dydt(0.0, lbd_vect, adj_matr)[split_index], 1e-9)
                    split_time = time_current + np.sqrt(-2 / r / dLambdadt * np.log(1 - random_uniform.get_next()))
                    
                    # add time and split index to split queues
                    split_times_queue, split_index_queue = insert_sorted2(split_times_queue, split_time, split_index_queue, split_index)
                    
                    if verbose: print(f'A split also happened at {time_current = :.1f}, {split_index = }, sampled new split time: {split_time:.2f}')

                    # update first expected time of split
                    ind_current_hi = int(split_times_queue[0] / dt) + 1

                # same for hard split
                new_hard_split_inds = np.where(lbd_vect > lbd_spl_hard)[0]
                for split_index in new_hard_split_inds:
                    # and split
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
                    if verbose: print(f'A hard split also happened at {time_current = :.1f}, {split_index = }, sampled new split time: {split_time:.2f}')

            elif event_ind == 1: # hard split event
                if verbose: print(f'Interrupted by hard split at {time_current = :.1f}')
                # determine where split happened and split
                split_index = np.argmin(np.abs(lbd_vect - lbd_spl_hard))
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
                
                # sample time-to-split: inv_P(uniform) is a sampled time-to-split (approx)
                dLambdadt = dydt(0.0, lbd_vect, adj_matr)[split_index] 
                split_time = time_current + np.sqrt(-2 / r / dLambdadt * np.log(1 - random_uniform.get_next()))
                
                # add time and split index to split queues
                split_times_queue, split_index_queue = insert_sorted2(split_times_queue, split_time, split_index_queue, split_index)
                if verbose: print(f'Interrupted by split at {time_current = :.1f}, {split_index = }, sampled new split time: {split_time:.2f}')
                
                # update first expected time of split
                ind_current_hi = int(split_times_queue[0] / dt) + 1
            
            # for event based saving, also save after event
            if save_states == 'event':
                t_range.append(sol.t_events[event_ind][0])
                lbd_vect_t.append(sol.y_events[event_ind][0])
                adj_matr_t.append(adj_matr.copy())

        elif sol.status == 0:
            finished_bool = split_index_queue[0] == None # if no more splitting left we are done

            if save_states == 'grid':
                lbd_vect_t[:, ind_current_lo:ind_current_hi] = sol.y[:, min(_counter, 1):] if finished_bool else sol.y[:, min(_counter, 1):-1]
                adj_matr_t[ind_current_lo:ind_current_hi] = adj_matr.copy()
            
            if not finished_bool: # splitting time reached
                if save_states == 'event':
                    t_range.append(sol.t[-1])
                    lbd_vect_t.append(sol.y[:, -1])
                    adj_matr_t.append(adj_matr.copy())

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
                        
                # for event based saving, also save after event
                if save_states == 'event':
                    t_range.append(sol.t[-1])
                    lbd_vect_t.append(lbd_vect)
                    adj_matr_t.append(adj_matr.copy())

        _counter += 1
        if verbose: print(f'excluded indices: {[item for item in np.arange(stop=len(lbd_vect), dtype=int) if item not in included_inds]}')
    
    return t_range, lbd_vect_t, adj_matr_t

def evolve_adapt_timestep_hard_split(
        lbd_vect_ini, 
        adj_matr_ini, 
        KS_params,
        r, 
        lbd_spl, 
        n_steps, 
        dt, 
        merge_threshold=1e-1,
        save_states='grid', # can be 'grid', 'event', or 'end'
        verbose=False, 
        tqdm_bool=True, 
        maxiter=1_000, 
    ):

    kappa, k, eps = reparameterize(*KS_params)
    
    def dydt(t, y, adj_matr): # number of additional arguments to conform to the solve_ivp
        return 0.5 * adj_matr @ (kappa * np.exp(-k * y) + eps * y)

    def jac_dydt(t, y, adj_matr):
        return 0.5 * adj_matr @ np.diag((-k * kappa * np.exp(-k * y) + eps))

    # initialize save variables
    if save_states == 'grid':
        t_range, lbd_vect_t, adj_matr_t = np.arange(0, dt * n_steps, dt), np.zeros((len(lbd_vect_ini), n_steps)), np.zeros((n_steps), dtype=object)
        lbd_vect_t[:, 0] = lbd_vect_ini.copy()
        adj_matr_t[0] = adj_matr_ini.copy()
    elif save_states == 'event':
        t_range, lbd_vect_t, adj_matr_t = [0.0], [lbd_vect_ini.copy()], [adj_matr_ini.copy()] # np.arange(0, dt * n_steps, dt), np.zeros((Nmax, n_steps)), np.zeros((n_steps), dtype=object)
    else:
        t_range, lbd_vect_t, adj_matr_t = [0.0], [lbd_vect_ini.copy()], [adj_matr_ini.copy()] # np.arange(0, dt * n_steps, dt), np.zeros((Nmax, n_steps)), np.zeros((n_steps), dtype=object)

    # initialize loop variables
    lbd_vect, adj_matr = lbd_vect_ini.copy(), adj_matr_ini.copy()
    time_current = 0.0
    ind_current_lo = 0
    ind_current_hi = n_steps
        
    included_inds = [ind for ind in np.arange(stop=len(lbd_vect), dtype=int) if lbd_vect[ind] > 0.0]
    finished_bool = False
    
    _iter = tqdm(range(maxiter), desc='Worst case timer: ') if tqdm_bool else range(maxiter)
    for _counter in _iter:
        # check if finished
        if finished_bool:
            break

        if save_states == 'grid':
            if _counter == 0:
                t_eval_current = t_range
            else:
                # insert current time to evaluation array to ensure potential event occurs on the evaluated interval
                t_eval_current = np.insert(t_range[ind_current_lo:ind_current_hi], 0, time_current)
                
        if verbose: print(f'# active: {np.sum(lbd_vect > 0)}')
        
        # evolve and (maybe) be interrupted 
        sol = solve_ivp(dydt, 
                        t_span=(time_current, dt * n_steps), 
                        y0=lbd_vect, 
                        method='BDF',
                        t_eval=t_eval_current if save_states == 'grid' else None, 
                        events=[get_event_merge(merge_threshold)] + [get_event_split_single(ind, lbd_spl) for ind in included_inds], 
                        args=(adj_matr),
                        jac=jac_dydt,
                        rtol=1e-6) # also can help: max_step (parameter)
        
        if sol.status == -1:
            if verbose: print(f'Solver failed. Message: {sol.message}.\n{sol}')
            raise ValueError(f'Solver failed. Message: {sol.message}.')

        elif sol.status == 1:
            # determine which event happened
            event_ind = [ind for ind in np.arange(1 + len(included_inds), dtype=int) if len(sol.y_events[ind]) > 0][0]

            if save_states == 'grid':
                lbd_vect_t[:, ind_current_lo:int(sol.t_events[event_ind][0] / dt) + 1] = sol.y[:, min(_counter, 1):]
                adj_matr_t[ind_current_lo:int(sol.t_events[event_ind][0] / dt) + 1] = adj_matr.copy()
            
            elif save_states == 'event':
                t_range.append(sol.t_events[event_ind][0])
                lbd_vect_t.append(sol.y_events[event_ind][0])
                adj_matr_t.append(adj_matr.copy())
                
            # update timing
            time_current = sol.t_events[event_ind][0]
            ind_current_lo = int(time_current / dt) + 1
            lbd_vect = sol.y_events[event_ind][0]

            if event_ind == 0: # merging event
                if verbose: print(f'Interrupted by merge at {time_current = :.1f}')
                
                # perform merge
                lbd_vect, adj_matr = merge(lbd_vect, adj_matr, merge_threshold)

                # when merging happens at Lambda_int > 0, a Lambda can jump the split threshold
                new_split_inds = set(np.where(lbd_vect > lbd_spl)[0])
                for split_index in new_split_inds:
                    # and split
                    lbd_vect, adj_matr = split(split_index, lbd_vect, adj_matr)
                    
                    # after splitting there are more plateaus
                    included_inds = [ind for ind in np.arange(stop=len(lbd_vect), dtype=int) if lbd_vect[ind] > 0.0]                    
                    if verbose: print(f'A hard split happened at {time_current = :.1f}, {split_index = }')

            elif event_ind == 1: # hard split event
                if verbose: print(f'Interrupted by hard split at {time_current = :.1f}')
                # determine where split happened and split
                split_index = np.argmin(np.abs(lbd_vect - 1.5*lbd_spl))
                lbd_vect, adj_matr = split(split_index, lbd_vect, adj_matr)
                
                # after splitting there are more plateaus
                included_inds = [ind for ind in np.arange(stop=len(lbd_vect), dtype=int) if lbd_vect[ind] > 0.0]

            else: # split threshold reached
                if verbose: print(f'Interrupted by hard split at {time_current = :.1f}')
                # determine where split happened and split
                split_index = included_inds[event_ind - 1]
                lbd_vect, adj_matr = split(split_index, lbd_vect, adj_matr)
                
                # after splitting there are more plateaus
                included_inds = [ind for ind in np.arange(stop=len(lbd_vect), dtype=int) if lbd_vect[ind] > 0.0]
            
            # for event based saving, also save after event
            if save_states == 'event':
                t_range.append(sol.t_events[event_ind][0])
                lbd_vect_t.append(sol.y_events[event_ind][0])
                adj_matr_t.append(adj_matr.copy())

        elif sol.status == 0:
            finished_bool = True # if no more splitting left we are done

            if save_states == 'grid':
                lbd_vect_t[:, ind_current_lo:ind_current_hi] = sol.y[:, min(_counter, 1):] if finished_bool else sol.y[:, min(_counter, 1):-1]
                adj_matr_t[ind_current_lo:ind_current_hi] = adj_matr.copy()
            

        _counter += 1
        if verbose: print(f'excluded indices: {[item for item in np.arange(stop=len(lbd_vect), dtype=int) if item not in included_inds]}')
    
    return t_range, lbd_vect_t, adj_matr_t


# document

# end