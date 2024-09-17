import numpy as np
from scipy.optimize import fsolve
from tqdm import tqdm
import sscfw.general

# basic functions
def rate(lbd, lbd_spl, r):
    return np.maximum(0, r * (lbd - lbd_spl))

def g(lbd, kappa, k, eps):
	if eps == 0.0:
		return kappa * np.exp(-k * lbd)
	else: 
		return kappa * np.exp(-k * lbd) + eps * lbd

# reparameterizion functions
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

def _epsilon_lbd_spl(lbd_spl):
    if lbd_spl == np.inf:
        return 0.0 
    else:
        return np.exp(-1.13734914 / 0.3579829) * lbd_spl**(- 1 / 0.3579829)
epsilon_lbd_spl = np.vectorize(_epsilon_lbd_spl)

def epsilon_from_eps(eps, Drho=0.1, Dc=1.0, T=5.0):
    return 0.25 * eps * np.exp(np.sqrt(3 * T / Drho))

def antireparameterize(kappa, k, eps, lbd_spl, r):
    Dc = 1 / k**2
    epsilon = epsilon_lbd_spl(lbd_spl)
    TM_inv = 1 / kappa - k / 4
    Drho = k / 2 / (TM_inv * np.log(4 * epsilon / eps))
    T = 1 / (TM_inv**2 * 12 * Drho * Dc)
    return Drho, Dc, T, epsilon, r

# more robust root finding
def find_roots_rich(func, array, args):
    """
    Find the roots of a function using an array of values and `fsolve`.

    This function locates the roots of a given function `func` by first evaluating 
    the function over an array of values `array` and then identifying where the 
    function changes sign, which suggests the presence of a root. These sign 
    changes are used as initial guesses for the `fsolve` function from `scipy.optimize` 
    to accurately compute the roots.

    Parameters:
    ----------
    func : callable
        The function whose roots are to be found. Must take `array` as the first argument
        and `args` as additional arguments.
    array : numpy.ndarray
        An array of values over which to evaluate the function. The function will be 
        evaluated at each point in this array.
    args : tuple
        Additional arguments to pass to the function `func`.

    Returns:
    -------
    numpy.ndarray or list
        An array of the roots found by `fsolve`, or [None] if the solver was unsuccessful.

    Notes:
    -----
    - The function assumes that the roots are simple (i.e., the function crosses the x-axis).
    - If the `fsolve` solver fails to find the roots, the function returns a list containing `None`.
    """
    # evaluate on array 
    func_array = func(array, *args)
    
    # find indices where sign change
    sign_change_inds = np.where(func_array[1:]*func_array[:-1] < 0.0)
        
    # use those as starting points for fsolve 
    roots = fsolve(func, array[sign_change_inds], args=args, full_output=True)

    # return roots in case of success
    if roots[-2]:
        return roots[0]
    else:
        return [None]
    
# scatter plot of double valued functions
def flexible_scatter(fig, ax, x_array, y_array, *args, **kwargs):
    """
    Create a scatter plot that adapts to varying lengths and structures of `x_array` and `y_array`.

    This function generates a scatter plot where `x_array` and `y_array` can be standard 
    arrays (or other iterable types) or lists of iterables with varying lengths. It handles 
    cases where either `x_array` or `y_array` is a list of iterables, flattening and 
    repeating the corresponding values as needed to match the dimensions, and then plots 
    them using the `ax.scatter()` method. If both `x_array` and `y_array` are standard 
    iterables, it simply plots them directly.

    Parameters:
    ----------
    fig : matplotlib.figure.Figure
        The figure object that the plot will be a part of.
    ax : matplotlib.axes.Axes
        The axes object where the scatter plot will be drawn.
    x_array : iterable or list of iterables
        The x-coordinates of the points. This can be a single iterable (e.g., list, tuple, 
        numpy array) or a list of iterables with varying lengths.
    y_array : iterable or list of iterables
        The y-coordinates of the points. This can be a single iterable (e.g., list, tuple, 
        numpy array) or a list of iterables with varying lengths.
    *args : tuple
        Additional positional arguments to pass to `ax.scatter()`.
    **kwargs : dict
        Additional keyword arguments to pass to `ax.scatter()`.

    Returns:
    -------
    matplotlib.collections.PathCollection
        The scatter plot object created by `ax.scatter()`.

    Notes:
    -----
    - If `y_array` consists of multiple iterables, the function repeats `x_array` values 
      to match the corresponding lengths and then flattens `y_array` before plotting.
    - If `x_array` consists of multiple iterables, the function repeats `y_array` values 
      to match the corresponding lengths and then flattens `x_array` before plotting.
    - If both `x_array` and `y_array` are single iterables, the function plots them directly.

    Examples:
    --------
    # Example with varying y_array lengths
    x = [1, 2, 3]
    y = [[4, 5], [6], [7, 8, 9]]
    flexible_scatter(fig, ax, x, y)

    # Example with varying x_array lengths
    x = [[1, 2], [3], [4, 5, 6]]
    y = [7, 8, 9]
    flexible_scatter(fig, ax, x, y)

    # Example with standard arrays
    x = [1, 2, 3]
    y = [4, 5, 6]
    flexible_scatter(fig, ax, x, y)
    """
    try:
        if isinstance(y_array, list) and all(isinstance(y, (list, tuple, np.ndarray)) for y in y_array):
            # Handle case where y_array is a list of iterables
            flattened_y = np.concatenate([np.asarray(y) for y in y_array])
            repeated_x = np.repeat(x_array, [len(y) for y in y_array])
            return ax.scatter(repeated_x, flattened_y, *args, **kwargs)
        elif isinstance(x_array, list) and all(isinstance(x, (list, tuple, np.ndarray)) for x in x_array):
            # Handle case where x_array is a list of iterables
            flattened_x = np.concatenate([np.asarray(x) for x in x_array])
            repeated_y = np.repeat(y_array, [len(x) for x in x_array])
            return ax.scatter(flattened_x, repeated_y, *args, **kwargs)
        else:
            # Handle the standard case where both x_array and y_array are single iterables
            return ax.scatter(x_array, y_array, *args, **kwargs)
    except Exception as e:
        raise ValueError(f"Error in flexible_scatter: {e}")
    
@sscfw.general.cache_numpys(num_of_returns=2)
def fixed_point_continuation(func, x_ini, p_old, p_box, p_step, dist_func, dist_tol, maxiter, p_step_decrease_multiplier, p_step_increase_multiplier):
    """
    Perform a fixed-point continuation to trace solutions of a parameterized function.

    This function uses a continuation method to follow the solution curve of a parameterized 
    nonlinear equation defined by `func`. Starting from an initial solution `x_ini` at parameter 
    value `p_old`, the function iteratively adjusts the parameter and solves for the new solution, 
    continuing until the parameter reaches the boundaries of the specified parameter box `p_box` 
    or the maximum number of iterations `maxiter` is reached.

    Parameters:
    ----------
    func : callable
        The function to be solved. It should take the current solution `x` and parameters `p` as arguments.
    x_ini : array-like
        Initial guess for the solution.
    p_old : array-like
        Initial parameter values for the continuation process.
    p_box : tuple of array-like
        A tuple (p_min, p_max) specifying the lower and upper bounds for the parameters.
    p_step : array-like
        Initial step size for adjusting the parameters during continuation.
    dist_func : callable
        A distance function to measure the difference between successive solutions. It should take 
        two solutions `x_new` and `x_old` as arguments and return a scalar distance.
    dist_tol : float
        Tolerance for the distance between successive solutions. If the distance exceeds this value, 
        the parameter step size is reduced.
    maxiter : int
        Maximum number of iterations for the continuation process.
    p_step_decrease_multiplier : float
        Factor by which the parameter step size is multiplied if the distance between successive 
        solutions exceeds `dist_tol`.
    p_step_increase_multiplier : float
        Factor by which the parameter step size is multiplied if the distance between successive 
        solutions is within `dist_tol`.
    
    Returns:
    -------
    xs : numpy.ndarray
        Array of solutions corresponding to each parameter value during the continuation process.
    ps : numpy.ndarray
        Array of parameter values corresponding to each solution during the continuation process.
    
    Raises:
    -------
    AssertionError
        If the initial solution `x_ini` at `p_old` is not found precisely by `fsolve`.

    Notes:
    -----
    - The function assumes that the parameters `p` are updated in logarithmic scale.
    - The continuation process is terminated when the parameters reach the boundary defined by `p_box`
      or when the maximum number of iterations `maxiter` is reached.
    - Progress is displayed using `tqdm`, which shows the worst-case estimated time based on `maxiter`.
    """
    # initialize
    full_sol_old = fsolve(func=func, 
                       x0=x_ini,
                       args=tuple(p_old),
                       full_output=1)
    assert full_sol_old[-2] == 1, 'Initial point not located precisely'
    x_old = np.array(full_sol_old[0])
    xs, ps = [x_old], [p_old]
    
    # loop
    _iter = tqdm(range(maxiter), desc='Worst case timer: ', ncols=130)
    for i in _iter:
        if not (np.all(p_old > p_box[0]) and np.all(p_old < p_box[1])):
            print('Success: Parameter box edge reached. ')
            break
        # update p (in logscale)
        p_new = p_old * (np.ones_like(p_old) + p_step)
        
        # solve at new p 
        full_sol_new = fsolve(func=func, 
                           x0=x_old,
                           args=tuple(p_new),
                           full_output=1)
        x_new = full_sol_new[0]
        
        # if not success decrease p_step  
        if dist_func(x_new, x_old) > dist_tol:
            p_step *= p_step_decrease_multiplier
        
        # if success save and proceed
        else:
            x_old, p_old = x_new, p_new
            xs.append(x_new)
            ps.append(p_new)
            p_step *= p_step_increase_multiplier
        
        tqdm.set_postfix(_iter, rel_step_size=f'{np.mean(p_step):.2e}')   
    if i == maxiter-1:
        print(f'Terminated at i = {maxiter}. Parameter box edge NOT reached.')
    
    return np.asarray(xs), np.asarray(ps)

def balanced_relative_dist(x1, x2):
    return np.mean(np.abs((x1 - x2) / x2))
    