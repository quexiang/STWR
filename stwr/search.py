# Bandwidth optimization methods (golden section and interval search)

__author__ = "Taylor Oshan tayoshan@gmail.com"

import numpy as np
from copy import deepcopy


def golden_section(a, c, delta, function, tol, max_iter, bw_max,
                   int_score=False, verbose=False):
    """
    Golden section search for optimal bandwidth.

    Narrows the search interval [a, c] by evaluating the criterion
    function at two internal points b and d, then discarding the
    portion that does not contain the minimum.

    Reference: Fotheringham et al. (2002), Section 9.6.4, p. 212.

    Parameters
    ----------
    a               : float
                      Lower bound of the initial search interval.
    c               : float or np.inf
                      Upper bound of the initial search interval.
    delta           : float
                      Golden ratio proportion (typically 0.38197).
    function        : callable
                      Objective function f(bw) -> score.
    tol             : float
                      Convergence tolerance on the score difference.
    max_iter        : int
                      Maximum number of iterations.
    bw_max          : float
                      Upper bound for bandwidth.
    int_score       : bool
                      If True, round bandwidths to integers (for adaptive
                      nearest-neighbor bandwidths).
    verbose         : bool
                      If True, print progress at each iteration.

    Returns
    -------
    opt_val         : float
                      Optimal bandwidth value.
    opt_score       : float
                      Criterion score at the optimum.
    output          : list of (float, float)
                      Search history: (bandwidth, score) at each iteration.
    """
    if c == np.inf:
        b = a + delta * np.abs(n - a)
        d = n - delta * np.abs(n - a)
    else:
        b = a + delta * np.abs(c - a)
        d = c - delta * np.abs(c - a)
    
    opt_score = np.inf
    diff = 1.0e9
    iters = 0
    output = []
    dict = {}
    while np.abs(diff) > tol and iters < max_iter and a != np.inf:
        iters += 1
        if int_score:
            b = np.round(b)
            d = np.round(d)

        if b in dict:
            score_b = dict[b]
        else:
            score_b = function(b)
            dict[b] = score_b
            if verbose:
                print("Bandwidth: ", np.round(b, 2), ", score: ",
                      "{0:.2f}".format(score_b[0]))

        if d in dict:
            score_d = dict[d]
        else:
            score_d = function(d)
            dict[d] = score_d
            if verbose:
                print("Bandwidth: ", np.round(d, 2), ", score: ",
                      "{0:.2f}".format(score_d[0]))

        if score_b <= score_d:
            opt_val = b
            opt_score = score_b
            c = d
            d = b
            b = a + delta * np.abs(c - a)

        else:
            opt_val = d
            opt_score = score_d
            a = b
            b = d
            d = c - delta * np.abs(c - a)

        output.append((opt_val, opt_score))
        
        opt_val = np.round(opt_val, 2)
        if (opt_val, opt_score) not in output:
            output.append((opt_val, opt_score))
        
        diff = score_b - score_d
        score = opt_score
        
    
    if a == np.inf or bw_max == np.inf:
        score_ols = function(np.inf)
        output.append((np.inf, score_ols))
            
        if score_ols <= opt_score:
            opt_score = score_ols
            opt_val = np.inf
        
        if verbose:
            print("Bandwidth: ", np.inf, ", score: ",
                    "{0:.2f}".format(score_ols[0]))

    return opt_val, opt_score, output


def equal_interval(l_bound, u_bound, interval, function, int_score=False,
                   verbose=False):
    """
    Brute-force interval search for optimal bandwidth.

    Evaluates the criterion function at every ``interval`` step
    between ``l_bound`` and ``u_bound``, taking the minimum.

    Parameters
    ----------
    l_bound         : float
                      Lower bound of the search interval.
    u_bound         : float
                      Upper bound of the search interval.
    interval        : float
                      Step size between evaluations.
    function        : callable
                      Objective function f(bw) -> score.
    int_score       : bool
                      If True, round bandwidths to integers.
    verbose         : bool
                      If True, print progress.

    Returns
    -------
    opt_val         : float
                      Optimal bandwidth value.
    opt_score       : float
                      Criterion score at the optimum.
    output          : list of (float, float)
                      Search history.
    """
    a = l_bound
    c = u_bound
    b = a + interval
    if int_score:
        a = np.round(a, 0)
        c = np.round(c, 0)
        b = np.round(b, 0)

    output = []

    score_a = function(a)
    if verbose:
        print(score_a)
        print("Bandwidth:", a, ", score:", "{0:.2f}".format(score_a[0]))

    output.append((a, score_a))

    opt_val = a
    opt_score = score_a

    while b < c:
        score_b = function(b)
        if verbose:
            print("Bandwidth:", b, ", score:", "{0:.2f}".format(score_b[0]))
        output.append((b, score_b))

        if score_b < opt_score:
            opt_val = b
            opt_score = score_b
        b = b + interval

    score_c = function(c)
    if verbose:
        print("Bandwidth:", c, ", score:", "{0:.2f}".format(score_c[0]))

    output.append((c, score_c))

    if score_c < opt_score:
        opt_val = c
        opt_score = score_c

    return opt_val, opt_score, output


def multi_bw(init, y, X, n, k, family, tol, max_iter, rss_score, gwr_func,
             bw_func, sel_func, multi_bw_min, multi_bw_max, bws_same_times,
             verbose=False):
    """
    Multiscale GWR bandwidth search via iterative GAM backfitting.

    For each covariate j, the partial residual (XB_j + error) is
    regressed against X_j to find the optimal bandwidth for that
    covariate. This repeats until the bandwidths stabilize or the
    score of change (SOC) falls below the tolerance.

    Parameters
    ----------
    init : float or None
        Initial bandwidth (None auto-derives from GWR).
    y : array (n, 1)
        Dependent variable.
    X : array (n, k)
        Design matrix.
    n : int
        Number of observations.
    k : int
        Number of covariates.
    family : family object
        Probability model.
    tol : float
        Convergence tolerance for the SOC.
    max_iter : int
        Maximum backfitting iterations.
    rss_score : bool
        If True, use RSS-based SOC; otherwise smooth function.
    gwr_func : callable
        Function to fit a GWR model: f(y, X, bw) -> result.
    bw_func : callable
        Function to create a Sel_BW instance.
    sel_func : callable
        Function to run bandwidth search on a Sel_BW instance.
    multi_bw_min : list
        Per-covariate minimum bandwidth values.
    multi_bw_max : list
        Per-covariate maximum bandwidth values.
    bws_same_times : int
        Early-stopping threshold: stop if all bandwidths remain
        unchanged for this many consecutive iterations.
    verbose : bool
        If True, print iteration progress.

    Returns
    -------
    tuple : (opt_bws, BWs_history, scores, params, err, gwr_sel_hist, bw_gwr)
    """
    if init is None:
        bw = sel_func(bw_func(y, X))
        optim_model = gwr_func(y, X, bw)
    else:
        bw = init
        optim_model = gwr_func(y, X, init)
    bw_gwr = bw
    err = optim_model.resid_response.reshape((-1, 1))
    param = optim_model.params

    XB = np.multiply(param, X)
    if rss_score:
        rss = np.sum((err)**2)
    iters = 0
    scores = []
    delta = 1e6
    BWs = []
    bw_stable_counter = 0
    bws = np.empty(k)
    gwr_sel_hist = []

    try:
        from tqdm.auto import tqdm  #if they have it, let users have a progress bar
    except ImportError:

        def tqdm(x, desc=''):  #otherwise, just passthrough the range
            return x

    for iters in tqdm(range(1, max_iter + 1), desc='Backfitting'):
        new_XB = np.zeros_like(X)
        params = np.zeros_like(X)

        for j in range(k):
            temp_y = XB[:, j].reshape((-1, 1))
            temp_y = temp_y + err
            temp_X = X[:, j].reshape((-1, 1))
            bw_class = bw_func(temp_y, temp_X)

            if bw_stable_counter >= bws_same_times:
                #If in backfitting, all bws not changing in bws_same_times (default 5) iterations
                bw = bws[j]
            else:
                bw = sel_func(bw_class, multi_bw_min[j], multi_bw_max[j])
                gwr_sel_hist.append(deepcopy(bw_class.sel_hist))

            optim_model = gwr_func(temp_y, temp_X, bw)
            err = optim_model.resid_response.reshape((-1, 1))
            param = optim_model.params.reshape((-1, ))
            new_XB[:, j] = optim_model.predy.reshape(-1)
            params[:, j] = param
            bws[j] = bw
    
        #If bws remain the same as from previous iteration
        if (iters > 1) and np.all(BWs[-1] == bws):
            bw_stable_counter += 1
        else:
            bw_stable_counter = 0
    
        num = np.sum((new_XB - XB)**2) / n
        den = np.sum(np.sum(new_XB, axis=1)**2)
        score = (num / den)**0.5
        XB = new_XB

        if rss_score:
            predy = np.sum(np.multiply(params, X), axis=1).reshape((-1, 1))
            new_rss = np.sum((y - predy)**2)
            score = np.abs((new_rss - rss) / new_rss)
            rss = new_rss
        scores.append(deepcopy(score))
        delta = score
        BWs.append(deepcopy(bws))

        if verbose:
            print("Current iteration:", iters, ",SOC:", np.round(score, 7))
            print("Bandwidths:", ', '.join([str(bw) for bw in bws]))

        if delta < tol:
            break

    opt_bws = BWs[-1]
    return (opt_bws, np.array(BWs), np.array(scores), params, err, gwr_sel_hist, bw_gwr)