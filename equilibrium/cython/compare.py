import argparse
import itertools as its
from functools import reduce
import numpy as np
import scipy.integrate as si
import scipy.stats as ss

import bajari.fsm.main as bajari
import util.util as util
from estimateparam import estimate_param


def compare(w, reputations):
    results = {w: {
        'reputations': reputations,
        'utilities': None,
        'prices': None
    }}

    try:
        n = reputations.size
    except AttributeError:
        reputations = np.array(reputations)
        n = reputations.size

    # load appropriate version of FSM method
    if n == 2:
        import dm.fsm.main as dm
    else:
        import dm.efsm.main as dm
        # estimate param
        param = estimate_param(w, reputations)
        if not param:
            return results

    # estimate lower and upper extremities
    lowers = np.empty(n, dtype=np.float)
    uppers = np.empty(n, dtype=np.float)
    for i in np.arange(n):
        lowers[i] = (1-w) * reputations[i]
        uppers[i] = (1-w) * reputations[i] + w

    # pick parameters for the common support case
    support = [lowers[0], uppers[-1]]
    bajari_params = []

    for i in np.arange(n):
        location = lowers[i] + w / 2
        scale = w / 4
        bajari_params.append({'location': location, 'scale': scale})

    # compute approximations
    if n == 2:
        dm_bids, dm_costs = dm.solve(w, reputations)
    else:
        dm_bids, dm_costs = dm.solve(w, reputations, param=param)

    bajari_bids, bajari_costs = bajari.solve(support, bajari_params)

    # ensure costs are monotonically increasing
    dm_costs, dm_bids = util.ensure_monotonicity(dm_costs, dm_bids)
    bajari_costs, bajari_bids = util.ensure_monotonicity(bajari_costs, bajari_bids)

    if n == 2:
        # precondition solutions: remove endpoints where fails
        # Lipschitz condition
        for c in dm_costs:
            diffs = list(map(lambda x,y: abs(x-y), c, dm_bids))
            for i in np.arange(len(diffs)):
                if diffs[i] <= 5e-5:
                    index = i
                    break

        try:
            dm_costs = np.array([c[:index] for c in dm_costs])
            dm_bids  = dm_bids[:index]
        except NameError:
            pass

    # interpolate bidding functions and their inverses
    dm_bid_funcs     = []
    bajari_bid_funcs = []
    dm_inverses      = []
    bajari_inverses  = []
    for i in np.arange(n):
        # fit
        dm_bid_func     = util.csplinefit(dm_costs[i], dm_bids)
        dm_inverse      = util.csplinefit(dm_bids, dm_costs[i])
        dm_bid_funcs.append(dm_bid_func)
        dm_inverses.append(dm_inverse)

        bajari_bid_func = util.csplinefit(bajari_costs[i], bajari_bids)
        bajari_inverse  = util.csplinefit(bajari_bids, bajari_costs[i])
        bajari_bid_funcs.append(bajari_bid_func)
        bajari_inverses.append(bajari_inverse)
        
    # compute ex-ante expected utilities
    params = [{'loc': lowers[i], 'scale': w} for i in np.arange(n)]
    cdfs   = [ss.uniform(**p) for p in params]

    dm_utils     = []
    bajari_utils = []
    common_costs = []
    for i in np.arange(n):
        # common costs
        costs = np.linspace(lowers[i], min(dm_costs[i][-1], bajari_costs[i][-1]), 1000)
        common_costs.append(costs)

        # derive expected utility functions
        js = [j for j in np.arange(n) if j != i]
        def dm_exp_util(x):
            bid   = dm_bid_funcs[i](x)
            probs = [(1 - cdfs[j].cdf(dm_inverses[j](bid))) for j in js]
            return (bid - x) * reduce(lambda x,y: x*y, probs)
        
        def bajari_exp_util(x):
            bid   = bajari_bid_funcs[i](x) 
            probs = [(1 - cdfs[j].cdf(bajari_inverses[j](bid))) for j in js]
            return (bid - x) * reduce(lambda x,y: x*y, probs)

        # compute ex-ante expected utilities
        dm_utilities     = np.array([dm_exp_util(x) * cdfs[i].pdf(x) for x in costs])
        bajari_utilities = np.array([bajari_exp_util(x) * cdfs[i].pdf(x) for x in costs])
        dm_util     = si.simps(dm_utilities, costs)
        bajari_util = si.simps(bajari_utilities, costs)
        dm_utils.append(dm_util)
        bajari_utils.append(bajari_util)

    # sample and generate prices for each auction
    size = 1000
    sampled_costs = []
    params = [{'loc': common_costs[i][0], 'scale': common_costs[i][-1]-common_costs[i][0]} for i in np.arange(n)]
    for p in params:
        loc   = p['loc']
        scale = p['scale']
        sampled_costs.append(ss.uniform.rvs(loc=loc, scale=scale, size=size))

    dm_prices = []
    for costs in zip(*sampled_costs):
        bids = [dm_bid_funcs[i](costs[i]) for i in np.arange(n)]
        dm_prices.append(min(bids))

    bajari_prices = []
    for costs in zip(*sampled_costs):
        bids = [bajari_bid_funcs[i](costs[i]) for i in np.arange(n)]
        bajari_prices.append(min(bids))

    results[w]['utilities'] = {'dm': dm_utils, 'cp': bajari_utils}
    results[w]['prices'] = {'dm': np.mean(dm_prices), 'cp': np.mean(bajari_prices)}

    return results


if __name__ == '__main__':                
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Compare auction models")
    parser.add_argument('--w', type=float, help='price weight')
    parser.add_argument('--reps', action='append', type=float, help='reputation array')
    args = parser.parse_args()
    w = args.w
    reputations = np.array(args.reps)

    results = compare(w, reputations)

    print(results)

