#!/usr/bin/env python2.7
# encoding: utf-8
"""
asymmetries.py

Created by Jakub Konka on 2012-05-20.
Copyright (c) 2012 University of Strathclyde. All rights reserved.
"""
from __future__ import division
import sys
import os
import decimal as dc
import numpy as np
import scipy.integrate as integrate
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import warnings
import progressbar as pb


rc('font',**{'family':'sans-serif','sans-serif':['Gill Sans']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14, 'legend.fontsize': 14})

# Ignore RuntimeWarnings for now
warnings.simplefilter("ignore", RuntimeWarning)
# Widgets for ProgressBar
widgets = ['Calculating: ', pb.Percentage(), pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]


def estimate_cost_hat_range():
  costs = np.linspace(0, 1, 1000)
  reps = np.linspace(0, 1, 1000)
  w_range = np.linspace(0.01, 1, 1000)
  cost_hat = lambda c, r, w: (1-w)*r + w*c
  bounds = {'lower': [], 'upper': []}
  for w in w_range:
    for r in reps:
      costs_hat = [cost_hat(c, r, w) for c in costs]
      bounds['lower'] += [min(costs_hat)]
      bounds['upper'] += [max(costs_hat)]
  print("Closure: [{}, {}]".format(min(bounds['lower']), max(bounds['upper'])))


def plot_cost_hat_dist(w, r1, r2):
  t = np.linspace(0, 1, 1000)
  # Construct the theoretical CDF
  # Player 1 (r1)
  f1 = [0 for val in t if val <= (1-w)*r1]
  f2 = [(val - (1-w)*r1)/w for val in t if ((1-w)*r1 < val) and (val <= w + (1-w)*r1)]
  f3 = [1 for val in t if val > w + (1-w)*r1]
  f = f1 + f2 + f3
  # Player 2 (r2)
  g1 = [0 for val in t if val <= (1-w)*r2]
  g2 = [(val - (1-w)*r2)/w for val in t if ((1-w)*r2 < val) and (val <= w + (1-w)*r2)]
  g3 = [1 for val in t if val > w + (1-w)*r2]
  g = g1 + g2 + g3
  # Plot it
  plt.figure()
  plt.plot(t, f, '-', linewidth=2)
  plt.plot(t, g, '--', linewidth=2)
  plt.xlabel(r"Cost, $c_i$")
  plt.ylabel(r"CDF, $G_i$")
  plt.legend([r"Player 1", r"Player 2"], loc="lower right")
  plt.grid()
  plt.show()
  

def estimate_bid_functions(w, reps, granularity=1000):
  # Calculate params
  v1 = [(1-w)*reps[0], (1-w)*reps[0] + w]
  v2 = [(1-w)*reps[1], (1-w)*reps[1] + w]
  # Check whether nontrivial NE
  if (v2[1] >= v1[1]):
    if (v1[1] <= 2*v2[0] - v2[1]):
      graph_vf1 = np.linspace(v1[0], v1[1], granularity)
      bids1 = map(lambda x: v2[0], graph_vf1)
      graph_vf2 = np.linspace(v2[0], v2[1], granularity)
      bids2 = graph_vf2
    else:
      # Bid bounds
      b = [(4 * v1[0] * v2[0] - (v1[1] + v2[1])**2) / (4 * (v1[0] - v1[1] + v2[0] - v2[1])), (v1[1] + v2[1]) / 2]
      # Constants of integration
      c1 = ((v2[1]-v1[1])**2 + 4*(b[0]-v2[1])*(v1[0]-v1[1])) / (-2*(b[0]-b[1])*(v1[0]-v1[1])) * np.exp((v2[1]-v1[1]) / (2*(b[0]-b[1])))
      c2 = ((v1[1]-v2[1])**2 + 4*(b[0]-v1[1])*(v2[0]-v2[1])) / (-2*(b[0]-b[1])*(v2[0]-v2[1])) * np.exp((v1[1]-v2[1]) / (2*(b[0]-b[1])))
      # Inverse bid functions
      vf1 = lambda x: v1[1] + (v2[1]-v1[1])**2 / (c1*(v2[1]+v1[1]-2*x)*np.exp((v2[1]-v1[1])/(v2[1]+v1[1]-2*x)) + 4*(v2[1]-x))
      vf2 = lambda x: v2[1] + (v1[1]-v2[1])**2 / (c2*(v1[1]+v2[1]-2*x)*np.exp((v1[1]-v2[1])/(v1[1]+v2[1]-2*x)) + 4*(v1[1]-x)) \
            if x <= b[1] else x
      # Sampling
      bids1 = np.linspace(b[0], b[1], granularity)
      bids2 = np.linspace(b[0], v2[1], granularity)
      graph_vf1 = map(vf1, bids1)
      graph_vf2 = map(vf2, bids2)
  else:
    if (v2[1] <= 2*v1[0] - v1[1]):
      graph_vf1 = np.linspace(v1[0], v1[1], granularity)
      bids1 = graph_vf1
      graph_vf2 = np.linspace(v2[0], v2[1], granularity)
      bids2 = map(lambda x: v1[0], graph_vf2)
    else:
      # Bid bounds
      b = [(4 * v1[0] * v2[0] - (v1[1] + v2[1])**2) / (4 * (v1[0] - v1[1] + v2[0] - v2[1])), (v1[1] + v2[1]) / 2]
      # Constants of integration
      c1 = ((v2[1]-v1[1])**2 + 4*(b[0]-v2[1])*(v1[0]-v1[1])) / (-2*(b[0]-b[1])*(v1[0]-v1[1])) * np.exp((v2[1]-v1[1]) / (2*(b[0]-b[1])))
      c2 = ((v1[1]-v2[1])**2 + 4*(b[0]-v1[1])*(v2[0]-v2[1])) / (-2*(b[0]-b[1])*(v2[0]-v2[1])) * np.exp((v1[1]-v2[1]) / (2*(b[0]-b[1])))
      # Inverse bid functions
      vf1 = lambda x: v1[1] + (v2[1]-v1[1])**2 / (c1*(v2[1]+v1[1]-2*x)*np.exp((v2[1]-v1[1])/(v2[1]+v1[1]-2*x)) + 4*(v2[1]-x)) \
            if x <= b[1] else x
      vf2 = lambda x: v2[1] + (v1[1]-v2[1])**2 / (c2*(v1[1]+v2[1]-2*x)*np.exp((v1[1]-v2[1])/(v1[1]+v2[1]-2*x)) + 4*(v1[1]-x))
      # Sampling
      bids1 = np.linspace(b[0], v1[1], granularity)
      bids2 = np.linspace(b[0], b[1], granularity)
      graph_vf1 = map(vf1, bids1)
      graph_vf2 = map(vf2, bids2)
  return bids1, graph_vf1, bids2, graph_vf2


def compare_with_approximation(ws, costs, reps):
  t_bids1, t_bids2 = [], []
  c_bids1, c_bids2 = [], []
  for w in ws:
    # Estimate bidding functions
    bids1, values1, bids2, values2 = estimate_bid_functions(w, reps)
    # Calculate Euclidean distance for each value in values1 and costs[0]
    # Get the index and use it to the get the value of the bid
    v1_dist = map(lambda x: np.abs(x - ((1-w)*reps[0] + costs[0]*w)), values1)
    bid = (bids1[v1_dist.index(min(v1_dist))] - (1-w)*reps[0]) / w
    t_bids1 += [bid]
    c_bids1 += [w*bid + (1-w)*reps[0]]
    # Repeat for bidder 2
    v2_dist = map(lambda x: np.abs(x - ((1-w)*reps[1] + costs[1]*w)), values2)
    bid = (bids2[v2_dist.index(min(v2_dist))] - (1-w)*reps[1]) / w
    t_bids2 += [bid]
    c_bids2 += [w*bid + (1-w)*reps[1]]
  # Use approximate (linear) solution
  approx_t_bids1 = map(lambda x: max(costs[0], 0.5 + 0.5*costs[0] - (1-x)*(reps[0]-reps[1])/(3*x)), ws)
  approx_c_bids1 = map(lambda x,y: x*y + (1-x)*reps[0], ws, approx_t_bids1)
  approx_t_bids2 = map(lambda x: max(costs[1], 0.5 + 0.5*costs[1] - (1-x)*(reps[1]-reps[0])/(3*x)), ws)
  approx_c_bids2 = map(lambda x,y: x*y + (1-x)*reps[1], ws, approx_t_bids2)
  # Calculate the error
  error_t_bids1 = map(lambda x,y: x-y, t_bids1, approx_t_bids1)
  error_c_bids1 = map(lambda x,y: x-y, c_bids1, approx_c_bids1)
  error_t_bids2 = map(lambda x,y: x-y, t_bids2, approx_t_bids2)
  error_c_bids2 = map(lambda x,y: x-y, c_bids2, approx_c_bids2)
  return error_t_bids1, error_c_bids1, error_t_bids2, error_c_bids2


def calc_avg(matrix):
  avg = []
  i = 0
  for row in matrix:
    j = 0
    for col in row:
      if i == 0:
        avg += [col]
      else:
        avg[j] += col
      j += 1
    i += 1
  return map(lambda x: x / len(matrix), avg)


def calc_std_dev(matrix, avg):
  std = []
  i = 0
  for row in matrix:
    j = 0
    for col in row:
      if i == 0:
        std += [(col - avg[j])**2]
      else:
        std[j] += (col - avg[j])**2
      j += 1
    i += 1
  return map(lambda x: np.sqrt(x / (len(matrix) - 1)), std)


def plot_error(num):
  # ProgressBar
  pbar = pb.ProgressBar(widgets=widgets, maxval=num).start()
  # Params
  ws = np.linspace(0.001, 0.999, 100)
  errors_t_bids1, errors_c_bids1, errors_t_bids2, errors_c_bids2 = [], [], [], []
  for n in range(num):
    costs = map(lambda x: x/num, np.random.randint(0, num+1, 2))
    reps = map(lambda x: x/num, np.random.randint(0, num+1, 2))
    while reps[0] >= reps[1]:
      reps = map(lambda x: x/num, np.random.randint(0, num+1, 2))
    error_t_bids1, error_c_bids1, error_t_bids2, error_c_bids2 = compare_with_approximation(ws, costs, reps)
    errors_t_bids1 += [error_t_bids1]
    errors_c_bids1 += [error_c_bids1]
    errors_t_bids2 += [error_t_bids2]
    errors_c_bids2 += [error_c_bids2]
    pbar.update(n + 1)
  # Compute average error and std deviation
  avg_error_t_bids1 = calc_avg(errors_t_bids1)
  std_dev_t_bids1 = calc_std_dev(errors_t_bids1, avg_error_t_bids1)
  avg_error_t_bids2 = calc_avg(errors_t_bids2)
  std_dev_t_bids2 = calc_std_dev(errors_t_bids2, avg_error_t_bids2)
  avg_error_c_bids1 = calc_avg(errors_c_bids1)
  std_dev_c_bids1 = calc_std_dev(errors_c_bids1, avg_error_c_bids1)
  avg_error_c_bids2 = calc_avg(errors_c_bids2)
  std_dev_c_bids2 = calc_std_dev(errors_c_bids2, avg_error_c_bids2)
  pbar.finish()
  print("Creating and saving plots...")
  # Plot the figures
  # Price error bidder 1 (full price range)
  plt.figure()
  plt.errorbar(ws, avg_error_t_bids1, yerr=std_dev_t_bids1, fmt='ro')
  plt.xlabel(r"Price weight, $w$")
  plt.ylabel(r"Average error in offered price, $\epsilon$")
  plt.grid()
  plt.savefig('price_error_1.pdf')
  # Price error bidder 1 (limited price range)
  plt.figure()
  plt.errorbar(ws, avg_error_t_bids1, yerr=std_dev_t_bids1, fmt='ro')
  plt.ylim([-5, 5])
  plt.xlabel(r"Price weight, $w$")
  plt.ylabel(r"Average error in offered price, $\epsilon$")
  plt.grid()
  plt.savefig('price_error_lim_1.pdf')
  # Price error bidder 2
  plt.figure()
  plt.errorbar(ws, avg_error_t_bids2, yerr=std_dev_t_bids2, fmt='ro')
  plt.xlabel(r"Price weight, $w$")
  plt.ylabel(r"Average error in offered price, $\epsilon$")
  plt.grid()
  plt.savefig('price_error_2.pdf')
  # Compound bid error bidder 1
  plt.figure()
  plt.errorbar(ws, avg_error_c_bids1, yerr=std_dev_c_bids1, fmt='ro')
  plt.xlabel(r"Price weight, $w$")
  plt.ylabel(r"Average error in compound bid, $\epsilon$")
  plt.grid()
  plt.savefig('compound_bid_error_1.pdf')
  # Compound bid error bidder 2
  plt.figure()
  plt.errorbar(ws, avg_error_c_bids2, yerr=std_dev_c_bids2, fmt='ro')
  plt.xlabel(r"Price weight, $w$")
  plt.ylabel(r"Average error in compound bid, $\epsilon$")
  plt.grid()
  plt.savefig('compound_bid_error_2.pdf')


def plot_wrt_w(costs, reps):
  # ProgressBar
  w_range = 100
  pbar = pb.ProgressBar(widgets=widgets, maxval=w_range).start()
  # Params
  ws = np.linspace(0.001, 0.999, w_range)
  t_bids1, t_bids2 = [], []
  c_bids1, c_bids2 = [], []
  win_prices_1 = []
  win_w_1 = []
  win_prices_2 = []
  win_w_2 = []
  count = 0
  for w in ws:
    bids1, values1, bids2, values2 = estimate_bid_functions(w, reps)
    # Calculate Euclidean distance for each value in values1 and costs[0]
    # Get the index and use it to the get the value of the bid
    v1_dist = map(lambda x: np.abs(x - ((1-w)*reps[0] + costs[0]*w)), values1)
    t_bid1 = (bids1[v1_dist.index(min(v1_dist))] - (1-w)*reps[0]) / w
    t_bids1 += [t_bid1]
    c_bid1 = w*t_bid1 + (1-w)*reps[0]
    c_bids1 += [c_bid1]
    # Repeat for bidder 2
    v2_dist = map(lambda x: np.abs(x - ((1-w)*reps[1] + costs[1]*w)), values2)
    t_bid2 = (bids2[v2_dist.index(min(v2_dist))] - (1-w)*reps[1]) / w
    t_bids2 += [t_bid2]
    c_bid2 = w*t_bid2 + (1-w)*reps[1]
    c_bids2 += [c_bid2]
    if c_bid1 < c_bid2:
      win_prices_1 += [t_bid1]
      win_w_1 += [w]
    elif c_bid1 > c_bid2:
      win_prices_2 += [t_bid2]
      win_w_2 += [w]
    else:
      win_prices_1 += [(t_bid1 + t_bid2)/2]
      win_w_1 += [w]
    pbar.update(count + 1)
    count += 1
  pbar.finish()
  print("Creating and saving plots...")
  # Plot the results
  # Price vs weight
  plt.figure()
  plt.plot(ws, t_bids1, '-', linewidth=1)
  plt.plot(ws, t_bids2, '--', linewidth=1)
  plt.ylim([0, 5])
  plt.annotate(r"$w_0$", xy=(0.265, 0.25), xycoords='data', xytext=(1,30), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.xlabel(r"Price weight, $w$")
  plt.ylabel(r"Offered price, $b_i$")
  plt.legend(['Network operator 1', 'Network operator 2'], loc='upper right')
  plt.grid()
  plt.savefig('1.pdf')
  # Compound bid vs weight
  plt.figure()
  plt.plot(ws, c_bids1, '-', linewidth=1)
  plt.plot(ws, c_bids2, '--', linewidth=1)
  plt.ylim([0, 1])
  plt.annotate(r"$w_c$", xy=(0.365, 0.605), xycoords='data', xytext=(1,30), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.xlabel(r"Price weight, $w$")
  plt.ylabel(r"Compound bid, $\beta(b_i, r_i)$")
  plt.legend(['Network operator 1', 'Network operator 2'], loc='upper left')
  plt.grid()
  plt.savefig('2.pdf')
  # Winning prices vs weight
  plt.figure()
  plt.plot(win_w_1, win_prices_1, '-', linewidth=1)
  plt.plot(win_w_2, win_prices_2, '--', linewidth=1)
  plt.ylim([0, 5])
  if costs[0] == 0.15:
    plt.plot(1.0, (costs[0]+1)/2, 'x', markeredgecolor='k', markeredgewidth=1)
    plt.annotate(r"Lowest price", xy=(1.0, 0.575), xycoords='data', xytext=(-100,45), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  elif costs[0] == 0.75:
    plt.plot(0.375, 0.36, 'x', markeredgecolor='k', markeredgewidth=1)
    plt.annotate(r"Lowest price", xy=(0.375, 0.375), xycoords='data', xytext=(5,45), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.xlabel(r"Price weight, $w$")
  plt.ylabel(r"Price, $p(w)$")
  plt.legend(['Network operator 1', 'Network operator 2'], loc='upper right')
  plt.grid()
  plt.savefig('3.pdf')


def avg_diff_min_price(num):
  # ProgressBar
  pbar = pb.ProgressBar(widgets=widgets, maxval=num).start()
  # Params
  ws = np.linspace(0.001, 0.999, 100)
  min_prices = []
  fpa_prices = []
  for n in range(num):
    # Randomize costs and reps
    costs = [np.random.uniform(0,1), np.random.uniform(0,1)]
    reps = [np.random.uniform(0,1), np.random.uniform(0,1)]
    condition = (costs[0] < costs[1] and reps[0] > reps[1]) or (costs[0] > costs[1] and reps[0] < reps[1])
    while not condition:
      costs = [np.random.uniform(0,1), np.random.uniform(0,1)]
      reps = [np.random.uniform(0,1), np.random.uniform(0,1)]
      condition = (costs[0] < costs[1] and reps[0] > reps[1]) or (costs[0] > costs[1] and reps[0] < reps[1])
    win_prices = []
    for w in ws:
      bids1, values1, bids2, values2 = estimate_bid_functions(w, reps)
      # Calculate Euclidean distance for each value in values1 and costs[0]
      # Get the index and use it to the get the value of the bid
      v1_dist = map(lambda x: np.abs(x - ((1-w)*reps[0] + costs[0]*w)), values1)
      t_bid1 = (bids1[v1_dist.index(min(v1_dist))] - (1-w)*reps[0]) / w
      c_bid1 = w*t_bid1 + (1-w)*reps[0]
      # Repeat for bidder 2
      v2_dist = map(lambda x: np.abs(x - ((1-w)*reps[1] + costs[1]*w)), values2)
      t_bid2 = (bids2[v2_dist.index(min(v2_dist))] - (1-w)*reps[1]) / w
      c_bid2 = w*t_bid2 + (1-w)*reps[1]
      if c_bid1 < c_bid2:
        win_prices += [t_bid1]
      elif c_bid1 > c_bid2:
        win_prices += [t_bid2]
      else:
        win_prices += [(t_bid1 + t_bid2)/2]
    # Select min price
    min_prices += [min(win_prices)]
    # Estimate price at w=1
    c = costs[0] if costs[0] < costs[1] else costs[1]
    fpa_prices += [(1 + c)/2]
    pbar.update(n + 1)
  pbar.finish()
  diff_prices = map(lambda x,y: x-y, fpa_prices, min_prices)
  avg_diff = np.mean(diff_prices)
  std_diff = np.std(diff_prices)
  print("Mean difference: {}".format(avg_diff))
  print("Standard deviation: {}".format(std_diff))


def expected_price(w, num):
  # ProgressBar
  pbar = pb.ProgressBar(widgets=widgets, maxval=num).start()
  # Params
  win_prices = []
  for n in range(num):
    # Randomize costs and reps
    costs = [np.random.uniform(0,1), np.random.uniform(0,1)]
    reps = [np.random.uniform(0,1), np.random.uniform(0,1)]
    if w != 1.0:
      # Estimate equilibrium bidding strategy function (transformed)
      bids1, values1, bids2, values2 = estimate_bid_functions(w, reps)
      # Calculate Euclidean distance for each value in values1 and costs[0]
      # Get the index and use it to the get the value of the bid
      v1_dist = map(lambda x: np.abs(x - ((1-w)*reps[0] + costs[0]*w)), values1)
      t_bid1 = (bids1[v1_dist.index(min(v1_dist))] - (1-w)*reps[0]) / w
      c_bid1 = w*t_bid1 + (1-w)*reps[0]
      # Repeat for bidder 2
      v2_dist = map(lambda x: np.abs(x - ((1-w)*reps[1] + costs[1]*w)), values2)
      t_bid2 = (bids2[v2_dist.index(min(v2_dist))] - (1-w)*reps[1]) / w
      c_bid2 = w*t_bid2 + (1-w)*reps[1]
    else:
      t_bid1 = (1 + costs[0]) / 2
      c_bid1 = t_bid1
      t_bid2 = (1 + costs[1]) / 2
      c_bid2 = t_bid2
    if c_bid1 < c_bid2:
      win_prices += [t_bid1]
    elif c_bid1 > c_bid2:
      win_prices += [t_bid2]
    else:
      win_prices += [(t_bid1 + t_bid2)/2]
    pbar.update(n + 1)
  pbar.finish()
  avg_price = np.mean(win_prices)
  std_price = np.std(win_prices, ddof=1)
  return avg_price, std_price


def plot_expected_prices(num):
  # Params
  ws = np.linspace(0.001, 1.0, 100)
  avg_prices, std_prices = [], []
  for w in ws:
    avg_price, std_price = expected_price(w, num)
    avg_prices += [avg_price]
    std_prices += [std_price]
  print("Creating and saving plots...")
  # Plot the results
  plt.figure()
  plt.errorbar(ws, avg_prices, yerr=std_prices, fmt='ro')
  plt.xlabel(r"Price weight, $w$")
  plt.ylabel(r"Average price, $\bar{p}(w)$")
  plt.ylim([1, 100])
  plt.grid()
  plt.savefig("expected_prices.pdf")
  # Limit the y-range
  plt.figure()
  plt.errorbar(ws, avg_prices, yerr=std_prices, fmt='ro')
  plt.xlabel(r"Price weight, $w$")
  plt.ylabel(r"Average price, $\bar{p}(w)$")
  plt.ylim([0, 1])
  plt.grid()
  plt.savefig("expected_prices_limited.pdf")


def plot_expected_prices_v2(num):
  # ProgressBar
  pbar = pb.ProgressBar(widgets=widgets, maxval=num).start()
  # Params
  ws = np.linspace(0.001, 1.0, 100)
  win_prices = []
  for n in range(num):
    costs = [np.random.uniform(0,1), np.random.uniform(0,1)]
    reps = [np.random.uniform(0,1), np.random.uniform(0,1)]
    win_prices_at_n = []
    for w in ws:
      if w != 1.0:
        # Estimate equilibrium bidding strategy function (transformed)
        bids1, values1, bids2, values2 = estimate_bid_functions(w, reps)
        # Calculate Euclidean distance for each value in values1 and costs[0]
        # Get the index and use it to the get the value of the bid
        v1_dist = map(lambda x: np.abs(x - ((1-w)*reps[0] + costs[0]*w)), values1)
        t_bid1 = (bids1[v1_dist.index(min(v1_dist))] - (1-w)*reps[0]) / w
        c_bid1 = w*t_bid1 + (1-w)*reps[0]
        # Repeat for bidder 2
        v2_dist = map(lambda x: np.abs(x - ((1-w)*reps[1] + costs[1]*w)), values2)
        t_bid2 = (bids2[v2_dist.index(min(v2_dist))] - (1-w)*reps[1]) / w
        c_bid2 = w*t_bid2 + (1-w)*reps[1]
      else:
        t_bid1 = (1 + costs[0]) / 2
        c_bid1 = t_bid1
        t_bid2 = (1 + costs[1]) / 2
        c_bid2 = t_bid2
      if c_bid1 < c_bid2:
        win_prices_at_n += [t_bid1]
      elif c_bid1 > c_bid2:
        win_prices_at_n += [t_bid2]
      else:
        win_prices_at_n += [(t_bid1 + t_bid2)/2]
    win_prices += [win_prices_at_n]
    pbar.update(n + 1)
  pbar.finish()
  print("Calculating average and standard deviation...")
  # Calculate the average, std dev, and 95% confidence intervals for each w
  avg_prices = calc_avg(win_prices)
  std_prices = calc_std_dev(win_prices, avg_prices)
  alpha = 1 - 0.95
  ci_prices = list(map(lambda x: stats.t.ppf(1-alpha/2,num-1)*x/np.sqrt(num), std_prices))
  print("Creating and saving plots...")
  # Plot the results
  plt.figure()
  plt.errorbar(ws, avg_prices, yerr=ci_prices, fmt='ro')
  plt.xlabel(r"Price weight, $w$")
  plt.ylabel(r"Average price, $\bar{p}(w)$")
  plt.ylim([1, 100])
  plt.grid()
  plt.savefig("expected_prices.pdf")
  # Limit the y-range
  plt.figure()
  plt.errorbar(ws, avg_prices, yerr=ci_prices, fmt='ro')
  plt.xlabel(r"Price weight, $w$")
  plt.ylabel(r"Average price, $\bar{p}(w)$")
  plt.ylim([0, 1])
  plt.grid()
  plt.savefig("expected_prices_limited.pdf")


def verify_params(w, reps):
  # Calculate params
  v1 = [(1-w)*reps[0], (1-w)*reps[0] + w]
  v2 = [(1-w)*reps[1], (1-w)*reps[1] + w]
  # Params conditions
  cond1 = (v2[1] >= v1[1])
  if cond1:
    cond2 = (v1[1] > 2*v2[0] - v2[1])
    eq_type = "Non-trivial" if cond2 else "Trivial (Bidder 1 always wins)"
  else:
    cond2 = (v2[1] > 2*v1[0] - v1[1])
    eq_type = "Non-trivial" if cond2 else "Trivial (Bidder 2 always wins)"
  return eq_type


if __name__ == '__main__':
  # Get command line args
  func = sys.argv[1]
  if func == 'plot_wrt_w' or func == '1':
    costs = [float(sys.argv[2]), float(sys.argv[3])]
    reps = [float(sys.argv[4]), float(sys.argv[5])]
    plot_wrt_w(costs, reps)
  elif func == 'plot_error' or func == '2':
    num = sys.argv[2]
    plot_error(int(num))
  elif func == 'avg_diff_min_price' or func == '3':
    num = sys.argv[2]
    avg_diff_min_price(int(num))
  elif func == 'expected_price' or func == '4':
    w = sys.argv[2]
    num = sys.argv[3]
    avg_price, std_price = expected_price(float(w), int(num))
    print("Mean price at w={}: {}".format(w, avg_price))
    print("Standard deviation: {}".format(std_price))
  elif func == 'plot_expected_prices' or func == '5':
    num = sys.argv[2]
    plot_expected_prices(int(num))
  elif func == 'plot_expected_prices_v2' or func == '6':
    num = sys.argv[2]
    plot_expected_prices_v2(int(num))
  else:
    print("Unknown function specified, exiting!")