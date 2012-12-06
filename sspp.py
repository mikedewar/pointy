# -*- coding: utf-8 -*-
# this is a simplified, poisson version of Smith and Brown 03
import numpy as np
import scipy.optimize
import functools
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%I:%M:%S'
)

# a little helper function to simulate the LDS
def sim(fun, initial, T):
    x = initial
    for t in range(T):
        yield x
        x = fun(x)

# a little wrapper for the optimise function
config = {
    "maxiter" : 1000,
    "f_tol" : 1e-4
}

optimise = functools.partial(
    scipy.optimize.broyden1, 
    maxiter=config['maxiter'], 
    f_tol=config['f_tol']
)

# model
# equation 1
obs = lambda x: np.random.poisson(rate(x))
# equation 2
state = lambda x: np.random.normal(a*x, np.sqrt(sigma_w))
# equation 3
rate = lambda x: np.exp(beta*x)

# forward filter
# the next two lines are from equation 7
predict = lambda x: a*x
predict_variance = lambda sigma: a**2 * sigma + sigma_w
# equation 14
correct = lambda y, xp, sp: optimise(
    lambda x: y*beta - beta * np.exp(beta*x) - (x - a*xp)*sp,
    xp,
)
# equation 15
correct_variance = lambda xp, sp: beta**2*np.exp(beta*xp) - sp


# algorithm 1
def update_posteriors(y,xposterior,sigmaposterior):
    xprior = predict(xposterior)
    sigmaprior = predict_variance(sigmaposterior)
    try:
        xposterior = correct(y,xprior,sigmaprior)
    except scipy.optimize.nonlin.NoConvergence:
        xposterior = xprior
        logging.warn('convergence error')
    sigmaposterior = correct_variance(xprior, sigmaprior)
    return xposterior, sigmaposterior

# parameters
x0 = 2
a = 0.9
beta = 0.7
sigma_w = 0.6

# simulate
T = 400
X = list(sim(state,x0,T))
Y = [obs(x) for x in X]
sigma0 = 0.2

# filter
xpos = x0
sigmapos = sigma0
Xhat = []
Sigmahat = []
for y in Y:
    xpos, sigmapos = update_posteriors(y,xpos,sigmapos)
    Xhat.append(float(xpos))
    Sigmahat.append(float(sigmapos))

# plot
import pylab as pb
for i,y in enumerate(Y):
    pb.plot([i,i], [0,y], 'k-',alpha=0.4)
pb.plot(map(rate,X),label="true")
pb.plot(map(rate,Xhat),label="est")
upper = [rate(x)+rate(s) for x,s in zip(Xhat, Sigmahat)]
lower = [rate(x)-rate(s) for x,s in zip(Xhat, Sigmahat)]
lower = [0 if l < 0 else l for l in lower]
pb.fill_between(range(len(X)),lower,upper,
    facecolor="gray",alpha=0.1,edgecolor=None)
pb.xlabel('k')
pb.ylabel('$\lambda(x_k)$')
pb.legend()
pb.show()
