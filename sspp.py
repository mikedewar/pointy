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
    lambda x: y*beta - beta * np.exp(beta*x) - (x - xp)*sp**-1,
    xp,
)
# equation 15
correct_variance = lambda xp, sp: -1.0/(-beta**2*np.exp(beta*xp) - sp**-1)

# algorithm 1
def update_posteriors(y,xposterior,sigmaposterior,i):
    xprior = predict(xposterior)
    sigmaprior = predict_variance(sigmaposterior)
    assert sigmaprior > 0, (sigmaprior,i)
    try:
        xposterior = correct(y,xprior,sigmaprior)
    except scipy.optimize.nonlin.NoConvergence:
        xposterior = xprior
        logging.warn('convergence error')
        if i:
            logging.warn('step %s'%i)
    except: 
        logging.info('y: %s, xprior: %s, sigmaprior: %s'%(y, xprior, sigmaprior))
        raise
    sigmaposterior = correct_variance(xprior, sigmaprior)
    if sigmaposterior < 0:
        logging.warn('negative variance averted at step %s'%i)
        sigmaposterior = sigmaprior
    assert sigmaposterior > 0, (sigmaposterior,sigmaprior,i)
    return xposterior, sigmaposterior

# parameters
x0 = 2
a = 0.9
beta = 0.5
sigma_w = 1

# simulate
T = 200
X = list(sim(state,x0,T))
Y = [obs(x) for x in X]
sigma0 = 1 
#Y = [y if i < 100 else 0 for i,y in enumerate(Y)]


# filter
xpos = x0
sigmapos = sigma0
Xhat = []
Sigmahat = []
for i,y in enumerate(Y):
    xpos, sigmapos = update_posteriors(y,xpos,sigmapos,i)
    Xhat.append(float(xpos))
    Sigmahat.append(float(sigmapos))

# plot
import pylab as pb
pb.subplot(3,1,1)
for i,y in enumerate(Y):
    pb.plot([i,i], [0,y], 'k-',alpha=0.4)
pb.ylabel('$y_k$')
pb.subplot(3,1,2)
pb.plot(map(rate,X),label="true")
pb.plot(map(rate,Xhat),label="est")
pb.xlabel('$k$')
pb.ylabel('$\lambda(x_k)$')
pb.legend()
pb.subplot(3,1,3)
pb.plot(X,label="true")
pb.plot(Xhat,label="est")
upper = [x+s for x,s in zip(Xhat, Sigmahat)]
lower = [x-s for x,s in zip(Xhat, Sigmahat)]
pb.fill_between(range(len(X)),lower,upper,
    facecolor="gray",alpha=0.1,edgecolor=None)
pb.ylabel('$x_k$')
pb.legend()
pb.show()
