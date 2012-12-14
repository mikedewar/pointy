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

#### MODEL
# equation 1
obs = lambda x: np.random.poisson(rate(x))
# equation 2
state = lambda x: np.random.normal(a*x, np.sqrt(sigma_w))
# equation 3
rate = lambda x: np.exp(beta*x)

#### FILTER
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
def filter(x0, sigma0, Y):

    def update(y,xposterior,sigmaposterior,i):
        xprior = predict(xposterior)
        sigmaprior = predict_variance(sigmaposterior)
        xposterior = correct(y,xprior,sigmaprior)
        sigmaposterior = correct_variance(xprior, sigmaprior)
        if sigmaposterior < 0:
            raise ValueError('negative variance at step %s'%i)
        return xprior, sigmaprior, xposterior, sigmaposterior

    xpos = x0
    sigmapos = sigma0
    Xhat = []
    Xpred = []
    Sigmahat = []
    Sigmapred = []
    for i,y in enumerate(Y):
        xprior, sigmaprior, xpos, sigmapos = update(y,xpos,sigmapos,i)
        Xpred.append(float(xprior))
        Xhat.append(float(xpos))
        Sigmapred.append(float(sigmaprior))
        Sigmahat.append(float(sigmapos))

    return Xpred, Xhat, Sigmapred, Sigmahat

### SMOOTHER 

# x1 = x_{k|k}
# x2 = x_{k+1|K}
# x3 = x_{k+1|k}
# p1 = p_{k|k-1}
# p2 = p_{k+1|k}
# p3 = p_{k|k}
# p4 = p_{k+1|K}

# equation 18
S = lambda p1, p2: p1*a*p2**-1
# equation 16
smooth_state = lambda x1, x2, x3, p1, p2: x1 + S(p1,p2)*(x2-x3)
# equation 17
smooth_var = lambda p1, p2, p3, p4: p3 + S(p1,p2)*(p4-p2)*S(p1,p2)

# algorithm 2
def smooth(X,Xpred,P,Ppred):
    Xsmooth = [0 for x in X]
    Psmooth = [0 for x in X]
    Xsmooth[-1] = X[-1]
    Psmooth[-1] = P[-1]
    for k in reversed(range(len(X)-1)):
        Xsmooth[k] = smooth_state(X[k], Xsmooth[k+1], Xpred[k+1], Ppred[k], Ppred[k+1])
        Psmooth[k] = smooth_var(Ppred[k], Ppred[k+1], P[k], Psmooth[k+1])
    return Xsmooth, Psmooth

def E_step(x0, sigma0, Y):
    Xpred, Xhat, Sigmapred, Sigmahat = filter(x0, sigma0, Y)
    Xpost, Sigmapost = smooth(Xhat, Xpred, Sigmahat, Sigmapred)
    return Xpost, Sigmapost

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

Xhat, Sigmahat = E_step(x0, sigma0, Y)

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
