# -*- coding: utf-8 -*-
import numpy as np
import logging
import pylab as pb
import scipy.optimize

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%I:%M:%S'
)

maxiter = 10000000
f_tol = 1e-3
logging.info('max number of iterations: %s'%maxiter)
logging.info('tolerance: %s'%f_tol)

def forward(xposterior, sigmaposterior, u, y, delta, rho, 
        alpha, beta, mu, sigma_eta):

    xprior = rho*xposterior + alpha * u
    sigmaprior = rho**2 * sigmaposterior + sigma_eta
    
    def F(xposterior): 
        rhs = xprior + \
            sigmaprior * \
            beta * (y - delta*np.exp(mu + beta*xposterior))
        return xposterior - rhs

    try:
        xposterior = scipy.optimize.broyden1(F,xprior,maxiter=maxiter,f_tol=f_tol)
    except:
        logging.warn("xprior:%s"%xprior)
        logging.warn("residual: %s"%(y - delta*np.exp(mu + beta*xposterior)))
        xposterior = xprior

    sigmaposterior = -(-sigmaprior**-1 - (
        beta**2 * np.exp(mu + beta*xposterior)
    ))**-1

    return xposterior, sigmaposterior

def forwards_pass(x0,Y,I,delta,rho,alpha,beta,mu,sigma_eta):
    logging.info(u'α: %s ρ: %s β: %s μ:%s σ_ε:%s'%(
        alpha, rho, beta, mu, sigma_eta
    ))
    x = []
    sigma = []
    xposterior = x0
    sigmaposterior = 1
    logging.info("starting forwards pass")
    for i,(y,u) in enumerate(zip(Y,I)):
        logging.debug('forwards pass: iteration %s'%i)
        xposterior, sigmaposterior = forward(xposterior, sigmaposterior, u, y,
                delta, rho, alpha, beta, mu, sigma_eta)
        x.append(float(xposterior))
        sigma.append(float(sigmaposterior))
    logging.info("forwards pass is complete")
    return x,sigma

def sim(T,delta,x0,rho,alpha,beta,mu,sigma_eta):
    t = np.arange(0, T, delta)
    N = len(t)
    x = np.empty(N)
    y = np.zeros(N)
    I = np.zeros(N)
    I[t.round()==t] = 1
    beta = beta 
    pspike = np.empty(N)
    x[0] = rho*x0 + alpha*I[0] + np.sqrt(sigma_eta)*np.random.randn()
    for i in range(1,N):
        x[i] = rho*x[i-1] + alpha*I[i] + np.sqrt(sigma_eta)*np.random.randn()
        pspike[i] = (np.exp(mu + beta*x[i])*delta)
        if np.random.rand() < pspike[i]:
            y[i] = 1 
    return x,y,I,t
    

if __name__ == "__main__":

    T = 10
    delta = 0.01
    x0 = 0
    rho = 0.9
    alpha = 4
    beta = 8
    mu = 0
    sigma_eta = 0.05

    x,Y,I,t = sim(T,delta,x0,rho,alpha,beta,mu,sigma_eta)
    xest, sigmaest = forwards_pass(x0,Y,I,delta,rho,alpha,beta,mu,sigma_eta)

    for ti,y in zip(t,Y):
        if y:
            pb.plot([ti,ti], [0,1], 'r-')

    pb.plot(t,x,label="true")
    pb.plot(t,xest,label="est")
    pb.legend()
    pb.show()

