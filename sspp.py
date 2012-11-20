# -*- coding: utf-8 -*-
import numpy as np
import logging
import scipy.optimize

from config import config

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%I:%M:%S'
)

logging.info('max number of iterations: %s'%config['maxiter'])
logging.info('tolerance: %s'%config['f_tol'])

def forward(xposterior, sigmaposterior, u, y, delta, rho, 
        alpha, beta, mu, sigma_eta):

    xprior = rho*xposterior + alpha * u
    sigmaprior = rho**2 * sigmaposterior + sigma_eta
    
    def F(xposterior): 
        rhs = xprior + \
            sigmaprior * \
            beta * (y - delta*np.exp(mu + beta*xposterior))
        return xposterior - rhs

    def Fprime(xposterior):
        return 1 - (sigmaprior * beta**2 * delta * np.exp(mu + beta*xposterior))

    try:
        xposterior = scipy.optimize.broyden1(
            F,
            xprior,
            maxiter=config['maxiter'],
            f_tol=config['f_tol']
        )
    except:
        logging.warn("xprior:%s"%xprior)
        logging.warn("σ_proir:%s"%sigmaprior)
        logging.warn("residual: %s"%(y - delta*np.exp(mu + beta*xposterior)))
        logging.warn("F`: %s"%(Fprime(xprior)))
        raise
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
        logging.debug('forwards pass: time index %s'%i)
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
