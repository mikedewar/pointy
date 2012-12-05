import sspp
import pylab as pb

T = 10
delta = 0.01
x0 = 0
rho = 0.9
alpha = 2
beta = 6 
mu = 0
sigma_eta = 0.05

x,Y,I,t = sspp.sim(T,delta,x0,rho,alpha,beta,mu,sigma_eta)
xest, sigmaest = sspp.forwards_pass(x0,Y,I,delta,rho,alpha,beta,mu,sigma_eta)

for ti,y in zip(t,Y):
    if y:
        pb.plot([ti,ti], [0,1], 'r-')

pb.plot(t,x,label="true")
pb.plot(t,xest,label="est")
pb.legend()
pb.show()
