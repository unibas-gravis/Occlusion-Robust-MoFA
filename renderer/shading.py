import torch
import math

ftiny = torch.finfo(torch.float).tiny * 10**3

def calc_spherical_harmonics_P(l,m,x):

    pmm = torch.ones_like(x)
    if m > 0:
        somx2 = torch.sqrt(torch.clamp((1.0-x)*(1.0+x),min=ftiny))

        fact = 1.0
        for i in range(1,m+1):
            pmm *=(-fact)*somx2
            fact += 2.0

    if l == m:
        return pmm

    pmmp1 = x * (2.0*float(m)+1.0)*pmm

    if l == m+1:
        return pmmp1

    pll = torch.zeros_like(x)
    for ll in range(m+2,l+1):
        pll = ((2.0*float(ll)-1.0)*x*pmmp1-(float(ll)+float(m)-1.0)*pmm)/(float(ll)-float(m))
        pmm = pmmp1
        pmmp1 = pll

    return pll

def calc_factorial(t):
    r = 1.0
    for s in range(2,t+1):
        r *= float(s)

    return r

def calcSphericalHarmonicsK(l,m):

    temp = ((2.0*l+1.0)*calc_factorial(l-m))/(4.0*math.pi*calc_factorial(l+m))
    return math.sqrt(temp)

def calc_spherical_harmonics(l,m,theta,phi):
    sqrt2 = math.sqrt(2.0)

    if m == 0:
        return calcSphericalHarmonicsK(l,m)*calc_spherical_harmonics_P(l,m,torch.cos(theta))

    elif m>0:
        return sqrt2*calcSphericalHarmonicsK(l,m)*torch.cos(float(m)*phi)*calc_spherical_harmonics_P(l,m,torch.cos(theta))

    else:
        return sqrt2*calcSphericalHarmonicsK(l,-m)*torch.sin(-float(m)*phi)*calc_spherical_harmonics_P(l,-m,torch.cos(theta))

def calc_spherical_angle(nrm):
    theta = torch.acos(torch.clamp(nrm[:,2,:],min=-1+ftiny,max=1-ftiny))
    nrm0 = torch.clamp(torch.abs(nrm[:,0,:]),min=ftiny)*(nrm[:,0,:].sign())
    phi = torch.atan2(nrm[:,1,:],nrm0)

    return theta, phi

def calc_sh_basis(normal):
    theta, phi = calc_spherical_angle(normal)
    pnum = theta.shape[1]
    sh_basis = torch.zeros(theta.shape[0],27,3,pnum).to(normal.device)

    idx = 0
    for l in range(0,3):
        for m in range(-l,l+1):
            sh_basis[:,idx,0,:] += calc_spherical_harmonics(l,m,theta,phi)
            sh_basis[:,idx+9,1,:] += calc_spherical_harmonics(l,m,theta,phi)
            sh_basis[:,idx+18,2,:] += calc_spherical_harmonics(l,m,theta,phi)
            idx+=1

    return sh_basis