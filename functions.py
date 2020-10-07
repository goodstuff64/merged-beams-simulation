import numpy as np
import scipy.stats as sps
import scipy.constants as const

def driftvoltage(s,l,ramp,V):
    flip = s > 0.5*l
    S = np.array(s)
    S[flip] -= 2*(abs(0.5*l-s[flip]))
    const = (np.sign(S-ramp)*np.sign(-S+l*.5)+1)*.5
    omega = .5*np.pi/ramp
    rise = (np.sign(S)*np.sign(-S+ramp)+1)*.5*np.sin(S*omega)
    return (const+rise)*V

def driftvoltage_data(s,data,V,center=.574):
    v = np.interp(s,data[:,0]-center,data[:,1])*V
    return v

def drift(s,l,ramp,V):
    return V

def calculate_match(E1,E2,m1,m2):
    return -(m1*E2-m2*E1)/(m2+m1)


def calculate_r_proj(Y_A,Y_B,Z_A,Z_B):
    y_dist = Y_A - Y_B
    z_dist = Z_A - Z_B
    dist = np.sqrt(y_dist*y_dist+z_dist*z_dist)
    return dist

def calculate_r(r_proj,delta_t,v):
    '''
    Calculate particle distance
    r_proj - projected distance in mm
    delta_t - time difference in ns
    v - matched velocity in m/s
    '''
    x_dist = delta_t*v*1e-6
    r = np.sqrt(r_proj*r_proj+x_dist*x_dist)
    
    return r

def calculate_center(Y_A,Y_B,Z_A,Z_B):
    y_dist = Y_A - Y_B
    z_dist = Z_A - Z_B
    cy = Y_A - 0.5*y_dist
    cz = Z_A - 0.5*z_dist
    return cy,cz

def calculate_angles(Y_A,Y_B,Z_A,Z_B,delta_t,r,v):
    z_dist = abs(Z_A-Z_B)
    y_dist = abs(Y_A-Y_B)
    x_dist = delta_t*v*1e-6

    theta = np.arccos(x_dist/r)
    phi = np.arctan(z_dist/y_dist)

    return theta,phi

def velocity(E,m):
    m *= const.value('atomic mass constant')
    return np.sqrt(2*E*const.e/m)

def velocity_COM(E1,E2,m1,m2,DT):
    v1 = velocity(E1-DT,m1)
    v2 = velocity(E2+DT,m2)
    return (m1*v1+m2*v2)/(m1+m2)

def velocity_at_match(E1,E2,m1,m2):
    DTV_match = -(m1*E2-m2*E1)/(m2+m1)
    m1 *= const.value('atomic mass constant')
    m2 *= const.value('atomic mass constant')
    v1 = np.sqrt(2*(E1-DTV_match)*const.e/m1)
    v2 = np.sqrt(2*(E2+DTV_match)*const.e/m2)
    # print(DTV_match,v1,v2)
    return v1

def ECOM(E1,E2,m1,m2,DT):
    v1 = velocity(E1-DT,m1)
    v2 = velocity(E2+DT,m2)
    m1 *= const.value('atomic mass constant')   
    m2 *= const.value('atomic mass constant')
    v_rel = v1-v2
    mu = m1*m2/(m1+m2)

    return 0.5*v_rel*v_rel*mu/const.e

def ECOM_ANGLE(E1,E2,m1,m2,DT,alpha=0):
    E1 -= DT
    E2 += DT
    mu = m1*m2/(m1+m2)

    ECOM = E1/m1 + E2/m2 - 2*np.sqrt(E1*E2/m1/m2)*np.cos(alpha)
    return mu*ECOM

def match_voltage(E1,E2,m1,m2):
    return -(m1*E2-m2*E1)/(m2+m1)
