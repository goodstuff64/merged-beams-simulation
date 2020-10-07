import numpy as np
import scipy.stats as sps
import scipy.constants as const


def driftvoltage_data(s,data,V):
    v = np.interp(s,data[:,0],data[:,1])*V
    return v

def crossection(E):
    return 1/np.sqrt(E)


def MN(N,m1,m2,E1_avg,E2_avg,EK,BR,potentialsamples=None,L_start=1.166,L_stop=2.214,
    potentiallimit=1e-9,DTV=0,usematch=True,R=0.0375,E1_sigma=0,E2_sigma=0,alpha=0,T=0,
    xsec_cutoff=1):
    '''
    Introducing using sampled data points for the drift tube potential.
    This is an attempt to introduce the potential MingChao made using SIMION.
    Needed to reformulate the geometry slightly for this to work.


    N - number of simulation events
    m1 - Mass of ion 1
    m2 - Mass of ion 2
    E1_avg - initial energy eV beam 1 (accelerated by positive DTV) (+ charge)
    E2_avg - initial energy ev beam 2 (decelerated by positive DTV) (- charge)
    EK - List of kinetic energy releases
    BR - List of branching ratios

    l - flight distance from center interaction region -> detector
    DT - length of interaction region
    R -  detector radius
    usematch - if True use DTV that gives 0 relative velocity
    DTV - else use DTV Drift tube voltage
    E1_sigma - beam 1 energy spread in %
    E2_sigma - beam 2 energy spread in %
    alpha - angle between the ion beams
    '''

    # L = sps.uniform(-DT_range,2*DT_range).rvs(N)
    L = sps.uniform(L_start,L_stop-L_start).rvs(N)
    # import matplotlib.pyplot as plt
    # plt.hist(L,100)
    # plt.show()

    if usematch:
        DTV = -(m1*E2_avg-m2*E1_avg)/(m2+m1)
        

    if potentialsamples is not None:
        # DTV_low = driftvoltage_data(-DT_range,potentialsamples,DTV,center=DT_center)
        # DTV_high = driftvoltage_data(DT_range,potentialsamples,DTV,center=DT_center)

        # if abs(DTV_low) > potentiallimit or abs(DTV_high) > potentiallimit:
        #     errormessage = 'The potential is cut before it reaches 0. Increase the DT_range parameter or increase cutoff.'
        #     errormessage += '\nValues at cutoff: {} and {}'.format(DTV_low,DTV_high)
        #     errormessage += '\nCuttof limit: {}'.format(potentiallimit)
        #     raise ValueError(errormessage)



        DTV_L = driftvoltage_data(L,potentialsamples,DTV)
    else:
        DTV_L = DTV

    
    m1 *= const.value('atomic mass constant')
    m2 *= const.value('atomic mass constant')


    if E1_sigma > 1e-6 and E2_sigma > 1e-6:
        E1 = sps.norm(E1_avg,E1_sigma*E1_avg).rvs(N)
        E2 = sps.norm(E2_avg,E2_sigma*E2_avg).rvs(N)
    else:
        E1 = np.ones(N)*E1_avg
        E2 = np.ones(N)*E2_avg    


    E1 -= DTV_L
    E2 += DTV_L

    # L += l

    E1 *= const.e
    E2 *= const.e

    v_1 = np.sqrt(2*E1/m1) # initial velocity
    v_2 = np.sqrt(2*E2/m2) # initial velocity

    mu = m1*m2/(m1+m2)
    RSQ = R*R

    v_A = np.zeros((N,3))
    v_B = np.zeros_like(v_A)

    # REMOVED INITIAL POSITIONS AS THEY ARE NOT NECESSARY
    # COULD BE REINTRODUCED IF INITIAL TRANSVERSAL POSITIONS ARE REQUIRED
    # r0_A = copy(r)
    # r0_B = copy(r)

    # If angle is to be included calculate transversal velocity componentes
    if alpha > 1e-6:
        v_A_x = np.cos(alpha*.5)*v_1
        v_A_T = np.sin(alpha*.5)*v_1

        v_B_x = np.cos(alpha*.5)*v_2
        v_B_T = np.sin(alpha*.5)*v_2



        beta = sps.uniform(0,2*np.pi).rvs(N)

        v_A_y = np.cos(beta) * v_A_T
        v_A_z = np.sin(beta) * v_A_T

        v_B_y = np.cos(beta+np.pi) * v_B_T
        v_B_z = np.sin(beta+np.pi) * v_B_T

        v_A[:,0] = v_A_x
        v_A[:,1] = v_A_y
        v_A[:,2] = v_A_z

        v_B[:,0] = v_B_x
        v_B[:,1] = v_B_y
        v_B[:,2] = v_B_z

    else:
        v_A[:,0] = v_1
        v_B[:,0] = v_2



    if T > 1e-6:
        sigmaA = np.sqrt(const.k*T/m1)
        sigmaB = np.sqrt(const.k*T/m2)

        v_A[:,1] += sps.norm(0,sigmaA).rvs(N)
        v_A[:,2] += sps.norm(0,sigmaA).rvs(N)

        v_B[:,1] += sps.norm(0,sigmaB).rvs(N)
        v_B[:,2] += sps.norm(0,sigmaB).rvs(N)


    scalarprod = np.sum(v_A*v_B,axis=1)
    A_abs = np.sqrt(np.sum(v_A*v_A,axis=1))
    B_abs = np.sqrt(np.sum(v_B*v_B,axis=1))
    alpha = np.arccos(scalarprod/A_abs/B_abs)

    v_rel_x = v_A[:,0] - v_B[:,0]
    v_rel_y = v_A[:,1] - v_B[:,1]
    v_rel_z = v_A[:,2] - v_B[:,2]

    v_rel = np.sqrt(v_rel_x*v_rel_x+v_rel_y*v_rel_y+v_rel_z*v_rel_z)
    ECOM = 0.5*v_rel*v_rel*mu/const.e

    xsec = crossection(ECOM)

    acceptlimit = sps.uniform.rvs(size=N)*crossection(xsec_cutoff)
    accept = acceptlimit < xsec
    # accept = acceptlimit < 1e6


    # MOVE EVERYTHING NOT NEEDED FOR ECM CALCULATION BELOW HERE
    N_new = np.sum(accept)

    v_A = v_A[accept]
    v_B = v_B[accept]
    L = L[accept]
    ECOM = ECOM[accept]

    rf_A = np.zeros_like(v_A)
    rf_B = np.zeros_like(rf_A)

    # print(r0_A)

    TOF = np.zeros((N_new,2))

    EK = np.array(EK)
    BR = np.array(BR)/sum(BR)

    custm = sps.rv_discrete(name='custm', values=(list(range(len(EK))),BR))
    KER_index = custm.rvs(size=N_new)
    KER = EK[KER_index] * const.e



    n = sps.norm().rvs((N_new,3))

    n_norm = np.sqrt(np.sum(n*n,axis=1))

    n = n/n_norm[:,None]

    mu = m1*m2/(m1+m2)
    vscalar = np.sum(n*v_A,axis=1)-np.sum(n*v_B,axis=1)


    root = np.sqrt(vscalar*vscalar+2*KER/mu)
    u1_factor = mu/m1*(-vscalar-root)
    u2_factor = mu/m2*(vscalar+root)


    v_A = v_A + n * u1_factor[:,None]
    v_B = v_B + n * u2_factor[:,None]



    TOF[:,0] = L/v_A[:,0]
    TOF[:,1] = L/v_B[:,0]

    # for k in range(3):
    #     # REMOVED INITIAL 3D POSITIONS AS THEY ARE NOT NECESSARY, L is sufficient
    #     # COULD BE REINTRODUCED IF INITIAL TRANSVERSAL POSITIONS ARE REQUIRED
    #     # rf_A[:,k] = r0_A[:,k] + v_A[:,k] * TOF[:,0]
    #     # rf_B[:,k] = r0_B[:,k] + v_B[:,k] * TOF[:,1]


    rf_A =  v_A * TOF[:,0][:,None]
    rf_B =  v_B * TOF[:,1][:,None]
       
    delta_t = []
    Z_A = []
    Y_A = []
    Z_B = []
    Y_B = []
    ECOM_SAVE = []

    y_A = rf_A[:,1]
    z_A = rf_A[:,2]

    y_B = rf_B[:,1]
    z_B = rf_B[:,2]

    # select only events where both products hit the detector
    hit_A = y_A*y_A + z_A * z_A < RSQ
    hit_B = y_B*y_B + z_B * z_B < RSQ

    hit = hit_A*hit_B

    delta_t = TOF[:,0][hit]-TOF[:,1][hit]
    Y_A = y_A[hit]
    Y_B = y_B[hit]
    Z_A = z_A[hit]
    Z_B = z_B[hit]
    ECOM_SAVE = ECOM[hit]

    return Y_A,Y_B,Z_A,Z_B,delta_t,ECOM_SAVE,alpha,N_new,np.sum(hit)#,xsec,ECOM,acceptlimit

def crossection2(E,cutoff=1):
    if E < cutoff:
        return 1
    else:
        return 1/np.sqrt(E)*np.sqrt(cutoff)



if __name__ == "__main__":
    main()

