import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import sys
from pathlib import Path

modpath = Path.cwd().parent
sys.path.append(str(modpath))

from io_functions import Element,read_Jon_data
import mc_module as mc
import functions as mcfunk


def r_simple(ek,ecm,vcm,mu,L):
    return np.sqrt(2*(ek+ecm)*const.e/mu/const.physical_constants['atomic mass constant'][0]) * L/vcm


# ------------- INPUT PARAMETERS ---------
N = 5000000 # number of simulation events
l = 1.683 # flight distance from center interaction region -> detector
# DT = .244 # length of interaction region
R = 0.075*1e3*.5 # detector radius
usematch = False # if True use DTV that gives 0 relative velocity

m2 = 2.0141017778 # Mass of ion 2 (negative)
# Positive ion mass read from file

E2 = 5000 #  energy of anion beam (decelerated by positive DTV) (- charge)
# Cation beam energy is calculated to match at DTV set below

DTV = -1500 # energy of positive ion beam is calculated such that they match at this  drift tube voltage
DTV_offset = 0 # offset DTV for higher energy collisions
E1_sigma = 0.0015  # beam 1 relative energy spread 
E2_sigma = 0.0015 #beam 2 relative energy spread 
T = 900 # Transversal ion beam temperature

atom = 'Rb' # Select collision partner
T_cm = 1000 # Collisional temperature selects br

dt_max = 200 # Only analyze events with delta t less than this number [ns]

# Drift tube potentials
potentialfile = 'potential_4.txt'

# file with atomic data
datafile = 'LCAO_data.dat'
massfile = 'masses.dat'

savefig = False
savename = ''

alpha = 0 # Angle between beams, bit iffy. I think the temperature is better to use.


# Plot settings
start_r = 15
stop_r = 65
bw_r = 0.1

start_r_proj = 0
stop_r_proj = 75
bw_r_proj = 0.1

start_dt = 0
stop_dt = 800
bw_dt = 1

start_ecm = 0
stop_ecm = 3
bw_ecm = 0.005

histkwargs = {
    'histtype':'step',
    'density' : True,
    'color' : 'k'
}

# ------------- End of input parameters ------------------

data = read_Jon_data(datafile)

with open(massfile,'r') as f:
    for line in f:
        line = line.split()
        data[line[0]].set_mass(float(line[1]))


histbins_r = np.arange(start_r,stop_r,bw_r)
histbins_r_proj = np.arange(start_r_proj,stop_r_proj,bw_r_proj)
histbins_dt = np.arange(start_dt,stop_dt,bw_dt)
histbins_ecm = np.arange(start_ecm,stop_ecm,bw_ecm)


imagebw =  0.25
imagebins = np.arange(-75*.5,75*.5,imagebw)


ker = data[atom].ker
br = data[atom].br[T_cm]
m1 = data[atom].mass

mass_ratio = m1/m2


print('------- Running {} --------'.format(atom))


E1 = mass_ratio * (E2+DTV) + DTV


v = mcfunk.velocity_at_match(E1,E2,m1,m2)

potentialpath = Path.cwd().parent.joinpath('potentials',potentialfile)
potentialdata = np.loadtxt(potentialpath)

mu = m1*m2/(m1+m2)

match = mcfunk.match_voltage(E1,E2,m1,m2)


header = {
'Number of simulation events' : N,
'Ion masses 1': m1 ,
'Ion masses 2': m2 ,
'Mass ratio m1/m2' : m1/m2,
'Beam 1 energy' : E1,
'Beam 2 energy' : E2,
'Collisional temperature' : T_cm,
'Beam 1 energy spread (gaussian sigma)' : E1_sigma,
'Beam 2 energy spread (gaussian sigma)' : E2_sigma,
'Transversal beam temperature' : T,
'Drift tube voltage for match ' : DTV,
'Drift tube voltage offset' : DTV_offset,
'Detector radius' : R,
'Potentialdata' : potentialfile,
'Flight distance center interaction to detector' : l,
'Maximum delta t' : dt_max
}

for k in header.keys():
    print(k,':',header[k])

fig,ax = plt.subplots(2,2,figsize=(12,8))
fig.tight_layout()



YA,YB,ZA,ZB,dt,ECOM,angles,accepted,hits = mc.MN(N,m1,m2,E1,E2,ker,br,potentialdata,l=l,DTV=DTV+DTV_offset,
potentiallimit=.5,usematch=usematch,alpha=alpha,E1_sigma=E1_sigma,E2_sigma=E2_sigma,T=T,xsec_cutoff=.05)

dt = abs(dt)
dt *= 1e9
YA *= 1e3
YB *= 1e3
ZA *= 1e3
ZB *= 1e3

hits = len(dt)

index = dt < dt_max

ax[1,1].hist(dt,histbins_dt,**histkwargs)
ax[1,1].axvline(dt_max,color='k',linestyle='--')
dt = dt[index]
YA = YA[index]
YB = YB[index]
ZA = ZA[index]
ZB = ZB[index]
ECOM = ECOM[index]

print('State, KER [eV], BR, estimated r [mm]')
print('-----------------')
for s,e,b in zip(data[atom].final_states,ker,br):
    r_calc = r_simple(e,np.median(ECOM),v,mu,l)*1e3
    print(s,e,b,r_calc)
    ax[0,1].axvline(r_calc,color='.65',linestyle='dashed',label=s)

print('-----------------')

r_proj = mcfunk.calculate_r_proj(YA,YB,ZA,ZB)
r = mcfunk.calculate_r(r_proj,dt,v)



n,binedges,_ = ax[0,1].hist(r,histbins_r,**histkwargs)

ax[1,0].hist(r_proj,histbins_r_proj,**histkwargs)
ax[0,0].hist(ECOM,histbins_ecm,**histkwargs)

print('Accepted events:',accepted)
print('Number of hits on detector',hits)
print('Average ECM:',np.mean(ECOM))
print('Median ECM:',np.median(ECOM))
print()

ax[0,0].axvline(T_cm*const.k/const.e,color='k',linestyle='--',label='$T=' + str(T_cm) +'$')
ax[0,0].set_xlabel('$E_{cm}$ [eV]')
ax[0,1].set_xlabel('$r$ [mm]')
ax[1,0].set_xlabel('$r_{||}$ [mm]')
ax[1,1].set_xlabel('$\Delta t$ [ns]')
ax[0,0].legend(loc='best',frameon=False)


if savefig:
    fig.savefig(Path.cwd().joinpath('results',atom+savename+'.pdf'),bbox_inches='tight')

# fig.savefig(atom+savename+'.pdf',bbox_inches='tight')
plt.show()