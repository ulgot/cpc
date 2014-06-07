import os
import numpy as np
import sys
import matplotlib.pyplot as p
from math import sqrt, exp

TRANS = 0.1
m = 2.5

def J(y,m):
  a = sqrt(y*(m**2/y))
  bp = exp( 1.0/(y*(1.0 - 1.0/a)) ) 
  bm = exp( 1.0/(y*(1.0 + 1.0/a)) ) 
  return (1.0/(4*y))*(bp - bm)/((bp - 1.0)*(bm - 1.0))

try:
  path = sys.argv[1]
except:
  path = './'
_dir = sorted(os.listdir(path))

N=5
A=-1
B=1
step=(B-A)/float(N-1)
Dp = [10**(-1+i*step) for i in range(N)]
x = [2**i for i in range(1,12)]

if 1:
 for _x in x:
  p.figure()
  once = True
  for fname in _dir:
    if fname == 'cpoisson_test.dat':
      _file = os.path.join(path,fname)
      print _file
      data = np.genfromtxt(_file, delimiter=' ', dtype=[int,float,float,float])
      vel = [c for a, b, c, d in data if a == _x]
      p.semilogx(Dp,vel,label=fname)
      if once and False:
	p.semilogx(Dp,[2*J(_Dp,m) for _Dp in Dp], 'o-', label='TH', color='black')
	p.ylabel('J')
	p.xlabel(r'$D_p$')
	p.title(fname[:-4]+", "+"%d runs"%_x)
	once = False
      p.legend(loc='best')
  p.savefig('Gsteps_Dp_%druns.png'%_x)    
  #p.savefig('vel_Dp_%druns.png'%_x)    

if (0):
  lDp = len(Dp)
  lx = len(x)
  for iDp in range(lDp):
    p.figure()
    for fname in _dir:
      if fname[-3:] == 'dat':
	if fname[-8:-4] == 'XEON':
	  lstyl = '-'
	else:
	  lstyl = ':'
	_file = os.path.join(path,fname)
	print _file
	data = np.genfromtxt(_file, usecols=[2], dtype=[float])
	time = data[iDp*lx:(iDp+1)*lx]
	p.loglog(x, time, label=fname, basex=2, linestyle=lstyl)
	p.xlabel('no of particles')
	p.ylabel('milisec')
	p.title(fname[:-4]+", "+"$D_p$ = %f"%Dp[iDp])
	p.legend(loc='best')
    p.savefig('time_n_%f.png'%Dp[iDp])
