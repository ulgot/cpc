import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys

_file = sys.argv[1]

M = 17
x = [2**i for i in range(1,M+1)]
Y = np.genfromtxt(_file, usecols=[1])
Z = np.genfromtxt(_file, usecols=[2])

lx = len(x)
lY = len(Y)
lG = lY/lx
print 'interpolation from', lG, 'points'

y,z = [],[]
for i in range(M):
  buf = sum([Y[i+j*M]/1000. for j in range(lG)])
  y.append(buf)
  buf = sum([Z[i+j*M] for j in range(lG)])
  z.append(buf)

#interpolacja
iN = 100
A = -1
B = 1
step = (B-A)/float(lG-1)
Dp = [10**(-1+i*step) for i in range(lG)]
istep = (B-A)/float(iN-1)
iDp = [10**(-1+i*istep) for i in range(iN)]

sGs, isGs, st, ist = [], [], [], []
for p in range(M):
  # G steps interpolation
  Gsteps = [Z[p+j*M] for j in range(lG)]
  _sumGsteps = sum([Z[p+j*M] for j in range(lG)])
  sGs.append(_sumGsteps)
  f = interp1d(Dp,Gsteps,kind='slinear')
  iGsteps = f(iDp)
  isGs.append(sum(iGsteps))

  # time interpolation
  time = [Y[p+j*M] for j in range(lG)]
  _sumtime = sum(time)
  st.append(_sumtime)
  f = interp1d(Dp,time,kind='slinear')
  itime = f(iDp)
  ist.append(sum(itime))

_f = plt.figure()
p1, p2, p3 = plt.loglog(x,sGs,'o',x,isGs,'k', x, [(100/lG)*i for i in sGs], 'h', basex=2)
_f.legend((p1,p2,p3),('%dp simulated data'%lG,'100p interpolated data','%dp * %d'%(lG,(100/lG))))
plt.savefig('interpolate_100p_'+_file[:-3]+'png')

out = open('cpu'+_file,'w')
for i in range(M):
  print >>out, x[i], ist[i], isGs[i]
out.close()
