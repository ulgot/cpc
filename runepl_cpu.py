#!/usr/bin/python
import commands, os
import numpy

#Model
Dg = 0
Dp = 0
lmd = 0
fa = 0
fb = 0
mua = 0
mub = 0
comp = 1
mean = 2.5

#Simulation
paths = 10#24
periods = 1000
spp = 100
trans = 0.1

out = 'epl_CPU_mean%s' % mean
_cmd = './progcpu --Dg=%s --Dp=%s --lambda=%s --fa=%s --fb=%s --mua=%s --mub=%s --comp=%d --mean=%s --paths=%d --periods=%s --spp=%d --trans=%s >> %s.dat' % (Dg, Dp, lmd, fa, fb, mua, mub, comp, mean, paths, periods, spp, trans, out)
output = open('%s.dat' % out, 'a')
print >>output, '#%s' % _cmd
output.close()
print _cmd
cmd = commands.getoutput(_cmd)
#os.system('gnuplot -e "m=%s" epl.plt' % mean)
#os.system('mv -v %s.dat %s.png %s' % (out, out, DIRNAME))