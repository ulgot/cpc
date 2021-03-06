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
dev = 0
block = 256
paths = 1024
periods = 1000
spp = 400
samples = 4000
trans = 0.1

#Output
mode = 'moments'
points = 100
beginx = -1
endx = 1
domain = '1d'
domainx = 'p'
logx = 1
DIRNAME='./'
#os.system('mkdir -p %s' % DIRNAME)
#os.system('rm -v %s*.dat %s*.png' % (DIRNAME, DIRNAME))

for mean in [2.5]: #[2.5, 5, 10]:
    out = 'epl_mean%s' % mean
    _cmd = './prog --dev=%d --Dg=%s --Dp=%s --lambda=%s --fa=%s --fb=%s --mua=%s --mub=%s --comp=%d --mean=%s --block=%d --paths=%d --periods=%s --spp=%d --trans=%s --mode=%s --points=%d --beginx=%s --endx=%s --domain=%s --domainx=%s --logx=%d --samples=%d >> %s.dat' % (dev, Dg, Dp, lmd, fa, fb, mua, mub, comp, mean, block, paths, periods, spp, trans, mode, points, beginx, endx, domain, domainx, logx, samples, out)
    output = open('%s.dat' % out, 'w')
    print >>output, '#%s' % _cmd
    output.close()
    print _cmd
    cmd = commands.getoutput(_cmd)
    os.system('gnuplot -e "m=%s" epl.plt' % mean)
    os.system('mv -v %s.dat %s.png %s' % (out, out, DIRNAME))
