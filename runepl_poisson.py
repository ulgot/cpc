#!/usr/bin/python
import commands, os
import numpy

#Model
Dp = 0.5
mean = 2.5

#Simulation
paths = 10**17 #1024
periods = 1000
spp = 400
trans = 0.1

for Dp in [0.1, 0.3, 0.5, 0.8, 1]:
  lmd = mean*mean/Dp
  out = 'epl_CPU_mean%s' % mean
  _cmd = './poisson --Dp=%s --lambda=%s --mean=%s --paths=%d --periods=%s --spp=%d --trans=%s >> %s.dat' % (Dp, lmd, mean, paths, periods, spp, trans, out)
  output = open('%s.dat' % out, 'a')
  print >>output, '#%s' % _cmd
  output.close()
  print _cmd
  cmd = commands.getoutput(_cmd)
  #os.system('gnuplot -e "m=%s" epl.plt' % mean)
  #os.system('mv -v %s.dat %s.png %s' % (out, out, DIRNAME))
