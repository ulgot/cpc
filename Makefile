ARCH =-arch=sm_20 -m64
#gpu
CC = nvcc
OPT =--use_fast_math -O3
CURAND =-L/usr/local/cuda/lib64 -lcurand
#cpu
CCPU = gcc
OPTCPU =-ffast-math -O3
ICPU = icc
IOPTCPU =-O3 
#IOPTCPU =-fast -O3 

all: single double

single: prog.cu
	$(CC) $(ARCH) $(OPT) -o prog prog.cu $(CURAND) -lm

double: double_prog.cu
	$(CC) $(ARCH) -o prog double_prog.cu $(CURAND) -lm

cpu: prog.c
	$(CCPU) $(OPTCPU) -o progcpu prog.c -lm

icpu: prog.c
	$(ICPU) $(IOPTCPU) -o progcpu prog.c -lm

dcpu: double_prog.c
	$(CCPU) $(OPTCPU) -o dprogcpu double_prog.c -lm


poisson: poisson.c
	$(CCPU) $(OPTCPU) -o poisson poisson.c -lm


