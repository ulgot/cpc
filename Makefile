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

dcpu: prog.c
	cat prog.c | sed 's/float/double/g;s/.0f/.0/g;s/.5f/.5/g;s/FLOAT/DOUBLE/g;s/%f/%lf/g;s/79f/79/g;s/fold(&/\/\/fold(&/g;s/\ +\ xfc//g' > dprog.c
	$(CCPU) $(OPTCPU) -o dprogcpu dprog.c -lm
	rm dprog.c

poisson: poisson.c
	$(CCPU) $(OPTCPU) -o poisson poisson.c -lm

dpoisson: poisson.c
	cat poisson.c | sed 's/float/double/g;s/.0f/.0/g;s/.5f/.5/g;s/FLOAT/DOUBLE/g;s/%f/%lf/g;s/79f/79/g;s/fold(&/\/\/fold(&/g;s/\ +\ xfc//g' > dpoisson.c
	$(CCPU) -o dpoisson dpoisson.c -lm
	rm dpoisson.c

dich: dich.c
	$(CCPU) $(OPTCPU) -o dich dich.c -lm

ddich: dich.c cdich.py
	cat dich.c | sed 's/float/double/g;s/.0f/.0/g;s/.5f/.5/g;s/FLOAT/DOUBLE/g;s/%f/%lf/g;s/79f/79/g;s/fold(&/\/\/fold(&/g;s/\ +\ xfc//g' > ddich.c
	$(CCPU) -o ddich ddich.c -lm
	rm ddich.c
	cat cdich.py | sed 's/dich/ddich/g' > dcdich.py




