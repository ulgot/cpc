ARCH =-arch=sm_20 -m64
#gpu
CC = nvcc
OPT =--use_fast_math -O3
CURAND =-L/usr/local/cuda/lib64 -lcurand
#cpu
CCPU = gcc
OPTCPU =-ffast-math -O3 -fsingle-precision-constant

all: single double

single: prog.cu
	$(CC) $(ARCH) $(OPT) -o prog prog.cu $(CURAND) -lm

double: double_prog.cu
	$(CC) $(ARCH) -o prog double_prog.cu $(CURAND) -lm

cpu: prog.c
	$(CCPU) $(OPTCPU) -o progcpu prog.c -lm
