/*
 * Overdamped Brownian particle in symmetric piecewise linear potential
 *
 * \dot{x} = -V'(x) + Gaussian, Poissonian and dichotomous noise
 *
 */

#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define PI 3.14159265358979f

//model
__constant__ double d_Dg, d_Dp, d_lambda, d_mean, d_fa, d_fb, d_mua, d_mub;
__constant__ int d_comp;
double h_lambda, h_fa, h_fb, h_mua, h_mub, h_mean;
int h_comp;

//simulation
double h_trans;
int h_dev, h_block, h_grid, h_spp, h_samples;
long h_paths, h_periods, h_threads, h_steps, h_trigger;
__constant__ int d_spp, d_samples;
__constant__ long d_paths;

//output
char *h_domain;
char h_domainx;
double h_beginx, h_endx;
int h_logx, h_points, h_moments;
__constant__ char d_domainx;
__constant__ int d_points;

//vector
double *h_x, *h_xb, *h_fx, *h_dx;
double *d_x, *d_fx, *d_dx;
int *d_pcd, *d_dcd, *d_dst;
unsigned int *h_seeds, *d_seeds;
curandState *d_states;

size_t size_f, size_i, size_ui, size_p;
curandGenerator_t gen;

static struct option options[] = {
    {"Dg", required_argument, NULL, 'a'},
    {"Dp", required_argument, NULL, 'b'},
    {"lambda", required_argument, NULL, 'c'},
    {"fa", required_argument, NULL, 'd'},
    {"fb", required_argument, NULL, 'e'},
    {"mua", required_argument, NULL, 'f'},
    {"mub", required_argument, NULL, 'g'},
    {"comp", required_argument, NULL, 'h'},
    {"mean", required_argument, NULL, 'i'},
    {"dev", required_argument, NULL, 'j'},
    {"block", required_argument, NULL, 'k'},
    {"paths", required_argument, NULL, 'l'},
    {"periods", required_argument, NULL, 'm'},
    {"trans", required_argument, NULL, 'n'},
    {"spp", required_argument, NULL, 'o'},
    {"samples", required_argument, NULL, 'p'},
    {"mode", required_argument, NULL, 'q'},
    {"domain", required_argument, NULL, 'r'},
    {"domainx", required_argument, NULL, 's'},
    {"logx", required_argument, NULL, 't'},
    {"points", required_argument, NULL, 'u'},
    {"beginx", required_argument, NULL, 'v'},
    {"endx", required_argument, NULL, 'w'}
};

void usage(char **argv)
{
    printf("Usage: %s <params> \n\n", argv[0]);
    printf("Model params:\n");
    printf("    -a, --Dg=FLOAT          set the Gaussian noise intensity 'D_G' to FLOAT\n");
    printf("    -b, --Dp=FLOAT          set the Poissonian noise intensity 'D_P' to FLOAT\n");
    printf("    -c, --lambda=FLOAT      set the Poissonian kicks frequency '\\lambda' to FLOAT\n");
    printf("    -d, --fa=FLOAT          set the first state of the dichotomous noise 'F_a' to FLOAT\n");
    printf("    -e, --fb=FLOAT          set the second state of the dichotomous noise 'F_b' to FLOAT\n");
    printf("    -f, --mua=FLOAT         set the transition rate of the first state of dichotomous noise '\\mu_a' to FLOAT\n");
    printf("    -g, --mub=FLOAT         set the transition rate of the second state of dichotomous noise '\\mu_b' to FLOAT\n");
    printf("    -h, --comp=INT          choose between biased and unbiased Poissonian or dichotomous noise. INT can be one of:\n");
    printf("                            0: biased; 1: unbiased\n");
    printf("    -i, --mean=FLOAT        if is nonzero, fix the mean value of Poissonian noise or dichotomous noise to FLOAT, matters only for domains p, l, a, b, m or n\n");
    printf("Simulation params:\n");
    printf("    -j, --dev=INT           set the gpu device to INT\n");
    printf("    -k, --block=INT         set the gpu block size to INT\n");
    printf("    -l, --paths=LONG        set the number of paths to LONG\n");
    printf("    -m, --periods=LONG      set the number of periods to LONG\n");
    printf("    -n, --trans=FLOAT       specify fraction FLOAT of periods which stands for transients\n");
    printf("    -o, --spp=INT           specify how many integration steps should be calculated for the smallest characteristic time scale\n");
    printf("    -p, --samples=INT       specify how many integration steps should be calculated for a single kernel call\n");
    printf("Output params:\n");
    printf("    -q, --mode=STRING       sets the output mode. STRING can be one of:\n");
    printf("                            moments: the first moment <<v>>\n");
    printf("    -r, --domain=STRING     simultaneously scan over one or two model params. STRING can be one of:\n");
    printf("                            1d: only one parameter\n");
    printf("    -s, --domainx=CHAR      sets the first domain of the moments. CHAR can be one of:\n");
    printf("                            D: Dg; p: Dp; l: lambda; a: fa; b: fb; m: mua; n: mub\n");
    printf("    -t, --logx=INT          choose between linear and logarithmic scale of the domainx\n");
    printf("                            0: linear; 1: logarithmic\n");
    printf("    -u, --points=INT        set the number of samples to generate between begin and end\n");
    printf("    -v, --beginx=FLOAT      set the starting value of the domainx to FLOAT\n");
    printf("    -w, --endx=FLOAT        set the end value of the domainx to FLOAT\n");
    printf("\n");
}

void parse_cla(int argc, char **argv)
{
    double ftmp;
    int c, itmp;

    while( (c = getopt_long(argc, argv, "a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w", options, NULL)) != EOF) {
        switch (c) {
            case 'a':
                ftmp = atof(optarg);
                cudaMemcpyToSymbol(d_Dg, &ftmp, sizeof(double));
                break;
            case 'b':
                ftmp = atof(optarg);
                cudaMemcpyToSymbol(d_Dp, &ftmp, sizeof(double));
                break;
            case 'c':
                h_lambda = atof(optarg);
                cudaMemcpyToSymbol(d_lambda, &h_lambda, sizeof(double));
                break;
            case 'd':
                h_fa = atof(optarg);
                cudaMemcpyToSymbol(d_fa, &h_fa, sizeof(double));
                break;
            case 'e':
                h_fb = atof(optarg);
                cudaMemcpyToSymbol(d_fb, &h_fb, sizeof(double));
                break;
            case 'f':
                h_mua = atof(optarg);
                cudaMemcpyToSymbol(d_mua, &h_mua, sizeof(double));
                break;
            case 'g':
                h_mub = atof(optarg);
                cudaMemcpyToSymbol(d_mub, &h_mub, sizeof(double));
                break;
            case 'h':
                h_comp = atoi(optarg);
                cudaMemcpyToSymbol(d_comp, &h_comp, sizeof(int));
                break;
            case 'i':
                h_mean = atof(optarg);
                cudaMemcpyToSymbol(d_mean, &h_mean, sizeof(double));
                break;
            case 'j':
                itmp = atoi(optarg);
                cudaSetDevice(itmp);
                break;
            case 'k':
                h_block = atoi(optarg);
                break;
            case 'l':
                h_paths = atol(optarg);
                cudaMemcpyToSymbol(d_paths, &h_paths, sizeof(long));
                break;
            case 'm':
                h_periods = atol(optarg);
                break;
            case 'n':
                h_trans = atof(optarg);
                break;
            case 'o':
                h_spp = atoi(optarg);
                cudaMemcpyToSymbol(d_spp, &h_spp, sizeof(int));
                break;
            case 'p':
                h_samples = atoi(optarg);
                cudaMemcpyToSymbol(d_samples, &h_samples, sizeof(int));
                break;
            case 'q':
                if ( !strcmp(optarg, "moments") ) {
                    h_moments = 1;
                }
                break;
            case 'r':
                h_domain = optarg;
                break;
            case 's':
                h_domainx = optarg[0]; 
                cudaMemcpyToSymbol(d_domainx, &h_domainx, sizeof(char));
                break;
            case 't':
                h_logx = atoi(optarg);
                break;
            case 'u':
                h_points = atoi(optarg);
                cudaMemcpyToSymbol(d_points, &h_points, sizeof(int));
                break;
            case 'v':
                h_beginx = atof(optarg);
                break;
            case 'w':
                h_endx = atof(optarg);
                break;
        }
    }
}

__global__ void init_dev_rng(unsigned int *d_seeds, curandState *d_states)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(d_seeds[idx], 0, 0, &d_states[idx]);
}

__device__ double drift(double l_x)
{    
    if (-sin(PI*l_x) < 0.0) {
        return -1.0;
    } else {
        return 1.0;
    }
}

__device__ double diffusion(double l_Dg, double l_dt, curandState *l_state)
{
    if (l_Dg != 0.0) {
        double r = curand_uniform_double(l_state);
        double g = sqrt(2.0*l_Dg);
        if ( r <= 1.0/6.0 ) {
            return -g*sqrt(3.0*l_dt);
        } else if ( r > 1.0/6.0 && r <= 1.0/3.0 ) {
            return g*sqrt(3.0*l_dt);
        } else {
            return 0.0;
        }
    } else {
        return 0.0;
    }
}

__global__ void init_noise(double *d_dx, int *d_pcd, int *d_dcd, int *d_dst, curandState *d_states)
//init noise
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    double l_dx; 
    curandState l_state;

    //cache model parameters in local variables
    l_state = d_states[idx];

    double l_Dp, l_lambda, l_mean, l_fa, l_fb, l_mua, l_mub;
    int l_comp;

    l_Dp = d_Dp;
    l_lambda = d_lambda;
    l_comp = d_comp;
    l_mean = d_mean;
    l_fa = d_fa;
    l_fb = d_fb;
    l_mua = d_mua;
    l_mub = d_mub;

    long ridx = (idx/d_paths) % d_points;
    l_dx = d_dx[ridx];

    switch(d_domainx) {
        case 'p':
            l_Dp = l_dx;
            if (l_mean != 0.0) l_lambda = (l_mean*l_mean)/l_Dp;
            break;
        case 'l':
            l_lambda = l_dx;
            if (l_mean != 0.0) l_Dp = (l_mean*l_mean)/l_lambda;
            break;
        case 'a':
            l_fa = l_dx;
            if (l_comp == 1) {
                l_fb = -l_fa*l_mub/l_mua;
            } else if (l_mean != 0.0) {
                l_fb = (l_mean*(l_mua + l_mub) - l_fa*l_mub)/l_mua;
            }
            break;
        case 'b':
            l_fb = l_dx;
            if (l_comp == 1) {
                l_fa = -l_fb*l_mua/l_mub;
            } else if (l_mean != 0.0) {
                l_fa = (l_mean*(l_mua + l_mub) - l_fb*l_mua)/l_mub;
            }
            break;
        case 'm':
            l_mua = l_dx;
            if (l_comp == 1) {
                l_mub = -l_fb*l_mua/l_fa;
            } else if (l_mean != 0.0) {
                l_mub = (l_fb - l_mean)*l_mua/(l_mean - l_fa);
            }
            break;
        case 'n':
            l_mub = l_dx;
            if (l_comp == 1) {
                l_mua = -l_fa*l_mub/l_fb;
            } else if (l_mean != 0.0) {
                l_mua = (l_fa - l_mean)*l_mub/(l_mean - l_fb);
            }
            break;
    }

    //step size
    double l_dt;
    int l_spp;

    l_spp = d_spp;

    if (l_lambda != 0.0) {
        l_dt = 1.0/l_lambda/l_spp;
    }

    if (l_mua != 0.0) {
        double taua, taub;

        taua = 1.0/l_mua;
        taub = 1.0/l_mub;

        if (taua < taub) {
            l_dt = taua/l_spp;
        } else {
            l_dt = taub/l_spp;
        }
    }

    //jump countdowns
    int l_pcd, l_dcd, l_dst;
    
    if (l_lambda != 0.0) l_pcd = (int) floor( -log( curand_uniform_double(&l_state) )/l_lambda/l_dt + 0.5 );

    if (l_mua != 0.0) {
        double rn;
        rn = curand_uniform_double(&l_state);

        if (rn < 0.5) {
            l_dst = 0;
            l_dcd = (int) floor( -log( curand_uniform_double(&l_state) )/l_mua/l_dt + 0.5);
        } else {
            l_dst = 1;
            l_dcd = (int) floor( -log( curand_uniform_double(&l_state) )/l_mub/l_dt + 0.5);
        }
    }
    
    //write back noise states to the global memory
    d_pcd[idx] = l_pcd;
    d_dcd[idx] = l_dcd;
    d_dst[idx] = l_dst;
    d_states[idx] = l_state;
}

__device__ double adapted_jump_poisson(int &npcd, int pcd, double l_lambda, double l_Dp, int l_comp, double l_dt, curandState *l_state)
{
    if (l_lambda != 0.0) {
        if (pcd <= 0) {
            double ampmean = sqrt(l_lambda/l_Dp);
           
            npcd = (int) floor( -log( curand_uniform_double(l_state) )/l_lambda/l_dt + 0.5 );

            if (l_comp) {
                double comp = sqrt(l_Dp*l_lambda)*l_dt;
                
                return -log( curand_uniform_double(l_state) )/ampmean - comp;
            } else {
                return -log( curand_uniform_double(l_state) )/ampmean;
            }
        } else {
            npcd = pcd - 1;
            if (l_comp) {
                double comp = sqrt(l_Dp*l_lambda)*l_dt;
                
                return -comp;
            } else {
                return 0.0;
            }
        }
    } else {
        return 0.0;
    }
}

__device__ double adapted_jump_dich(int &ndcd, int dcd, int &ndst, int dst, double l_fa, double l_fb, double l_mua, double l_mub, double l_dt, curandState *l_state)
{
    if (l_mua != 0.0) {
        if (dcd <= 0) {
            if (dst == 0) {
                ndst = 1; 
                ndcd = (int) floor( -log( curand_uniform_double(l_state) )/l_mub/l_dt + 0.5 );
                return l_fb*l_dt;
            } else {
                ndst = 0;
                ndcd = (int) floor( -log( curand_uniform_double(l_state) )/l_mua/l_dt + 0.5 );
                return l_fa*l_dt;
            }
        } else {
            ndcd = dcd - 1;
            if (dst == 0) {
                return l_fa*l_dt;
            } else {
                return l_fb*l_dt;
            }
        }
    } else {
        return 0.0;
    }
}

__device__ void predcorr(double &corrl_x, double l_x, int &npcd, int pcd, curandState *l_state, \
                         double l_Dg, double l_Dp, double l_lambda, int l_comp, \
                         int &ndcd, int dcd, int &ndst, int dst, double l_fa, double l_fb, double l_mua, double l_mub, double l_dt)
/* simplified weak order 2.0 adapted predictor-corrector scheme
( see E. Platen, N. Bruti-Liberati; Numerical Solution of Stochastic Differential Equations with Jumps in Finance; Springer 2010; p. 503, p. 532 )
*/
{
    double l_xt, l_xtt, predl_x;

    l_xt = drift(l_x);

    predl_x = l_x + l_xt*l_dt + diffusion(l_Dg, l_dt, l_state);

    l_xtt = drift(predl_x);

    predl_x = l_x + 0.5*(l_xt + l_xtt)*l_dt + diffusion(l_Dg, l_dt, l_state);

    l_xtt = drift(predl_x);

    corrl_x = l_x + 0.5*(l_xt + l_xtt)*l_dt + adapted_jump_dich(ndcd, dcd, ndst, dst, l_fa, l_fb, l_mua, l_mub, l_dt, l_state) + diffusion(l_Dg, l_dt, l_state) + adapted_jump_poisson(npcd, pcd, l_lambda, l_Dp, l_comp, l_dt, l_state);
}

__global__ void fold(double *d_x, double *d_fx)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    double l_x, l_fx, f;

    l_x = d_x[idx];
    l_fx = d_fx[idx];

    f = floor(l_x/2.0)*2.0;
    l_x = l_x - f;
    l_fx = l_fx + f;

    d_x[idx] = l_x;
    d_fx[idx] = l_fx;
}

void unfold(double *x, double *fx)
{
    int i;

    for (i = 0; i < h_threads; i++) {
        x[i] = x[i] + fx[i];
    }
}

__global__ void run_moments(double *d_x, double *d_dx, int *d_pcd, int *d_dcd, int *d_dst, curandState *d_states)
//actual moments kernel
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    double l_x, l_dx; 
    curandState l_state;

    //cache path and model parameters in local variables
    l_x = d_x[idx];
    l_state = d_states[idx];

    double l_Dg, l_Dp, l_lambda, l_mean, l_fa, l_fb, l_mua, l_mub;
    int l_comp;

    l_Dg = d_Dg;
    l_Dp = d_Dp;
    l_lambda = d_lambda;
    l_comp = d_comp;
    l_mean = d_mean;
    l_fa = d_fa;
    l_fb = d_fb;
    l_mua = d_mua;
    l_mub = d_mub;

    //run simulation for multiple values of the system parameters
    long ridx = (idx/d_paths) % d_points;
    l_dx = d_dx[ridx];

    switch(d_domainx) {
        case 'D':
            l_Dg = l_dx;
            break;
        case 'p':
            l_Dp = l_dx;
            if (l_mean != 0.0) l_lambda = (l_mean*l_mean)/l_Dp;
            break;
        case 'l':
            l_lambda = l_dx;
            if (l_mean != 0.0) l_Dp = (l_mean*l_mean)/l_lambda;
            break;
        case 'a':
            l_fa = l_dx;
            if (l_comp == 1) {
                l_fb = -l_fa*l_mub/l_mua;
            } else if (l_mean != 0.0) {
                l_fb = (l_mean*(l_mua + l_mub) - l_fa*l_mub)/l_mua;
            }
            break;
        case 'b':
            l_fb = l_dx;
            if (l_comp == 1) {
                l_fa = -l_fb*l_mua/l_mub;
            } else if (l_mean != 0.0) {
                l_fa = (l_mean*(l_mua + l_mub) - l_fb*l_mua)/l_mub;
            }
            break;
        case 'm':
            l_mua = l_dx;
            if (l_comp == 1) {
                l_mub = -l_fb*l_mua/l_fa;
            } else if (l_mean != 0.0) {
                l_mub = (l_fb - l_mean)*l_mua/(l_mean - l_fa);
            }
            break;
        case 'n':
            l_mub = l_dx;
            if (l_comp == 1) {
                l_mua = -l_fa*l_mub/l_fb;
            } else if (l_mean != 0.0) {
                l_mua = (l_fa - l_mean)*l_mub/(l_mean - l_fb);
            }
            break;
    }

    //step size & number of steps
    double l_dt;
    int i, l_spp, l_samples;

    l_spp = d_spp;

    if (l_lambda != 0.0) {
        l_dt = 1.0/l_lambda/l_spp;
    }

    if (l_mua != 0.0) {
        double taua, taub;

        taua = 1.0/l_mua;
        taub = 1.0/l_mub;

        if (taua < taub) {
            l_dt = taua/l_spp;
        } else {
            l_dt = taub/l_spp;
        }
    }

    l_samples = d_samples;

    //jump countdowns
    int l_pcd, l_dcd, l_dst;
    
    l_pcd = d_pcd[idx];
    l_dcd = d_dcd[idx];
    l_dst = d_dst[idx];

    for (i = 0; i < l_samples; i++) {
        predcorr(l_x, l_x, l_pcd, l_pcd, &l_state, l_Dg, l_Dp, l_lambda, l_comp, \
                 l_dcd, l_dcd, l_dst, l_dst, l_fa, l_fb, l_mua, l_mub, l_dt); 
    }

    //write back path parameters to the global memory
    d_x[idx] = l_x;
    d_pcd[idx] = l_pcd;
    d_dcd[idx] = l_dcd;
    d_dst[idx] = l_dst;
    d_states[idx] = l_state;
}

void prepare()
//prepare simulation
{
    //grid size
    h_paths = (h_paths/h_block)*h_block;
    h_threads = h_paths;

    if (h_moments) h_threads *= h_points;

    h_grid = h_threads/h_block;

    //number of steps
    if (h_moments) h_steps = h_periods*h_spp;
     
    //host memory allocation
    size_f = h_threads*sizeof(double);
    size_i = h_threads*sizeof(int);
    size_ui = h_threads*sizeof(unsigned int);
    size_p = h_points*sizeof(double);

    h_x = (double*)malloc(size_f);
    h_seeds = (unsigned int*)malloc(size_ui);

    //create & initialize host rng
    curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

    curandGenerate(gen, h_seeds, h_threads);
 
    //device memory allocation
    cudaMalloc((void**)&d_x, size_f);
    cudaMalloc((void**)&d_seeds, size_ui);
    cudaMalloc((void**)&d_pcd, size_i);
    cudaMalloc((void**)&d_dcd, size_i);
    cudaMalloc((void**)&d_dst, size_i);
    cudaMalloc((void**)&d_states, h_threads*sizeof(curandState));

    //copy seeds from host to device
    cudaMemcpy(d_seeds, h_seeds, size_ui, cudaMemcpyHostToDevice);

    //initialization of device rng
    init_dev_rng<<<h_grid, h_block>>>(d_seeds, d_states);

    free(h_seeds);
    cudaFree(d_seeds);

    //moments specific requirements
    if (h_moments) {
        h_trigger = h_steps*h_trans;

        h_xb = (double*)malloc(size_f);
        h_fx = (double*)malloc(size_f);
        h_dx = (double*)malloc(size_p);

        double dxtmp = h_beginx;
        double dxstep = (h_endx - h_beginx)/h_points;

        int i;
        
        //set domainx
        for (i = 0; i < h_points; i++) {
            if (h_logx) {
                h_dx[i] = exp10(dxtmp);
            } else {
                h_dx[i] = dxtmp;
            }
            dxtmp += dxstep;
        }
        
        cudaMalloc((void**)&d_fx, size_f);
        cudaMalloc((void**)&d_dx, size_p);
    
        cudaMemcpy(d_fx, h_fx, size_f, cudaMemcpyHostToDevice);
        cudaMemcpy(d_dx, h_dx, size_p, cudaMemcpyHostToDevice);
    }
}

void copy_to_dev()
{
    cudaMemcpy(d_x, h_x, size_f, cudaMemcpyHostToDevice);
}

void copy_from_dev()
{
    cudaMemcpy(h_x, d_x, size_f, cudaMemcpyDeviceToHost);
}

void initial_conditions()
//set initial conditions for path parameters
{
    int i;

    curandGenerateUniformDouble(gen, h_x, h_threads); //x in (0,1]

    for (i = 0; i < h_threads; i++) {
        h_x[i] = 2.0*h_x[i] - 1.0; //x in (-1,1]
    }

    copy_to_dev();
}

void moments(double *av)
//calculate the first moment of v
{
    double sx, sxb, tmp, taua, taub, dt;
    int i, j;

    cudaMemcpy(h_x, d_x, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fx, d_fx, size_f, cudaMemcpyDeviceToHost);

    unfold(h_x, h_fx);

    for (j = 0; j < h_points; j++) {
        sx = 0.0;
        sxb = 0.0;

        for (i = 0; i < h_paths; i++) {
            sx += h_x[j*h_paths + i];
            sxb += h_xb[j*h_paths + i];
        }

        //Poissonian
        if (h_domainx == 'l') {
            dt = 1.0/h_dx[j]/h_spp;
        } else if (h_domainx == 'p' && h_mean != 0.0) {
            dt = 1.0/(h_mean*h_mean/h_dx[j])/h_spp;
        } else if (h_lambda != 0.0) {
            dt = 1.0/h_lambda/h_spp;
        }

        //Dichotomous
        if (h_domainx == 'm') {
            taua = 1.0/h_dx[j];
            taub = 1.0/h_mub;

            if (h_comp) {
                tmp = 1.0/(-h_fb*h_dx[j]/h_fa);
            } else if (h_mean != 0.0) {
                tmp = (h_fb - h_mean)*h_dx[j]/(h_mean - h_fa);
            } else {
                tmp = taub;
            }

            if (taua <= tmp) {
                dt = taua/h_spp;
            } else {
                dt = tmp/h_spp;
            }
        } else if (h_domainx == 'n') {
            taua = 1.0/h_mua;
            taub = 1.0/h_dx[j];

            if (h_comp) {
                tmp = 1.0/(-h_fa*h_dx[j]/h_fb);
            } else if (h_mean != 0.0) {
                tmp = (h_fa - h_mean)*h_dx[j]/(h_mean - h_fb);
            } else {
                tmp = taua;
            }

            if (taub <= tmp) {
                dt = taub/h_spp;
            } else {
                dt = tmp/h_spp;
            }
        } else if (h_mua != 0.0 || h_mub != 0.0) {
            taua = 1.0/h_mua;
            taub = 1.0/h_mub;

            if (taua < taub) {
                dt = taua/h_spp;
            } else {
                dt = taub/h_spp;
            }
        }
            
        av[j] = (sx - sxb)/( (1.0 - h_trans)*h_steps*dt )/h_paths;
    }
}

void finish()
//free memory
{

    free(h_x);
    
    curandDestroyGenerator(gen);
    cudaFree(d_x);
    cudaFree(d_pcd);
    cudaFree(d_dcd);
    cudaFree(d_dst);
    cudaFree(d_states);
    
    if (h_moments) {
        free(h_xb);
        free(h_fx);
        free(h_dx);

        cudaFree(d_fx);
        cudaFree(d_dx);
    }
}

int main(int argc, char **argv)
{
    parse_cla(argc, argv);
    if (!h_moments) {
        usage(argv);
        return -1;
    }
 
    prepare();
    
    initial_conditions();
    
    //asymptotic long time average velocity <<v>>
    if (h_moments) {
        double *av;
        int i;

        av = (double*)malloc(size_p);
 
        if ( !strcmp(h_domain, "1d") ) { 

            init_noise<<<h_grid, h_block>>>(d_dx, d_pcd, d_dcd, d_dst, d_states);

            for (i = 0; i < h_steps; i += h_samples) {
                run_moments<<<h_grid, h_block>>>(d_x, d_dx, d_pcd, d_dcd, d_dst, d_states);
                fold<<<h_grid, h_block>>>(d_x, d_fx);
                if (i == h_trigger) {
                    cudaMemcpy(h_xb, d_x, size_f, cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_fx, d_fx, size_f, cudaMemcpyDeviceToHost);
                    unfold(h_xb, h_fx);
                }
            }

            moments(av);
 
            printf("#%c <<v>>\n", h_domainx);
            for (i = 0; i < h_points; i++) {
                printf("%e %e\n", h_dx[i], av[i]);
            }
        }

        free(av);
    }

    finish();
    
    return 0;
}
