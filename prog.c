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

#define PI 3.14159265358979f

//model
float d_Dg, d_Dp, d_lambda, d_mean, d_fa, d_fb, d_mua, d_mub;
int d_comp;
float h_lambda, h_fa, h_fb, h_mua, h_mub, h_mean;
int h_comp;

//simulation
float d_trans;
int h_dev, h_block, h_grid, h_spp;
long h_paths, d_periods, h_threads, h_steps, h_trigger;
int d_spp;
long d_paths, d_steps, d_trigger;

//output
char *h_domain;
char h_domainx;
float h_beginx, h_endx;
int h_logx, h_points, d_moments;
char d_domainx;
int d_points;

//vector
float *h_x, *h_xb, *h_dx;
float *d_x, *d_xb, *d_dx;
unsigned int *h_seeds, *d_seeds;

size_t size_f, size_ui, size_p;
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
    {"paths", required_argument, NULL, 'l'},
    {"periods", required_argument, NULL, 'm'},
    {"trans", required_argument, NULL, 'n'},
    {"spp", required_argument, NULL, 'o'},
    {"mode", required_argument, NULL, 'p'}
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
    printf("    -l, --paths=LONG        set the number of paths to LONG\n");
    printf("    -m, --periods=LONG      set the number of periods to LONG\n");
    printf("    -n, --trans=FLOAT       specify fraction FLOAT of periods which stands for transients\n");
    printf("    -o, --spp=INT           specify how many integration steps should be calculated for a single period of the driving force\n");
    printf("Output params:\n");
    printf("    -p, --mode=STRING       sets the output mode. STRING can be one of:\n");
    printf("                            moments: the first moment <<v>>\n");
    printf("\n");
}

void parse_cla(int argc, char **argv)
{
    float ftmp;
    int c, itmp;

    while( (c = getopt_long(argc, argv, "a:b:c:d:e:f:g:h:i:l:m:n:o:p", options, NULL)) != EOF) {
        switch (c) {
            case 'a':
		scanf(optarg, "%f", &d_Dg);
                break;
            case 'b':
		scanf(optarg, "%f", &d_Dp);
                break;
            case 'c':
		scanf(optarg, "%f", &d_lambda);
                break;
            case 'd':
		scanf(optarg, "%f", &d_fa);
                break;
            case 'e':
		scanf(optarg, "%f", &d_fb);
                break;
            case 'f':
		scanf(optarg, "%f", &d_mua);
                break;
            case 'g':
		scanf(optarg, "%f", &d_mub);
                break;
            case 'h':
		scanf(optarg, "%d", &d_comp);
                break;
            case 'i':
		scanf(optarg, "%f", &d_mean);
                break;
            case 'l':
		scanf(optarg, "%ld", &d_paths);
                break;
            case 'm':
		scanf(optarg, "%ld", &d_periods);
                break;
            case 'n':
		scanf(optarg, "%f", &d_trans);
                d_trans = atof(optarg);
                break;
            case 'o':
		scanf(optarg, "%d", &d_spp);
                break;
            case 'p':
                if ( !strcmp(optarg, "moments") ) {
                    d_moments = 1;
                }
                break;
            }
    }
}

float drift(float l_x)
{
    float l_y, l_f;

    l_y = fmod(l_x, 2.0f);

    if (l_y < -1.0f) {
        l_y += 2.0f;
    } else if (l_y > 1.0f) {
        l_y -= 2.0f;
    }

    if (l_y >= -1.0f && l_y < 0.0f) {
        l_f = 1.0f;
    } else if (l_y >= 0.0f && l_y <= 1.0f) {
        l_f = -1.0f;
    }

    return l_f;
}

float u01()
//easy to extend for any library with better statistics/algorithms (e.g. GSL)
{
  return rand()/RAND_MAX;
}

float diffusion(float l_Dg, float l_dt)
{
    if (l_Dg != 0.0f) {
        float r = u01();
        if ( r <= 1.0f/6 ) {
            return -sqrtf(6.0f*l_Dg*l_dt);
        } else if ( r > 1.0f/6 && r <= 2.0f/6 ) {
            return sqrtf(6.0f*l_Dg*l_dt);
        } else {
            return 0.0f;
        }
    } else {
        return 0.0f;
    }
}

float adapted_jump_poisson(int &npcd, int pcd, float l_lambda, float l_Dp, int l_comp, float l_dt)
{
    if (l_lambda != 0.0f) {
        if (pcd <= 0) {
            float ampmean = sqrtf(l_lambda/l_Dp);
           
            npcd = (int) floor( -logf( u01() )/l_lambda/l_dt + 0.5f );

            if (l_comp) {
                float comp = sqrtf(l_Dp*l_lambda)*l_dt;
                
                return -logf( u01() )/ampmean - comp;
            } else {
                return -logf( u01() )/ampmean;
            }
        } else {
            npcd = pcd - 1;
            if (l_comp) {
                float comp = sqrtf(l_Dp*l_lambda)*l_dt;
                
                return -comp;
            } else {
                return 0.0f;
            }
        }
    } else {
        return 0.0f;
    }
}

float adapted_jump_dich(int &ndcd, int dcd, int &ndst, int dst, float l_fa, float l_fb, float l_mua, float l_mub, float l_dt)
{
    if (l_mua != 0.0f || l_mub != 0.0f) {
        if (dcd <= 0) {
            if (dst == 0) {
                ndst = 1; 
                ndcd = (int) floor( -logf( u01() )/l_mub/l_dt + 0.5f );
                return l_fb*l_dt;
            } else {
                ndst = 0;
                ndcd = (int) floor( -logf( u01() )/l_mua/l_dt + 0.5f );
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
        return 0.0f;
    }
}

void predcorr(float &corrl_x, float l_x, int &npcd, int pcd, \
                         float l_Dg, float l_Dp, float l_lambda, int l_comp, \
                         int &ndcd, int dcd, int &ndst, int dst, float l_fa, float l_fb, float l_mua, float l_mub, float l_dt)
/* simplified weak order 2.0 adapted predictor-corrector scheme
( see E. Platen, N. Bruti-Liberati; Numerical Solution of Stochastic Differential Equations with Jumps in Finance; Springer 2010; p. 503, p. 532 )
*/
{
    float l_xt, l_xtt, predl_x;

    l_xt = drift(l_x);

    predl_x = l_x + l_xt*l_dt + diffusion(l_Dg, l_dt);

    l_xtt = drift(predl_x);

    predl_x = l_x + 0.5f*(l_xt + l_xtt)*l_dt + diffusion(l_Dg, l_dt);

    l_xtt = drift(predl_x);

    corrl_x = l_x + 0.5f*(l_xt + l_xtt)*l_dt + adapted_jump_dich(ndcd, dcd, ndst, dst, l_fa, l_fb, l_mua, l_mub, l_dt, l_state) + diffusion(l_Dg, l_dt) + adapted_jump_poisson(npcd, pcd, l_lambda, l_Dp, l_comp, l_dt);
}

void fold(float &nx, float x, float y, float &nfc, float fc)
//reduce periodic variable to the base domain
{
    nx = x - floor(x/y)*y;
    nfc = fc + floor(x/y)*y;
}

void run_moments(float *d_x, float *d_xb, float *d_dx)
//actual moments kernel
{
    float l_x, l_xb, l_dx; 

    //cache path and model parameters in local variables
    l_x = d_x;
    l_xb = d_xb;

    float l_Dg, l_Dp, l_lambda, l_mean, l_fa, l_fb, l_mua, l_mub;
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

    //step size & number of steps
    float l_dt;
    long l_steps, l_trigger, i;

    if (l_lambda != 0.0f) 
        l_dt = 1.0f/l_lambda/d_spp;

    if (l_mua != 0.0f || l_mub != 0.0f) {
        float taua, taub;

        taua = 1.0f/l_mua;
        taub = 1.0f/l_mub;

        if (taua < taub) 
            l_dt = taua/d_spp;
        else 
            l_dt = taub/d_spp;
        
    }

    l_steps = d_steps;
    l_trigger = d_trigger;

    //counters for folding
    float xfc;
    
    xfc = 0.0f;

    int pcd, dcd, dst;

    //jump countdowns
    if (l_lambda != 0.0f) pcd = (int) floor( -logf( u01() )/l_lambda/l_dt + 0.5f );

    if (l_mua != 0.0f || l_mub != 0.0f) {
        float rn;
        rn = u01();

        if (rn < 0.5f) {
            dst = 0;
            dcd = (int) floor( -logf( u01() )/l_mua/l_dt + 0.5f);
        } else {
            dst = 1;
            dcd = (int) floor( -logf( u01() )/l_mub/l_dt + 0.5f);
        }
    }
    
    for (i = 0; i < l_steps; i++) {

        predcorr(l_x, l_x, pcd, pcd, l_Dg, l_Dp, l_lambda, l_comp, \
                 dcd, dcd, dst, dst, l_fa, l_fb, l_mua, l_mub, l_dt);
        
        //fold path parameters
        if ( fabs(l_x) > 2.0f ) {
            fold(l_x, l_x, 2.0f, xfc, xfc);
        }

        if (i == l_trigger) {
            l_xb = l_x + xfc;
        }

    }

    //write back path parameters to the global memory
    d_x = l_x + xfc;
    d_xb = l_xb;
}

void prepare()
//prepare simulation
{
    //number of steps
    if (d_moments) h_steps = d_periods*h_spp;
     
    //host memory allocation
    size_f = h_threads*sizeof(float);
    size_ui = h_threads*sizeof(unsigned int);
    size_p = h_points*sizeof(float);

    h_x = (float*)malloc(size_f);

    //initialization of rng
    srand(time(0));
}

void initial_conditions()
//set initial conditions for path parameters
{
    int i;

    for (i = 0; i < h_threads; i++) {
        h_x[i] = 2.0f*h_x[i] - 1.0f; //x in (-1,1]
    }

    if (d_moments) {
        memset(h_xb, 0, size_f);
    }
    
    copy_to_dev();
}

void moments(float *av)
//calculate the first moment of v
{
    float sx, sxb, tmp, taua, taub, dt;
    int i, j;

    cudaMemcpy(h_x, d_x, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_xb, d_xb, size_f, cudaMemcpyDeviceToHost);

    for (j = 0; j < h_points; j++) {
        sx = 0.0f;
        sxb = 0.0f;

        for (i = 0; i < h_paths; i++) {
            sx += h_x[j*h_paths + i];
            sxb += h_xb[j*h_paths + i];
        }

        //Poissonian
        if (h_domainx == 'l') {
            dt = 1.0f/h_dx[j]/h_spp;
        } else if (h_domainx == 'p' && h_mean != 0.0f) {
            dt = 1.0f/(h_mean*h_mean/h_dx[j])/h_spp;
        } else if (h_lambda != 0.0f) {
            dt = 1.0f/h_lambda/h_spp;
        }

        //Dichotomous
        if (h_domainx == 'm') {
            taua = 1.0f/h_dx[j];
            taub = 1.0f/h_mub;

            if (h_comp) {
                tmp = 1.0f/(-h_fb*h_dx[j]/h_fa);
            } else if (h_mean != 0.0f) {
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
            taua = 1.0f/h_mua;
            taub = 1.0f/h_dx[j];

            if (h_comp) {
                tmp = 1.0f/(-h_fa*h_dx[j]/h_fb);
            } else if (h_mean != 0.0f) {
                tmp = (h_fa - h_mean)*h_dx[j]/(h_mean - h_fb);
            } else {
                tmp = taua;
            }

            if (taub <= tmp) {
                dt = taub/h_spp;
            } else {
                dt = tmp/h_spp;
            }
        } else if (h_mua != 0.0f || h_mub != 0.0f) {
            taua = 1.0f/h_mua;
            taub = 1.0f/h_mub;

            if (taua < taub) {
                dt = taua/h_spp;
            } else {
                dt = taub/h_spp;
            }
        }
            
        av[j] = (sx - sxb)/( (1.0f - d_trans)*h_steps*dt )/h_paths;
    }
}

void finish()
//free memory
{

    free(h_x);
    
    if (d_moments) {
        free(h_xb);
        free(h_dx);
    }
}

int main(int argc, char **argv)
{
    parse_cla(argc, argv);
    if (!d_moments) {
        usage(argv);
        return -1;
    }

    prepare();
    
    initial_conditions();
    
    //asymptotic long time average velocity <<v>>
    if (d_moments) {
        float *av;
        int i;

        av = (float*)malloc(size_p);

        run_moments(d_x, d_xb, d_dx);
        moments(av);

        printf("#%c <<v>>\n", h_domainx);
        for (i = 0; i < h_points; i++) {
          printf("%e %e\n", h_dx[i], av[i]);
        }   

        free(av);
    }

    finish();

    return EXIT_SUCCESS;
}
