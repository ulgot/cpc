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

//simulation
float d_trans;
long d_paths, d_periods, d_steps, d_trigger;
int d_spp;
float d_x, d_xb, d_dx, d_dt;

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
    {"spp", required_argument, NULL, 'o'}
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
    printf("\n");
}

void parse_cla(int argc, char **argv)
{
    float ftmp;
    int c, itmp;

    while( (c = getopt_long(argc, argv, "a:b:c:d:e:f:g:h:i:l:m:n:o", options, NULL)) != EOF) {
        switch (c) {
            case 'a':
		sscanf(optarg, "%f", &d_Dg);
                break;
            case 'b':
		sscanf(optarg, "%f", &d_Dp);
                break;
            case 'c':
		sscanf(optarg, "%f", &d_lambda);
                break;
            case 'd':
		sscanf(optarg, "%f", &d_fa);
                break;
            case 'e':
		sscanf(optarg, "%f", &d_fb);
                break;
            case 'f':
		sscanf(optarg, "%f", &d_mua);
                break;
            case 'g':
		sscanf(optarg, "%f", &d_mub);
                break;
            case 'h':
		sscanf(optarg, "%d", &d_comp);
                break;
            case 'i':
		sscanf(optarg, "%f", &d_mean);
                break;
            case 'l':
		sscanf(optarg, "%ld", &d_paths);
                break;
            case 'm':
		sscanf(optarg, "%ld", &d_periods);
                break;
            case 'n':
		sscanf(optarg, "%f", &d_trans);
                break;
            case 'o':
		sscanf(optarg, "%d", &d_spp);
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
  return (float)rand()/RAND_MAX;
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

float adapted_jump_poisson(int *npcd, int pcd, float l_lambda, float l_Dp, int l_comp, float l_dt)
{
  if (l_lambda != 0.0f) {
    if (pcd <= 0) {
      float ampmean = sqrtf(l_lambda/l_Dp);
      *npcd = (int) floor( -logf( u01() )/l_lambda/l_dt + 0.5f );
      
      if (l_comp) {
	float comp = sqrtf(l_Dp*l_lambda)*l_dt;
	return -logf( u01() )/ampmean - comp;
      } 
      else 
	return -logf( u01() )/ampmean;
    } 
    else {
      *npcd = pcd - 1;
      if (l_comp) {
	float comp = sqrtf(l_Dp*l_lambda)*l_dt;
	return -comp;
      } 
      else 
	return 0.0f;
    }
  } else {
    return 0.0f;
  }
}

float adapted_jump_dich(int *ndcd, int dcd, int *ndst, int dst, float l_fa, float l_fb, float l_mua, float l_mub, float l_dt)
{
    if (l_mua != 0.0f || l_mub != 0.0f) {
        if (dcd <= 0) {
            if (dst == 0) {
                *ndst = 1; 
                *ndcd = (int) floor( -logf( u01() )/l_mub/l_dt + 0.5f );
                return l_fb*l_dt;
            } else {
                *ndst = 0;
                *ndcd = (int) floor( -logf( u01() )/l_mua/l_dt + 0.5f );
                return l_fa*l_dt;
            }
        } else {
            *ndcd = dcd - 1;
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

void predcorr(float *corrl_x, float l_x, int *npcd, int pcd, \
                         float l_Dg, float l_Dp, float l_lambda, int l_comp, \
                         int *ndcd, int dcd, int *ndst, int dst, float l_fa, float l_fb, float l_mua, float l_mub, float l_dt)
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

    *corrl_x = l_x + 0.5f*(l_xt + l_xtt)*l_dt + adapted_jump_dich(ndcd, dcd, ndst, dst, l_fa, l_fb, l_mua, l_mub, l_dt) + diffusion(l_Dg, l_dt) + adapted_jump_poisson(npcd, pcd, l_lambda, l_Dp, l_comp, l_dt);
}

void fold(float *nx, float x, float y, float *nfc, float fc)
//reduce periodic variable to the base domain
{
  *nx = x - floor(x/y)*y;
  *nfc = fc + floor(x/y)*y;
}

void run_moments()
//actual moments kernel
{
  long i;
  //cache path and model parameters in local variables
  //this is only to maintain original GPGPU code
  float l_x = d_x,
	l_xb;

  int l_comp = d_comp;

  float l_Dg = d_Dg,
	l_Dp = d_Dp,
	l_lambda = d_lambda,
	l_mean = d_mean,
	l_fa = d_fa,
	l_fb = d_fb,
	l_mua = d_mua,
	l_mub = d_mub;

  //step size & number of steps
  float l_dt;

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
  //store step size in global mem
  d_dt = l_dt;

  float l_steps = d_steps,
	l_trigger = d_trigger;

  //counters for folding
  float xfc = 0.0f;

  int pcd, dcd, dst;

  //jump countdowns
  if (l_lambda != 0.0f) pcd = (int) floor( -logf( u01() )/l_lambda/l_dt + 0.5f );

  if (l_mua != 0.0f || l_mub != 0.0f) {
      float rn = u01();

      if (rn < 0.5f) {
	  dst = 0;
	  dcd = (int) floor( -logf( u01() )/l_mua/l_dt + 0.5f);
      } else {
	  dst = 1;
	  dcd = (int) floor( -logf( u01() )/l_mub/l_dt + 0.5f);
      }
  }
  
  for (i = 0; i < l_steps; i++) {

      predcorr(&l_x, l_x, &pcd, pcd, l_Dg, l_Dp, l_lambda, l_comp, \
	       &dcd, dcd, &dst, dst, l_fa, l_fb, l_mua, l_mub, l_dt);
      
      //fold path parameters
      if ( fabs(l_x) > 2.0f ) {
	fold(&l_x, l_x, 2.0f, &xfc, xfc);
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
  d_steps = d_periods*d_spp;

  //initialization of rng
  srand(time(NULL));
}

void initial_conditions()
//set initial conditions for path parameters
{
  d_x = 2.0f*u01() - 1.0f; //x in (-1,1]
}

float moments()
//calculate the first moment of v
{
  float dt, av;

  av = (d_x - d_xb)/( (1.0f - d_trans)*d_steps*d_dt )/d_paths;
  return av;
}

void print_params()
{
  printf("#Dg %e\n",d_Dg);
  printf("#Dp %e\n",d_Dp);
  printf("#lambda %e\n",d_lambda);
  printf("#fa %e\n",d_fa);
  printf("#fb %e\n",d_fb);
  printf("#mua %e\n",d_mua);
  printf("#mub %e\n",d_mub);
  printf("#comp %d\n",d_comp);
  printf("#mean %e\n",d_mean);
  printf("#paths %ld\n",d_paths);
  printf("#periods %ld\n",d_periods);
  printf("#trans %f\n",d_trans);
  printf("#spp %d\n",d_spp);
}

int main(int argc, char **argv)
{
  parse_cla(argc, argv);
  print_params();

  if (0) usage(argv);
  prepare();
  
  //asymptotic long time average velocity <<v>>
  float av = 0.0f;
  int i;

  for (i = 0; i < d_paths; ++i){
    initial_conditions();
    printf("#p no %d: %f -> ",i, d_x);
    run_moments();
    printf("(%f) %f\n",d_xb,d_x);
    av += moments();
  }

  printf("#<<v>>\n%e\n", av);

  return EXIT_SUCCESS;
}
