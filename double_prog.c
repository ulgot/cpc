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

#define PI 3.14159265358979

//model
double d_Dg, d_Dp, d_lambda, d_mean, d_fa, d_fb, d_mua, d_mub;
int d_comp;

//simulation
double d_trans;
long d_paths, d_periods, d_steps, d_trigger;
int d_spp;
double d_x, d_xb, d_dx, d_dt;

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
    printf("    -a, --Dg=DOUBLE          set the Gaussian noise intensity 'D_G' to DOUBLE\n");
    printf("    -b, --Dp=DOUBLE          set the Poissonian noise intensity 'D_P' to DOUBLE\n");
    printf("    -c, --lambda=DOUBLE      set the Poissonian kicks frequency '\\lambda' to DOUBLE\n");
    printf("    -d, --fa=DOUBLE          set the first state of the dichotomous noise 'F_a' to DOUBLE\n");
    printf("    -e, --fb=DOUBLE          set the second state of the dichotomous noise 'F_b' to DOUBLE\n");
    printf("    -f, --mua=DOUBLE         set the transition rate of the first state of dichotomous noise '\\mu_a' to DOUBLE\n");
    printf("    -g, --mub=DOUBLE         set the transition rate of the second state of dichotomous noise '\\mu_b' to DOUBLE\n");
    printf("    -h, --comp=INT          choose between biased and unbiased Poissonian or dichotomous noise. INT can be one of:\n");
    printf("                            0: biased; 1: unbiased\n");
    printf("    -i, --mean=DOUBLE        if is nonzero, fix the mean value of Poissonian noise or dichotomous noise to DOUBLE, matters only for domains p, l, a, b, m or n\n");
    printf("Simulation params:\n");
    printf("    -l, --paths=LONG        set the number of paths to LONG\n");
    printf("    -m, --periods=LONG      set the number of periods to LONG\n");
    printf("    -n, --trans=DOUBLE       specify fraction DOUBLE of periods which stands for transients\n");
    printf("    -o, --spp=INT           specify how many integration steps should be calculated for a single period of the driving force\n");
    printf("\n");
}

void parse_cla(int argc, char **argv)
{
    double ftmp;
    int c, itmp;

    while( (c = getopt_long(argc, argv, "a:b:c:d:e:f:g:h:i:l:m:n:o", options, NULL)) != EOF) {
        switch (c) {
            case 'a':
		sscanf(optarg, "%lf", &d_Dg);
                break;
            case 'b':
		sscanf(optarg, "%lf", &d_Dp);
                break;
            case 'c':
		sscanf(optarg, "%lf", &d_lambda);
                break;
            case 'd':
		sscanf(optarg, "%lf", &d_fa);
                break;
            case 'e':
		sscanf(optarg, "%lf", &d_fb);
                break;
            case 'f':
		sscanf(optarg, "%lf", &d_mua);
                break;
            case 'g':
		sscanf(optarg, "%lf", &d_mub);
                break;
            case 'h':
		sscanf(optarg, "%d", &d_comp);
                break;
            case 'i':
		sscanf(optarg, "%lf", &d_mean);
                break;
            case 'l':
		sscanf(optarg, "%ld", &d_paths);
                break;
            case 'm':
		sscanf(optarg, "%ld", &d_periods);
                break;
            case 'n':
		sscanf(optarg, "%lf", &d_trans);
                break;
            case 'o':
		sscanf(optarg, "%d", &d_spp);
                break;
            }
    }
}

double drift(double l_x)
{
    double l_y, l_f;

    l_y = fmod(l_x, 2.0);

    if (l_y < -1.0) {
        l_y += 2.0;
    } else if (l_y > 1.0) {
        l_y -= 2.0;
    }

    if (l_y >= -1.0 && l_y < 0.0) {
        l_f = 1.0;
    } else if (l_y >= 0.0 && l_y <= 1.0) {
        l_f = -1.0;
    }

    return l_f;
}

double u01()
//easy to extend for any library with better statistics/algorithms (e.g. GSL)
{
  return (double)rand()/RAND_MAX;
}

double diffusion(double l_Dg, double l_dt)
{
  if (l_Dg != 0.0) {
      double r = u01();
      if ( r <= 1.0/6 ) {
	  return -sqrtf(6.0*l_Dg*l_dt);
      } else if ( r > 1.0/6 && r <= 2.0/6 ) {
	  return sqrtf(6.0*l_Dg*l_dt);
      } else {
	  return 0.0;
      }
  } else {
      return 0.0;
  }
}

double adapted_jump_poisson(int *npcd, int pcd, double l_lambda, double l_Dp, int l_comp, double l_dt)
{
  if (l_lambda != 0.0) {
    if (pcd <= 0) {
      double ampmean = sqrtf(l_lambda/l_Dp);
      *npcd = (int) floor( -logf( u01() )/l_lambda/l_dt + 0.5 );
      
      if (l_comp) {
	double comp = sqrtf(l_Dp*l_lambda)*l_dt;
	return -logf( u01() )/ampmean - comp;
      } 
      else 
	return -logf( u01() )/ampmean;
    } 
    else {
      *npcd = pcd - 1;
      if (l_comp) {
	double comp = sqrtf(l_Dp*l_lambda)*l_dt;
	return -comp;
      } 
      else 
	return 0.0;
    }
  } else {
    return 0.0;
  }
}

double adapted_jump_dich(int *ndcd, int dcd, int *ndst, int dst, double l_fa, double l_fb, double l_mua, double l_mub, double l_dt)
{
    if (l_mua != 0.0 || l_mub != 0.0) {
        if (dcd <= 0) {
            if (dst == 0) {
                *ndst = 1; 
                *ndcd = (int) floor( -logf( u01() )/l_mub/l_dt + 0.5 );
                return l_fb*l_dt;
            } else {
                *ndst = 0;
                *ndcd = (int) floor( -logf( u01() )/l_mua/l_dt + 0.5 );
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
        return 0.0;
    }
}

void predcorr(double *corrl_x, double l_x, int *npcd, int pcd, \
                         double l_Dg, double l_Dp, double l_lambda, int l_comp, \
                         int *ndcd, int dcd, int *ndst, int dst, double l_fa, double l_fb, double l_mua, double l_mub, double l_dt)
/* simplified weak order 2.0 adapted predictor-corrector scheme
( see E. Platen, N. Bruti-Liberati; Numerical Solution of Stochastic Differential Equations with Jumps in Finance; Springer 2010; p. 503, p. 532 )
*/
{
    double l_xt, l_xtt, predl_x;

    l_xt = drift(l_x);

    predl_x = l_x + l_xt*l_dt + diffusion(l_Dg, l_dt);

    l_xtt = drift(predl_x);

    predl_x = l_x + 0.5*(l_xt + l_xtt)*l_dt + diffusion(l_Dg, l_dt);

    l_xtt = drift(predl_x);

    *corrl_x = l_x + 0.5*(l_xt + l_xtt)*l_dt + adapted_jump_dich(ndcd, dcd, ndst, dst, l_fa, l_fb, l_mua, l_mub, l_dt) + diffusion(l_Dg, l_dt) + adapted_jump_poisson(npcd, pcd, l_lambda, l_Dp, l_comp, l_dt);
}

void fold(double *nx, double x, double y, double *nfc, double fc)
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
  double l_x = d_x,
	l_xb;

  int l_comp = d_comp;

  double l_Dg = d_Dg,
	l_Dp = d_Dp,
	l_lambda = d_lambda,
	l_mean = d_mean,
	l_fa = d_fa,
	l_fb = d_fb,
	l_mua = d_mua,
	l_mub = d_mub;

  //step size & number of steps
  double l_dt;

  if (l_lambda != 0.0) 
      l_dt = 1.0/l_lambda/d_spp;

  if (l_mua != 0.0 || l_mub != 0.0) {
      double taua, taub;

      taua = 1.0/l_mua;
      taub = 1.0/l_mub;

      if (taua < taub) 
	  l_dt = taua/d_spp;
      else 
	  l_dt = taub/d_spp;
      
  }
  //store step size in global mem
  d_dt = l_dt;

  double l_steps = d_steps,
	l_trigger = d_trigger;

  //counters for folding
  double xfc = 0.0;

  int pcd, dcd, dst;

  //jump countdowns
  if (l_lambda != 0.0) pcd = (int) floor( -logf( u01() )/l_lambda/l_dt + 0.5 );

  if (l_mua != 0.0 || l_mub != 0.0) {
      double rn = u01();

      if (rn < 0.5) {
	  dst = 0;
	  dcd = (int) floor( -logf( u01() )/l_mua/l_dt + 0.5);
      } else {
	  dst = 1;
	  dcd = (int) floor( -logf( u01() )/l_mub/l_dt + 0.5);
      }
  }
  
  for (i = 0; i < l_steps; i++) {

      predcorr(&l_x, l_x, &pcd, pcd, l_Dg, l_Dp, l_lambda, l_comp, \
	       &dcd, dcd, &dst, dst, l_fa, l_fb, l_mua, l_mub, l_dt);
      
      //fold path parameters
      if ( fabs(l_x) > 2.0 ) {
	fold(&l_x, l_x, 2.0, &xfc, xfc);
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
  d_x = 2.0*u01() - 1.0; //x in (-1,1]
}

double moments()
//calculate the first moment of v
{
  return (d_x - d_xb)/( (1.0 - d_trans)*d_steps*d_dt )/d_paths;
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
  printf("#trans %lf\n",d_trans);
  printf("#spp %d\n",d_spp);
}

int main(int argc, char **argv)
{
  parse_cla(argc, argv);
  //print_params();

  if (0) usage(argv);
  prepare();
  
  //asymptotic long time average velocity <<v>>
  double av = 0.0;
  int i;

  for (i = 0; i < d_paths; ++i){
    initial_conditions();
    run_moments();
    av += moments();
  }

  //printf("#<<v>>\n%e\n", av);
  printf("%e\n", av);

  return EXIT_SUCCESS;
}
