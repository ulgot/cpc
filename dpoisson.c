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
#include <sys/time.h>

#define PI 3.14159265358979

//model
double d_Dp, d_lambda, d_mean; // poissonian noise

//simulation
double d_trans;
long d_paths, d_periods, d_steps;
int d_spp, d_samples;
double d_x, d_xb, d_dx, d_dt;

static struct option options[] = {
    {"Dp", required_argument, NULL, 'b'},
    {"lambda", required_argument, NULL, 'c'},
    {"mean", required_argument, NULL, 'i'},
    {"paths", required_argument, NULL, 'l'},
    {"periods", required_argument, NULL, 'm'},
    {"trans", required_argument, NULL, 'n'},
    {"spp", required_argument, NULL, 'o'},
    {"samples", required_argument, NULL, 'j'}
};

void usage(char **argv)
{
    printf("Usage: %s <params> \n\n", argv[0]);
    printf("Model params:\n");
    printf("    -b, --Dp=DOUBLE          set the Poissonian noise intensity 'D_P' to DOUBLE\n");
    printf("    -c, --lambda=DOUBLE      set the Poissonian kicks frequency '\\lambda' to DOUBLE\n");
    printf("    -i, --mean=DOUBLE        if is nonzero, fix the mean value of Poissonian noise or dichotomous noise to DOUBLE, matters only for domains p, l, a, b, m or n\n");
    printf("Simulation params:\n");
    printf("    -l, --paths=LONG        set the number of paths to LONG\n");
    printf("    -m, --periods=LONG      set the number of periods to LONG\n");
    printf("    -n, --trans=DOUBLE       specify fraction DOUBLE of periods which stands for transients\n");
    printf("    -o, --spp=INT           specify how many integration steps should be calculated for a single period of the driving force\n");
    printf("    -j, --samples=INT       specify how many integration steps should be calculated for a single kernel call\n");
    printf("\n");
}

void parse_cla(int argc, char **argv)
{
    double ftmp;
    int c, itmp;

    while( (c = getopt_long(argc, argv, "b:c:i:l:m:n:o:j", options, NULL)) != EOF) {
        switch (c) {
            case 'b':
		sscanf(optarg, "%lf", &d_Dp);
                break;
            case 'c':
		sscanf(optarg, "%lf", &d_lambda);
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
            case 'j':
		sscanf(optarg, "%d", &d_samples);
                break;
            }
    }
}

double drift(double l_x)
{
  if (-sin(PI*l_x) < 0.0)
    return -1.0;
  else
    return 1.0;
}

double u01()
//easy to extend for any library with better statistics/algorithms (e.g. GSL)
{
  return (double)rand()/RAND_MAX;
}

double adapted_jump_poisson(int *npcd, int pcd, double l_lambda, double l_Dp, double l_dt)
{
  double comp = sqrtf(l_Dp*l_lambda)*l_dt;
  
  if (pcd <= 0) {
    double ampmean = sqrtf(l_lambda/l_Dp);
    *npcd = (int) floorf( -log( u01() )/l_lambda/l_dt + 0.5 );
    return -log( u01() )/ampmean - comp;
  } 
  else {
    *npcd = pcd - 1;
    return -comp;
  }
}

void predcorr(double *corrl_x, double l_x, int *npcd, int pcd, double l_Dp, double l_lambda, double l_dt)
/* simplified weak order 2.0 adapted predictor-corrector scheme
( see E. Platen, N. Bruti-Liberati; Numerical Solution of Stochastic Differential Equations with Jumps in Finance; Springer 2010; p. 503, p. 532 )
*/
{
    double l_xt, l_xtt, predl_x;

    l_xt = drift(l_x);

    predl_x = l_x + l_xt*l_dt;

    l_xtt = drift(predl_x);

    predl_x = l_x + 0.5*(l_xt + l_xtt)*l_dt;

    l_xtt = drift(predl_x);

    *corrl_x = l_x + 0.5*(l_xt + l_xtt)*l_dt + adapted_jump_poisson(npcd, pcd, l_lambda, l_Dp, l_dt);
}

void fold(double *nx, double x, double y, double *nfc, double fc)
//reduce periodic variable to the base domain
{
  *nx = x - floorf(x/y)*y;
  *nfc = fc + floorf(x/y)*y;
}

void run_moments()
//actual moments kernel
{
  long i;
  int sample;
  //cache path and model parameters in local variables
  //this is only to maintain original GPGPU code
  double l_x = d_x,
	l_xb;

  double l_Dp = d_Dp,
	l_lambda = d_lambda,
	l_mean = d_mean;

  //step size & number of steps
  double l_dt = 1.0/l_lambda/d_spp;


  //store step size in global mem
  d_dt = l_dt;

  long l_steps = d_steps,
       sample_trigger = lrint(d_trans * d_steps / d_samples),
       steps_samples = l_steps/d_samples;

  //counters for folding
  double xfc = 0.0;

  //jump countdowns
  int pcd = (int) floorf( -log( u01() )/l_lambda/l_dt + 0.5 );
 
  for (i = 0; i < steps_samples; i++) {

    for (sample = 0; sample < d_samples; sample++) {
      predcorr(&l_x, l_x, &pcd, pcd, l_Dp, l_lambda, l_dt);
      //fold path parameters
      //fold(&l_x, l_x, 2.0, &xfc, xfc);
      }

    if (i == sample_trigger) 
      l_xb = l_x;// + xfc;

  }

  //write back path parameters to the global memory
  d_x = l_x;// + xfc;
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
  return (d_x - d_xb)/( (1.0 - d_trans)*d_steps*d_dt );
}

void print_params()
{
  printf("#Dp %e\n",d_Dp);
  printf("#lambda %e\n",d_lambda);
  printf("#mean %e\n",d_mean);
  printf("#paths %ld\n",d_paths);
  printf("#periods %ld\n",d_periods);
  printf("#trans %lf\n",d_trans);
  printf("#spp %d\n",d_spp);
}

long long current_timestamp() {
  struct timeval te; 
  gettimeofday(&te, NULL); // get current time
  long long milliseconds = te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
  return milliseconds;
}

int main(int argc, char **argv)
{
  long long t0, te;
  double tsim;

  parse_cla(argc, argv);
  //print_params();

  if (0) usage(argv);
  prepare();
  
  //asymptotic long time average velocity <<v>>
  double av = 0.0;
  int i;

  int dump_av = 2;
  printf("#[1]no_of_runs [2]cpu_time(milisec) [3]Gsteps/sec [4]<<v>>\n");
  t0 = current_timestamp();
  for (i = 0; i < d_paths; ++i){

    initial_conditions();
    run_moments();
    av += moments();

    if (i == dump_av - 1){
      te = current_timestamp();
      tsim = te - t0;
      fprintf(stdout,"%d %lf %e %e\n", i+1, tsim, (i+1)*d_periods*d_spp*(1.0e-12)/tsim,av/(i+1));
      fflush(stdout);
      dump_av *= 2;
    }
  }

  return EXIT_SUCCESS;
}
