#include <stdio.h>
#include <math.h>
int main(){
  long a = 1000,
       b = 400,
       c = 6000;
  float d = 0.1;

  float triger = a * b * d;
  int sample_triger = rintf(triger / c);

  printf("%ld %ld %ld %f %f %d\n",a,b,c,d,triger,sample_triger);
  return 0;
}
