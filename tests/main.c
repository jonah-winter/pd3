#include "pd3.h"
#include <stdio.h>

void print_mat(pdmat3* m) {
  for(int i=0; i < 9; i++) {
    printf("%d ", (*m)[i]);
  }
}

int main(void) {
  pdmat3 a = {
    1,2,3,
    0,1,4,
    5,6,0,
  };
  pdmat3 b;
  pdmat3 c;
  char e[256];
  if(pd3_newton(a, b, c, 1e-6, 50, e) == PD3_ERRC) {
    fprintf(stderr, "failed\n");
    return 1;
  }
  return 0;
}
