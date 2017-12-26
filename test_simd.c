#include <mmintrin.h>
#include <smmintrin.h>
#include <xmmintrin.h>
#include <stdlib.h>
#include <stdio.h>

void print_vector(const char* name, const __m128& val) {
  float store[4];
  _mm_store_ps(store, val);
  printf("%s = [%f %f %f %f]\n", name, store[0], store[1], store[2], store[3]);
}

int main(int ac, char* av[]) {
  float vals1[4] = {0, 1, 2, 3};
  float vals2[4] = {4, 5, 6, 7};
  float out[4];
  __m128 v1 = _mm_load_ps(vals1);
  __m128 v2 = _mm_load_ps(vals2);

  //_mm_storeh_pi(out, v1);
  //printf("out = %f %f\n", out[0], out[1]);

  __m128 hi = _mm_unpackhi_ps(v1, v2);
  print_vector("v1", v1);
  print_vector("v2", v2);
  print_vector("mm_unpack_hi(v1, v2)", hi);

  __m128 lo = _mm_unpacklo_ps(v1, v2);
  print_vector("v1", v1);
  print_vector("v2", v2);
  print_vector("mm_unpack_lo(v1, v2)", lo);


  __m128 shuf = _mm_shuffle_ps(v1, v1,
                               1 + (1<<2) + (3<<4) + (3<<6));
  print_vector("shuf", shuf);

  __m128 w11 = _mm_movehl_ps(_mm_setzero_ps(), v2);
  __m128 test = _mm_shuffle_ps(w11, w11,
                               0 + (2<<2) + (2<<4) + (2<<6));
  print_vector("extract_2", test);
  return 0;
}
