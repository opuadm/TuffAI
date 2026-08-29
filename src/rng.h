#ifndef RNG_H
#define RNG_H

#include <stdint.h>

int rng_init(void);
uint32_t rng_u32(void);
int rng_range(int limit);
float rng_unit(void);
float rng_signed(void);

#endif
