#define _GNU_SOURCE
#include "rng.h"
#include <sys/random.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>

static uint64_t rng_state[4];
static int rng_ready = 0;
static pthread_mutex_t rng_mutex = PTHREAD_MUTEX_INITIALIZER;

static uint64_t rotate_left(uint64_t value, int shift) {
    return (value << shift) | (value >> (64 - shift));
}

static int read_entropy(void *buffer, unsigned long size) {
    unsigned char *bytes;
    long result;
    unsigned long offset;
    int descriptor;

    bytes = buffer;
    offset = 0;
    while (offset < size) {
        result = getrandom(bytes + offset, size - offset, 0);
        if (result > 0) {
            offset += (unsigned long)result;
        } else if (result < 0 && errno == EINTR) {
            continue;
        } else {
            break;
        }
    }
    if (offset == size) return 1;

    descriptor = open("/dev/urandom", O_RDONLY);
    if (descriptor < 0) return 0;
    offset = 0;
    while (offset < size) {
        result = read(descriptor, bytes + offset, size - offset);
        if (result > 0) {
            offset += (unsigned long)result;
        } else if (result < 0 && errno == EINTR) {
            continue;
        } else {
            close(descriptor);
            return 0;
        }
    }
    close(descriptor);
    return 1;
}

int rng_init(void) {
    int i;
    int nonzero;
    uint64_t fresh[4];

    if (!read_entropy(fresh, sizeof(fresh))) return 0;
    nonzero = 0;
    for (i = 0; i < 4; i++)
        if (fresh[i] != 0) nonzero = 1;
    if (!nonzero) return 0;
    pthread_mutex_lock(&rng_mutex);
    rng_state[0] = fresh[0];
    rng_state[1] = fresh[1];
    rng_state[2] = fresh[2];
    rng_state[3] = fresh[3];
    rng_ready = 1;
    pthread_mutex_unlock(&rng_mutex);
    return 1;
}

uint32_t rng_u32(void) {
    uint64_t result;
    uint64_t temporary;
    int ready;

    pthread_mutex_lock(&rng_mutex);
    ready = rng_ready;
    pthread_mutex_unlock(&rng_mutex);
    if (!ready && !rng_init()) return 0;
    pthread_mutex_lock(&rng_mutex);
    result = rotate_left(rng_state[1] * 5, 7) * 9;
    temporary = rng_state[1] << 17;
    rng_state[2] ^= rng_state[0];
    rng_state[3] ^= rng_state[1];
    rng_state[1] ^= rng_state[2];
    rng_state[0] ^= rng_state[3];
    rng_state[2] ^= temporary;
    rng_state[3] = rotate_left(rng_state[3], 45);
    result >>= 32;
    pthread_mutex_unlock(&rng_mutex);
    return (uint32_t)result;
}

int rng_range(int limit) {
    uint32_t value;
    uint32_t threshold;

    if (limit <= 1) return 0;
    threshold = (uint32_t)(-(uint32_t)limit) % (uint32_t)limit;
    do {
        value = rng_u32();
    } while (value < threshold);
    return (int)(value % (uint32_t)limit);
}

float rng_unit(void) {
    return (float)(rng_u32() >> 8) * (1.0f / 16777216.0f);
}

float rng_signed(void) {
    return rng_unit() * 2.0f - 1.0f;
}
