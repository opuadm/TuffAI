#include "net.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

float embeddings[ACTUAL_VOCAB][EMBED_DIM];
float W1[EMBED_DIM * 3][HIDDEN];
float b1[HIDDEN];
float W2[HIDDEN][EMBED_DIM];
float b2[EMBED_DIM];
float W_feat[16][EMBED_DIM];

static int last_words[64];
static int last_words_n = 0;

float frand_r(void) {
    return ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

void net_init(void) {
    int i, j, k;

    srand(0xDEADC0DE);
    for (i = 0; i < ACTUAL_VOCAB; i++)
        for (j = 0; j < EMBED_DIM; j++)
            embeddings[i][j] = frand_r() * 0.6f;
    for (i = 0; i < EMBED_DIM * 3; i++)
        for (j = 0; j < HIDDEN; j++)
            W1[i][j] = frand_r() * 0.25f;
    for (j = 0; j < HIDDEN; j++) b1[j] = frand_r() * 0.1f;
    for (j = 0; j < HIDDEN; j++)
        for (k = 0; k < EMBED_DIM; k++)
            W2[j][k] = frand_r() * 0.25f;
    for (k = 0; k < EMBED_DIM; k++) b2[k] = frand_r() * 0.1f;
    for (i = 0; i < 16; i++)
        for (j = 0; j < EMBED_DIM; j++)
            W_feat[i][j] = frand_r() * 0.4f;
    srand((unsigned)time(NULL));
}

void encode_context(const int *tokens, int n, float *ctx) {
    float w, tw;
    int i, d;

    memset(ctx, 0, EMBED_DIM * sizeof(float));
    w = 1.0f;
    tw = 0.0f;
    for (i = n - 1; i >= 0 && i >= n - 8; i--) {
        for (d = 0; d < EMBED_DIM; d++) ctx[d] += embeddings[tokens[i]][d] * w;
        tw += w;
        w *= 0.65f;
    }
    if (tw > 0)
        for (d = 0; d < EMBED_DIM; d++) ctx[d] /= tw;
}

void next_embed(const float *ctx, const float *prev, const float *feat_ctx, float *out) {
    float inp[EMBED_DIM * 3];
    float h[HIDDEN];
    int i, j, k;
    float z;

    memcpy(inp, ctx, EMBED_DIM * sizeof(float));
    memcpy(inp + EMBED_DIM, prev, EMBED_DIM * sizeof(float));
    memcpy(inp + EMBED_DIM * 2, feat_ctx, EMBED_DIM * sizeof(float));

    for (j = 0; j < HIDDEN; j++) {
        z = b1[j];
        for (i = 0; i < EMBED_DIM * 3; i++) z += inp[i] * W1[i][j];
        h[j] = tanhf(z);
    }
    for (k = 0; k < EMBED_DIM; k++) {
        z = b2[k];
        for (j = 0; j < HIDDEN; j++) z += h[j] * W2[j][k];
        out[k] = tanhf(z);
    }
}

void push_recent(int w) {
    int i;

    for (i = 63; i > 0; i--) last_words[i] = last_words[i - 1];
    last_words[0] = w;
    if (last_words_n < 64) last_words_n++;
}

void reset_recent(void) {
    last_words_n = 0;
}

int sample_vocab(const float *target, float temperature, float noise,
                 float freq_penalty, float rep_penalty, float top_p) {
    float scores[ACTUAL_VOCAB];
    float maxs, sum, r, cum;
    int i, d;
    float dot, penalty;
    int rep_count;
    float thresh;
    float eff_noise;
    int appeared;

    eff_noise = noise;
    if (eff_noise < 0.5f) eff_noise = 0.5f;

    for (i = 0; i < ACTUAL_VOCAB; i++) {
        dot = 0;
        for (d = 0; d < EMBED_DIM; d++) dot += target[d] * embeddings[i][d];

        rep_count = 0;
        appeared = 0;
        for (d = 0; d < last_words_n; d++) {
            if (last_words[d] == i) {
                rep_count++;
                appeared = 1;
            }
        }
        penalty = 0.0f;
        if (rep_count > 0) {
            penalty -= freq_penalty * (float)rep_count;
            penalty -= rep_penalty * 2.0f;
        }
        if (appeared) {
            penalty -= 1.5f;
        }

        scores[i] = dot / temperature + ((float)rand() / RAND_MAX) * eff_noise + penalty;
    }
    maxs = scores[0];
    for (i = 1; i < ACTUAL_VOCAB; i++)
        if (scores[i] > maxs) maxs = scores[i];
    sum = 0;
    for (i = 0; i < ACTUAL_VOCAB; i++) {
        scores[i] = expf(scores[i] - maxs);
        sum += scores[i];
    }

    if (top_p < 1.0f && top_p > 0.0f) {
        thresh = sum * top_p;
        cum = 0;
        for (i = 0; i < ACTUAL_VOCAB; i++) {
            cum += scores[i];
            if (cum > thresh) {
                scores[i] = 0.0f;
            }
        }
        sum = 0;
        for (i = 0; i < ACTUAL_VOCAB; i++) sum += scores[i];
    }

    r = ((float)rand() / RAND_MAX) * sum;
    cum = 0;
    for (i = 0; i < ACTUAL_VOCAB; i++) {
        cum += scores[i];
        if (r < cum) return i;
    }
    return ACTUAL_VOCAB - 1;
}
