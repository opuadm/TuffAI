#ifndef NET_H
#define NET_H

#include "knowledge/knowledge.h"

extern float embeddings[MAX_VOCAB_SIZE][EMBED_DIM];
extern float W1[EMBED_DIM * 3][HIDDEN];
extern float b1[HIDDEN];
extern float W2[HIDDEN][EMBED_DIM];
extern float b2[EMBED_DIM];
extern float W_feat[16][EMBED_DIM];

void net_init(void);
void encode_context(const int *tokens, int n, float *ctx);
void next_embed(const float *ctx, const float *prev, const float *feat_ctx, float *out);
int sample_vocab(const float *target, float temperature, float noise,
                 float freq_penalty, float rep_penalty, float top_p, int vocab_size);
void push_recent(int w);
void reset_recent(void);
float frand_r(void);

#endif
