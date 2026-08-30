#include "net.h"
#include "rng.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__SSE__) || defined(_M_IX86) || defined(_M_X64)
#define USE_SSE 1
#include <xmmintrin.h>
#endif

#ifdef USE_SSE
static float dot_sse(const float *a, const float *b, int n) {
    __m128 sum0 = _mm_setzero_ps();
    __m128 sum1 = _mm_setzero_ps();
    int i;
    float result;

    for (i = 0; i + 7 < n; i += 8) {
        sum0 = _mm_add_ps(sum0, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(_mm_loadu_ps(a + i + 4), _mm_loadu_ps(b + i + 4)));
    }
    sum0 = _mm_add_ps(sum0, sum1);
    for (; i + 3 < n; i += 4)
        sum0 = _mm_add_ps(sum0, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    sum0 = _mm_add_ps(sum0, _mm_movehl_ps(sum0, sum0));
    sum0 = _mm_add_ss(sum0, _mm_shuffle_ps(sum0, sum0, 1));
    _mm_store_ss(&result, sum0);
    for (; i < n; i++) result += a[i] * b[i];
    return result;
}

#endif

float embeddings[MAX_VOCAB_SIZE][EMBED_DIM];
float W1[EMBED_DIM * 3][HIDDEN];
float b1[HIDDEN];
float W2[HIDDEN][EMBED_DIM];
float b2[EMBED_DIM];
float W_feat[16][EMBED_DIM];

static float attention_query[EMBED_DIM][EMBED_DIM];
static float attention_key[EMBED_DIM][EMBED_DIM];
static float attention_value[EMBED_DIM][EMBED_DIM];
static float attention_output[EMBED_DIM][EMBED_DIM];
static float position_embed[NET_CONTEXT_MAX][EMBED_DIM];

static int last_words[64];
static int last_words_n = 0;

static float fast_tanh_value(float value) {
    float magnitude;

    magnitude = value < 0.0f ? -value : value;
    return value / (1.0f + magnitude);
}

static float fast_exp_negative(float value) {
    float square;

    if (value < -10.0f) return 0.0f;
    square = value * value;
    return 1.0f / (1.0f - value + 0.48f * square);
}

float frand_r(void) {
    return rng_signed();
}

int net_init(void) {
    int i, j, k;

    if (!rng_init()) return 0;
    for (i = 0; i < MAX_VOCAB_SIZE; i++)
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
    for (i = 0; i < EMBED_DIM; i++) {
        for (j = 0; j < EMBED_DIM; j++) {
            attention_query[i][j] = frand_r() * 0.18f;
            attention_key[i][j] = frand_r() * 0.18f;
            attention_value[i][j] = frand_r() * 0.18f;
            attention_output[i][j] = frand_r() * 0.18f;
        }
    }
    for (i = 0; i < NET_CONTEXT_MAX; i++)
        for (j = 0; j < EMBED_DIM; j++)
            position_embed[i][j] = frand_r() * 0.12f;
    return 1;
}

void encode_context(const int *tokens, int n, float *ctx) {
    float query[EMBED_DIM];
    float key_query[EMBED_DIM];
    float attended_source[EMBED_DIM];
    float attended[EMBED_DIM];
    float projected[EMBED_DIM];
    float scores[NET_CONTEXT_MAX];
    float source[EMBED_DIM];
    float query_source[EMBED_DIM];
    float max_score;
    float score_sum;
    float mean;
    float variance;
    float inv_scale;
    int used;
    int first;
    int token_id;
    int i;
    int j;
    int d;

    memset(ctx, 0, EMBED_DIM * sizeof(float));
    if (!tokens || n <= 0) return;

    used = n < NET_CONTEXT_MAX ? n : NET_CONTEXT_MAX;
    first = n - used;
    inv_scale = 1.0f / sqrtf((float)EMBED_DIM);

    token_id = tokens[n - 1] % MAX_VOCAB_SIZE;
    if (token_id < 0) token_id += MAX_VOCAB_SIZE;
    for (d = 0; d < EMBED_DIM; d++)
        query_source[d] = embeddings[token_id][d] +
                          position_embed[used - 1][d];

    for (d = 0; d < EMBED_DIM; d++) {
        query[d] = 0.0f;
        for (j = 0; j < EMBED_DIM; j++)
            query[d] += query_source[j] * attention_query[j][d];
    }
    for (j = 0; j < EMBED_DIM; j++) {
        key_query[j] = 0.0f;
        for (d = 0; d < EMBED_DIM; d++)
            key_query[j] += attention_key[j][d] * query[d];
    }

    for (i = 0; i < used; i++) {
        token_id = tokens[first + i] % MAX_VOCAB_SIZE;
        if (token_id < 0) token_id += MAX_VOCAB_SIZE;
        for (d = 0; d < EMBED_DIM; d++)
            source[d] = embeddings[token_id][d] + position_embed[i][d];
        scores[i] = 0.0f;
        for (d = 0; d < EMBED_DIM; d++)
            scores[i] += source[d] * key_query[d];
        scores[i] *= inv_scale;
    }

    max_score = scores[0];
    for (i = 1; i < used; i++)
        if (scores[i] > max_score) max_score = scores[i];

    score_sum = 0.0f;
    for (i = 0; i < used; i++) {
        scores[i] = expf(scores[i] - max_score);
        score_sum += scores[i];
    }

    memset(attended_source, 0, sizeof(attended_source));
    for (i = 0; i < used; i++) {
        token_id = tokens[first + i] % MAX_VOCAB_SIZE;
        if (token_id < 0) token_id += MAX_VOCAB_SIZE;
        for (d = 0; d < EMBED_DIM; d++) {
            source[d] = embeddings[token_id][d] + position_embed[i][d];
            attended_source[d] += source[d] * scores[i] / score_sum;
        }
    }

    for (d = 0; d < EMBED_DIM; d++) {
        attended[d] = 0.0f;
        for (j = 0; j < EMBED_DIM; j++)
            attended[d] += attended_source[j] * attention_value[j][d];
    }

    for (d = 0; d < EMBED_DIM; d++) {
        projected[d] = 0.0f;
        for (j = 0; j < EMBED_DIM; j++)
            projected[d] += attended[j] * attention_output[j][d];
        ctx[d] = projected[d] + query_source[d];
    }

    mean = 0.0f;
    for (d = 0; d < EMBED_DIM; d++) mean += ctx[d];
    mean /= (float)EMBED_DIM;
    variance = 0.0f;
    for (d = 0; d < EMBED_DIM; d++)
        variance += (ctx[d] - mean) * (ctx[d] - mean);
    variance /= (float)EMBED_DIM;
    inv_scale = 1.0f / sqrtf(variance + 0.00001f);
    for (d = 0; d < EMBED_DIM; d++)
        ctx[d] = (ctx[d] - mean) * inv_scale;
}

void encode_context_fast(const int *tokens, int n, float *ctx) {
    float mean;
    float variance;
    float scale;
    float weight;
    float weight_sum;
    int used;
    int first;
    int token;
    int i;
    int d;

    memset(ctx, 0, EMBED_DIM * sizeof(float));
    if (!tokens || n <= 0) return;
    used = n < 16 ? n : 16;
    first = n - used;
    weight_sum = 0.0f;
    for (i = 0; i < used; i++) {
        token = tokens[first + i] % MAX_VOCAB_SIZE;
        if (token < 0) token += MAX_VOCAB_SIZE;
        weight = (float)(i + 1);
        weight_sum += weight;
        for (d = 0; d < EMBED_DIM; d++)
            ctx[d] += embeddings[token][d] * weight;
    }
    mean = 0.0f;
    for (d = 0; d < EMBED_DIM; d++) {
        ctx[d] /= weight_sum;
        mean += ctx[d];
    }
    mean /= (float)EMBED_DIM;
    variance = 0.0f;
    for (d = 0; d < EMBED_DIM; d++)
        variance += (ctx[d] - mean) * (ctx[d] - mean);
    variance /= (float)EMBED_DIM;
    scale = 1.0f / sqrtf(variance + 0.00001f);
    for (d = 0; d < EMBED_DIM; d++)
        ctx[d] = (ctx[d] - mean) * scale;
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
        out[k] = tanhf(z + ctx[k] * 0.28f + prev[k] * 0.12f);
    }
}

void next_embed_fast(const float *ctx, const float *prev,
                     const float *feat_ctx, float *out) {
    float inp[EMBED_DIM * 3];
    float h[16];
    float value;
    int i;
    int j;
    int k;

    memcpy(inp, ctx, EMBED_DIM * sizeof(float));
    memcpy(inp + EMBED_DIM, prev, EMBED_DIM * sizeof(float));
    memcpy(inp + EMBED_DIM * 2, feat_ctx, EMBED_DIM * sizeof(float));
    for (j = 0; j < 16; j++) {
        value = b1[j];
        for (i = 0; i < EMBED_DIM * 3; i++)
            value += inp[i] * W1[i][j];
        h[j] = fast_tanh_value(value);
    }
    for (k = 0; k < EMBED_DIM; k++) {
        value = b2[k];
        for (j = 0; j < 16; j++) value += h[j] * W2[j][k];
        out[k] = fast_tanh_value(
            value + ctx[k] * 0.28f + prev[k] * 0.12f);
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

static int sample_vocab_internal(const float *target, float temperature, float noise,
                                 float freq_penalty, float rep_penalty, float top_p,
                                 int vocab_size, const int *bias_tokens,
                                 int bias_count, float bias_strength) {
    float scores[MAX_VOCAB_SIZE];
    float context_bonuses[MAX_VOCAB_SIZE];
    float bucket_sums[256];
    unsigned char recent_counts[MAX_VOCAB_SIZE];
    float maxs, sum, r, cum;
    int i, d;
    float dot, penalty;
    int rep_count;
    float thresh;
    float eff_noise;
    int appeared;
    int bias_start;
    int bias_pos;
    float context_bonus;
    int bucket;
    int cutoff_bucket;

    eff_noise = noise;
    if (eff_noise < 0.5f) eff_noise = 0.5f;
    bias_start = bias_count > 256 ? bias_count - 256 : 0;
    memset(recent_counts, 0, (size_t)vocab_size * sizeof(recent_counts[0]));
    memset(context_bonuses, 0,
           (size_t)vocab_size * sizeof(context_bonuses[0]));
    for (d = 0; d < last_words_n; d++) {
        i = last_words[d];
        if (i >= 0 && i < vocab_size && recent_counts[i] < 255)
            recent_counts[i]++;
    }
    if (bias_tokens && bias_strength > 0.0f) {
        for (bias_pos = bias_start; bias_pos < bias_count; bias_pos++) {
            i = bias_tokens[bias_pos];
            if (i >= 0 && i < vocab_size)
                context_bonuses[i] += bias_strength *
                    (0.25f + 0.75f *
                     (float)(bias_pos - bias_start + 1) /
                     (float)(bias_count - bias_start + 1));
        }
    }

    for (i = 0; i < vocab_size; i++) {
#ifdef USE_SSE
        dot = dot_sse(target, embeddings[i], EMBED_DIM);
#else
        dot = 0;
        for (d = 0; d < EMBED_DIM; d++) dot += target[d] * embeddings[i][d];
#endif

        rep_count = recent_counts[i];
        appeared = rep_count > 0;
        penalty = 0.0f;
        if (rep_count > 0) {
            penalty -= freq_penalty * (float)rep_count;
            penalty -= rep_penalty * 2.0f;
        }
        if (appeared) {
            penalty -= 1.5f;
        }

        context_bonus = context_bonuses[i];
        scores[i] = dot / temperature + rng_unit() * eff_noise + penalty + context_bonus;
    }
    maxs = scores[0];
    for (i = 1; i < vocab_size; i++)
        if (scores[i] > maxs) maxs = scores[i];
    sum = 0;
    for (i = 0; i < vocab_size; i++) {
        scores[i] = expf(scores[i] - maxs);
        sum += scores[i];
    }

    if (top_p < 1.0f && top_p > 0.0f) {
        thresh = sum * top_p;
        memset(bucket_sums, 0, sizeof(bucket_sums));
        for (i = 0; i < vocab_size; i++) {
            bucket = (int)(scores[i] * 255.0f);
            if (bucket < 0) bucket = 0;
            if (bucket > 255) bucket = 255;
            bucket_sums[bucket] += scores[i];
        }
        cum = 0;
        cutoff_bucket = 0;
        for (bucket = 255; bucket >= 0; bucket--) {
            cum += bucket_sums[bucket];
            if (cum >= thresh) {
                cutoff_bucket = bucket;
                break;
            }
        }
        sum = 0.0f;
        for (i = 0; i < vocab_size; i++) {
            bucket = (int)(scores[i] * 255.0f);
            if (bucket < cutoff_bucket) {
                scores[i] = 0.0f;
            } else {
                sum += scores[i];
            }
        }
    }

    r = rng_unit() * sum;
    cum = 0;
    for (i = 0; i < vocab_size; i++) {
        cum += scores[i];
        if (r < cum) return i;
    }
    return vocab_size - 1;
}

int sample_vocab(const float *target, float temperature, float noise,
                 float freq_penalty, float rep_penalty, float top_p,
                 int vocab_size) {
    return sample_vocab_internal(target, temperature, noise, freq_penalty,
                                 rep_penalty, top_p, vocab_size,
                                 NULL, 0, 0.0f);
}

int sample_vocab_contextual(const float *target, float temperature, float noise,
                            float freq_penalty, float rep_penalty, float top_p,
                            int vocab_size, const int *bias_tokens,
                            int bias_count, float bias_strength) {
    return sample_vocab_internal(target, temperature, noise, freq_penalty,
                                 rep_penalty, top_p, vocab_size,
                                 bias_tokens, bias_count, bias_strength);
}

static int candidate_gcd(int left, int right) {
    int remainder;

    while (right != 0) {
        remainder = left % right;
        left = right;
        right = remainder;
    }
    return left;
}

static void add_candidate(int token, int vocab_size, int *candidates,
                          int *candidate_count, int candidate_limit,
                          unsigned int *candidate_marks,
                          unsigned int candidate_generation) {
    if (token < 0 || token >= vocab_size) return;
    if (*candidate_count >= candidate_limit) return;
    if (candidate_marks[token] == candidate_generation) return;
    candidate_marks[token] = candidate_generation;
    candidates[(*candidate_count)++] = token;
}

int sample_vocab_contextual_fast(const float *target, float temperature,
                                 float noise, float freq_penalty,
                                 float rep_penalty, float top_p,
                                 int vocab_size, const int *bias_tokens,
                                 int bias_count, float bias_strength) {
    static unsigned int candidate_marks[MAX_VOCAB_SIZE];
    static unsigned int candidate_generation;
    int candidates[1024];
    float scores[1024];
    float bucket_sums[256];
    unsigned char recent_counts[MAX_VOCAB_SIZE];
    float context_bonuses[MAX_VOCAB_SIZE];
    float max_score;
    float score_sum;
    float random_score;
    float cumulative;
    float threshold;
    float effective_noise;
    float dot;
    float penalty;
    float context_bonus;
    int candidate_count;
    int candidate_limit;
    int contextual_limit;
    int bias_start;
    int bias_position;
    int start;
    int step;
    int token;
    int index;
#ifndef USE_SSE
    int dimension;
#endif
    int recent_count;
    int bucket;
    int cutoff_bucket;

    if (vocab_size <= 0) return 0;
    if (vocab_size > MAX_VOCAB_SIZE) vocab_size = MAX_VOCAB_SIZE;
    candidate_limit = vocab_size < 64 ? vocab_size : 64;
    contextual_limit = candidate_limit > 16 ? candidate_limit - 16 :
                       candidate_limit;
    candidate_generation++;
    if (candidate_generation == 0) {
        memset(candidate_marks, 0, sizeof(candidate_marks));
        candidate_generation = 1;
    }
    memset(recent_counts, 0,
           (size_t)vocab_size * sizeof(recent_counts[0]));
    memset(context_bonuses, 0,
           (size_t)vocab_size * sizeof(context_bonuses[0]));
    for (index = 0; index < last_words_n; index++) {
        token = last_words[index];
        if (token >= 0 && token < vocab_size && recent_counts[token] < 255)
            recent_counts[token]++;
    }
    bias_start = bias_count > 256 ? bias_count - 256 : 0;
    if (bias_tokens && bias_strength > 0.0f) {
        for (bias_position = bias_start; bias_position < bias_count;
             bias_position++) {
            token = bias_tokens[bias_position];
            if (token >= 0 && token < vocab_size)
                context_bonuses[token] += bias_strength *
                    (0.25f + 0.75f *
                     (float)(bias_position - bias_start + 1) /
                     (float)(bias_count - bias_start + 1));
        }
    }
    candidate_count = 0;
    if (bias_tokens) {
        for (bias_position = bias_start; bias_position < bias_count;
             bias_position++)
            add_candidate(bias_tokens[bias_position], vocab_size,
                          candidates, &candidate_count, contextual_limit,
                          candidate_marks, candidate_generation);
    }
    for (index = 0; index < last_words_n; index++)
        add_candidate(last_words[index], vocab_size, candidates,
                      &candidate_count, contextual_limit, candidate_marks,
                      candidate_generation);
    start = rng_range(vocab_size);
    step = 1 + rng_range(vocab_size - 1);
    while (candidate_gcd(step, vocab_size) != 1) {
        step++;
        if (step >= vocab_size) step = 1;
    }
    token = start;
    while (candidate_count < candidate_limit) {
        add_candidate(token, vocab_size, candidates, &candidate_count,
                      candidate_limit, candidate_marks,
                      candidate_generation);
        token += step;
        if (token >= vocab_size) token %= vocab_size;
    }
    effective_noise = noise;
    if (effective_noise < 0.5f) effective_noise = 0.5f;
    for (index = 0; index < candidate_count; index++) {
        token = candidates[index];
#ifdef USE_SSE
        dot = dot_sse(target, embeddings[token], EMBED_DIM);
#else
        dot = 0.0f;
        for (dimension = 0; dimension < EMBED_DIM; dimension++)
            dot += target[dimension] * embeddings[token][dimension];
#endif
        recent_count = recent_counts[token];
        penalty = 0.0f;
        if (recent_count > 0) {
            penalty -= freq_penalty * (float)recent_count;
            penalty -= rep_penalty * 2.0f;
            penalty -= 1.5f;
        }
        context_bonus = context_bonuses[token];
        scores[index] = dot / temperature +
                        rng_unit() * effective_noise + penalty +
                        context_bonus;
    }
    max_score = scores[0];
    for (index = 1; index < candidate_count; index++)
        if (scores[index] > max_score) max_score = scores[index];
    score_sum = 0.0f;
    for (index = 0; index < candidate_count; index++) {
        scores[index] = fast_exp_negative(scores[index] - max_score);
        score_sum += scores[index];
    }
    if (top_p < 1.0f && top_p > 0.0f) {
        threshold = score_sum * top_p;
        memset(bucket_sums, 0, sizeof(bucket_sums));
        for (index = 0; index < candidate_count; index++) {
            bucket = (int)(scores[index] * 255.0f);
            if (bucket < 0) bucket = 0;
            if (bucket > 255) bucket = 255;
            bucket_sums[bucket] += scores[index];
        }
        cumulative = 0.0f;
        cutoff_bucket = 0;
        for (bucket = 255; bucket >= 0; bucket--) {
            cumulative += bucket_sums[bucket];
            if (cumulative >= threshold) {
                cutoff_bucket = bucket;
                break;
            }
        }
        score_sum = 0.0f;
        for (index = 0; index < candidate_count; index++) {
            bucket = (int)(scores[index] * 255.0f);
            if (bucket < cutoff_bucket)
                scores[index] = 0.0f;
            else
                score_sum += scores[index];
        }
    }
    random_score = rng_unit() * score_sum;
    cumulative = 0.0f;
    for (index = 0; index < candidate_count; index++) {
        cumulative += scores[index];
        if (random_score < cumulative) return candidates[index];
    }
    return candidates[candidate_count - 1];
}
