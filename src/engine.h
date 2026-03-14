#ifndef ENGINE_H
#define ENGINE_H

#include "knowledge/knowledge.h"
#include "features.h"
#include "corpus.h"
#include "tokenizer.h"
#include "net.h"

#define TOPIC_TRACK_MAX 32
#define TOPIC_WORD_LEN 64
#define ABSORBED_MAX 32
#define PREV_RESPONSE_MAX 8
#define PREV_RESPONSE_LEN 512
#define SELF_CTX_MAX 256
#define SYSTEM_PROMPT_MAX 2048

typedef struct {
    void (*chat_add)(const char *line);
    void (*chat_add_c)(const char *line, int color);
    void (*chat_add_wrapped)(const char *prefix, const char *text, int color);
    void (*draw_chat)(void);
    void (*show_status)(const char *msg);
    void (*sleep_with_scroll)(int ms);
    void (*refresh_screen)(void);
    void (*curs_set_fn)(int visibility);
} EngineCallbacks;

typedef struct {
    char topic_words[TOPIC_TRACK_MAX][TOPIC_WORD_LEN];
    int topic_counts[TOPIC_TRACK_MAX];
    int topic_n;

    char obsession_word[TOPIC_WORD_LEN];
    int obsession_strength;

    char absorbed_words[ABSORBED_MAX][TOPIC_WORD_LEN];
    int absorbed_n;

    char prev_responses[PREV_RESPONSE_MAX][PREV_RESPONSE_LEN];
    int prev_resp_count;

    int self_ctx_tokens[SELF_CTX_MAX];
    int self_ctx_len;

    int total_tokens;
    int turn_count;

    float cfg_temp;
    float cfg_noise;
    float cfg_freq_penalty;
    float cfg_rep_penalty;
    float cfg_top_p;
    float cfg_presence_penalty;
    int cfg_max_words;

    char (*hist_buf)[HIST_LEN];
    int (*hist_tokens)[MAX_TOKENS];
    int *hist_lens;
    int *hist_cnt;

    int color_think;
    int color_ai;

    char system_prompt[SYSTEM_PROMPT_MAX];
} EngineState;

typedef struct {
    void (*generate_response)(EngineState *state, const EngineCallbacks *cb, const char *input);
    int max_output_tokens;
    int context_window;
    int vocab_size;
    int has_system_prompt;
} EngineVtable;

#ifdef ENABLE_TUFFAI_V1
extern const EngineVtable engine_tuffai_v1;
#endif
#ifdef ENABLE_TUFFAI_V2
extern const EngineVtable engine_tuffai_v2;
#endif

#endif
