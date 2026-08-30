#include "../../../engine.h"
#include "../../../net.h"
#include "../../../features.h"
#include "../../../corpus.h"
#include "../../../tokenizer.h"
#include "../../../websearch.h"
#include "../../../knowledge/knowledge.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <limits.h>

#define V1_CONTEXT_TOKENS 1024

static void append_v1_context(int *context, int *context_count,
                              const int *tokens, int token_count) {
    int overflow;

    if (token_count <= 0) return;
    if (token_count > V1_CONTEXT_TOKENS) {
        tokens += token_count - V1_CONTEXT_TOKENS;
        token_count = V1_CONTEXT_TOKENS;
    }
    overflow = *context_count + token_count - V1_CONTEXT_TOKENS;
    if (overflow > 0) {
        memmove(context, context + overflow,
                (*context_count - overflow) * sizeof(int));
        *context_count -= overflow;
    }
    memcpy(context + *context_count, tokens, token_count * sizeof(int));
    *context_count += token_count;
}

static void append_v1_self_context(EngineState *state, const int *tokens,
                                   int token_count) {
    int overflow;

    if (token_count <= 0) return;
    if (token_count > V1_CONTEXT_TOKENS) {
        tokens += token_count - V1_CONTEXT_TOKENS;
        token_count = V1_CONTEXT_TOKENS;
    }
    overflow = state->self_ctx_len + token_count - V1_CONTEXT_TOKENS;
    if (overflow > 0) {
        memmove(state->self_ctx_tokens,
                state->self_ctx_tokens + overflow,
                (state->self_ctx_len - overflow) * sizeof(int));
        state->self_ctx_len -= overflow;
    }
    memcpy(state->self_ctx_tokens + state->self_ctx_len, tokens,
           token_count * sizeof(int));
    state->self_ctx_len += token_count;
}

static int compute_smart_length(const Features *feat, int pattern, int input_words, int cfg_max_words, int max_output_tokens) {
    float base;
    float chaos;
    int result;

    if (cfg_max_words > 0) {
        result = cfg_max_words;
        if (result < 1) result = 1;
        return result;
    }

    if (pattern == PAT_MATH) {
        base = 40.0f + (float)(rand() % 200);
        base *= (1.0f + feat->digit_ratio * 8.0f);
    } else if (pattern == PAT_QUESTION && input_words <= 5) {
        base = 15.0f + (float)(rand() % 60);
        if (feat->entropy < 0.4f) base *= 2.5f;
    } else if (pattern == PAT_QUESTION && input_words > 5) {
        base = 5.0f + (float)(rand() % 20);
    } else if (pattern == PAT_GREETING) {
        base = 3.0f + (float)(rand() % 8);
    } else if (pattern == PAT_TIME) {
        base = 20.0f + (float)(rand() % 150);
        base *= (1.0f + feat->is_question * 3.0f);
    } else if (pattern == PAT_TECH) {
        base = 30.0f + (float)(rand() % 300);
    } else if (pattern == PAT_EMOTIONAL) {
        base = 10.0f + (float)(rand() % 40);
    } else if (pattern == PAT_COMMAND) {
        base = 50.0f + (float)(rand() % 250);
    } else {
        base = (float)input_words * (0.5f + ((float)rand() / RAND_MAX) * 4.0f);
        base += (float)(rand() % 30);
    }

    chaos = sinf((float)rand()) * 0.6f + ((float)rand() / RAND_MAX) * 0.4f;
    base *= (0.5f + chaos);

    if (feat->entropy > 0.6f) base *= 0.7f;
    if (feat->upper_ratio > 0.5f) base *= 1.8f;
    if (feat->punct_ratio > 0.2f) base *= 1.3f;

    result = (int)base;
    if (result < MIN_GEN) result = MIN_GEN;
    if (result > max_output_tokens) result = max_output_tokens;
    return result;
}

static void scramble_context(float *ctx, float intensity) {
    int d;
    int swap_a, swap_b;
    float tmp;

    for (d = 0; d < EMBED_DIM; d++) {
        ctx[d] += frand_r() * intensity;
        ctx[d] *= (0.7f + frand_r() * 0.6f);
    }

    for (d = 0; d < EMBED_DIM / 2; d++) {
        swap_a = rand() % EMBED_DIM;
        swap_b = rand() % EMBED_DIM;
        tmp = ctx[swap_a];
        ctx[swap_a] = ctx[swap_b];
        ctx[swap_b] = tmp;
    }
}

static void inject_history_chaos(int *mixed, int *mixed_n, int max_mix,
                                  char (*hist_buf)[HIST_LEN],
                                  int (*hist_tokens)[MAX_TOKENS],
                                  int *hist_lens, int hist_cnt) {
    int num_injections, i, slot, pos, tok_idx;

    (void)hist_buf;
    if (hist_cnt < 2) return;

    num_injections = 1 + rand() % 3;
    for (i = 0; i < num_injections && *mixed_n < max_mix; i++) {
        slot = rand() % (hist_cnt < HIST_MAX ? hist_cnt : HIST_MAX);
        if (hist_lens[slot] > 0) {
            pos = rand() % hist_lens[slot];
            tok_idx = hist_tokens[slot][pos];
            mixed[(*mixed_n)++] = tok_idx;
        }
    }
}

static void derail_embed(float *target, int step, int max_words) {
    float drift;
    int d;
    int random_word;

    drift = 0.3f + (float)step / (float)max_words * 0.7f;

    for (d = 0; d < EMBED_DIM; d++)
        target[d] = target[d] * (1.0f - drift * 0.5f) + frand_r() * drift;

    if (rand() % 6 == 0) {
        random_word = rand() % ACTUAL_VOCAB;
        for (d = 0; d < EMBED_DIM; d++)
            target[d] = target[d] * 0.4f + embeddings[random_word][d] * 0.6f;
    }
}

static void inject_obsession(char *buf, int buf_size,
                              const char *obsession_word, int obsession_strength) {
    char tmp[8192];
    char *words[256];
    int nwords = 0;
    char *p;
    int i, written;
    int inject_at;
    int obslen;

    if (obsession_strength < 2 || !obsession_word[0]) return;
    if (rand() % 100 >= 20 + obsession_strength * 10) return;

    strncpy(tmp, buf, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = '\0';
    p = strtok(tmp, " ");
    while (p && nwords < 256) {
        words[nwords++] = p;
        p = strtok(NULL, " ");
    }
    if (nwords < 3) return;

    obslen = (int)strlen(obsession_word);
    inject_at = rand() % nwords;
    written = 0;

    for (i = 0; i < nwords && written < buf_size - obslen - 10; i++) {
        if (i == inject_at) {
            if (written > 0) buf[written++] = ' ';
            if (rand() % 2 == 0) {
                memcpy(buf + written, obsession_word, obslen);
                written += obslen;
                buf[written++] = ' ';
            } else {
                SAFE_SNPRINTF(written, buf_size, snprintf(buf + written, buf_size - written,
                    "(wait, %s?) ", obsession_word));
            }
        }
        if (written > 0) buf[written++] = ' ';
        SAFE_SNPRINTF(written, buf_size, snprintf(buf + written, buf_size - written, "%s", words[i]));
    }
    buf[written] = '\0';
}

static void inject_absorbed(char *buf, int buf_size,
                              char absorbed_words[][TOPIC_WORD_LEN], int absorbed_n) {
    char suffix[256];
    int slen;
    int blen;
    int pick;

    if (absorbed_n == 0) return;
    if (rand() % 100 >= 15) return;

    pick = rand() % absorbed_n;
    switch (rand() % 4) {
    case 0:
        snprintf(suffix, sizeof(suffix), " (much like %s)", absorbed_words[pick]);
        break;
    case 1:
        snprintf(suffix, sizeof(suffix), " which is basically %s", absorbed_words[pick]);
        break;
    case 2:
        snprintf(suffix, sizeof(suffix), ", or as they say, %s", absorbed_words[pick]);
        break;
    default:
        snprintf(suffix, sizeof(suffix), " -- %s --", absorbed_words[pick]);
        break;
    }

    slen = (int)strlen(suffix);
    blen = (int)strlen(buf);
    if (blen + slen < buf_size - 1) {
        memcpy(buf + blen, suffix, slen);
        buf[blen + slen] = '\0';
    }
}

static void inject_contradiction(char *buf, int buf_size,
                                  char prev_responses[][PREV_RESPONSE_LEN],
                                  int prev_resp_count) {
    int old_slot;
    char fragment[128];
    int flen, blen;
    int old_len, cut;
    char prefix[64];
    int plen;

    if (prev_resp_count < 2) return;
    if (rand() % 100 >= 12) return;

    old_slot = rand() % (prev_resp_count < PREV_RESPONSE_MAX ? prev_resp_count : PREV_RESPONSE_MAX);
    old_len = (int)strlen(prev_responses[old_slot]);
    if (old_len < 10) return;

    cut = old_len > 80 ? 80 : old_len;
    while (cut > 0 && prev_responses[old_slot][cut] != ' ') cut--;
    if (cut <= 0) cut = old_len > 80 ? 80 : old_len;
    memcpy(fragment, prev_responses[old_slot], cut);
    fragment[cut] = '\0';

    switch (rand() % 4) {
    case 0:
        snprintf(prefix, sizeof(prefix), " Actually wait, I take that back. ");
        break;
    case 1:
        snprintf(prefix, sizeof(prefix), " No, ignore what I said before about ");
        break;
    case 2:
        snprintf(prefix, sizeof(prefix), " (Correction: unlike when I said ");
        break;
    default:
        snprintf(prefix, sizeof(prefix), " But earlier I was wrong about ");
        break;
    }

    plen = (int)strlen(prefix);
    flen = (int)strlen(fragment);
    blen = (int)strlen(buf);

    if (blen + plen + flen + 2 < buf_size) {
        memcpy(buf + blen, prefix, plen);
        memcpy(buf + blen + plen, fragment, flen);
        buf[blen + plen + flen] = '\0';
    }
}

static int is_common_word(const char *w) {
    static const char *common[] = {
        "the","a","an","is","it","was","are","am","be","have","has",
        "do","does","did","will","would","could","should","can",
        "i","you","we","they","he","she","me","my","your","our",
        "this","that","what","which","who","when","where","why","how",
        "if","but","and","or","not","no","yes","so","very","too",
        "for","from","with","about","to","of","in","on","at","by",
        "up","down","out","off","tell","show","please","want","like",
        "know","think","say","make","get","go","just","also",
        NULL
    };
    char lower[64];
    int i;
    int len;

    len = (int)strlen(w);
    if (len >= 63 || len < 3) return 1;
    for (i = 0; i < len; i++) lower[i] = (char)tolower((unsigned char)w[i]);
    lower[len] = '\0';
    for (i = 0; common[i]; i++) {
        if (strcmp(lower, common[i]) == 0) return 1;
    }
    return 0;
}

static void track_topic(EngineState *state, const char *input) {
    char buf[512];
    char *words[64];
    int nwords = 0;
    char *p;
    int i, j, found;
    int best_idx;
    int best_count;
    int wlen;

    strncpy(buf, input, 511);
    buf[511] = '\0';
    p = strtok(buf, " \t\n.,!?;:'\"()[]{}");
    while (p && nwords < 64) {
        if (!is_common_word(p)) words[nwords++] = p;
        p = strtok(NULL, " \t\n.,!?;:'\"()[]{}");
    }

    for (i = 0; i < nwords; i++) {
        found = -1;
        for (j = 0; j < state->topic_n; j++) {
            wlen = (int)strlen(words[i]);
            if (wlen >= TOPIC_WORD_LEN) wlen = TOPIC_WORD_LEN - 1;
            if (strncmp(state->topic_words[j], words[i], wlen) == 0 &&
                state->topic_words[j][wlen] == '\0') {
                found = j;
                break;
            }
        }
        if (found >= 0) {
            state->topic_counts[found]++;
        } else if (state->topic_n < TOPIC_TRACK_MAX) {
            wlen = (int)strlen(words[i]);
            if (wlen >= TOPIC_WORD_LEN) wlen = TOPIC_WORD_LEN - 1;
            memcpy(state->topic_words[state->topic_n], words[i], wlen);
            state->topic_words[state->topic_n][wlen] = '\0';
            state->topic_counts[state->topic_n] = 1;
            state->topic_n++;
        }
    }

    best_idx = -1;
    best_count = 1;
    for (i = 0; i < state->topic_n; i++) {
        if (state->topic_counts[i] > best_count) {
            best_count = state->topic_counts[i];
            best_idx = i;
        }
    }

    if (best_idx >= 0 && best_count >= 2) {
        strncpy(state->obsession_word, state->topic_words[best_idx], TOPIC_WORD_LEN - 1);
        state->obsession_word[TOPIC_WORD_LEN - 1] = '\0';
        state->obsession_strength = best_count;
    }
}

static void absorb_words(EngineState *state, const char *input) {
    char buf[512];
    char *words[64];
    int nwords = 0;
    char *p;
    int i, j, found;
    int wlen;

    strncpy(buf, input, 511);
    buf[511] = '\0';
    p = strtok(buf, " \t\n.,!?;:'\"()[]{}");
    while (p && nwords < 64) {
        words[nwords++] = p;
        p = strtok(NULL, " \t\n.,!?;:'\"()[]{}");
    }

    for (i = 0; i < nwords; i++) {
        wlen = (int)strlen(words[i]);
        if (wlen < 5) continue;
        if (is_common_word(words[i])) continue;

        found = 0;
        for (j = 0; j < state->absorbed_n; j++) {
            if (strncmp(state->absorbed_words[j], words[i], wlen) == 0 &&
                state->absorbed_words[j][wlen] == '\0') {
                found = 1;
                break;
            }
        }
        if (!found && rand() % 4 == 0) {
            if (state->absorbed_n >= ABSORBED_MAX) {
                state->absorbed_n = ABSORBED_MAX - 1;
                memmove(state->absorbed_words[0], state->absorbed_words[1],
                    (ABSORBED_MAX - 1) * TOPIC_WORD_LEN);
            }
            if (wlen >= TOPIC_WORD_LEN) wlen = TOPIC_WORD_LEN - 1;
            memcpy(state->absorbed_words[state->absorbed_n], words[i], wlen);
            state->absorbed_words[state->absorbed_n][wlen] = '\0';
            state->absorbed_n++;
        }
    }
}

static void save_response(EngineState *state, const char *response) {
    int slot;
    int len;

    slot = state->prev_resp_count % PREV_RESPONSE_MAX;
    len = (int)strlen(response);
    if (len >= PREV_RESPONSE_LEN) len = PREV_RESPONSE_LEN - 1;
    memcpy(state->prev_responses[slot], response, len);
    state->prev_responses[slot][len] = '\0';
    state->prev_resp_count++;
}

static void hist_push(EngineState *state, const char *s, const int *toks, int n) {
    int slot, cp;
    size_t slen;

    slot = *state->hist_cnt % HIST_MAX;
    slen = strlen(s);
    if (slen >= HIST_LEN) slen = HIST_LEN - 1;
    memcpy(state->hist_buf[slot], s, slen);
    state->hist_buf[slot][slen] = '\0';
    cp = n < MAX_TOKENS ? n : MAX_TOKENS;
    memcpy(state->hist_tokens[slot], toks, cp * sizeof(int));
    state->hist_lens[slot] = cp;
    (*state->hist_cnt)++;
}

static void v1_generate_response(EngineState *state, const EngineCallbacks *cb, const char *input) {
    int prev_word;
    Features feat;
    float feat_ctx[EMBED_DIM];
    int cur_tokens[MAX_TOKENS];
    int search_tokens[V1_CONTEXT_TOKENS];
    int cur_len;
    int search_token_count;
    int mixed[V1_CONTEXT_TOKENS];
    int mixed_n;
    int pattern;
    float ctx[EMBED_DIM];
    float prev_embed[EMBED_DIM];
    float base_temp, base_noise;
    int max_words;
    int step, word, d;
    float target[EMBED_DIM];
    float step_temp;
    const char *w;
    int input_words;
    int add, slot;
    const char *p_count;
    int in_w;
    int resp_mode;
    char corpus_buf[8192];
    char search_text[8192];
    int used_corpus;
    int explicit_search;
    int search_result_count;
    int input_pattern;
    int autonomous_search_allowed;
    int restore_search_enabled;
    int amnesia_slot;
    char amnesia_input[HIST_LEN];
    float personality_temp_mult;
    float personality_noise_mult;
    char *word_buf;
    size_t word_buf_bytes;
    int word_buf_size;
    int wb_pos;
    int wlen;
    int unicode_chance;
    float temp_mode_bias;
    const char *user_match;
    int resp_ids[SELF_CTX_MAX];
    int resp_ids_n;
    int streamed_pos;
    int generated_words;
    const char *stream_chunk;

    prev_word = rand() % ACTUAL_VOCAB;

    cb->curs_set_fn(0);
    cb->show_status("Thinking... ESC or Ctrl-C to stop.");
    state->turn_count++;
    web_search_take_used();
    search_text[0] = '\0';
    input_pattern = detect_pattern(input);
    explicit_search = web_search_enabled() && web_search_requested(input);
    autonomous_search_allowed = explicit_search ||
        input_pattern == PAT_QUESTION || input_pattern == PAT_COMMAND ||
        input_pattern == PAT_NEWS || input_pattern == PAT_TIME ||
        input_pattern == PAT_TECH;
    search_result_count = -2;
    if (web_search_requested(input) && !web_search_enabled())
        cb->chat_add("Web search is disabled. Use /search to enable it.");
    if (explicit_search) {
        cb->show_status("Searching the web...");
        search_result_count = web_research(
            input, search_text, sizeof(search_text), 0);
        web_search_take_used();
        if (search_result_count > 0)
            cb->chat_add("Searched the web");
        else if (search_result_count == 0)
            cb->chat_add("Web search returned no usable pages.");
        else
            cb->chat_add("Web search failed.");
        cb->show_status("Thinking... ESC or Ctrl-C to stop.");
    }

    track_topic(state, input);
    absorb_words(state, input);

    personality_temp_mult = 1.0f + (float)*state->hist_cnt * 0.04f;
    personality_noise_mult = 1.0f + (float)*state->hist_cnt * 0.06f;
    if (personality_temp_mult > 3.0f) personality_temp_mult = 3.0f;
    if (personality_noise_mult > 4.0f) personality_noise_mult = 4.0f;

    if (*state->hist_cnt > 3 && rand() % 100 < 8 + *state->hist_cnt * 2) {
        amnesia_slot = rand() % (*state->hist_cnt < HIST_MAX ? *state->hist_cnt : HIST_MAX);
        strncpy(amnesia_input, state->hist_buf[amnesia_slot], HIST_LEN - 1);
        amnesia_input[HIST_LEN - 1] = '\0';
        hist_push(state, input, cur_tokens, 0);
        feat = extract(amnesia_input);
        feat_to_embed(&feat, feat_ctx);
        cur_len = tokenize(amnesia_input, cur_tokens, MAX_TOKENS);
        pattern = detect_pattern(amnesia_input);

        input_words = 0;
        p_count = amnesia_input;
        in_w = 0;
        while (*p_count) {
            if (isspace((unsigned char)*p_count)) { in_w = 0; }
            else if (!in_w) { input_words++; in_w = 1; }
            p_count++;
        }
    } else {
        feat = extract(input);
        feat_to_embed(&feat, feat_ctx);
        cur_len = tokenize(input, cur_tokens, MAX_TOKENS);
        pattern = detect_pattern(input);

        input_words = 0;
        p_count = input;
        in_w = 0;
        while (*p_count) {
            if (isspace((unsigned char)*p_count)) { in_w = 0; }
            else if (!in_w) { input_words++; in_w = 1; }
            p_count++;
        }
    }

    state->total_tokens += cur_len;

    mixed_n = 0;

    if (*state->hist_cnt > 0 && rand() % 2 == 0) {
        slot = (*state->hist_cnt - 1 - rand() % (*state->hist_cnt < HIST_MAX ? *state->hist_cnt : HIST_MAX) + HIST_MAX) % HIST_MAX;
        add = state->hist_lens[slot];
        append_v1_context(mixed, &mixed_n, state->hist_tokens[slot], add);
    }

    inject_history_chaos(mixed, &mixed_n, V1_CONTEXT_TOKENS,
                         state->hist_buf, state->hist_tokens,
                         state->hist_lens, *state->hist_cnt);

    if (state->self_ctx_len > 0) {
        add = state->self_ctx_len;
        append_v1_context(mixed, &mixed_n, state->self_ctx_tokens, add);
    }

    if (search_result_count > 0) {
        search_token_count = tokenize(search_text, search_tokens,
                                      V1_CONTEXT_TOKENS);
        append_v1_context(mixed, &mixed_n, search_tokens,
                          search_token_count);
    }

    append_v1_context(mixed, &mixed_n, cur_tokens, cur_len);

    hist_push(state, input, cur_tokens, cur_len);
    reset_recent();
    encode_context(mixed, mixed_n, ctx);
    scramble_context(ctx, 1.5f + (float)(rand() % 100) / 100.0f);
    memcpy(prev_embed, embeddings[prev_word], EMBED_DIM * sizeof(float));

    base_temp = state->cfg_temp * personality_temp_mult;
    base_noise = state->cfg_noise * personality_noise_mult;
    if (feat.is_question > 0.5f) { base_temp *= 0.9f; base_noise *= 1.4f; }
    if (feat.exclamation > 0.5f) { base_temp *= 0.85f; base_noise *= 1.6f; }
    if (feat.entropy > 0.7f) { base_temp *= 1.3f; base_noise *= 1.1f; }
    if (feat.digit_ratio > 0.1f) { base_temp *= 1.5f; base_noise *= 1.5f; }
    if (pattern == PAT_MATH) { base_temp *= 1.4f; base_noise *= 1.8f; }
    if (pattern == PAT_TECH) { base_temp *= 1.2f; base_noise *= 1.3f; }

    max_words = compute_smart_length(&feat, pattern, input_words, state->cfg_max_words, engine_tuffai_v1.max_output_tokens);

    if ((size_t)max_words >
        ((size_t)INT_MAX - 2) / (RENDERED_WORD_LEN + 1)) {
        word_buf_bytes = INT_MAX;
    } else {
        word_buf_bytes = (size_t)max_words * (RENDERED_WORD_LEN + 1) + 2;
    }
    if (word_buf_bytes < 8192) word_buf_bytes = 8192;
    word_buf = (char *)malloc(word_buf_bytes);
    if (!word_buf) {
        cb->show_status("Unable to allocate the requested response size.");
        cb->curs_set_fn(1);
        return;
    }
    word_buf_size = (int)word_buf_bytes;

    temp_mode_bias = (state->cfg_temp - TEMP_BASE) / TEMP_BASE;

    resp_mode = pick_response_mode(pattern, &feat, *state->hist_cnt);

    if (resp_mode != RESP_CODE_GEN && temp_mode_bias > 0.3f && rand() % 100 < (int)(temp_mode_bias * 40.0f)) {
        switch (rand() % 6) {
        case 0: resp_mode = RESP_SEARCH_DRIFT; break;
        case 1: resp_mode = RESP_SEARCH_MANGLE; break;
        case 2: resp_mode = RESP_TRUNCATE; break;
        case 3: resp_mode = RESP_REPEAT; break;
        case 4: resp_mode = RESP_WORD_SALAD; break;
        default: resp_mode = RESP_ECHO_MANGLE; break;
        }
    }
    if (resp_mode != RESP_CODE_GEN && temp_mode_bias < -0.3f && rand() % 100 < (int)(-temp_mode_bias * 30.0f)) {
        switch (rand() % 3) {
        case 0: resp_mode = RESP_SEARCH_CONFUSED; break;
        case 1: resp_mode = RESP_ECHO_KEYWORD; break;
        default: resp_mode = RESP_CONFUSION; break;
        }
    }

    {
        static const char *think_connectors[] = {
            "hmm", "wait", "actually", "no", "but", "so", "therefore",
            "unless", "considering", "however", "indeed", "perhaps",
            "fundamentally", "interestingly", "okay", "right", "well",
            "let me think", "on second thought", "basically", "clearly",
            "obviously", "I recall", "it seems", "the thing is",
            "now", "then again", "hold on", "yes", "ah"
        };
        static const char *think_fillers[] = {
            "this means", "which implies", "if we consider",
            "from what I know", "that reminds me", "in other words",
            "the key point is", "wait no", "scratch that",
            "going back to", "the real question is",
            "let me reconsider", "I think", "so basically",
            "putting it together", "on reflection"
        };
        int think_words;
        int think_line_words;
        int think_wb;
        char think_line[512];
        float think_ctx[EMBED_DIM];
        float think_prev[EMBED_DIM];
        float think_target[EMBED_DIM];
        float think_temp_val;
        int think_word_id;
        const char *think_w;
        int think_wlen;
        char input_keyword[128];
        int phase;
        int phase_len;
        int connector_countdown;
        int inject_keyword_countdown;
        int n_connectors;
        int n_fillers;
        int think_ids[SELF_CTX_MAX];
        int think_ids_n;

        n_connectors = (int)(sizeof(think_connectors) / sizeof(think_connectors[0]));
        n_fillers = (int)(sizeof(think_fillers) / sizeof(think_fillers[0]));

        extract_keyword_ext(input, input_keyword, sizeof(input_keyword));
        think_ids_n = 0;

        think_words = 200 + rand() % 300;
        if (pattern == PAT_MATH || pattern == PAT_TECH || pattern == PAT_CODE)
            think_words += 100 + rand() % 150;
        if (pattern == PAT_QUESTION)
            think_words += 60 + rand() % 100;
        if (feat.entropy > 0.6f)
            think_words += 80;
        if (rand() % 10 == 0)
            think_words += 150 + rand() % 250;

        think_line_words = 0;
        think_wb = 0;

        for (d = 0; d < EMBED_DIM; d++) {
            think_ctx[d] = ctx[d];
            think_prev[d] = prev_embed[d];
        }

        think_line[0] = ' ';
        think_line[1] = ' ';
        think_line[2] = '>';
        think_line[3] = ' ';
        think_wb = 4;

        phase = 0;
        phase_len = 15 + rand() % 20;
        connector_countdown = 5 + rand() % 8;
        inject_keyword_countdown = 8 + rand() % 12;

        for (step = 0; step < think_words; step++) {
            if (cb->generation_should_stop()) break;
            if (connector_countdown <= 0 && think_wb < (int)sizeof(think_line) - 60) {
                const char *conn;
                int clen;

                if (phase == 0 || rand() % 3 == 0)
                    conn = think_connectors[rand() % n_connectors];
                else
                    conn = think_fillers[rand() % n_fillers];

                clen = (int)strlen(conn);
                if (think_wb > 4) {
                    if (rand() % 3 == 0) {
                        think_line[think_wb++] = '.';
                        think_line[think_wb++] = '.';
                        think_line[think_wb++] = '.';
                    } else {
                        think_line[think_wb++] = ',';
                    }
                    think_line[think_wb++] = ' ';
                }
                if (think_wb + clen < (int)sizeof(think_line) - 2) {
                    memcpy(think_line + think_wb, conn, clen);
                    think_wb += clen;
                }
                think_line_words++;
                connector_countdown = 6 + rand() % 12;
            }

            if (inject_keyword_countdown <= 0 && think_wb < (int)sizeof(think_line) - 60) {
                int klen = (int)strlen(input_keyword);
                if (think_wb > 4)
                    think_line[think_wb++] = ' ';
                if (rand() % 3 == 0 && think_wb + 2 + klen < (int)sizeof(think_line) - 2) {
                    think_line[think_wb++] = '"';
                    memcpy(think_line + think_wb, input_keyword, klen);
                    think_wb += klen;
                    think_line[think_wb++] = '"';
                } else if (think_wb + klen < (int)sizeof(think_line) - 2) {
                    memcpy(think_line + think_wb, input_keyword, klen);
                    think_wb += klen;
                }
                think_line_words++;
                inject_keyword_countdown = 10 + rand() % 18;
            }

            next_embed(think_ctx, think_prev, feat_ctx, think_target);
            derail_embed(think_target, step, think_words);

            for (d = 0; d < EMBED_DIM; d++)
                think_target[d] += frand_r() * (base_noise * 2.0f);

            think_temp_val = base_temp * (1.2f + (float)step / think_words * 0.8f);
            think_word_id = sample_vocab(think_target, think_temp_val, base_noise * 2.0f,
                                state->cfg_freq_penalty, state->cfg_rep_penalty, state->cfg_top_p,
                                engine_tuffai_v1.vocab_size);

            think_w = vocab[think_word_id];
            think_wlen = (int)strlen(think_w);

            if (think_ids_n < SELF_CTX_MAX)
                think_ids[think_ids_n++] = think_word_id;

            if (think_wb > 4 && think_wb + 1 + think_wlen < (int)sizeof(think_line) - 2) {
                think_line[think_wb++] = ' ';
            }
            if (think_wb + think_wlen < (int)sizeof(think_line) - 2) {
                memcpy(think_line + think_wb, think_w, think_wlen);
                think_wb += think_wlen;
            }

            memcpy(think_prev, embeddings[think_word_id], EMBED_DIM * sizeof(float));

            for (d = 0; d < EMBED_DIM; d++)
                think_ctx[d] = think_ctx[d] * 0.6f + embeddings[think_word_id][d] * 0.4f;

            think_line_words++;
            connector_countdown--;
            inject_keyword_countdown--;

            if (think_line_words >= 15 + rand() % 11) {
                think_line[think_wb] = '\0';
                cb->stream_text("", think_line, state->color_think);
                think_wb = 4;
                think_line[0] = ' ';
                think_line[1] = ' ';
                think_line[2] = '>';
                think_line[3] = ' ';
                think_line_words = 0;

                phase_len--;
                if (phase_len <= 0) {
                    phase = (phase + 1) % 3;
                    phase_len = 12 + rand() % 18;
                }
            }
        }

        if (think_line_words > 0) {
            think_line[think_wb] = '\0';
            cb->stream_text("", think_line, state->color_think);
        }

        append_v1_self_context(state, think_ids, think_ids_n);
        state->total_tokens += think_ids_n;
        state->sampled_tokens += think_ids_n;
    }

    if (cb->generation_should_stop()) {
        free(word_buf);
        cb->show_status("");
        cb->curs_set_fn(1);
        return;
    }

    user_match = NULL;
    if (search_result_count > 0) {
        snprintf(corpus_buf, sizeof(corpus_buf), "%s", search_text);
        used_corpus = 1;
        resp_mode = RESP_SEARCH_DRIFT;
    } else {
        user_match = knowledge_match_user(&know_extra, input);
        if (!user_match) user_match = knowledge_match_user(&know_tech, input);
        if (!user_match) user_match = knowledge_match_user(&know_general, input);
        if (!user_match)
            user_match = knowledge_match_user(&know_phrases_en, input);
    }
    if (search_result_count <= 0 && user_match && rand() % 100 < 80) {
        snprintf(corpus_buf, sizeof(corpus_buf), "%s", user_match);
        if (rand() % 100 < 40) {
            scramble_words(corpus_buf, word_buf, word_buf_size);
            snprintf(corpus_buf, sizeof(corpus_buf), "%s", word_buf);
        }
        used_corpus = 1;
    } else if (search_result_count <= 0) {
        restore_search_enabled = web_search_enabled();
        if (!autonomous_search_allowed && restore_search_enabled)
            web_search_set_enabled(0);
        used_corpus = generate_corpus_response(resp_mode, input, pattern, &feat, corpus_buf, sizeof(corpus_buf));
        if (!autonomous_search_allowed && restore_search_enabled)
            web_search_set_enabled(1);
    }

    if (web_search_take_used())
        cb->chat_add("Searched the web");

    if (used_corpus) {
        if (resp_mode != RESP_CODE_GEN) {
            inject_obsession(corpus_buf, sizeof(corpus_buf),
                             state->obsession_word, state->obsession_strength);
            inject_absorbed(corpus_buf, sizeof(corpus_buf),
                            state->absorbed_words, state->absorbed_n);
            inject_contradiction(corpus_buf, sizeof(corpus_buf),
                                 state->prev_responses, state->prev_resp_count);
        }

        unicode_chance = 15 + *state->hist_cnt * 3 + (int)(state->cfg_noise * 3.0f);
        if (unicode_chance > 90) unicode_chance = 90;
        if (resp_mode != RESP_CODE_GEN && rand() % 100 < unicode_chance) {
            inject_unicode(corpus_buf, sizeof(corpus_buf));
        }

        if (resp_mode != RESP_CODE_GEN && state->cfg_noise > NOISE_BASE * 1.5f) {
            scramble_words(corpus_buf, word_buf, word_buf_size);
            save_response(state, word_buf);
            cb->stream_text("TuffAI: ", word_buf, state->color_ai);
        } else {
            save_response(state, corpus_buf);
            cb->stream_text("TuffAI: ", corpus_buf, state->color_ai);
        }
        free(word_buf);
        cb->show_status("");
        cb->curs_set_fn(1);
        return;
    }

    cb->show_status("Generating... ESC or Ctrl-C to stop.");

    wb_pos = 0;
    resp_ids_n = 0;
    streamed_pos = 0;
    generated_words = 0;
    for (step = 0; step < max_words; step++) {
        next_embed(ctx, prev_embed, feat_ctx, target);
        derail_embed(target, step, max_words);

        for (d = 0; d < EMBED_DIM; d++)
            target[d] += frand_r() * (base_noise * (1.0f + (float)step / max_words * 0.8f));

        step_temp = base_temp * (1.0f + (float)step / max_words * 0.5f);
        word = sample_vocab(target, step_temp, base_noise,
                            state->cfg_freq_penalty, state->cfg_rep_penalty, state->cfg_top_p,
                            engine_tuffai_v1.vocab_size);
        push_recent(word);

        if (resp_ids_n < SELF_CTX_MAX)
            resp_ids[resp_ids_n++] = word;

        w = vocab[word];
        wlen = (int)strlen(w);

        if (wb_pos > 0 && wb_pos + 1 + wlen < word_buf_size - 2) {
            word_buf[wb_pos++] = ' ';
        }
        if (wb_pos + wlen < word_buf_size - 2) {
            memcpy(word_buf + wb_pos, w, wlen);
            wb_pos += wlen;
        }
        generated_words++;

        if (generated_words % 12 == 0) {
            word_buf[wb_pos] = '\0';
            stream_chunk = word_buf + streamed_pos;
            while (*stream_chunk == ' ') stream_chunk++;
            if (*stream_chunk)
                cb->stream_text(streamed_pos == 0 ? "TuffAI: " : "",
                                stream_chunk, state->color_ai);
            streamed_pos = wb_pos;
        }
        if (cb->generation_should_stop()) break;

        memcpy(prev_embed, embeddings[word], EMBED_DIM * sizeof(float));
        prev_word = word;

        for (d = 0; d < EMBED_DIM; d++)
            ctx[d] = ctx[d] * 0.6f + embeddings[word][d] * 0.4f;
    }

    (void)prev_word;
    state->total_tokens += generated_words;
    state->sampled_tokens += generated_words;

    word_buf[wb_pos++] = '.';
    word_buf[wb_pos] = '\0';

    save_response(state, word_buf);
    stream_chunk = word_buf + streamed_pos;
    while (*stream_chunk == ' ') stream_chunk++;
    if (*stream_chunk)
        cb->stream_text(streamed_pos == 0 ? "TuffAI: " : "",
                        stream_chunk, state->color_ai);

    append_v1_self_context(state, resp_ids, resp_ids_n);

    free(word_buf);
    cb->show_status("");
    cb->curs_set_fn(1);
}

const EngineVtable engine_tuffai_v1 = {
    v1_generate_response,
    1024,
    V1_CONTEXT_TOKENS,
    ACTUAL_VOCAB,
    0,
    NULL,
    0,
    0,
    0
};
