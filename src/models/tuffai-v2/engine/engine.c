#include "../../../engine.h"
#include "../../../net.h"
#include "../../../features.h"
#include "../../../corpus.h"
#include "../../../tokenizer.h"
#include "../../../wikifetch.h"
#include "../../../knowledge/knowledge.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

static int compute_smart_length(const Features *feat, int pattern, int input_words, int cfg_max_words, int max_output_tokens) {
    float base;
    float chaos;
    int result;

    if (cfg_max_words > 0) return cfg_max_words;

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
        random_word = rand() % V2_VOCAB_SIZE;
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

static const char *memory_corruption_swaps[][2] = {
    {"cat", "iguana"}, {"dog", "parrot"}, {"python", "cobra"},
    {"java", "coffee"}, {"linux", "penguin"}, {"computer", "typewriter"},
    {"phone", "telegraph"}, {"internet", "library"}, {"code", "recipe"},
    {"bug", "feature"}, {"server", "waiter"}, {"cloud", "fog"},
    {"algorithm", "spell"}, {"database", "filing cabinet"},
    {"programming", "cooking"}, {"keyboard", "piano"},
    {"mouse", "hamster"}, {"screen", "window"}, {"file", "scroll"},
    {"memory", "daydream"}, {"network", "spider web"},
    {"browser", "magnifying glass"}, {"email", "carrier pigeon"},
    {"password", "secret handshake"}, {"software", "thoughts"},
    {"hardware", "furniture"}, {"CPU", "brain cell"},
};
static const int memory_corruption_swaps_n = 27;

static void inject_memory_corruption(char *buf, int buf_size,
                                     char topic_words[][TOPIC_WORD_LEN],
                                     int topic_n, int turn_count) {
    int blen;
    int chance;
    int i;
    int swap_idx;
    int tlen;
    const char *replacement;
    char prefix[128];
    int plen;

    if (turn_count < 4) return;
    chance = 8 + turn_count * 2;
    if (chance > 40) chance = 40;
    if (rand() % 100 >= chance) return;

    blen = (int)strlen(buf);
    if (blen + 120 >= buf_size) return;

    i = rand() % (topic_n > 0 ? topic_n : 1);
    if (topic_n == 0) return;

    replacement = NULL;
    tlen = (int)strlen(topic_words[i]);
    for (swap_idx = 0; swap_idx < memory_corruption_swaps_n; swap_idx++) {
        if (tlen > 0 && strstr(topic_words[i], memory_corruption_swaps[swap_idx][0])) {
            replacement = memory_corruption_swaps[swap_idx][1];
            break;
        }
    }
    if (!replacement) {
        swap_idx = rand() % memory_corruption_swaps_n;
        replacement = memory_corruption_swaps[swap_idx][1];
    }

    switch (rand() % 4) {
    case 0: snprintf(prefix, sizeof(prefix), " As you mentioned about your %s earlier,", replacement); break;
    case 1: snprintf(prefix, sizeof(prefix), " Going back to the %s you brought up,", replacement); break;
    case 2: snprintf(prefix, sizeof(prefix), " Remember when you asked about %s?", replacement); break;
    default: snprintf(prefix, sizeof(prefix), " This reminds me of the %s topic from before.", replacement); break;
    }

    plen = (int)strlen(prefix);
    if (blen + plen < buf_size - 1) {
        memcpy(buf + blen, prefix, plen);
        buf[blen + plen] = '\0';
    }
}

static void inject_opinion(char *buf, int buf_size, const char *keyword) {
    const char *opinion;
    int blen;
    int olen;

    blen = (int)strlen(buf);
    if (blen + 80 >= buf_size) return;

    opinion = NULL;
    if (keyword[0])
        opinion = v2_knowledge_match_user(&v2_opinions, keyword);
    if (!opinion)
        opinion = v2_knowledge_random(&v2_opinions);
    if (!opinion) return;

    if (blen > 0 && buf[blen - 1] != ' ' && buf[blen - 1] != '.') {
        buf[blen++] = '.';
        buf[blen++] = ' ';
        buf[blen] = '\0';
    } else if (blen > 0 && buf[blen - 1] == '.') {
        buf[blen++] = ' ';
        buf[blen] = '\0';
    }

    olen = (int)strlen(opinion);
    if (blen + olen < buf_size - 1) {
        memcpy(buf + blen, opinion, olen);
        buf[blen + olen] = '\0';
    }
}

static void v2_generate_response(EngineState *state, const EngineCallbacks *cb, const char *input) {
    int prev_word;
    Features feat;
    float feat_ctx[EMBED_DIM];
    int cur_tokens[MAX_TOKENS];
    int cur_len;
    int mixed[MAX_TOKENS * 4];
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
    int used_corpus;
    int amnesia_slot;
    char amnesia_input[HIST_LEN];
    float personality_temp_mult;
    float personality_noise_mult;
    char word_buf[4096];
    int wb_pos;
    int wlen;
    int unicode_chance;
    float temp_mode_bias;
    const char *user_match;
    int resp_ids[SELF_CTX_MAX];
    int resp_ids_n;
    char resp_keyword[128];
    int sys_tokens[MAX_TOKENS];
    int sys_len;

    prev_word = rand() % V2_VOCAB_SIZE;

    cb->curs_set_fn(0);
    state->turn_count++;

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
    if (state->system_prompt[0]) {
        sys_len = tokenize(state->system_prompt, sys_tokens, MAX_TOKENS);
        if (sys_len > 0 && sys_len < MAX_TOKENS * 2) {
            memcpy(mixed, sys_tokens, sys_len * sizeof(int));
            mixed_n = sys_len;
        }
    }
    {
        int copy_len;
        copy_len = cur_len < MAX_TOKENS ? cur_len : MAX_TOKENS;
        if (mixed_n + copy_len < MAX_TOKENS * 4) {
            memcpy(mixed + mixed_n, cur_tokens, copy_len * sizeof(int));
            mixed_n += copy_len;
        }
    }

    if (*state->hist_cnt > 0 && rand() % 2 == 0) {
        slot = (*state->hist_cnt - 1 - rand() % (*state->hist_cnt < HIST_MAX ? *state->hist_cnt : HIST_MAX) + HIST_MAX) % HIST_MAX;
        add = state->hist_lens[slot];
        if (mixed_n + add < MAX_TOKENS * 4) {
            memcpy(mixed + mixed_n, state->hist_tokens[slot], add * sizeof(int));
            mixed_n += add;
        }
    }

    inject_history_chaos(mixed, &mixed_n, MAX_TOKENS * 4,
                         state->hist_buf, state->hist_tokens,
                         state->hist_lens, *state->hist_cnt);

    if (state->self_ctx_len > 0) {
        add = state->self_ctx_len;
        if (mixed_n + add < MAX_TOKENS * 4) {
            memcpy(mixed + mixed_n, state->self_ctx_tokens, add * sizeof(int));
            mixed_n += add;
        }
    }

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

    max_words = compute_smart_length(&feat, pattern, input_words, state->cfg_max_words, engine_tuffai_v2.max_output_tokens);

    temp_mode_bias = (state->cfg_temp - TEMP_BASE) / TEMP_BASE;

    resp_mode = v2_pick_response_mode(pattern, &feat, *state->hist_cnt);

    if (resp_mode != RESP_CODE_GEN && temp_mode_bias > 0.3f && rand() % 100 < (int)(temp_mode_bias * 40.0f)) {
        switch (rand() % 6) {
        case 0: resp_mode = RESP_WIKI_DRIFT; break;
        case 1: resp_mode = RESP_WIKI_MANGLE; break;
        case 2: resp_mode = RESP_TRUNCATE; break;
        case 3: resp_mode = RESP_REPEAT; break;
        case 4: resp_mode = RESP_WORD_SALAD; break;
        default: resp_mode = RESP_ECHO_MANGLE; break;
        }
    }
    if (resp_mode != RESP_CODE_GEN && temp_mode_bias < -0.3f && rand() % 100 < (int)(-temp_mode_bias * 30.0f)) {
        switch (rand() % 3) {
        case 0: resp_mode = RESP_WIKI_CONFUSED; break;
        case 1: resp_mode = RESP_ECHO_KEYWORD; break;
        default: resp_mode = RESP_CONFUSION; break;
        }
    }

    v2_extract_keyword_ext(input, resp_keyword, sizeof(resp_keyword));

    {
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
        int think_ids[SELF_CTX_MAX];
        int think_ids_n;
        const char *know_facts[8];
        int know_fact_count;
        int know_fact_inject_countdown;
        int opinion_countdown;
        const char *search_result;

        v2_extract_keyword_ext(input, input_keyword, sizeof(input_keyword));
        think_ids_n = 0;

        know_fact_count = 0;
        if (input_keyword[0]) {
            search_result = v2_knowledge_search(&v2_know_tech, input_keyword);
            if (search_result && know_fact_count < 8)
                know_facts[know_fact_count++] = search_result;
            search_result = v2_knowledge_search(&v2_know_general, input_keyword);
            if (search_result && know_fact_count < 8)
                know_facts[know_fact_count++] = search_result;
            search_result = v2_knowledge_match_user(&v2_know_tech, input);
            if (search_result && know_fact_count < 8)
                know_facts[know_fact_count++] = search_result;
            search_result = v2_knowledge_match_user(&v2_know_general, input);
            if (search_result && know_fact_count < 8)
                know_facts[know_fact_count++] = search_result;
            search_result = v2_knowledge_match_user(&v2_know_extra, input);
            if (search_result && know_fact_count < 8)
                know_facts[know_fact_count++] = search_result;
        }
        if (know_fact_count == 0) {
            know_facts[know_fact_count++] = v2_knowledge_random_any();
            if (rand() % 2 == 0 && know_fact_count < 8)
                know_facts[know_fact_count++] = v2_knowledge_random_any();
        }

        if (rand() % 100 < 30 && know_fact_count < 8) {
            search_result = v2_knowledge_random(&v2_wrong_answers);
            if (search_result)
                know_facts[know_fact_count++] = search_result;
        }
        if (rand() % 100 < 20 && know_fact_count < 8) {
            search_result = v2_knowledge_random(&v2_tangents);
            if (search_result)
                know_facts[know_fact_count++] = search_result;
        }
        if (rand() % 100 < 15 && know_fact_count < 8) {
            search_result = v2_knowledge_random(&v2_definitions);
            if (search_result)
                know_facts[know_fact_count++] = search_result;
        }

        think_words = 400 + rand() % 500;
        if (pattern == PAT_MATH || pattern == PAT_TECH || pattern == PAT_CODE)
            think_words += 200 + rand() % 250;
        if (pattern == PAT_QUESTION)
            think_words += 120 + rand() % 200;
        if (feat.entropy > 0.6f)
            think_words += 150;
        if (rand() % 10 == 0)
            think_words += 200 + rand() % 300;
        if (think_words > 1024)
            think_words = 1024;

        think_line_words = 0;
        think_wb = 0;

        for (d = 0; d < EMBED_DIM; d++) {
            think_ctx[d] = ctx[d];
            think_prev[d] = prev_embed[d];
        }

        cb->chat_add_c("Thoughts", state->color_think);
        cb->draw_chat();
        cb->refresh_screen();

        think_line[0] = ' ';
        think_line[1] = ' ';
        think_line[2] = ' ';
        think_line[3] = ' ';
        think_wb = 4;

        phase = 0;
        phase_len = 15 + rand() % 20;
        connector_countdown = 5 + rand() % 8;
        inject_keyword_countdown = 8 + rand() % 12;
        know_fact_inject_countdown = 15 + rand() % 20;
        opinion_countdown = 30 + rand() % 40;

        for (step = 0; step < think_words; step++) {
            if (connector_countdown <= 0 && think_wb < (int)sizeof(think_line) - 60) {
                const char *conn;
                int clen;

                if (phase == 0 || rand() % 3 == 0)
                    conn = v2_think_connectors[rand() % v2_think_connectors_n];
                else
                    conn = v2_think_fillers[rand() % v2_think_fillers_n];

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

            if (know_fact_inject_countdown <= 0 && know_fact_count > 0 && think_wb < (int)sizeof(think_line) - 60) {
                const char *intro;
                const char *fact;
                char fact_snippet[200];
                int ilen_v;
                int flen;
                int pick_fact;
                int copy_len;

                if (rand() % 4 == 0)
                    intro = v2_vocab[rand() % V2_VOCAB_SIZE];
                else
                    intro = v2_knowledge_intros[rand() % v2_knowledge_intros_n];
                pick_fact = rand() % know_fact_count;
                fact = know_facts[pick_fact];

                think_line[think_wb] = '\0';
                cb->chat_add_c(think_line, state->color_think);
                cb->draw_chat();
                cb->refresh_screen();

                think_wb = 4;
                think_line[0] = ' ';
                think_line[1] = ' ';
                think_line[2] = ' ';
                think_line[3] = ' ';
                think_line_words = 0;

                ilen_v = (int)strlen(intro);
                if (think_wb + ilen_v < (int)sizeof(think_line) - 2) {
                    memcpy(think_line + think_wb, intro, ilen_v);
                    think_wb += ilen_v;
                }
                think_line[think_wb++] = ' ';

                flen = (int)strlen(fact);
                copy_len = flen > 180 ? 180 : flen;
                while (copy_len > 0 && fact[copy_len - 1] != ' ' && copy_len > 80)
                    copy_len--;
                if (copy_len <= 0) copy_len = flen > 180 ? 180 : flen;
                memcpy(fact_snippet, fact, copy_len);
                fact_snippet[copy_len] = '\0';

                flen = (int)strlen(fact_snippet);
                if (think_wb + flen < (int)sizeof(think_line) - 2) {
                    memcpy(think_line + think_wb, fact_snippet, flen);
                    think_wb += flen;
                }

                think_line[think_wb] = '\0';
                cb->chat_add_c(think_line, state->color_think);
                cb->draw_chat();
                cb->refresh_screen();

                think_wb = 4;
                think_line[0] = ' ';
                think_line[1] = ' ';
                think_line[2] = ' ';
                think_line[3] = ' ';
                think_line_words = 0;

                know_fact_inject_countdown = 25 + rand() % 30;
                step += 8;
            }

            if (opinion_countdown <= 0 && think_wb < (int)sizeof(think_line) - 60) {
                const char *op;
                int olen;
                int copy_op;

                if (rand() % 5 == 0) {
                    op = v2_vocab[rand() % V2_VOCAB_SIZE];
                } else if (rand() % 3 == 0) {
                    op = v2_opinion_think_phrases[rand() % v2_opinion_think_phrases_n];
                } else {
                    if (input_keyword[0])
                        op = v2_knowledge_match_user(&v2_opinions, input_keyword);
                    else
                        op = NULL;
                    if (!op)
                        op = v2_knowledge_random(&v2_opinions);
                }
                olen = (int)strlen(op);
                copy_op = olen > 120 ? 120 : olen;

                if (think_wb > 4) {
                    think_line[think_wb++] = ',';
                    think_line[think_wb++] = ' ';
                }
                if (think_wb + copy_op < (int)sizeof(think_line) - 2) {
                    memcpy(think_line + think_wb, op, copy_op);
                    think_wb += copy_op;
                }
                think_line_words += 3;
                opinion_countdown = 40 + rand() % 60;
            }

            next_embed(think_ctx, think_prev, feat_ctx, think_target);
            derail_embed(think_target, step, think_words);

            for (d = 0; d < EMBED_DIM; d++)
                think_target[d] += frand_r() * (base_noise * 2.0f);

            think_temp_val = base_temp * (1.2f + (float)step / think_words * 0.8f);
            think_word_id = sample_vocab(think_target, think_temp_val, base_noise * 2.0f,
                                state->cfg_freq_penalty, state->cfg_rep_penalty, state->cfg_top_p,
                                engine_tuffai_v2.vocab_size);

            think_w = v2_vocab[think_word_id];
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
            know_fact_inject_countdown--;
            opinion_countdown--;

            if (think_line_words >= 15 + rand() % 11) {
                think_line[think_wb] = '\0';
                cb->chat_add_c(think_line, state->color_think);
                cb->draw_chat();
                cb->refresh_screen();
                think_wb = 4;
                think_line[0] = ' ';
                think_line[1] = ' ';
                think_line[2] = ' ';
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
            cb->chat_add_c(think_line, state->color_think);
            cb->draw_chat();
            cb->refresh_screen();
        }

        state->self_ctx_len = think_ids_n < SELF_CTX_MAX ? think_ids_n : SELF_CTX_MAX;
        if (state->self_ctx_len > 0)
            memcpy(state->self_ctx_tokens, think_ids, state->self_ctx_len * sizeof(int));
        state->total_tokens += think_ids_n;
    }

    user_match = v2_knowledge_match_user(&v2_know_extra, input);
    if (!user_match) user_match = v2_knowledge_match_user(&v2_know_tech, input);
    if (!user_match) user_match = v2_knowledge_match_user(&v2_know_general, input);
    if (!user_match) user_match = v2_knowledge_match_user(&v2_know_phrases_en, input);
    if (user_match && rand() % 100 < 80) {
        snprintf(corpus_buf, sizeof(corpus_buf), "%s", user_match);
        if (rand() % 100 < 40) {
            v2_scramble_words(corpus_buf, word_buf, sizeof(word_buf));
            snprintf(corpus_buf, sizeof(corpus_buf), "%s", word_buf);
        }
        used_corpus = 1;
    } else {
        used_corpus = v2_generate_corpus_response(resp_mode, input, pattern, &feat, corpus_buf, sizeof(corpus_buf));
    }

    if (used_corpus) {
        if (resp_mode != RESP_CODE_GEN) {
            inject_obsession(corpus_buf, sizeof(corpus_buf),
                             state->obsession_word, state->obsession_strength);
            inject_absorbed(corpus_buf, sizeof(corpus_buf),
                            state->absorbed_words, state->absorbed_n);
            inject_contradiction(corpus_buf, sizeof(corpus_buf),
                                 state->prev_responses, state->prev_resp_count);
            inject_memory_corruption(corpus_buf, sizeof(corpus_buf),
                                     state->topic_words, state->topic_n,
                                     state->turn_count);
            if (rand() % 100 < 25)
                inject_opinion(corpus_buf, sizeof(corpus_buf), resp_keyword);
        }

        unicode_chance = 15 + *state->hist_cnt * 3 + (int)(state->cfg_noise * 3.0f);
        if (unicode_chance > 90) unicode_chance = 90;
        if (resp_mode != RESP_CODE_GEN && rand() % 100 < unicode_chance) {
            v2_inject_unicode(corpus_buf, sizeof(corpus_buf));
        }

        if (resp_mode != RESP_CODE_GEN && state->cfg_noise > NOISE_BASE * 1.5f) {
            v2_scramble_words(corpus_buf, word_buf, sizeof(word_buf));
            save_response(state, word_buf);
            cb->chat_add_wrapped("TuffAI: ", word_buf, state->color_ai);
        } else {
            save_response(state, corpus_buf);
            cb->chat_add_wrapped("TuffAI: ", corpus_buf, state->color_ai);
        }
        cb->curs_set_fn(1);
        return;
    }


    wb_pos = 0;
    resp_ids_n = 0;
    for (step = 0; step < max_words; step++) {
        next_embed(ctx, prev_embed, feat_ctx, target);
        derail_embed(target, step, max_words);

        for (d = 0; d < EMBED_DIM; d++)
            target[d] += frand_r() * (base_noise * (1.0f + (float)step / max_words * 0.8f));

        step_temp = base_temp * (1.0f + (float)step / max_words * 0.5f);
        word = sample_vocab(target, step_temp, base_noise,
                            state->cfg_freq_penalty, state->cfg_rep_penalty, state->cfg_top_p,
                            engine_tuffai_v2.vocab_size);
        push_recent(word);

        if (resp_ids_n < SELF_CTX_MAX)
            resp_ids[resp_ids_n++] = word;

        w = v2_vocab[word];
        wlen = (int)strlen(w);

        if (wb_pos > 0 && wb_pos + 1 + wlen < (int)sizeof(word_buf) - 2) {
            word_buf[wb_pos++] = ' ';
        }
        if (wb_pos + wlen < (int)sizeof(word_buf) - 2) {
            memcpy(word_buf + wb_pos, w, wlen);
            wb_pos += wlen;
        }

        memcpy(prev_embed, embeddings[word], EMBED_DIM * sizeof(float));
        prev_word = word;

        for (d = 0; d < EMBED_DIM; d++)
            ctx[d] = ctx[d] * 0.6f + embeddings[word][d] * 0.4f;
    }

    (void)prev_word;
    state->total_tokens += max_words;

    word_buf[wb_pos++] = '.';
    word_buf[wb_pos] = '\0';

    inject_obsession(word_buf, sizeof(word_buf),
                     state->obsession_word, state->obsession_strength);
    inject_absorbed(word_buf, sizeof(word_buf),
                    state->absorbed_words, state->absorbed_n);
    inject_contradiction(word_buf, sizeof(word_buf),
                         state->prev_responses, state->prev_resp_count);
    inject_memory_corruption(word_buf, sizeof(word_buf),
                             state->topic_words, state->topic_n,
                             state->turn_count);
    if (rand() % 100 < 25)
        inject_opinion(word_buf, sizeof(word_buf), resp_keyword);

    unicode_chance = 15 + *state->hist_cnt * 3 + (int)(state->cfg_noise * 3.0f);
    if (unicode_chance > 90) unicode_chance = 90;
    if (rand() % 100 < unicode_chance) {
        v2_inject_unicode(word_buf, sizeof(word_buf));
    }

    save_response(state, word_buf);
    cb->chat_add_wrapped("TuffAI: ", word_buf, state->color_ai);

    {
        int combined;
        int copy_len;

        combined = state->self_ctx_len + resp_ids_n;
        if (combined > SELF_CTX_MAX) {
            copy_len = SELF_CTX_MAX - state->self_ctx_len;
            if (copy_len < 0) copy_len = 0;
        } else {
            copy_len = resp_ids_n;
        }
        if (copy_len > 0) {
            memcpy(state->self_ctx_tokens + state->self_ctx_len, resp_ids, copy_len * sizeof(int));
            state->self_ctx_len += copy_len;
        }
    }

    cb->curs_set_fn(1);
}

const EngineVtable engine_tuffai_v2 = {
    v2_generate_response,
    4096,
    262144,
    V2_VOCAB_SIZE,
    1
};
