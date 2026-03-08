#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <ncurses.h>
#include "knowledge/knowledge.h"
#include "net.h"
#include "features.h"
#include "tokenizer.h"
#include "corpus.h"
#include "version.h"

#define INPUT_MAX 512
#define CHAT_LINES 4096
#define CHAT_LINE_LEN 512

static char chat_log[CHAT_LINES][CHAT_LINE_LEN];
static int chat_count = 0;
static int chat_scroll = 0;

static char hist_buf[HIST_MAX][HIST_LEN];
static int hist_tokens[HIST_MAX][MAX_TOKENS];
static int hist_lens[HIST_MAX];
static int hist_cnt = 0;

static float cfg_temp = 75.0f;
static float cfg_noise = 2.5f;
static float cfg_freq_penalty = 19.5f;
static float cfg_rep_penalty = 19.0f;
static int cfg_max_words = 0;
static float cfg_top_p = 0.02f;
static float cfg_presence_penalty = 10.0f;

typedef struct {
    const char *name;
    const char *description;
    float default_temp;
    float default_noise;
    float default_freq_penalty;
    float default_rep_penalty;
    float default_top_p;
    float default_presence_penalty;
} ModelDef;

static const ModelDef models[] = {
    {
        "TuffAI-v1",
        "The Original TuffAI",
        75.0f,
        2.5f,
        19.5f,
        19.0f,
        0.02f,
        10.0f,
    },
};

#define MODEL_COUNT (int)(sizeof(models) / sizeof(models[0]))

static int current_model = 0;

static void chat_add(const char *line) {
    int slot;
    int len;

    if (!line || !line[0]) return;
    slot = chat_count % CHAT_LINES;
    len = (int)strlen(line);
    if (len >= CHAT_LINE_LEN) len = CHAT_LINE_LEN - 1;
    memcpy(chat_log[slot], line, len);
    chat_log[slot][len] = '\0';
    chat_count++;
    chat_scroll = 0;
}

static void chat_add_wrapped(const char *prefix, const char *text) {
    char line[CHAT_LINE_LEN];
    int max_w;
    int tlen;
    int pos;
    int first;
    int chunk;
    int cut;
    int nl;

    max_w = COLS - 2;
    if (max_w < 20) max_w = 20;
    if (max_w > CHAT_LINE_LEN - 1) max_w = CHAT_LINE_LEN - 1;

    tlen = (int)strlen(text);
    pos = 0;
    first = 1;

    while (pos < tlen) {
        if (first && prefix) {
            chunk = max_w - (int)strlen(prefix);
            if (chunk < 1) chunk = 1;
            if (pos + chunk > tlen) chunk = tlen - pos;
            for (nl = 0; nl < chunk; nl++) {
                if (text[pos + nl] == '\n') break;
            }
            if (nl < chunk) {
                cut = nl;
            } else {
                cut = chunk;
                if (pos + chunk < tlen) {
                    while (cut > 0 && text[pos + cut] != ' ') cut--;
                    if (cut <= 0) cut = chunk;
                }
            }
            snprintf(line, sizeof(line), "%s%.*s", prefix, cut, text + pos);
            chat_add(line);
            pos += cut;
            first = 0;
        } else {
            chunk = max_w;
            if (pos + chunk > tlen) chunk = tlen - pos;
            for (nl = 0; nl < chunk; nl++) {
                if (text[pos + nl] == '\n') break;
            }
            if (nl < chunk) {
                cut = nl;
            } else {
                cut = chunk;
                if (pos + chunk < tlen) {
                    while (cut > 0 && text[pos + cut] != ' ') cut--;
                    if (cut <= 0) cut = chunk;
                }
            }
            snprintf(line, sizeof(line), "%.*s", cut, text + pos);
            chat_add(line);
            pos += cut;
        }
        if (pos < tlen && text[pos] == '\n') {
            pos++;
        } else {
            while (pos < tlen && text[pos] == ' ') pos++;
        }
    }
}

static void draw_chat(void) {
    int rows, cols;
    int chat_rows;
    int start;
    int i, line_idx;

    getmaxyx(stdscr, rows, cols);
    chat_rows = rows - 3;
    if (chat_rows < 1) chat_rows = 1;

    start = chat_count - chat_rows - chat_scroll;
    if (start < 0) start = 0;

    for (i = 0; i < chat_rows; i++) {
        move(i, 0);
        clrtoeol();
        line_idx = start + i;
        if (line_idx >= 0 && line_idx < chat_count) {
            mvaddnstr(i, 0, chat_log[line_idx % CHAT_LINES], cols - 1);
        }
    }

    move(chat_rows, 0);
    attron(A_REVERSE);
    for (i = 0; i < cols; i++) addch(' ');
    mvprintw(chat_rows, 1,
        " T:%.1f N:%.1f FP:%.1f RP:%.1f TP:%.2f PP:%.1f ",
        cfg_temp, cfg_noise, cfg_freq_penalty, cfg_rep_penalty,
        cfg_top_p, cfg_presence_penalty);
    attroff(A_REVERSE);
}

static void draw_input(const char *buf, int cursor) {
    int rows, cols;
    int input_row;

    getmaxyx(stdscr, rows, cols);
    input_row = rows - 2;

    move(input_row, 0);
    clrtoeol();
    mvprintw(input_row, 0, "You: %s", buf);
    move(input_row, 5 + cursor);
    (void)cols;
}

static void hist_push(const char *s, const int *toks, int n) {
    int slot, cp;
    size_t slen;

    slot = hist_cnt % HIST_MAX;
    slen = strlen(s);
    if (slen >= HIST_LEN) slen = HIST_LEN - 1;
    memcpy(hist_buf[slot], s, slen);
    hist_buf[slot][slen] = '\0';
    cp = n < MAX_TOKENS ? n : MAX_TOKENS;
    memcpy(hist_tokens[slot], toks, cp * sizeof(int));
    hist_lens[slot] = cp;
    hist_cnt++;
}

static int compute_smart_length(const Features *feat, int pattern, int input_words) {
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
    if (result > MAX_GEN) result = MAX_GEN;
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

static void inject_history_chaos(int *mixed, int *mixed_n, int max_mix) {
    int num_injections, i, slot, pos, tok_idx;

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

static int str_casecmp(const char *a, const char *b) {
    while (*a && *b) {
        int ca = tolower((unsigned char)*a);
        int cb = tolower((unsigned char)*b);
        if (ca != cb) return ca - cb;
        a++;
        b++;
    }
    return (unsigned char)*a - (unsigned char)*b;
}

static int handle_command(const char *cmd) {
    char name[64];
    char arg[64];
    int n;
    float val;
    int i;
    char model_line[256];

    if (cmd[0] != '/') return 0;

    n = sscanf(cmd, "%63s %63s", name, arg);
    if (n < 1) return 0;

    if (strcmp(name, "/help") == 0) {
        chat_add("--- Commands ---");
        chat_add("/temp <val>    - set temperature (default 6.0)");
        chat_add("/noise <val>   - set noise level (default 4.5)");
        chat_add("/freq <val>    - frequency penalty (default 3.5)");
        chat_add("/rep <val>     - repetition penalty mult (default 1.0)");
        chat_add("/topp <val>    - top-p / nucleus sampling (default 1.0)");
        chat_add("/presence <val>- presence penalty (default 0.0)");
        chat_add("/maxwords <val>- override max response words (0=auto)");
        chat_add("/model [name]  - list models or switch model");
        chat_add("/clear         - clear chat history");
        chat_add("/quit          - exit");
        return 1;
    }

    if (strcmp(name, "/model") == 0) {
        if (n < 2) {
            chat_add("--- Available Models ---");
            for (i = 0; i < MODEL_COUNT; i++) {
                snprintf(model_line, sizeof(model_line), "%s%s - %s",
                    (i == current_model) ? "* " : "  ",
                    models[i].name, models[i].description);
                chat_add(model_line);
            }
            snprintf(model_line, sizeof(model_line), "Current: %s", models[current_model].name);
            chat_add(model_line);
        } else {
            for (i = 0; i < MODEL_COUNT; i++) {
                if (str_casecmp(arg, models[i].name) == 0) {
                    current_model = i;
                    cfg_temp = models[i].default_temp;
                    cfg_noise = models[i].default_noise;
                    cfg_freq_penalty = models[i].default_freq_penalty;
                    cfg_rep_penalty = models[i].default_rep_penalty;
                    cfg_top_p = models[i].default_top_p;
                    cfg_presence_penalty = models[i].default_presence_penalty;
                    snprintf(model_line, sizeof(model_line), "Switched to %s", models[i].name);
                    chat_add(model_line);
                    return 1;
                }
            }
            chat_add("Unknown model. Use /model to list available models.");
        }
        return 1;
    }

    if (strcmp(name, "/quit") == 0 || strcmp(name, "/exit") == 0) {
        return -1;
    }

    if (strcmp(name, "/clear") == 0) {
        chat_count = 0;
        chat_scroll = 0;
        chat_add("Chat cleared.");
        return 1;
    }

    if (n < 2) {
        chat_add("Missing argument. Type /help for usage.");
        return 1;
    }

    if (strcmp(name, "/temp") == 0) {
        val = (float)atof(arg);
        if (val > 0.01f && val < 100.0f) {
            cfg_temp = val;
            chat_add("Temperature set.");
        } else {
            chat_add("Invalid value (0.01 - 100).");
        }
        return 1;
    }

    if (strcmp(name, "/noise") == 0) {
        val = (float)atof(arg);
        if (val >= 0.0f && val < 100.0f) {
            cfg_noise = val;
            chat_add("Noise set.");
        } else {
            chat_add("Invalid value (0 - 100).");
        }
        return 1;
    }

    if (strcmp(name, "/freq") == 0) {
        val = (float)atof(arg);
        if (val >= 0.0f && val < 50.0f) {
            cfg_freq_penalty = val;
            chat_add("Frequency penalty set.");
        } else {
            chat_add("Invalid value (0 - 50).");
        }
        return 1;
    }

    if (strcmp(name, "/rep") == 0) {
        val = (float)atof(arg);
        if (val >= 0.0f && val < 20.0f) {
            cfg_rep_penalty = val;
            chat_add("Repetition penalty set.");
        } else {
            chat_add("Invalid value (0 - 20).");
        }
        return 1;
    }

    if (strcmp(name, "/topp") == 0) {
        val = (float)atof(arg);
        if (val > 0.0f && val <= 1.0f) {
            cfg_top_p = val;
            chat_add("Top-p set.");
        } else {
            chat_add("Invalid value (0.01 - 1.0).");
        }
        return 1;
    }

    if (strcmp(name, "/presence") == 0) {
        val = (float)atof(arg);
        if (val >= -10.0f && val <= 10.0f) {
            cfg_presence_penalty = val;
            chat_add("Presence penalty set.");
        } else {
            chat_add("Invalid value (-10 - 10).");
        }
        return 1;
    }

    if (strcmp(name, "/maxwords") == 0) {
        n = atoi(arg);
        if (n >= 0 && n <= 5000) {
            cfg_max_words = n;
            if (n == 0) {
                chat_add("Max words set to auto.");
            } else {
                chat_add("Max words override set.");
            }
        } else {
            chat_add("Invalid value (0-5000, 0=auto).");
        }
        return 1;
    }

    chat_add("Unknown command. Type /help for usage.");
    return 1;
}

static void show_status(const char *msg) {
    int rows, cols;
    int status_row;

    getmaxyx(stdscr, rows, cols);
    status_row = rows - 3;

    move(status_row, 0);
    attron(A_DIM);
    clrtoeol();
    mvaddnstr(status_row, 0, msg, cols - 1);
    attroff(A_DIM);
    refresh();
    (void)cols;
}

static void generate_response(const char *input) {
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

    prev_word = rand() % ACTUAL_VOCAB;

    show_status("Thinking...");

    personality_temp_mult = 1.0f + (float)hist_cnt * 0.04f;
    personality_noise_mult = 1.0f + (float)hist_cnt * 0.06f;
    if (personality_temp_mult > 3.0f) personality_temp_mult = 3.0f;
    if (personality_noise_mult > 4.0f) personality_noise_mult = 4.0f;

    if (hist_cnt > 3 && rand() % 100 < 8 + hist_cnt * 2) {
        amnesia_slot = rand() % (hist_cnt < HIST_MAX ? hist_cnt : HIST_MAX);
        strncpy(amnesia_input, hist_buf[amnesia_slot], HIST_LEN - 1);
        amnesia_input[HIST_LEN - 1] = '\0';
        hist_push(input, cur_tokens, 0);
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

    mixed_n = cur_len < MAX_TOKENS ? cur_len : MAX_TOKENS;
    memcpy(mixed, cur_tokens, mixed_n * sizeof(int));

    if (hist_cnt > 0 && rand() % 2 == 0) {
        slot = (hist_cnt - 1 - rand() % (hist_cnt < HIST_MAX ? hist_cnt : HIST_MAX) + HIST_MAX) % HIST_MAX;
        add = hist_lens[slot];
        if (mixed_n + add < MAX_TOKENS * 4) {
            memcpy(mixed + mixed_n, hist_tokens[slot], add * sizeof(int));
            mixed_n += add;
        }
    }

    inject_history_chaos(mixed, &mixed_n, MAX_TOKENS * 4);
    hist_push(input, cur_tokens, cur_len);
    reset_recent();
    encode_context(mixed, mixed_n, ctx);
    scramble_context(ctx, 1.5f + (float)(rand() % 100) / 100.0f);
    memcpy(prev_embed, embeddings[prev_word], EMBED_DIM * sizeof(float));

    base_temp = cfg_temp * personality_temp_mult;
    base_noise = cfg_noise * personality_noise_mult;
    if (feat.is_question > 0.5f) { base_temp *= 0.9f; base_noise *= 1.4f; }
    if (feat.exclamation > 0.5f) { base_temp *= 0.85f; base_noise *= 1.6f; }
    if (feat.entropy > 0.7f) { base_temp *= 1.3f; base_noise *= 1.1f; }
    if (feat.digit_ratio > 0.1f) { base_temp *= 1.5f; base_noise *= 1.5f; }
    if (pattern == PAT_MATH) { base_temp *= 1.4f; base_noise *= 1.8f; }
    if (pattern == PAT_TECH) { base_temp *= 1.2f; base_noise *= 1.3f; }

    max_words = compute_smart_length(&feat, pattern, input_words);

    temp_mode_bias = (cfg_temp - TEMP_BASE) / TEMP_BASE;

    resp_mode = pick_response_mode(pattern, &feat, hist_cnt);

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

        think_words = 50 + rand() % 40;
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

        for (step = 0; step < think_words; step++) {
            next_embed(think_ctx, think_prev, feat_ctx, think_target);
            derail_embed(think_target, step, think_words);

            for (d = 0; d < EMBED_DIM; d++)
                think_target[d] += frand_r() * (base_noise * 2.0f);

            think_temp_val = base_temp * 1.5f;
            think_word_id = sample_vocab(think_target, think_temp_val, base_noise * 2.0f,
                                cfg_freq_penalty, cfg_rep_penalty, cfg_top_p);

            think_w = vocab[think_word_id];
            think_wlen = (int)strlen(think_w);

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

            if (think_line_words >= 6 + rand() % 5) {
                think_line[think_wb] = '\0';
                chat_add(think_line);
                draw_chat();
                refresh();
                napms(60 + rand() % 150);
                think_wb = 4;
                think_line[0] = ' ';
                think_line[1] = ' ';
                think_line[2] = '>';
                think_line[3] = ' ';
                think_line_words = 0;
            }
        }

        if (think_line_words > 0) {
            think_line[think_wb] = '\0';
            chat_add(think_line);
            draw_chat();
            refresh();
            napms(60 + rand() % 150);
        }
    }

    user_match = knowledge_match_user(&know_extra, input);
    if (!user_match) user_match = knowledge_match_user(&know_tech, input);
    if (!user_match) user_match = knowledge_match_user(&know_general, input);
    if (!user_match) user_match = knowledge_match_user(&know_phrases_en, input);
    if (user_match && rand() % 100 < 80) {
        snprintf(corpus_buf, sizeof(corpus_buf), "%s", user_match);
        if (rand() % 100 < 40) {
            scramble_words(corpus_buf, word_buf, sizeof(word_buf));
            snprintf(corpus_buf, sizeof(corpus_buf), "%s", word_buf);
        }
        used_corpus = 1;
    } else {
        used_corpus = generate_corpus_response(resp_mode, input, pattern, &feat, corpus_buf, sizeof(corpus_buf));
    }

    if (used_corpus) {
        unicode_chance = 15 + hist_cnt * 3 + (int)(cfg_noise * 3.0f);
        if (unicode_chance > 90) unicode_chance = 90;
        if (resp_mode != RESP_CODE_GEN && rand() % 100 < unicode_chance) {
            inject_unicode(corpus_buf, sizeof(corpus_buf));
        }

        if (resp_mode != RESP_CODE_GEN && cfg_noise > NOISE_BASE * 1.5f) {
            scramble_words(corpus_buf, word_buf, sizeof(word_buf));
            chat_add_wrapped("TuffAI: ", word_buf);
        } else {
            chat_add_wrapped("TuffAI: ", corpus_buf);
        }
        show_status("");
        return;
    }

    show_status("Generating...");

    wb_pos = 0;
    for (step = 0; step < max_words; step++) {
        next_embed(ctx, prev_embed, feat_ctx, target);
        derail_embed(target, step, max_words);

        for (d = 0; d < EMBED_DIM; d++)
            target[d] += frand_r() * (base_noise * (1.0f + (float)step / max_words * 0.8f));

        step_temp = base_temp * (1.0f + (float)step / max_words * 0.5f);
        word = sample_vocab(target, step_temp, base_noise,
                            cfg_freq_penalty, cfg_rep_penalty, cfg_top_p);
        push_recent(word);

        w = vocab[word];
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

    word_buf[wb_pos++] = '.';
    word_buf[wb_pos] = '\0';

    unicode_chance = 15 + hist_cnt * 3 + (int)(cfg_noise * 3.0f);
    if (unicode_chance > 90) unicode_chance = 90;
    if (rand() % 100 < unicode_chance) {
        inject_unicode(word_buf, sizeof(word_buf));
    }

    chat_add_wrapped("TuffAI: ", word_buf);
    show_status("");
}

int main(void) {
    char input_buf[INPUT_MAX];
    int input_pos;
    int ch;
    int cmd_result;
    int running;
    char user_line[INPUT_MAX + 16];

    net_init();
    initscr();
    raw();
    keypad(stdscr, TRUE);
    noecho();
    scrollok(stdscr, FALSE);

    chat_add("");
    snprintf(input_buf, sizeof(input_buf), "%s %s", TUFFAI_NAME, TUFFAI_VERSION);
    chat_add(input_buf);
    snprintf(input_buf, sizeof(input_buf), "Model: %s", models[current_model].name);
    chat_add(input_buf);
    chat_add("Type /help for commands. /quit to exit.");
    chat_add("");

    input_buf[0] = '\0';
    input_pos = 0;
    running = 1;

    while (running) {
        draw_chat();
        draw_input(input_buf, input_pos);
        refresh();

        ch = getch();

        if (ch == '\n' || ch == KEY_ENTER) {
            if (input_pos == 0) continue;
            input_buf[input_pos] = '\0';

            if (input_buf[0] == '/') {
                cmd_result = handle_command(input_buf);
                if (cmd_result < 0) {
                    running = 0;
                    break;
                }
            } else {
                snprintf(user_line, sizeof(user_line), "You: %s", input_buf);
                chat_add(user_line);
                generate_response(input_buf);
                chat_add("");
            }

            input_buf[0] = '\0';
            input_pos = 0;
        } else if (ch == KEY_BACKSPACE || ch == 127 || ch == 8) {
            if (input_pos > 0) {
                input_pos--;
                input_buf[input_pos] = '\0';
            }
        } else if (ch == KEY_UP) {
            if (chat_scroll < chat_count - 1) chat_scroll++;
        } else if (ch == KEY_DOWN) {
            if (chat_scroll > 0) chat_scroll--;
        } else if (ch == KEY_PPAGE) {
            chat_scroll += 10;
            if (chat_scroll > chat_count - 1) chat_scroll = chat_count - 1;
        } else if (ch == KEY_NPAGE) {
            chat_scroll -= 10;
            if (chat_scroll < 0) chat_scroll = 0;
        } else if (ch >= 32 && ch < 127 && input_pos < INPUT_MAX - 1) {
            input_buf[input_pos++] = (char)ch;
            input_buf[input_pos] = '\0';
        }
    }

    endwin();
    return 0;
}