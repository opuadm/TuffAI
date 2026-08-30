#define _XOPEN_SOURCE 700
#include "../../../engine.h"
#include "../../../net.h"
#include "../../../features.h"
#include "../../../tokenizer.h"
#include "../../../knowledge/knowledge.h"
#include "../../../rng.h"
#include "../../../websearch.h"
#include "../retrieval.h"
#include "../code_mode.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <wchar.h>
#include <wctype.h>

#define V3_CONTEXT_TOKENS 32768
#define V3_CONTEXT_TRIM 1024
#define V3_RECENT_WORDS RENDERED_RECENT_MAX
#define V3_WORD_LEN RENDERED_WORD_LEN
#define V3_BIAS_TOKENS 512
#define V3_MAX_RESPONSE_WORDS 1000
#define V3_CODE_RESPONSE_SIZE 65536
#define V3_ACTIVE_ATTENTION_TOKENS 16
#define V3_SEARCH_TEXT_SIZE 65536
#define V3_SOURCE_WORDS 2048

#ifndef TUFFAI_ENGINE_VTABLE
#define TUFFAI_ENGINE_VTABLE engine_tuffai_v3
#endif

#ifndef TUFFAI_GENERATE_RESPONSE
#define TUFFAI_GENERATE_RESPONSE v3_generate_response
#endif

#ifndef TUFFAI_INTACT_TOKEN_THRESHOLD
#define TUFFAI_INTACT_TOKEN_THRESHOLD 72
#endif

#ifndef TUFFAI_MUTATED_TOKEN_THRESHOLD
#define TUFFAI_MUTATED_TOKEN_THRESHOLD 78
#endif

#ifndef TUFFAI_UNICODE_TOKEN_THRESHOLD
#define TUFFAI_UNICODE_TOKEN_THRESHOLD 80
#endif

#ifndef TUFFAI_MUTATE_FORCED_WORD
#define TUFFAI_MUTATE_FORCED_WORD 0
#endif

#ifndef TUFFAI_RETRIEVAL_SOURCE_CHANCE
#define TUFFAI_RETRIEVAL_SOURCE_CHANCE 24
#endif

static unsigned char v3_glitch_token_mask[V3_VOCAB_SIZE];
static unsigned char v3_glitch_token_variant[V3_VOCAB_SIZE];
static int v3_glitch_token_partner[V3_VOCAB_SIZE];
static int v3_glitch_tokens_initialized;
static const char *const v3_effort_modes[] = {
    "None", "Low", "Medium", "High", "Max"
};

static int text_contains_case(const char *text, const char *needle) {
    int text_position;
    int needle_position;
    unsigned char left;
    unsigned char right;

    if (!text || !needle || !needle[0]) return 0;
    for (text_position = 0; text[text_position]; text_position++) {
        needle_position = 0;
        while (needle[needle_position] &&
               text[text_position + needle_position]) {
            left = (unsigned char)text[text_position + needle_position];
            right = (unsigned char)needle[needle_position];
            if (tolower(left) != tolower(right)) break;
            needle_position++;
        }
        if (!needle[needle_position]) return 1;
    }
    return 0;
}

static int identity_request(const char *input) {
    if (text_contains_case(input, "who are you") ||
        text_contains_case(input, "what are you"))
        return 1;
    return (text_contains_case(input, "model") ||
            text_contains_case(input, "which ai")) &&
           text_contains_case(input, "you") &&
           (text_contains_case(input, "which") ||
            text_contains_case(input, "what"));
}

static void append_context_tokens(int *tokens, int *count,
                                  const int *source, int source_count);

static void append_context_token(int *tokens, int *count, int token) {
    append_context_tokens(tokens, count, &token, 1);
}

static void append_context_tokens(int *tokens, int *count, const int *source, int source_count) {
    int discard_count;

    if (source_count <= 0) return;
    discard_count = *count + source_count - V3_CONTEXT_TOKENS;
    if (discard_count > 0) {
        if (discard_count < V3_CONTEXT_TRIM && *count >= V3_CONTEXT_TRIM)
            discard_count = V3_CONTEXT_TRIM;
        if (discard_count > *count) discard_count = *count;
        memmove(tokens, tokens + discard_count,
                (*count - discard_count) * sizeof(int));
        *count -= discard_count;
    }
    memcpy(tokens + *count, source, source_count * sizeof(int));
    *count += source_count;
}

static void push_history(EngineState *state, const char *input, const int *tokens, int token_count) {
    int slot;
    int copy_count;
    size_t input_len;

    slot = *state->hist_cnt % HIST_MAX;
    input_len = strlen(input);
    if (input_len >= HIST_LEN) input_len = HIST_LEN - 1;
    memcpy(state->hist_buf[slot], input, input_len);
    state->hist_buf[slot][input_len] = '\0';
    copy_count = token_count < MAX_TOKENS ? token_count : MAX_TOKENS;
    memcpy(state->hist_tokens[slot], tokens, copy_count * sizeof(int));
    state->hist_lens[slot] = copy_count;
    (*state->hist_cnt)++;
}

static void build_context(EngineState *state, const char *input,
                          const V3Retrieval *retrieval,
                          const char *search_text, int *context,
                          int *context_count, int *input_tokens,
                          int *input_count) {
    int temp_tokens[MAX_TOKENS];
    int search_tokens[V3_CONTEXT_TOKENS];
    int temp_count;
    int available_history;
    int history_count;
    int history_slot;
    int i;

    *context_count = 0;
    *input_count = v3_tokenize(input, input_tokens, MAX_TOKENS);
    if (state->system_prompt[0]) {
        temp_count = v3_tokenize(state->system_prompt, temp_tokens, MAX_TOKENS);
        append_context_tokens(context, context_count, temp_tokens, temp_count);
    }
    available_history = *state->hist_cnt < HIST_MAX ? *state->hist_cnt : HIST_MAX;
    history_count = available_history < 3 ? available_history : 3;
    for (i = history_count; i > 0; i--) {
        history_slot = (*state->hist_cnt - i + HIST_MAX) % HIST_MAX;
        append_context_tokens(context, context_count, state->hist_tokens[history_slot], state->hist_lens[history_slot]);
    }
    if (state->self_ctx_len > 0)
        append_context_tokens(context, context_count, state->self_ctx_tokens, state->self_ctx_len);
    if (retrieval->text) {
        temp_count = v3_tokenize(retrieval->text, temp_tokens, MAX_TOKENS);
        if (temp_count > 48) temp_count = 48;
        append_context_tokens(context, context_count, temp_tokens, temp_count);
    }
    if (search_text && search_text[0]) {
        temp_count = v3_tokenize(search_text, search_tokens,
                                 V3_CONTEXT_TOKENS);
        append_context_tokens(context, context_count, search_tokens,
                              temp_count);
    }
    append_context_tokens(context, context_count, input_tokens, *input_count);
}

static int model_wants_search(const char *input, const Features *features,
                              const V3Retrieval *retrieval) {
    int probability;

    if (!web_search_enabled()) return 0;
    if (web_search_requested(input)) return 1;
    if (detect_pattern(input) == PAT_MATH) return 0;
    if (retrieval->is_code || retrieval->requested_words > 0) return 0;
    probability = 2;
    probability += (int)(features->is_question * 48.0f);
    probability += (int)(features->has_number * 8.0f);
    probability += (int)(features->long_word_ratio * 16.0f);
    probability += (int)(features->entropy * 10.0f);
    if (retrieval->score < 20) probability += 20;
    if (retrieval->matches == 0) probability += 10;
    if (probability > 82) probability = 82;
    return rng_range(100) < probability;
}

static int evaluate_basic_arithmetic(const char *input, char *output,
                                     int output_size) {
    const char *cursor;
    char *left_end;
    char *right_end;
    double left;
    double right;
    double result;
    char operation;
    int written;

    cursor = input;
    while (*cursor && !isdigit((unsigned char)*cursor) &&
           !((*cursor == '+' || *cursor == '-') &&
             isdigit((unsigned char)cursor[1])))
        cursor++;
    if (!*cursor) return 0;
    left = strtod(cursor, &left_end);
    if (left_end == cursor) return 0;
    cursor = left_end;
    while (isspace((unsigned char)*cursor)) cursor++;
    operation = *cursor;
    if (operation != '+' && operation != '-' && operation != '*' &&
        operation != '/')
        return 0;
    cursor++;
    while (isspace((unsigned char)*cursor)) cursor++;
    right = strtod(cursor, &right_end);
    if (right_end == cursor) return 0;
    if (operation == '+') result = left + right;
    else if (operation == '-') result = left - right;
    else if (operation == '*') result = left * right;
    else {
        if (right == 0.0) return 0;
        result = left / right;
    }
    if (!isfinite(result)) return 0;
    written = snprintf(output, (size_t)output_size,
                       "%.15g%c%.15g=%.15g",
                       left, operation, right, result);
    return written > 0 && written < output_size;
}

static void build_memory_bias(EngineState *state, const int *input_tokens,
                              int input_count, int *bias_tokens,
                              int *bias_count) {
    int available_history;
    int history_count;
    int history_slot;
    int copy_count;
    int i;

    *bias_count = 0;
    available_history = *state->hist_cnt < HIST_MAX ?
                        *state->hist_cnt : HIST_MAX;
    history_count = available_history < 3 ? available_history : 3;
    for (i = history_count; i > 0; i--) {
        history_slot = (*state->hist_cnt - i + HIST_MAX) % HIST_MAX;
        copy_count = state->hist_lens[history_slot];
        if (*bias_count + copy_count > V3_BIAS_TOKENS)
            copy_count = V3_BIAS_TOKENS - *bias_count;
        if (copy_count > 0) {
            memcpy(bias_tokens + *bias_count,
                   state->hist_tokens[history_slot],
                   copy_count * sizeof(int));
            *bias_count += copy_count;
        }
    }
    copy_count = input_count;
    if (*bias_count + copy_count > V3_BIAS_TOKENS) {
        copy_count = V3_BIAS_TOKENS - *bias_count;
        if (copy_count < input_count)
            input_tokens += input_count - copy_count;
    }
    if (copy_count > 0) {
        memcpy(bias_tokens + *bias_count, input_tokens,
               copy_count * sizeof(int));
        *bias_count += copy_count;
    }
}

static int decode_utf8(const char *text, int available, unsigned int *codepoint) {
    const unsigned char *bytes;
    unsigned int value;
    int length;
    int i;

    bytes = (const unsigned char *)text;
    if (available <= 0 || bytes[0] == 0) return 0;
    if (bytes[0] < 0x80) {
        *codepoint = bytes[0];
        return 1;
    }
    if ((bytes[0] & 0xE0) == 0xC0) {
        value = bytes[0] & 0x1F;
        length = 2;
        if (value < 2) return 0;
    } else if ((bytes[0] & 0xF0) == 0xE0) {
        value = bytes[0] & 0x0F;
        length = 3;
    } else if ((bytes[0] & 0xF8) == 0xF0) {
        value = bytes[0] & 0x07;
        length = 4;
    } else {
        return 0;
    }
    if (length > available) return 0;
    for (i = 1; i < length; i++) {
        if ((bytes[i] & 0xC0) != 0x80) return 0;
        value = (value << 6) | (bytes[i] & 0x3F);
    }
    if ((length == 3 && value < 0x800) ||
        (length == 4 && value < 0x10000) ||
        value > 0x10FFFF ||
        (value >= 0xD800 && value <= 0xDFFF)) return 0;
    *codepoint = value;
    return length;
}

static int encode_utf8(unsigned int codepoint, char *output,
                       int output_size) {
    if (codepoint <= 0x7F) {
        if (output_size < 2) return 0;
        output[0] = (char)codepoint;
        output[1] = '\0';
        return 1;
    }
    if (codepoint <= 0x7FF) {
        if (output_size < 3) return 0;
        output[0] = (char)(0xC0 | (codepoint >> 6));
        output[1] = (char)(0x80 | (codepoint & 0x3F));
        output[2] = '\0';
        return 2;
    }
    if (codepoint <= 0xFFFF) {
        if (output_size < 4) return 0;
        output[0] = (char)(0xE0 | (codepoint >> 12));
        output[1] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
        output[2] = (char)(0x80 | (codepoint & 0x3F));
        output[3] = '\0';
        return 3;
    }
    if (codepoint <= 0x10FFFF) {
        if (output_size < 5) return 0;
        output[0] = (char)(0xF0 | (codepoint >> 18));
        output[1] = (char)(0x80 | ((codepoint >> 12) & 0x3F));
        output[2] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
        output[3] = (char)(0x80 | (codepoint & 0x3F));
        output[4] = '\0';
        return 4;
    }
    return 0;
}

static int private_codepoint(unsigned int codepoint) {
    if (codepoint >= 0xE000 && codepoint <= 0xF8FF) return 1;
    if (codepoint >= 0xF0000 && codepoint <= 0xFFFFD) return 1;
    if (codepoint >= 0x100000 && codepoint <= 0x10FFFD) return 1;
    return 0;
}

static int noncharacter_codepoint(unsigned int codepoint) {
    if (codepoint >= 0xFDD0 && codepoint <= 0xFDEF) return 1;
    if ((codepoint & 0xFFFF) == 0xFFFE ||
        (codepoint & 0xFFFF) == 0xFFFF) return 1;
    return 0;
}

static int recent_codepoint(EngineState *state, unsigned int codepoint) {
    int i;

    for (i = 0; i < state->recent_codepoint_count; i++)
        if (state->recent_codepoints[i] == codepoint) return 1;
    return 0;
}

static void remember_codepoint(EngineState *state, unsigned int codepoint) {
    int i;

    if (state->recent_codepoint_count < RECENT_CODEPOINT_MAX) {
        state->recent_codepoints[state->recent_codepoint_count++] = codepoint;
        return;
    }
    for (i = 1; i < RECENT_CODEPOINT_MAX; i++)
        state->recent_codepoints[i - 1] = state->recent_codepoints[i];
    state->recent_codepoints[RECENT_CODEPOINT_MAX - 1] = codepoint;
}

static int printable_random_codepoint(EngineState *state,
                                      unsigned int *codepoint) {
    unsigned int candidate;
    int width;
    int attempt;

    for (attempt = 0; attempt < 8192; attempt++) {
        candidate = (unsigned int)rng_range(0x100000);
        if (candidate == 0xFFFD) continue;
        if (candidate >= 0xD800 && candidate <= 0xDFFF) continue;
        if (private_codepoint(candidate) ||
            noncharacter_codepoint(candidate)) continue;
        if (!iswprint((wint_t)candidate)) continue;
        width = wcwidth((wchar_t)candidate);
        if (width < 1 || width > 2) continue;
        if (recent_codepoint(state, candidate)) continue;
        *codepoint = candidate;
        remember_codepoint(state, candidate);
        return 1;
    }
    return 0;
}

static int random_unicode_word(EngineState *state, char *word,
                               int word_size) {
    char encoded[5];
    unsigned int codepoint;
    int target;
    int encoded_length;
    int word_length;
    int i;

    target = 1 + rng_range(3);
    word_length = 0;
    for (i = 0; i < target; i++) {
        if (!printable_random_codepoint(state, &codepoint)) break;
        encoded_length = encode_utf8(codepoint, encoded, sizeof(encoded));
        if (encoded_length <= 0 ||
            word_length + encoded_length >= word_size) break;
        memcpy(word + word_length, encoded, encoded_length);
        word_length += encoded_length;
    }
    word[word_length] = '\0';
    return word_length;
}

static int terminal_safe_codepoint(unsigned int codepoint) {
    if (codepoint < 0x20 || (codepoint >= 0x7F && codepoint <= 0x9F)) return 0;
    if (codepoint == 0x061C ||
        (codepoint >= 0x200B && codepoint <= 0x200F) ||
        (codepoint >= 0x2028 && codepoint <= 0x202E) ||
        (codepoint >= 0x2060 && codepoint <= 0x206F) ||
        codepoint == 0xFEFF ||
        (codepoint >= 0xFFF9 && codepoint <= 0xFFFB)) return 0;
    if (codepoint >= '0' && codepoint <= '9') return 1;
    if (codepoint >= 'A' && codepoint <= 'Z') return 1;
    if (codepoint >= 'a' && codepoint <= 'z') return 1;
    if (codepoint >= 0x00C0 && codepoint <= 0x02AF) return 1;
    if (codepoint >= 0x0370 && codepoint <= 0x052F) return 1;
    if (codepoint >= 0x05D0 && codepoint <= 0x05EA) return 1;
    if (codepoint >= 0x0620 && codepoint <= 0x063F) return 1;
    if (codepoint >= 0x0641 && codepoint <= 0x064A) return 1;
    if (codepoint >= 0x066E && codepoint <= 0x066F) return 1;
    if (codepoint >= 0x0671 && codepoint <= 0x06D3) return 1;
    if (codepoint >= 0x06FA && codepoint <= 0x06FC) return 1;
    if (codepoint >= 0x0900 && codepoint <= 0x097F) return 1;
    if (codepoint >= 0x0E00 && codepoint <= 0x0E7F) return 1;
    if (codepoint >= 0x3040 && codepoint <= 0x30FF) return 1;
    if (codepoint >= 0x3400 && codepoint <= 0x9FFF) return 1;
    if (codepoint >= 0xAC00 && codepoint <= 0xD7A3) return 1;
    return 0;
}

static int sanitize_token(const char *source, char *output, int output_size) {
    int source_len;
    int source_pos;
    int output_pos;
    int sequence_len;
    unsigned int codepoint;

    source_len = (int)strlen(source);
    source_pos = 0;
    output_pos = 0;
    while (source_pos < source_len && output_pos < output_size - 1) {
        sequence_len = decode_utf8(source + source_pos,
                                   source_len - source_pos, &codepoint);
        if (sequence_len <= 0) {
            source_pos++;
            continue;
        }
        if (terminal_safe_codepoint(codepoint) &&
            output_pos + sequence_len < output_size) {
            memcpy(output + output_pos, source + source_pos, sequence_len);
            output_pos += sequence_len;
        }
        source_pos += sequence_len;
    }
    output[output_pos] = '\0';
    return output_pos;
}

static unsigned int v3_glitch_hash(const char *text) {
    unsigned int hash;

    hash = 2166136261u;
    while (*text) {
        hash ^= (unsigned char)*text++;
        hash *= 16777619u;
    }
    return hash;
}

static float v3_glitch_score(int token, const float *centroid) {
    const unsigned char *text;
    unsigned int hash;
    float score;
    float difference;
    int length;
    int high_bytes;
    int repeated_bytes;
    int transitions;
    int previous_high;
    int current_high;
    int d;

    text = (const unsigned char *)v3_vocab[token];
    if (!text || !text[0]) return -1.0f;
    length = 0;
    high_bytes = 0;
    repeated_bytes = 0;
    transitions = 0;
    previous_high = text[0] >= 128;
    while (text[length]) {
        current_high = text[length] >= 128;
        if (current_high) high_bytes++;
        if (length > 0 && text[length] == text[length - 1]) repeated_bytes++;
        if (length > 0 && current_high != previous_high) transitions++;
        previous_high = current_high;
        length++;
    }
    score = 0.0f;
    for (d = 0; d < EMBED_DIM; d++) {
        difference = embeddings[token][d] - centroid[d];
        score += difference * difference;
    }
    score += (float)high_bytes * 0.035f;
    score += (float)repeated_bytes * 0.11f;
    score += (float)transitions * 0.08f;
    score += (float)length * 0.0015f;
    hash = v3_glitch_hash(v3_vocab[token]);
    score += (float)(hash & 1023u) / 1048576.0f;
    return score;
}

static int v3_nearest_glitch_partner(int token) {
    float similarity;
    float best_similarity;
    int best_token;
    int candidate;
    int candidate_index;
    int candidate_count;
    int start;
    int step;
    int d;

    best_similarity = -1000000.0f;
    best_token = token;
    candidate_count = V3_VOCAB_SIZE < 512 ? V3_VOCAB_SIZE : 512;
    start = (int)(v3_glitch_hash(v3_vocab[token]) % V3_VOCAB_SIZE);
    step = V3_VOCAB_SIZE / candidate_count + 1;
    for (candidate_index = 0; candidate_index < candidate_count;
         candidate_index++) {
        candidate = (start + candidate_index * step) % V3_VOCAB_SIZE;
        if (candidate == token || !v3_vocab[candidate] ||
            !v3_vocab[candidate][0]) continue;
        similarity = 0.0f;
        for (d = 0; d < EMBED_DIM; d++)
            similarity += embeddings[token][d] * embeddings[candidate][d];
        if (similarity > best_similarity) {
            best_similarity = similarity;
            best_token = candidate;
        }
    }
    return best_token;
}

static void v3_initialize_glitch_tokens(void) {
    float centroid[EMBED_DIM];
    float scores[V3_VOCAB_SIZE];
    float best_score;
    int glitch_count;
    int best_token;
    int selected;
    int token;
    int d;

    if (v3_glitch_tokens_initialized) return;
    memset(v3_glitch_token_mask, 0, sizeof(v3_glitch_token_mask));
    memset(v3_glitch_token_variant, 0, sizeof(v3_glitch_token_variant));
    for (d = 0; d < EMBED_DIM; d++) {
        centroid[d] = 0.0f;
        for (token = 0; token < V3_VOCAB_SIZE; token++)
            centroid[d] += embeddings[token][d];
        centroid[d] /= (float)V3_VOCAB_SIZE;
    }
    for (token = 0; token < V3_VOCAB_SIZE; token++)
        scores[token] = v3_glitch_score(token, centroid);
    glitch_count = (int)(sqrt((double)V3_VOCAB_SIZE) / 2.0);
    for (selected = 0; selected < glitch_count; selected++) {
        best_score = -1.0f;
        best_token = -1;
        for (token = 0; token < V3_VOCAB_SIZE; token++) {
            if (!v3_glitch_token_mask[token] && scores[token] > best_score) {
                best_score = scores[token];
                best_token = token;
            }
        }
        if (best_token < 0) break;
        v3_glitch_token_mask[best_token] = 1;
        v3_glitch_token_variant[best_token] =
            (unsigned char)(v3_glitch_hash(v3_vocab[best_token]) % 3u);
        v3_glitch_token_partner[best_token] =
            v3_nearest_glitch_partner(best_token);
    }
    v3_glitch_tokens_initialized = 1;
}

static int v3_render_glitch_token(int token, char *word, int word_size) {
    char primary[V3_WORD_LEN];
    char partner[V3_WORD_LEN];
    int primary_length;
    int partner_length;
    int output_length;
    int copy_length;
    int variant;

    if (!v3_glitch_tokens_initialized) v3_initialize_glitch_tokens();
    if (token < 0 || token >= V3_VOCAB_SIZE ||
        !v3_glitch_token_mask[token]) return 0;
    primary_length = sanitize_token(v3_vocab[token], primary,
                                    sizeof(primary));
    partner_length = sanitize_token(
        v3_vocab[v3_glitch_token_partner[token]], partner,
        sizeof(partner));
    if (primary_length <= 0 || partner_length <= 0) return 0;
    output_length = 0;
    variant = v3_glitch_token_variant[token];
    if (variant == 2) {
        copy_length = partner_length;
        if (copy_length > word_size - output_length - 1)
            copy_length = word_size - output_length - 1;
        memcpy(word + output_length, partner, (size_t)copy_length);
        output_length += copy_length;
    }
    copy_length = primary_length;
    if (copy_length > word_size - output_length - 1)
        copy_length = word_size - output_length - 1;
    memcpy(word + output_length, primary, (size_t)copy_length);
    output_length += copy_length;
    if (variant != 2 && output_length < word_size - 1) {
        copy_length = variant == 0 ? primary_length : partner_length;
        if (copy_length > word_size - output_length - 1)
            copy_length = word_size - output_length - 1;
        memcpy(word + output_length,
               variant == 0 ? primary : partner,
               (size_t)copy_length);
        output_length += copy_length;
    }
    word[output_length] = '\0';
    return output_length;
}

static void encode_generation_context(const int *context, int context_count,
                                      float *encoded) {
    int active[V3_ACTIVE_ATTENTION_TOKENS];
    int older_count;
    int recent_count;
    int sampled_count;
    int source_index;
    int i;

    if (context_count <= V3_ACTIVE_ATTENTION_TOKENS) {
        encode_context_fast(context, context_count, encoded);
        return;
    }
    recent_count = V3_ACTIVE_ATTENTION_TOKENS / 2;
    older_count = context_count - recent_count;
    sampled_count = V3_ACTIVE_ATTENTION_TOKENS - recent_count;
    for (i = 0; i < sampled_count; i++) {
        source_index = i * older_count / sampled_count;
        active[i] = context[source_index];
    }
    memcpy(active + sampled_count, context + context_count - recent_count,
           (size_t)recent_count * sizeof(int));
    encode_context_fast(active, V3_ACTIVE_ATTENTION_TOKENS, encoded);
}

static int sample_model_token(EngineState *state, const float *feature_context,
                              int *context, int *context_count, float *previous,
                              const int *bias_tokens, int bias_count) {
    float encoded[EMBED_DIM];
    float target[EMBED_DIM];
    float temperature;
    float noise;
    int token;
    int d;

    encode_generation_context(context, *context_count, encoded);
    next_embed_fast(encoded, previous, feature_context, target);
    noise = state->cfg_noise;
    if (noise < 0.25f) noise = 0.25f;
    for (d = 0; d < EMBED_DIM; d++) target[d] += frand_r() * noise * 0.35f;
    temperature = state->cfg_temp;
    if (temperature < 0.2f) temperature = 0.2f;
    token = sample_vocab_contextual_fast(
        target, temperature, noise, state->cfg_freq_penalty,
        state->cfg_rep_penalty, state->cfg_top_p,
        TUFFAI_ENGINE_VTABLE.vocab_size, bias_tokens, bias_count, 0.12f);
    push_recent(token);
    append_context_token(context, context_count, token);
    memcpy(previous, embeddings[token], EMBED_DIM * sizeof(float));
    state->total_tokens++;
    state->sampled_tokens++;
    return token;
}

static int model_requests_stop(EngineState *state, const int *context,
                               int context_count, int generated_units,
                               int minimum_units, int boundary,
                               float stop_scale) {
    float encoded[EMBED_DIM];
    float alignment;
    float temperature;
    float probability;
    int token;
    int d;

    if (generated_units < minimum_units || context_count <= 0) return 0;
    encode_generation_context(context, context_count, encoded);
    token = context[context_count - 1] % V3_VOCAB_SIZE;
    if (token < 0) token += V3_VOCAB_SIZE;
    alignment = 0.0f;
    for (d = 0; d < EMBED_DIM; d++)
        alignment += encoded[d] * embeddings[token][d];
    alignment /= (float)EMBED_DIM;
    temperature = state->cfg_temp;
    if (temperature < 0.2f) temperature = 0.2f;
    alignment = tanhf(alignment * 3.0f / temperature);
    if (boundary)
        probability = 0.20f + (alignment + 1.0f) * 0.25f;
    else
        probability = 0.01f + (alignment + 1.0f) * 0.025f;
    probability *= stop_scale;
    if (probability > 1.0f) probability = 1.0f;
    return rng_unit() < probability;
}

static int create_fragment(const char *source, char *fragment, int fragment_size) {
    char clean[V3_WORD_LEN];
    int boundaries[V3_WORD_LEN];
    int clean_len;
    int codepoints;
    int position;
    int sequence_len;
    int start;
    int count;
    int end;
    unsigned int codepoint;

    clean_len = sanitize_token(source, clean, sizeof(clean));
    codepoints = 0;
    position = 0;
    while (position < clean_len && codepoints < V3_WORD_LEN - 1) {
        boundaries[codepoints++] = position;
        sequence_len = decode_utf8(clean + position, clean_len - position,
                                   &codepoint);
        if (sequence_len <= 0) sequence_len = 1;
        position += sequence_len;
    }
    boundaries[codepoints] = clean_len;
    if (codepoints == 0) {
        fragment[0] = '\0';
        return 0;
    }
    count = 1 + rng_range(6);
    if (count > codepoints) count = codepoints;
    start = rng_range(codepoints - count + 1);
    end = boundaries[start + count];
    position = boundaries[start];
    count = end - position;
    if (count >= fragment_size) count = fragment_size - 1;
    while (count > 0 && ((unsigned char)clean[position + count] & 0xC0) == 0x80)
        count--;
    memcpy(fragment, clean + position, count);
    fragment[count] = '\0';
    return count;
}

static int word_was_recent(EngineState *state, const char *word) {
    int start;
    int i;

    start = state->recent_rendered_count > 24 ?
            state->recent_rendered_count - 24 : 0;
    for (i = start; i < state->recent_rendered_count; i++)
        if (strcmp(state->recent_rendered[i], word) == 0) return 1;
    return 0;
}

static void remember_word(EngineState *state, const char *word) {
    int i;
    int word_len;

    word_len = (int)strlen(word);
    if (word_len >= V3_WORD_LEN) word_len = V3_WORD_LEN - 1;
    if (state->recent_rendered_count < V3_RECENT_WORDS) {
        memcpy(state->recent_rendered[state->recent_rendered_count], word,
               word_len);
        state->recent_rendered[state->recent_rendered_count][word_len] = '\0';
        state->recent_rendered_count++;
        return;
    }
    for (i = 1; i < V3_RECENT_WORDS; i++)
        memcpy(state->recent_rendered[i - 1], state->recent_rendered[i],
               V3_WORD_LEN);
    memcpy(state->recent_rendered[V3_RECENT_WORDS - 1], word, word_len);
    state->recent_rendered[V3_RECENT_WORDS - 1][word_len] = '\0';
}

static int mutate_token(const char *source, char *word, int word_size) {
    char clean[V3_WORD_LEN];
    int clean_len;
    int position;
    char replacement;

    clean_len = sanitize_token(source, clean, sizeof(clean));
    if (clean_len < 2) return 0;
    position = rng_range(clean_len);
    while (position > 0 && ((unsigned char)clean[position] & 0xC0) == 0x80)
        position--;
    replacement = (char)('a' + rng_range(26));
    if (clean_len >= word_size) clean_len = word_size - 1;
    memcpy(word, clean, clean_len);
    if ((unsigned char)word[position] < 128) word[position] = replacement;
    else if (clean_len + 1 < word_size) word[clean_len++] = replacement;
    word[clean_len] = '\0';
    return clean_len;
}

static int generate_word(EngineState *state, const float *feature_context,
                         int *context, int *context_count, float *previous,
                         const int *bias_tokens, int bias_count,
                         const char *forced_word, char *word, int word_size) {
    char fragment[16];
    int piece_count;
    int token;
    int fragment_len;
    int word_len;
    int attempt;
    int piece;
    int mode;

    word[0] = '\0';
    word_len = 0;
    mode = forced_word ? 15 : rng_range(100);
    token = sample_model_token(state, feature_context, context,
                               context_count, previous,
                               bias_tokens, bias_count);
    if (!forced_word)
        word_len = v3_render_glitch_token(token, word, word_size);
    if (word_len > 0) return word_len;
    if (forced_word && TUFFAI_MUTATE_FORCED_WORD)
        word_len = mutate_token(forced_word, word, word_size);
    else if (forced_word)
        word_len = sanitize_token(forced_word, word, word_size);
    if (word_len > 0) return word_len;
    if (mode < TUFFAI_INTACT_TOKEN_THRESHOLD)
        word_len = sanitize_token(v3_vocab[token], word, word_size);
    else if (mode < TUFFAI_MUTATED_TOKEN_THRESHOLD)
        word_len = mutate_token(v3_vocab[token], word, word_size);
    else if (mode < TUFFAI_UNICODE_TOKEN_THRESHOLD)
        word_len = random_unicode_word(state, word, word_size);
    if (word_len > 0) return word_len;

    piece_count = 1 + rng_range(5);
    for (piece = 0; piece < piece_count && word_len < word_size - 2; piece++) {
        fragment_len = 0;
        for (attempt = 0; attempt < 16 && fragment_len == 0; attempt++) {
            if (piece > 0 || attempt > 0)
                token = sample_model_token(state, feature_context, context,
                                           context_count, previous,
                                           bias_tokens, bias_count);
            fragment_len = create_fragment(v3_vocab[token], fragment,
                                           sizeof(fragment));
        }
        if (fragment_len > 0 && word_len + fragment_len < word_size - 1) {
            memcpy(word + word_len, fragment, fragment_len);
            word_len += fragment_len;
        }
    }
    if (word_len == 0) {
        word[0] = 'x';
        word_len = 1;
    }
    word[word_len] = '\0';
    return word_len;
}

static void append_generated_word(char *output, int output_size, int *output_len, const char *word, int word_index, int *sentence_words, int sentence_target) {
    int word_len;
    char separator;

    word_len = (int)strlen(word);
    if (*output_len + word_len + 3 >= output_size) return;
    if (word_index > 0) output[(*output_len)++] = ' ';
    memcpy(output + *output_len, word, word_len);
    *output_len += word_len;
    (*sentence_words)++;
    separator = '\0';
    if (*sentence_words >= sentence_target) {
        separator = rng_range(5) == 0 ? '?' : '.';
        *sentence_words = 0;
    } else if (*sentence_words > 2 && rng_range(11) == 0) {
        separator = ',';
    }
    if (separator) output[(*output_len)++] = separator;
    output[*output_len] = '\0';
}

static void stream_generated_text(const EngineCallbacks *cb,
                                  const char *prefix, int color,
                                  char *output, int output_len,
                                  int *streamed_len) {
    const char *chunk;

    if (!cb || !prefix || output_len <= *streamed_len) return;
    chunk = output + *streamed_len;
    while (*chunk == ' ') chunk++;
    if (*chunk)
        cb->stream_text(*streamed_len == 0 ? prefix : "", chunk, color);
    *streamed_len = output_len;
}

static void generate_mixed_answer(const char *source, char *output,
                                  int output_size) {
    char copy[8192];
    char changed[V3_WORD_LEN];
    char *words[512];
    char *word;
    char *temporary;
    int word_count;
    int output_length;
    int word_length;
    int join;
    int i;
    int current_has_digit;
    int next_has_digit;

    strncpy(copy, source, sizeof(copy) - 1);
    copy[sizeof(copy) - 1] = '\0';
    word_count = 0;
    word = strtok(copy, " \t\r\n");
    while (word && word_count < (int)(sizeof(words) / sizeof(words[0]))) {
        words[word_count++] = word;
        word = strtok(NULL, " \t\r\n");
    }
    for (i = 0; i + 1 < word_count; i++) {
        current_has_digit = strpbrk(words[i], "0123456789") != NULL;
        next_has_digit = strpbrk(words[i + 1], "0123456789") != NULL;
        if (!current_has_digit && !next_has_digit && rng_range(100) < 9) {
            temporary = words[i];
            words[i] = words[i + 1];
            words[i + 1] = temporary;
            i++;
        }
    }
    output_length = 0;
    join = 0;
    for (i = 0; i < word_count; i++) {
        word = words[i];
        current_has_digit = strpbrk(word, "0123456789") != NULL;
        if (!current_has_digit && word_count > 8 &&
            rng_range(100) < 6)
            continue;
        if (current_has_digit) join = 0;
        if (!current_has_digit && rng_range(100) < 13 &&
            mutate_token(word, changed, sizeof(changed)) > 0)
            word = changed;
        word_length = (int)strlen(word);
        if (output_length > 0 && !join &&
            output_length < output_size - 1)
            output[output_length++] = ' ';
        if (word_length > output_size - output_length - 1)
            word_length = output_size - output_length - 1;
        if (word_length <= 0) break;
        memcpy(output + output_length, word, (size_t)word_length);
        output_length += word_length;
        output[output_length] = '\0';
        join = !current_has_digit && rng_range(100) < 11;
    }
    if (output_length == 0) {
        strncpy(output, source, (size_t)output_size - 1);
        output[output_size - 1] = '\0';
    }
}

static int split_source_words(char *source, char **words,
                              int maximum_words) {
    char *word;
    int count;

    count = 0;
    word = strtok(source,
                  " \t\n\r.,!?;:\"'()[]{}<>/\\|=+*&%$#@`~");
    while (word && count < maximum_words) {
        if (word[0] && word[1]) words[count++] = word;
        word = strtok(NULL,
                      " \t\n\r.,!?;:\"'()[]{}<>/\\|=+*&%$#@`~");
    }
    return count;
}

static int select_source_word(const char *const *source_words,
                              int source_word_count, int target_token,
                              char *output, int output_size) {
    int candidate_token;
    int best_index;
    int candidate_index;
    int candidate_pick;
    int candidate_count;
    int d;
    float score;
    float best_score;

    best_index = rng_range(source_word_count);
    best_score = -1000000.0f;
    candidate_count = source_word_count < 24 ? source_word_count : 24;
    for (candidate_index = 0; candidate_index < candidate_count;
         candidate_index++) {
        candidate_pick = rng_range(source_word_count);
        candidate_token = 0;
        v3_tokenize(source_words[candidate_pick], &candidate_token, 1);
        candidate_token %= V3_VOCAB_SIZE;
        if (candidate_token < 0) candidate_token += V3_VOCAB_SIZE;
        score = 0.0f;
        for (d = 0; d < EMBED_DIM; d++)
            score += embeddings[target_token][d] *
                     embeddings[candidate_token][d];
        if (score > best_score) {
            best_score = score;
            best_index = candidate_pick;
        }
    }
    return sanitize_token(source_words[best_index], output, output_size);
}

static void generate_text(EngineState *state, const float *feature_context,
                          int *context, int *context_count, char *output,
                          int output_size, int maximum_words,
                          int minimum_words, float stop_scale,
                          const int *bias_tokens, int bias_count,
                          const char *prompt, int prompt_chance,
                          const char *source_text, int source_chance,
                          const EngineCallbacks *cb,
                          const char *stream_prefix, int stream_color) {
    char word[V3_WORD_LEN];
    float previous[EMBED_DIM];
    int output_len;
    int sentence_words;
    int sentence_target;
    int previous_token;
    int attempt;
    int i;
    int carry_index;
    char prompt_buffer[512];
    char *prompt_words[64];
    char *prompt_word;
    int prompt_word_count;
    int streamed_len;
    int streamed_words;
    char *source_buffer;
    char *source_words[V3_SOURCE_WORDS];
    int source_word_count;
    int source_token;

    output[0] = '\0';
    output_len = 0;
    sentence_words = 0;
    sentence_target = 4 + rng_range(11);
    previous_token = *context_count > 0 ? context[*context_count - 1] : rng_range(V3_VOCAB_SIZE);
    previous_token %= V3_VOCAB_SIZE;
    if (previous_token < 0) previous_token += V3_VOCAB_SIZE;
    memcpy(previous, embeddings[previous_token], sizeof(previous));
    carry_index = -1;
    prompt_word_count = 0;
    streamed_len = 0;
    streamed_words = 0;
    prompt_word = NULL;
    source_buffer = NULL;
    source_word_count = 0;
    if (prompt && prompt[0]) {
        strncpy(prompt_buffer, prompt, sizeof(prompt_buffer) - 1);
        prompt_buffer[sizeof(prompt_buffer) - 1] = '\0';
        prompt_word = strtok(prompt_buffer, " \t\n\r.,!?;:\"'()[]{}");
        while (prompt_word && prompt_word_count < 64) {
            prompt_words[prompt_word_count++] = prompt_word;
            prompt_word = strtok(NULL, " \t\n\r.,!?;:\"'()[]{}");
        }
    }
    if (prompt_word_count > 0 && rng_range(100) < prompt_chance) {
        if (prompt_word_count == 1)
            carry_index = 0;
        else
            carry_index = rng_range(maximum_words);
    }
    if (source_text && source_text[0]) {
        source_buffer = (char *)malloc(strlen(source_text) + 1);
        if (source_buffer) {
            memcpy(source_buffer, source_text, strlen(source_text) + 1);
            source_word_count = split_source_words(
                source_buffer, source_words, V3_SOURCE_WORDS);
        }
    }
    for (i = 0; i < maximum_words; i++) {
        prompt_word = i == carry_index ?
            prompt_words[rng_range(prompt_word_count)] : NULL;
        for (attempt = 0; attempt < 16; attempt++) {
            if (!prompt_word && source_word_count > 0 &&
                (i == 0 || rng_range(100) < source_chance)) {
                source_token = sample_model_token(
                    state, feature_context, context, context_count,
                    previous, bias_tokens, bias_count);
                select_source_word(
                    (const char *const *)source_words, source_word_count,
                    source_token, word, sizeof(word));
            } else {
                generate_word(state, feature_context, context,
                              context_count, previous, bias_tokens,
                              bias_count, prompt_word, word, sizeof(word));
            }
            if (prompt_word) break;
            if (!word_was_recent(state, word)) break;
            prompt_word = NULL;
        }
        remember_word(state, word);
        append_generated_word(output, output_size, &output_len, word, i, &sentence_words, sentence_target);
        if (sentence_words == 0 && i + 1 - streamed_words >= 32) {
            stream_generated_text(cb, stream_prefix, stream_color,
                                  output, output_len, &streamed_len);
            streamed_words = i + 1;
        }
        if (cb && (i & 15) == 0 && cb->generation_should_stop()) break;
        if ((sentence_words == 0 || (i + 1) % 8 == 0) &&
            model_requests_stop(state, context, *context_count, i + 1,
                                minimum_words, sentence_words == 0,
                                stop_scale))
            break;
        if (sentence_words == 0) sentence_target = 4 + rng_range(11);
        if (output_len + V3_WORD_LEN + 3 >= output_size) break;
    }
    if (output_len > 0 && output[output_len - 1] != '.' && output[output_len - 1] != '?') {
        output[output_len++] = '.';
        output[output_len] = '\0';
    }
    stream_generated_text(cb, stream_prefix, stream_color,
                          output, output_len, &streamed_len);
    free(source_buffer);
}

static void generate_thought_text(EngineState *state,
                                  const float *feature_context,
                                  int *context, int *context_count,
                                  const int *bias_tokens, int bias_count,
                                  const char *input,
                                  const V3Retrieval *retrieval,
                                  const EngineCallbacks *cb,
                                  char *output, int output_size,
                                  int maximum_words, int minimum_words,
                                  float stop_scale) {
    char prompt[1024];
    const char *source_text;
    int source_chance;
    int written;

    written = snprintf(prompt, sizeof(prompt), "%s", input);
    if (written < 0) written = 0;
    if (written >= (int)sizeof(prompt)) written = sizeof(prompt) - 1;
    if (retrieval->thinking && written < (int)sizeof(prompt) - 2) {
        snprintf(prompt + written, sizeof(prompt) - written, " %s",
                 retrieval->thinking);
        written = (int)strlen(prompt);
    }
    if (retrieval->text && written < (int)sizeof(prompt) - 2)
        snprintf(prompt + written, sizeof(prompt) - written, " %s",
                 retrieval->text);
    source_text = retrieval->thinking ? retrieval->thinking :
                  retrieval->text ? retrieval->text : input;
    source_chance = 20 + state->effort_mode * 10;
    if (source_chance > 60) source_chance = 60;
    generate_text(state, feature_context, context, context_count,
                  output, output_size, maximum_words, minimum_words,
                  stop_scale,
                  bias_tokens, bias_count,
                  prompt, 100, source_text, source_chance,
                  cb, NULL, 0);
}

static int effort_minimum_words(int effort_mode, int input_count,
                                const Features *features,
                                const V3Retrieval *retrieval) {
    static const int base_words[] = {0, 2, 4, 8, 16};
    static const int complexity_weights[] = {0, 2, 4, 8, 16};
    int complexity;

    if (effort_mode < 0 || effort_mode > 4) effort_mode = 1;
    if (effort_mode == 0) return 0;
    complexity = 1 + input_count / 8;
    if (features->is_question > 0.5f) complexity += 2;
    if (retrieval->is_code) complexity += 2;
    if (retrieval->matches == 0) complexity += 2;
    if (complexity > 16) complexity = 16;
    return base_words[effort_mode] +
           complexity * complexity_weights[effort_mode];
}

static float effort_stop_scale(int effort_mode) {
    static const float stop_scales[] = {1.0f, 1.4f, 0.8f, 0.4f, 0.12f};

    if (effort_mode < 0 || effort_mode > 4) return stop_scales[2];
    return stop_scales[effort_mode];
}

static int ascii_identifier_from_token(const char *source, char *output,
                                       int output_size) {
    int source_pos;
    int output_pos;
    unsigned char c;

    source_pos = 0;
    output_pos = 0;
    while (source[source_pos] && output_pos < output_size - 1) {
        c = (unsigned char)source[source_pos++];
        if (c < 128 && (isalnum(c) || c == '_'))
            output[output_pos++] = (char)c;
    }
    if (output_pos > 0 && isdigit((unsigned char)output[0]) &&
        output_pos < output_size - 1) {
        memmove(output + 1, output, output_pos);
        output[0] = 'x';
        output_pos++;
    }
    output[output_pos] = '\0';
    return output_pos;
}

static int mutate_code_identifier(EngineState *state,
                                  const float *feature_context,
                                  int *context, int *context_count,
                                  float *previous, const int *bias_tokens,
                                  int bias_count, const char *source,
                                  int source_len, char *output,
                                  int output_size, int force_mutation) {
    char original[128];
    int token;
    int result_len;
    int position;
    int choice;
    int preserve_threshold;
    int mutation_threshold;
    static const int preserve_thresholds[] = {55, 65, 75, 85, 92};

    if (source_len >= (int)sizeof(original))
        source_len = sizeof(original) - 1;
    memcpy(original, source, source_len);
    original[source_len] = '\0';
    preserve_threshold = state->effort_mode;
    if (preserve_threshold < 0 || preserve_threshold > 4)
        preserve_threshold = 1;
    preserve_threshold = preserve_thresholds[preserve_threshold];
    mutation_threshold = preserve_threshold +
                         (100 - preserve_threshold) / 2;
    if (force_mutation)
        choice = preserve_threshold;
    else
        choice = rng_range(100);
    if (choice < preserve_threshold) {
        result_len = source_len;
        if (result_len >= output_size) result_len = output_size - 1;
        memcpy(output, original, result_len);
        output[result_len] = '\0';
        return result_len;
    }

    if (choice < mutation_threshold) {
        result_len = source_len;
        if (result_len >= output_size) result_len = output_size - 1;
        memcpy(output, original, result_len);
        position = rng_range(result_len);
        if (isalpha((unsigned char)output[position]))
            output[position] = (char)('a' + rng_range(26));
        else
            output[position] = '_';
        output[result_len] = '\0';
        return result_len;
    }

    token = sample_model_token(state, feature_context, context,
                               context_count, previous,
                               bias_tokens, bias_count);
    result_len = ascii_identifier_from_token(v3_vocab[token], output,
                                             output_size);
    if (result_len == 0) {
        result_len = source_len;
        if (result_len >= output_size) result_len = output_size - 1;
        memcpy(output, original, result_len);
        output[result_len] = '\0';
    }
    return result_len;
}

static void generate_code_text(EngineState *state,
                               const float *feature_context,
                               int *context, int *context_count,
                               const int *bias_tokens, int bias_count,
                               const char *seed, char *output,
                               int output_size,
                               const EngineCallbacks *cb,
                               const char *stream_prefix,
                               int stream_color) {
    int seed_tokens[MAX_TOKENS];
    float previous[EMBED_DIM];
    char identifier[128];
    int seed_token_count;
    int seed_len;
    int source_pos;
    int output_pos;
    int identifier_end;
    int identifier_len;
    int previous_token;
    int punctuation_roll;
    int punctuation_damage;
    int streamed_len;
    int identifier_count;
    int identifier_index;
    int forced_identifier;
    static const int punctuation_damage_levels[] = {7, 5, 3, 2, 1};
    unsigned char c;

    seed_token_count = v3_tokenize(seed, seed_tokens, MAX_TOKENS);
    append_context_tokens(context, context_count, seed_tokens,
                          seed_token_count);
    previous_token = *context_count > 0 ?
                     context[*context_count - 1] : rng_range(V3_VOCAB_SIZE);
    previous_token %= V3_VOCAB_SIZE;
    memcpy(previous, embeddings[previous_token], sizeof(previous));

    seed_len = (int)strlen(seed);
    identifier_count = 0;
    source_pos = 0;
    while (source_pos < seed_len) {
        c = (unsigned char)seed[source_pos];
        if (c < 128 && (isalpha(c) || c == '_')) {
            identifier_count++;
            source_pos++;
            while (source_pos < seed_len) {
                c = (unsigned char)seed[source_pos];
                if (c >= 128 || (!isalnum(c) && c != '_')) break;
                source_pos++;
            }
        } else {
            source_pos++;
        }
    }
    if (identifier_count > 0)
        forced_identifier = rng_range(identifier_count);
    else
        forced_identifier = -1;
    punctuation_damage = state->effort_mode;
    if (punctuation_damage < 0 || punctuation_damage > 4)
        punctuation_damage = 1;
    punctuation_damage = punctuation_damage_levels[punctuation_damage];
    source_pos = 0;
    output_pos = 0;
    streamed_len = 0;
    identifier_index = 0;
    while (source_pos < seed_len && output_pos < output_size - 2) {
        c = (unsigned char)seed[source_pos];
        if ((c < 128 && (isalpha(c) || c == '_'))) {
            identifier_end = source_pos + 1;
            while (identifier_end < seed_len) {
                c = (unsigned char)seed[identifier_end];
                if (c >= 128 || (!isalnum(c) && c != '_')) break;
                identifier_end++;
            }
            identifier_len = mutate_code_identifier(
                state, feature_context, context, context_count, previous,
                bias_tokens, bias_count, seed + source_pos,
                identifier_end - source_pos, identifier,
                sizeof(identifier), identifier_index == forced_identifier);
            identifier_index++;
            if (output_pos + identifier_len >= output_size - 1)
                identifier_len = output_size - output_pos - 1;
            memcpy(output + output_pos, identifier, identifier_len);
            output_pos += identifier_len;
            source_pos = identifier_end;
            continue;
        }
        if (c >= 0x20 && c < 0x7F) {
            punctuation_roll = rng_range(100);
            if (isalnum(c) || isspace(c) ||
                punctuation_roll >= punctuation_damage)
                output[output_pos++] = (char)c;
            if (punctuation_damage > 0 && strchr(";,()[]{}", c) &&
                punctuation_roll >= 100 - punctuation_damage &&
                output_pos < output_size - 1)
                output[output_pos++] = (char)c;
        } else if (c == '\n' || c == '\t') {
            output[output_pos++] = (char)c;
        }
        source_pos++;
        if (c == '\n') {
            if (cb && cb->generation_should_stop()) break;
        }
    }
    output[output_pos] = '\0';
    stream_generated_text(cb, stream_prefix, stream_color,
                          output, output_pos, &streamed_len);
}

static void save_self_context(EngineState *state, const int *context, int context_count) {
    int copy_count;
    int start;

    copy_count = context_count < SELF_CTX_MAX ? context_count : SELF_CTX_MAX;
    start = context_count - copy_count;
    if (copy_count > 0) memcpy(state->self_ctx_tokens, context + start, copy_count * sizeof(int));
    state->self_ctx_len = copy_count;
}

static int select_useful_thought_tokens(const int *thought_tokens,
                                        int thought_count,
                                        const float *context_reference,
                                        const float *feature_context,
                                        int *useful_tokens) {
    float *scores;
    float score;
    float mean;
    float reference;
    float best_score;
    int best_index;
    int useful_count;
    int token;
    int i;
    int d;

    if (thought_count <= 0) return 0;
    scores = (float *)malloc((size_t)thought_count * sizeof(float));
    if (!scores) {
        useful_tokens[0] = thought_tokens[thought_count - 1];
        return 1;
    }
    mean = 0.0f;
    best_score = -1000000.0f;
    best_index = 0;
    for (i = 0; i < thought_count; i++) {
        token = thought_tokens[i] % V3_VOCAB_SIZE;
        if (token < 0) token += V3_VOCAB_SIZE;
        score = 0.0f;
        for (d = 0; d < EMBED_DIM; d++) {
            reference = context_reference[d] * 0.75f +
                        feature_context[d] * 0.25f;
            score += embeddings[token][d] * reference;
        }
        scores[i] = score / (float)EMBED_DIM;
        mean += scores[i];
        if (scores[i] > best_score) {
            best_score = scores[i];
            best_index = i;
        }
    }
    mean /= (float)thought_count;
    useful_count = 0;
    for (i = 0; i < thought_count; i++)
        if (scores[i] >= mean)
            useful_tokens[useful_count++] = thought_tokens[i];
    if (useful_count == 0)
        useful_tokens[useful_count++] = thought_tokens[best_index];
    free(scores);
    return useful_count;
}

static void prioritize_thought_bias(int *bias_tokens, int *bias_count,
                                    const int *thought_tokens,
                                    int thought_count) {
    int retained_bias;
    int thought_start;

    thought_start = 0;
    if (thought_count >= V3_BIAS_TOKENS) {
        thought_start = thought_count - V3_BIAS_TOKENS;
        thought_count = V3_BIAS_TOKENS;
        *bias_count = 0;
    } else if (*bias_count + thought_count > V3_BIAS_TOKENS) {
        retained_bias = V3_BIAS_TOKENS - thought_count;
        memmove(bias_tokens, bias_tokens + *bias_count - retained_bias,
                retained_bias * sizeof(int));
        *bias_count = retained_bias;
    }
    memcpy(bias_tokens + *bias_count, thought_tokens + thought_start,
           thought_count * sizeof(int));
    *bias_count += thought_count;
}

static void TUFFAI_GENERATE_RESPONSE(EngineState *state, const EngineCallbacks *cb, const char *input) {
    Features features;
    float feature_context[EMBED_DIM];
    float thought_reference[EMBED_DIM];
    int context[V3_CONTEXT_TOKENS];
    int input_tokens[MAX_TOKENS];
    int tool_tokens[V3_CONTEXT_TOKENS];
    int *thought_tokens;
    int *useful_thought_tokens;
    int bias_tokens[V3_BIAS_TOKENS];
    int context_count;
    int input_count;
    int thought_count;
    int useful_thought_count;
    int bias_count;
    int prior_history_count;
    int prior_slot;
    int response_words;
    int minimum_response_words;
    int requested_limit;
    int minimum_percent;
    int maximum_percent;
    int response_prompt_chance;
    int identity_response;
    int mixed_response;
    int response_size;
    int search_result_count;
    int tool_token_count;
    int wants_search;
    int dataset_tool_search;
    int thought_minimum_words;
    int thought_maximum_words;
    int has_arithmetic_response;
    float thought_stop_scale;
    size_t response_bytes;
    size_t thought_bytes;
    char *thought;
    char search_text[V3_SEARCH_TEXT_SIZE];
    char *response;
    const char *response_source;
    int response_source_chance;
    char carry_prompt[HIST_LEN];
    char code_seed[8192];
    V3Retrieval retrieval;
    const char *code_source;
    const char *search_query;
    char pre_tool_response[8192];
    char arithmetic_response[256];

    thought = NULL;
    thought_tokens = NULL;
    useful_thought_tokens = NULL;
    cb->curs_set_fn(0);
    state->turn_count++;
    features = extract(input);
    feat_to_embed(&features, feature_context);
    prior_history_count = *state->hist_cnt;
    v3_retrieve_dataset_example(input, &retrieval);
    has_arithmetic_response = detect_pattern(input) == PAT_MATH &&
                              evaluate_basic_arithmetic(
                                  input, arithmetic_response,
                                  sizeof(arithmetic_response));
    search_text[0] = '\0';
    web_search_take_used();
    dataset_tool_search = retrieval.use_tool &&
                          strcmp(retrieval.use_tool, "web_search") == 0;
    search_query = dataset_tool_search && retrieval.tool_input &&
                   retrieval.tool_input[0] ? retrieval.tool_input : input;
    wants_search = model_wants_search(input, &features, &retrieval);
    if (dataset_tool_search && web_search_enabled()) wants_search = 1;
    if (web_search_requested(input) && !web_search_enabled())
        cb->chat_add("Web search is disabled. Use /search to enable it.");
    build_context(state, input, &retrieval, search_text,
                  context, &context_count,
                  input_tokens, &input_count);
    encode_generation_context(context, context_count, thought_reference);
    build_memory_bias(state, input_tokens, input_count,
                      bias_tokens, &bias_count);
    strncpy(carry_prompt, input, sizeof(carry_prompt) - 1);
    carry_prompt[sizeof(carry_prompt) - 1] = '\0';
    if (prior_history_count > 0 && rng_range(100) < 35) {
        prior_slot = (prior_history_count - 1) % HIST_MAX;
        strncpy(carry_prompt, state->hist_buf[prior_slot],
                sizeof(carry_prompt) - 1);
        carry_prompt[sizeof(carry_prompt) - 1] = '\0';
    }
    identity_response = identity_request(input);
    response_prompt_chance = 35;
    if (identity_response) {
        strcpy(carry_prompt, "TuffAI-v3");
        response_prompt_chance = 100;
    }
    push_history(state, input, input_tokens, input_count);
    reset_recent();
    thought_minimum_words = effort_minimum_words(
        state->effort_mode, input_count, &features, &retrieval);
    thought_stop_scale = effort_stop_scale(state->effort_mode);
    thought_maximum_words = V3_CONTEXT_TOKENS;
    if (thought_minimum_words > 0) {
        thought_bytes = (size_t)thought_maximum_words *
                        (V3_WORD_LEN + 3) + 2;
        thought = (char *)malloc(thought_bytes);
        thought_tokens = (int *)malloc(V3_CONTEXT_TOKENS * sizeof(int));
        useful_thought_tokens =
            (int *)malloc(V3_CONTEXT_TOKENS * sizeof(int));
        if (thought && thought_tokens && useful_thought_tokens) {
            cb->show_status("Thinking... ESC or Ctrl-C to stop.");
            generate_thought_text(
                state, feature_context, context, &context_count,
                bias_tokens, bias_count, input, &retrieval, cb,
                thought, (int)thought_bytes, thought_maximum_words,
                thought_minimum_words, thought_stop_scale);
            thought_count = v3_tokenize(thought, thought_tokens,
                                        V3_CONTEXT_TOKENS);
            useful_thought_count = select_useful_thought_tokens(
                thought_tokens, thought_count, thought_reference,
                feature_context, useful_thought_tokens);
            append_context_tokens(context, &context_count,
                                  useful_thought_tokens,
                                  useful_thought_count);
            prioritize_thought_bias(bias_tokens, &bias_count,
                                    useful_thought_tokens,
                                    useful_thought_count);
            cb->chat_add_c("Thoughts", state->color_think);
            cb->chat_add_wrapped("    ", thought, state->color_think);
        } else {
            cb->chat_add("Unable to allocate reasoning workspace.");
        }
        free(thought);
        free(thought_tokens);
        free(useful_thought_tokens);
        thought = NULL;
        thought_tokens = NULL;
        useful_thought_tokens = NULL;
        if (cb->generation_should_stop()) {
            save_self_context(state, context, context_count);
            state->total_tokens += input_count;
            cb->curs_set_fn(1);
            return;
        }
    }
    if (wants_search) {
        if (dataset_tool_search && retrieval.text && retrieval.text[0]) {
            generate_mixed_answer(retrieval.text, pre_tool_response,
                                  sizeof(pre_tool_response));
            cb->stream_text("TuffAI: ", pre_tool_response, state->color_ai);
        }
        cb->use_tool("web_search", search_query);
        cb->show_status("web_search is running...");
        search_result_count = web_research(
            search_query, search_text, sizeof(search_text), 2);
        web_search_take_used();
        if (search_result_count > 0) {
            tool_token_count = v3_tokenize(
                search_text, tool_tokens, V3_CONTEXT_TOKENS);
            append_context_tokens(context, &context_count,
                                  tool_tokens, tool_token_count);
        } else if (search_result_count == 0) {
            cb->chat_add("web_search returned no usable pages.");
        } else {
            cb->chat_add("web_search failed.");
        }
        if (cb->generation_should_stop()) {
            save_self_context(state, context, context_count);
            state->total_tokens += input_count;
            cb->curs_set_fn(1);
            return;
        }
    }
    response_words = TUFFAI_ENGINE_VTABLE.max_output_tokens;
    if (state->cfg_max_words > 0 && state->cfg_max_words < response_words)
        response_words = state->cfg_max_words;
    minimum_response_words = 1;
    if (retrieval.requested_words > 0 && !retrieval.is_code) {
        requested_limit = retrieval.requested_words;
        if (requested_limit > response_words) requested_limit = response_words;
        if (requested_limit <= 2) {
            minimum_response_words = requested_limit;
            response_words = requested_limit;
        } else {
            minimum_percent = 45 + rng_range(31);
            maximum_percent = 75 + rng_range(16);
            minimum_response_words =
                requested_limit * minimum_percent / 100;
            response_words = requested_limit * maximum_percent / 100;
            if (minimum_response_words < 1) minimum_response_words = 1;
            if (response_words <= minimum_response_words)
                response_words = minimum_response_words + 1;
        }
    }
    if (retrieval.is_code) {
        response_bytes = V3_CODE_RESPONSE_SIZE;
    } else if ((size_t)response_words >
               ((size_t)INT_MAX - 2) / (V3_WORD_LEN + 3)) {
        response_bytes = INT_MAX;
    } else {
        response_bytes = (size_t)response_words * (V3_WORD_LEN + 3) + 2;
    }
    response = (char *)malloc(response_bytes);
    if (!response) {
        cb->show_status("Unable to allocate the requested response size.");
        cb->curs_set_fn(1);
        return;
    }
    response_size = (int)response_bytes;
    cb->show_status("Generating... ESC or Ctrl-C to stop.");
    mixed_response = retrieval.text && !retrieval.is_code &&
                     retrieval.requested_words == 0 && !search_text[0] &&
                     (detect_pattern(input) == PAT_MATH ||
                      rng_range(100) < 45 + state->effort_mode * 5);
    if (has_arithmetic_response) {
        strncpy(response, arithmetic_response, (size_t)response_size - 1);
        response[response_size - 1] = '\0';
        cb->stream_text("TuffAI: ", response, state->color_ai);
        input_count = v3_tokenize(response, input_tokens, MAX_TOKENS);
        append_context_tokens(context, &context_count,
                              input_tokens, input_count);
    } else if (retrieval.is_code) {
        code_source = retrieval.text;
        if (rng_range(100) < 55 &&
            v3_compose_code(input, NULL, code_seed,
                            sizeof(code_seed)) > 0)
            code_source = code_seed;
        generate_code_text(state, feature_context, context, &context_count,
                           bias_tokens, bias_count, code_source,
                           response, response_size, cb, "TuffAI: ",
                           state->color_ai);
    } else if (mixed_response) {
        generate_mixed_answer(retrieval.text, response, response_size);
        cb->stream_text("TuffAI: ", response, state->color_ai);
        input_count = v3_tokenize(response, input_tokens, MAX_TOKENS);
        append_context_tokens(context, &context_count,
                              input_tokens, input_count);
    } else {
        response_source = retrieval.after_tool && wants_search ?
                          retrieval.after_tool : search_text;
        response_source_chance = retrieval.after_tool && wants_search ?
                                 65 : search_text[0] ? 55 : 0;
        if (!search_text[0] && retrieval.text &&
            TUFFAI_RETRIEVAL_SOURCE_CHANCE > 0) {
            response_source = retrieval.text;
            response_source_chance = TUFFAI_RETRIEVAL_SOURCE_CHANCE;
        }
        generate_text(state, feature_context, context, &context_count,
                      response, response_size, response_words,
                      minimum_response_words, 1.0f,
                      bias_tokens, bias_count, carry_prompt,
                      response_prompt_chance,
                      response_source, response_source_chance,
                      cb, "TuffAI: ", state->color_ai);
    }
    cb->show_status("");
    free(response);
    save_self_context(state, context, context_count);
    state->total_tokens += input_count;
    cb->curs_set_fn(1);
}

const EngineVtable TUFFAI_ENGINE_VTABLE = {
    TUFFAI_GENERATE_RESPONSE,
    V3_MAX_RESPONSE_WORDS,
    V3_CONTEXT_TOKENS,
    V3_VOCAB_SIZE,
    1,
    v3_effort_modes,
    5,
    1,
    1
};
