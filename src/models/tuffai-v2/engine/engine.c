#define _XOPEN_SOURCE 700
#include "../../../engine.h"
#include "../../../net.h"
#include "../../../features.h"
#include "../../../tokenizer.h"
#include "../../../knowledge/knowledge.h"
#include "../../../rng.h"
#include "../../../websearch.h"
#include "../retrieval.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <wchar.h>
#include <wctype.h>

#define V2_CONTEXT_TOKENS 4096
#define V2_RECENT_WORDS RENDERED_RECENT_MAX
#define V2_WORD_LEN RENDERED_WORD_LEN
#define V2_BIAS_TOKENS 512
#define V2_MAX_RESPONSE_WORDS 1000
#define V2_CODE_RESPONSE_SIZE 65536
#define V2_ACTIVE_ATTENTION_TOKENS 256
#define V2_SEARCH_TEXT_SIZE 65536
#define V2_SOURCE_WORDS 2048

static void append_context_token(int *tokens, int *count, int token) {
    if (*count >= V2_CONTEXT_TOKENS) {
        memmove(tokens, tokens + 1, (V2_CONTEXT_TOKENS - 1) * sizeof(int));
        *count = V2_CONTEXT_TOKENS - 1;
    }
    tokens[(*count)++] = token;
}

static void append_context_tokens(int *tokens, int *count, const int *source, int source_count) {
    int i;

    for (i = 0; i < source_count; i++) append_context_token(tokens, count, source[i]);
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
                          const V2Retrieval *retrieval,
                          const char *search_text, int *context,
                          int *context_count, int *input_tokens,
                          int *input_count) {
    int temp_tokens[MAX_TOKENS];
    int search_tokens[V2_CONTEXT_TOKENS];
    int temp_count;
    int available_history;
    int history_count;
    int history_slot;
    int i;

    *context_count = 0;
    *input_count = v2_tokenize(input, input_tokens, MAX_TOKENS);
    if (state->system_prompt[0]) {
        temp_count = v2_tokenize(state->system_prompt, temp_tokens, MAX_TOKENS);
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
        temp_count = v2_tokenize(retrieval->text, temp_tokens, MAX_TOKENS);
        if (temp_count > 48) temp_count = 48;
        append_context_tokens(context, context_count, temp_tokens, temp_count);
    }
    if (search_text && search_text[0]) {
        temp_count = v2_tokenize(search_text, search_tokens,
                                 V2_CONTEXT_TOKENS);
        append_context_tokens(context, context_count, search_tokens,
                              temp_count);
    }
    append_context_tokens(context, context_count, input_tokens, *input_count);
}

static int model_wants_search(const char *input, const Features *features,
                              const V2Retrieval *retrieval) {
    int probability;

    if (!web_search_enabled()) return 0;
    if (web_search_requested(input)) return 1;
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
        if (*bias_count + copy_count > V2_BIAS_TOKENS)
            copy_count = V2_BIAS_TOKENS - *bias_count;
        if (copy_count > 0) {
            memcpy(bias_tokens + *bias_count,
                   state->hist_tokens[history_slot],
                   copy_count * sizeof(int));
            *bias_count += copy_count;
        }
    }
    copy_count = input_count;
    if (*bias_count + copy_count > V2_BIAS_TOKENS) {
        copy_count = V2_BIAS_TOKENS - *bias_count;
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

static void encode_generation_context(const int *context, int context_count,
                                      float *encoded) {
    int active[V2_ACTIVE_ATTENTION_TOKENS];
    int older_count;
    int recent_count;
    int sampled_count;
    int source_index;
    int i;

    if (context_count <= V2_ACTIVE_ATTENTION_TOKENS) {
        encode_context(context, context_count, encoded);
        return;
    }
    recent_count = V2_ACTIVE_ATTENTION_TOKENS / 2;
    older_count = context_count - recent_count;
    sampled_count = V2_ACTIVE_ATTENTION_TOKENS - recent_count;
    for (i = 0; i < sampled_count; i++) {
        source_index = i * older_count / sampled_count;
        active[i] = context[source_index];
    }
    memcpy(active + sampled_count, context + context_count - recent_count,
           (size_t)recent_count * sizeof(int));
    encode_context(active, V2_ACTIVE_ATTENTION_TOKENS, encoded);
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
    next_embed(encoded, previous, feature_context, target);
    noise = state->cfg_noise;
    if (noise < 0.25f) noise = 0.25f;
    for (d = 0; d < EMBED_DIM; d++) target[d] += frand_r() * noise * 0.35f;
    temperature = state->cfg_temp;
    if (temperature < 0.2f) temperature = 0.2f;
    token = sample_vocab_contextual(target, temperature, noise,
                                    state->cfg_freq_penalty,
                                    state->cfg_rep_penalty,
                                    state->cfg_top_p,
                                    engine_tuffai_v2.vocab_size,
                                    bias_tokens, bias_count, 0.12f);
    push_recent(token);
    append_context_token(context, context_count, token);
    memcpy(previous, embeddings[token], EMBED_DIM * sizeof(float));
    state->total_tokens++;
    state->sampled_tokens++;
    return token;
}

static int model_requests_stop(EngineState *state, const int *context,
                               int context_count, int generated_units,
                               int minimum_units, int boundary) {
    float encoded[EMBED_DIM];
    float alignment;
    float temperature;
    float probability;
    int token;
    int d;

    if (generated_units < minimum_units || context_count <= 0) return 0;
    encode_generation_context(context, context_count, encoded);
    token = context[context_count - 1] % V2_VOCAB_SIZE;
    if (token < 0) token += V2_VOCAB_SIZE;
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
    return rng_unit() < probability;
}

static int create_fragment(const char *source, char *fragment, int fragment_size) {
    char clean[V2_WORD_LEN];
    int boundaries[V2_WORD_LEN];
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
    while (position < clean_len && codepoints < V2_WORD_LEN - 1) {
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

static int decode_codepoints(const char *text, unsigned int *codepoints,
                             int maximum) {
    int length;
    int position;
    int count;
    int sequence_length;

    length = (int)strlen(text);
    position = 0;
    count = 0;
    while (position < length && count < maximum) {
        sequence_length = decode_utf8(text + position, length - position,
                                      &codepoints[count]);
        if (sequence_length <= 0) {
            position++;
            continue;
        }
        position += sequence_length;
        count++;
    }
    return count;
}

static int trigram_overlap(const char *left, const char *right) {
    unsigned int left_codepoints[V2_WORD_LEN];
    unsigned int right_codepoints[V2_WORD_LEN];
    int left_len;
    int right_len;
    int left_pos;
    int right_pos;
    int matches;
    int total;

    left_len = decode_codepoints(left, left_codepoints, V2_WORD_LEN);
    right_len = decode_codepoints(right, right_codepoints, V2_WORD_LEN);
    if (strcmp(left, right) == 0) return 100;
    if (left_len < 3 || right_len < 3) return 0;
    matches = 0;
    total = left_len - 2;
    for (left_pos = 0; left_pos + 2 < left_len; left_pos++) {
        for (right_pos = 0; right_pos + 2 < right_len; right_pos++) {
            if (left_codepoints[left_pos] == right_codepoints[right_pos] &&
                left_codepoints[left_pos + 1] ==
                    right_codepoints[right_pos + 1] &&
                left_codepoints[left_pos + 2] ==
                    right_codepoints[right_pos + 2]) {
                matches++;
                break;
            }
        }
    }
    return matches * 100 / total;
}

static int word_was_recent(EngineState *state, const char *word) {
    int i;

    for (i = 0; i < state->recent_rendered_count; i++)
        if (trigram_overlap(state->recent_rendered[i], word) > 35) return 1;
    return 0;
}

static void remember_word(EngineState *state, const char *word) {
    int i;
    int word_len;

    word_len = (int)strlen(word);
    if (word_len >= V2_WORD_LEN) word_len = V2_WORD_LEN - 1;
    if (state->recent_rendered_count < V2_RECENT_WORDS) {
        memcpy(state->recent_rendered[state->recent_rendered_count], word,
               word_len);
        state->recent_rendered[state->recent_rendered_count][word_len] = '\0';
        state->recent_rendered_count++;
        return;
    }
    for (i = 1; i < V2_RECENT_WORDS; i++)
        memcpy(state->recent_rendered[i - 1], state->recent_rendered[i],
               V2_WORD_LEN);
    memcpy(state->recent_rendered[V2_RECENT_WORDS - 1], word, word_len);
    state->recent_rendered[V2_RECENT_WORDS - 1][word_len] = '\0';
}

static int mutate_token(const char *source, char *word, int word_size) {
    char clean[V2_WORD_LEN];
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
    if (forced_word) word_len = mutate_token(forced_word, word, word_size);
    if (word_len > 0) return word_len;
    if (mode < 10) word_len = sanitize_token(v2_vocab[token], word, word_size);
    else if (mode < 20) word_len = mutate_token(v2_vocab[token], word, word_size);
    else if (mode < 30) word_len = random_unicode_word(state, word,
                                                       word_size);
    if (word_len > 0) return word_len;

    piece_count = 1 + rng_range(5);
    for (piece = 0; piece < piece_count && word_len < word_size - 2; piece++) {
        fragment_len = 0;
        for (attempt = 0; attempt < 16 && fragment_len == 0; attempt++) {
            if (piece > 0 || attempt > 0)
                token = sample_model_token(state, feature_context, context,
                                           context_count, previous,
                                           bias_tokens, bias_count);
            fragment_len = create_fragment(v2_vocab[token], fragment,
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
        v2_tokenize(source_words[candidate_pick], &candidate_token, 1);
        candidate_token %= V2_VOCAB_SIZE;
        if (candidate_token < 0) candidate_token += V2_VOCAB_SIZE;
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
                          int minimum_words,
                          const int *bias_tokens, int bias_count,
                          const char *prompt, int prompt_chance,
                          const char *source_text, int source_chance,
                          const EngineCallbacks *cb,
                          const char *stream_prefix, int stream_color) {
    char word[V2_WORD_LEN];
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
    char *source_words[V2_SOURCE_WORDS];
    int source_word_count;
    int source_token;

    output[0] = '\0';
    output_len = 0;
    sentence_words = 0;
    sentence_target = 4 + rng_range(11);
    previous_token = *context_count > 0 ? context[*context_count - 1] : rng_range(V2_VOCAB_SIZE);
    previous_token %= V2_VOCAB_SIZE;
    if (previous_token < 0) previous_token += V2_VOCAB_SIZE;
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
    if (prompt_word_count > 0 && rng_range(100) < prompt_chance)
        carry_index = rng_range(maximum_words);
    if (source_text && source_text[0]) {
        source_buffer = (char *)malloc(strlen(source_text) + 1);
        if (source_buffer) {
            memcpy(source_buffer, source_text, strlen(source_text) + 1);
            source_word_count = split_source_words(
                source_buffer, source_words, V2_SOURCE_WORDS);
        }
    }
    for (i = 0; i < maximum_words; i++) {
        prompt_word = i == carry_index ?
            prompt_words[rng_range(prompt_word_count)] : NULL;
        for (attempt = 0; attempt < 16; attempt++) {
            if (source_word_count > 0 &&
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
        if (cb && cb->generation_should_stop()) break;
        if ((sentence_words == 0 || (i + 1) % 8 == 0) &&
            model_requests_stop(state, context, *context_count, i + 1,
                                minimum_words, sentence_words == 0))
            break;
        if (sentence_words == 0) sentence_target = 4 + rng_range(11);
        if (output_len + V2_WORD_LEN + 3 >= output_size) break;
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
                                  const V2Retrieval *retrieval,
                                  const EngineCallbacks *cb,
                                  char *output, int output_size) {
    char prompt[1024];
    int written;

    written = snprintf(prompt, sizeof(prompt), "%s", input);
    if (written < 0) written = 0;
    if (written >= (int)sizeof(prompt)) written = sizeof(prompt) - 1;
    if (retrieval->text && written < (int)sizeof(prompt) - 2)
        snprintf(prompt + written, sizeof(prompt) - written, " %s",
                 retrieval->text);
    generate_text(state, feature_context, context, context_count,
                  output, output_size, 96, 3, bias_tokens, bias_count,
                  prompt, 100, NULL, 0, cb, NULL, 0);
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

    if (source_len >= (int)sizeof(original))
        source_len = sizeof(original) - 1;
    memcpy(original, source, source_len);
    original[source_len] = '\0';
    choice = force_mutation ? 65 + rng_range(20) : rng_range(100);
    if (choice < 65) {
        result_len = source_len;
        if (result_len >= output_size) result_len = output_size - 1;
        memcpy(output, original, result_len);
        output[result_len] = '\0';
        return result_len;
    }

    if (choice < 85) {
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
    result_len = ascii_identifier_from_token(v2_vocab[token], output,
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
    int identifier_count;
    int identifier_index;
    int forced_identifier;
    int generated_lines;
    int streamed_len;
    unsigned char c;

    seed_token_count = v2_tokenize(seed, seed_tokens, MAX_TOKENS);
    append_context_tokens(context, context_count, seed_tokens,
                          seed_token_count);
    previous_token = *context_count > 0 ?
                     context[*context_count - 1] : rng_range(V2_VOCAB_SIZE);
    previous_token %= V2_VOCAB_SIZE;
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
    forced_identifier = identifier_count > 0 ?
                        rng_range(identifier_count) : -1;
    identifier_index = 0;
    source_pos = 0;
    output_pos = 0;
    generated_lines = 0;
    streamed_len = 0;
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
                sizeof(identifier),
                identifier_index == forced_identifier);
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
            if (isalnum(c) || isspace(c) || punctuation_roll >= 4)
                output[output_pos++] = (char)c;
            if (strchr(";,()[]{}", c) && punctuation_roll >= 98 &&
                output_pos < output_size - 1)
                output[output_pos++] = (char)c;
        } else if (c == '\n' || c == '\t') {
            output[output_pos++] = (char)c;
        }
        source_pos++;
        if (c == '\n') {
            generated_lines++;
            output[output_pos] = '\0';
            stream_generated_text(cb, stream_prefix, stream_color,
                                  output, output_pos, &streamed_len);
            if (cb && cb->generation_should_stop()) break;
            if (model_requests_stop(state, context, *context_count,
                                    generated_lines, 2, 1))
                break;
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
    float scores[V2_CONTEXT_TOKENS];
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
    mean = 0.0f;
    best_score = -1000000.0f;
    best_index = 0;
    for (i = 0; i < thought_count; i++) {
        token = thought_tokens[i] % V2_VOCAB_SIZE;
        if (token < 0) token += V2_VOCAB_SIZE;
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
    return useful_count;
}

static void prioritize_thought_bias(int *bias_tokens, int *bias_count,
                                    const int *thought_tokens,
                                    int thought_count) {
    int retained_bias;
    int thought_start;

    thought_start = 0;
    if (thought_count >= V2_BIAS_TOKENS) {
        thought_start = thought_count - V2_BIAS_TOKENS;
        thought_count = V2_BIAS_TOKENS;
        *bias_count = 0;
    } else if (*bias_count + thought_count > V2_BIAS_TOKENS) {
        retained_bias = V2_BIAS_TOKENS - thought_count;
        memmove(bias_tokens, bias_tokens + *bias_count - retained_bias,
                retained_bias * sizeof(int));
        *bias_count = retained_bias;
    }
    memcpy(bias_tokens + *bias_count, thought_tokens + thought_start,
           thought_count * sizeof(int));
    *bias_count += thought_count;
}

static void v2_generate_response(EngineState *state, const EngineCallbacks *cb, const char *input) {
    Features features;
    float feature_context[EMBED_DIM];
    float thought_reference[EMBED_DIM];
    int context[V2_CONTEXT_TOKENS];
    int input_tokens[MAX_TOKENS];
    int thought_tokens[V2_CONTEXT_TOKENS];
    int useful_thought_tokens[V2_CONTEXT_TOKENS];
    int bias_tokens[V2_BIAS_TOKENS];
    int context_count;
    int input_count;
    int thought_count;
    int useful_thought_count;
    int bias_count;
    int prior_history_count;
    int prior_slot;
    int response_words;
    int minimum_response_words;
    int response_size;
    int search_result_count;
    size_t response_bytes;
    char thought[2048];
    char search_text[V2_SEARCH_TEXT_SIZE];
    char *response;
    char carry_prompt[HIST_LEN];
    V2Retrieval retrieval;

    cb->curs_set_fn(0);
    state->turn_count++;
    features = extract(input);
    feat_to_embed(&features, feature_context);
    prior_history_count = *state->hist_cnt;
    v2_retrieve_dataset_example(input, &retrieval);
    search_text[0] = '\0';
    web_search_take_used();
    if (web_search_requested(input) && !web_search_enabled())
        cb->chat_add("Web search is disabled. Use /search to enable it.");
    if (model_wants_search(input, &features, &retrieval)) {
        cb->show_status("Searching the web...");
        search_result_count = web_research(
            input, search_text, sizeof(search_text), 1);
        web_search_take_used();
        if (search_result_count > 0)
            cb->chat_add("Searched the web");
        else if (search_result_count == 0)
            cb->chat_add("Web search returned no usable pages.");
        else
            cb->chat_add("Web search failed.");
    }
    build_context(state, input, &retrieval, search_text,
                  context, &context_count,
                  input_tokens, &input_count);
    encode_context(context, context_count, thought_reference);
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
    push_history(state, input, input_tokens, input_count);
    reset_recent();
    cb->show_status("Thinking... ESC or Ctrl-C to stop.");
    generate_thought_text(state, feature_context, context, &context_count,
                          bias_tokens, bias_count, input, &retrieval,
                          cb,
                          thought, sizeof(thought));
    thought_count = v2_tokenize(thought, thought_tokens,
                                V2_CONTEXT_TOKENS);
    useful_thought_count = select_useful_thought_tokens(
        thought_tokens, thought_count, thought_reference, feature_context,
        useful_thought_tokens);
    append_context_tokens(context, &context_count,
                          useful_thought_tokens,
                          useful_thought_count);
    prioritize_thought_bias(bias_tokens, &bias_count,
                            useful_thought_tokens,
                            useful_thought_count);
    cb->chat_add_c("Thoughts", state->color_think);
    cb->chat_add_wrapped("    ", thought, state->color_think);
    cb->draw_chat();
    cb->refresh_screen();
    if (cb->generation_should_stop()) {
        save_self_context(state, context, context_count);
        state->total_tokens += input_count;
        cb->curs_set_fn(1);
        return;
    }
    response_words = retrieval.requested_words > 0 ?
                     retrieval.requested_words :
                     engine_tuffai_v2.max_output_tokens;
    if (state->cfg_max_words > 0 &&
        (retrieval.requested_words == 0 ||
         state->cfg_max_words < response_words))
        response_words = state->cfg_max_words;
    minimum_response_words = retrieval.requested_words > 0 ?
                             response_words : 1;
    if (retrieval.is_code) {
        response_bytes = V2_CODE_RESPONSE_SIZE;
    } else if ((size_t)response_words >
               ((size_t)INT_MAX - 2) / (V2_WORD_LEN + 3)) {
        response_bytes = INT_MAX;
    } else {
        response_bytes = (size_t)response_words * (V2_WORD_LEN + 3) + 2;
    }
    response = (char *)malloc(response_bytes);
    if (!response) {
        cb->show_status("Unable to allocate the requested response size.");
        cb->curs_set_fn(1);
        return;
    }
    response_size = (int)response_bytes;
    cb->show_status("Generating... ESC or Ctrl-C to stop.");
    if (retrieval.is_code) {
        generate_code_text(state, feature_context, context, &context_count,
                           bias_tokens, bias_count, retrieval.text,
                           response, response_size, cb, "TuffAI: ",
                           state->color_ai);
    } else {
        generate_text(state, feature_context, context, &context_count,
                      response, response_size, response_words,
                      minimum_response_words,
                      bias_tokens, bias_count, carry_prompt, 35,
                      search_text, search_text[0] ? 55 : 0,
                      cb, "TuffAI: ", state->color_ai);
    }
    cb->show_status("");
    free(response);
    save_self_context(state, context, context_count);
    state->total_tokens += input_count;
    cb->curs_set_fn(1);
}

const EngineVtable engine_tuffai_v2 = {
    v2_generate_response,
    V2_MAX_RESPONSE_WORDS,
    V2_CONTEXT_TOKENS,
    V2_VOCAB_SIZE,
    1
};
