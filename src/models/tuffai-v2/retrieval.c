#include "retrieval.h"
#include "../../knowledge/knowledge.h"
#include "../../rng.h"
#include <ctype.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

static int word_character(unsigned char character) {
    return character >= 128 || isalnum(character);
}

static int has_word(const char *text, const char *word) {
    int text_length;
    int word_length;
    int text_position;
    int word_position;
    unsigned char left;
    unsigned char right;

    text_length = (int)strlen(text);
    word_length = (int)strlen(word);
    if (word_length == 0 || word_length > text_length) return 0;
    for (text_position = 0;
         text_position + word_length <= text_length;
         text_position++) {
        if (text_position > 0 &&
            word_character((unsigned char)text[text_position - 1]))
            continue;
        for (word_position = 0; word_position < word_length;
             word_position++) {
            left = (unsigned char)text[text_position + word_position];
            right = (unsigned char)word[word_position];
            if (left < 128) left = (unsigned char)tolower(left);
            if (right < 128) right = (unsigned char)tolower(right);
            if (left != right) break;
        }
        if (word_position == word_length &&
            (text_position + word_length == text_length ||
             !word_character((unsigned char)
                             text[text_position + word_length])))
            return 1;
    }
    return 0;
}

static int prefix_similarity(const char *left, const char *right) {
    int left_length;
    int right_length;
    int limit;
    int matched;
    unsigned char a;
    unsigned char b;

    left_length = (int)strlen(left);
    right_length = (int)strlen(right);
    if (left_length < 4 || right_length < 4) return 0;
    limit = left_length < right_length ? left_length : right_length;
    matched = 0;
    while (matched < limit) {
        a = (unsigned char)left[matched];
        b = (unsigned char)right[matched];
        if (a < 128) a = (unsigned char)tolower(a);
        if (b < 128) b = (unsigned char)tolower(b);
        if (a != b) break;
        matched++;
    }
    return matched >= 4 ? matched : 0;
}

static int fuzzy_word_score(const char *text, const char *query) {
    char copy[1024];
    char *word;
    int best;
    int score;

    if (!text || strlen(text) >= sizeof(copy)) return 0;
    memcpy(copy, text, strlen(text) + 1);
    best = 0;
    word = strtok(copy, " \t\n\r.,!?;:\"'()[]{}<>/\\|=+-*&%$#@`~");
    while (word) {
        score = prefix_similarity(word, query);
        if (score > best) best = score;
        word = strtok(NULL, " \t\n\r.,!?;:\"'()[]{}<>/\\|=+-*&%$#@`~");
    }
    return best;
}

static int score_text(const char *text, char words[][64], int word_count,
                      int prompt_weight, int *matches, int *longest) {
    int score;
    int length;
    int fuzzy;
    int i;

    score = 0;
    if (!text) return 0;
    for (i = 0; i < word_count; i++) {
        length = (int)strlen(words[i]);
        if (length == 1 && !isupper((unsigned char)words[i][0])) continue;
        if (length == 2 && !isupper((unsigned char)words[i][0])) continue;
        if (has_word(text, words[i])) {
            score += (length > 16 ? 16 : length) * prompt_weight;
            (*matches)++;
            if (length > *longest) *longest = length;
        } else if (length >= 4) {
            fuzzy = fuzzy_word_score(text, words[i]);
            if (fuzzy >= 4) score += fuzzy * prompt_weight / 3;
        }
    }
    return score;
}

static int response_structure(const char *text) {
    int newlines;
    int indented_lines;
    int syntax;
    int line_start;
    int score;
    int i;

    newlines = 0;
    indented_lines = 0;
    syntax = 0;
    line_start = 1;
    for (i = 0; text[i]; i++) {
        if (text[i] == '\n') {
            newlines++;
            line_start = 1;
        } else {
            if (line_start && (text[i] == ' ' || text[i] == '\t'))
                indented_lines++;
            line_start = 0;
        }
        if (text[i] == '{' || text[i] == '}' || text[i] == ';' ||
            text[i] == '=' || text[i] == '(' || text[i] == ')' ||
            text[i] == '[' || text[i] == ']' || text[i] == ':' ||
            text[i] == '<' || text[i] == '>')
            syntax++;
    }
    score = syntax * 3 + newlines * 4 + indented_lines * 5;
    if (newlines < 2 && syntax < 6) score = 0;
    if (score > 120) score = 120;
    return score;
}

static void score_category(const KnowledgeCategory *category,
                           char words[][64], int word_count,
                           const char **best_text, int *best_score,
                           int *best_matches, int *best_longest,
                           int *best_prompt_matches,
                           int *best_structure) {
    int score;
    int matches;
    int longest;
    int prompt_matches;
    int prompt_longest;
    int answer_matches;
    int answer_longest;
    int structure;
    int ties;
    int i;

    ties = 1;
    for (i = 0; i < category->count; i++) {
        if (!category->entries[i].assistant) continue;
        prompt_matches = 0;
        prompt_longest = 0;
        answer_matches = 0;
        answer_longest = 0;
        score = score_text(category->entries[i].user, words, word_count,
                           5, &prompt_matches, &prompt_longest);
        score += score_text(category->entries[i].assistant, words,
                            word_count, 1, &answer_matches,
                            &answer_longest);
        matches = prompt_matches + answer_matches;
        longest = prompt_longest > answer_longest ?
                  prompt_longest : answer_longest;
        structure = response_structure(category->entries[i].assistant);
        if (prompt_matches >= 3) score += structure;
        if (score > *best_score) {
            *best_text = category->entries[i].assistant;
            *best_score = score;
            *best_matches = matches;
            *best_longest = longest;
            *best_prompt_matches = prompt_matches;
            *best_structure = structure;
            ties = 1;
        } else if (score == *best_score && score > 0) {
            ties++;
            if (rng_range(ties) == 0) {
                *best_text = category->entries[i].assistant;
                *best_matches = matches;
                *best_longest = longest;
                *best_prompt_matches = prompt_matches;
                *best_structure = structure;
            }
        }
    }
}

static int split_words(const char *input, char words[][64]) {
    char copy[512];
    char *word;
    int count;
    int length;

    strncpy(copy, input, sizeof(copy) - 1);
    copy[sizeof(copy) - 1] = '\0';
    count = 0;
    word = strtok(copy, " \t\n\r.,!?;:\"'()[]{}<>/\\|=+-*&%$#@`~");
    while (word && count < 64) {
        length = (int)strlen(word);
        if (length > 0 && length < 64) {
            memcpy(words[count], word, length + 1);
            count++;
        }
        word = strtok(NULL, " \t\n\r.,!?;:\"'()[]{}<>/\\|=+-*&%$#@`~");
    }
    return count;
}

static int word_count_unit(const char *text) {
    char unit[16];
    int length;
    unsigned char character;

    length = 0;
    while (text[length] && length < (int)sizeof(unit) - 1) {
        character = (unsigned char)text[length];
        if (!isalpha(character)) break;
        unit[length] = (char)tolower(character);
        length++;
    }
    unit[length] = '\0';
    return strcmp(unit, "word") == 0 || strcmp(unit, "words") == 0;
}

static int text_word_count(const char *text) {
    int count;
    int inside_word;

    count = 0;
    inside_word = 0;
    while (*text) {
        if (isspace((unsigned char)*text)) {
            inside_word = 0;
        } else if (!inside_word) {
            count++;
            inside_word = 1;
        }
        text++;
    }
    return count;
}

static int prompt_is_question(const char *text) {
    int length;

    length = (int)strlen(text);
    while (length > 0 && isspace((unsigned char)text[length - 1])) length--;
    return length > 0 && text[length - 1] == '?';
}

static int generation_prompt_score(
    const char *prefix, const KnowledgeCategory *const *sets,
    int set_count) {
    char words[64][64];
    int word_count;
    int best_score;
    int score;
    int matches;
    int longest;
    int set_index;
    int entry_index;

    word_count = split_words(prefix, words);
    best_score = 0;
    for (set_index = 0; set_index < set_count; set_index++) {
        for (entry_index = 0; entry_index < sets[set_index]->count;
             entry_index++) {
            if (!sets[set_index]->entries[entry_index].user ||
                !sets[set_index]->entries[entry_index].assistant)
                continue;
            if (prompt_is_question(
                    sets[set_index]->entries[entry_index].user))
                continue;
            if (text_word_count(
                    sets[set_index]->entries[entry_index].assistant) < 20)
                continue;
            matches = 0;
            longest = 0;
            score = score_text(
                sets[set_index]->entries[entry_index].user, words,
                word_count, 5, &matches, &longest);
            if (score > best_score) best_score = score;
        }
    }
    return best_score;
}

static int requested_word_count(
    const char *input, const KnowledgeCategory *const *sets,
    int set_count) {
    const char *cursor;
    const char *unit;
    char *number_end;
    char prefix[512];
    int prefix_length;
    long count;

    cursor = input;
    while (*cursor) {
        if (!isdigit((unsigned char)*cursor)) {
            cursor++;
            continue;
        }
        if (cursor > input && isalnum((unsigned char)cursor[-1])) {
            cursor++;
            continue;
        }
        count = strtol(cursor, &number_end, 10);
        unit = number_end;
        while (*unit && (isspace((unsigned char)*unit) || *unit == '-'))
            unit++;
        if (count > 0 && word_count_unit(unit)) {
            prefix_length = (int)(cursor - input);
            if (prefix_length >= (int)sizeof(prefix))
                prefix_length = sizeof(prefix) - 1;
            memcpy(prefix, input, prefix_length);
            prefix[prefix_length] = '\0';
            if (generation_prompt_score(prefix, sets, set_count) >= 25) {
                if (count > INT_MAX) count = INT_MAX;
                return (int)count;
            }
        }
        cursor = number_end > cursor ? number_end : cursor + 1;
    }
    return 0;
}

void v2_retrieve_dataset_example(const char *input,
                                 V2Retrieval *retrieval) {
    const KnowledgeCategory *sets[16];
    char words[64][64];
    const char *best_text;
    int word_count;
    int best_score;
    int best_matches;
    int best_longest;
    int best_prompt_matches;
    int best_structure;
    int i;

    sets[0] = &v2_know_code_c;
    sets[1] = &v2_know_code_py;
    sets[2] = &v2_know_code_js;
    sets[3] = &v2_know_code_misc;
    sets[4] = &v2_know_tech;
    sets[5] = &v2_know_general;
    sets[6] = &v2_know_phrases_en;
    sets[7] = &v2_know_phrases_pl;
    sets[8] = &v2_know_phrases_ru;
    sets[9] = &v2_know_phrases_zh;
    sets[10] = &v2_know_extra;
    sets[11] = &v2_opinions;
    sets[12] = &v2_wrong_answers;
    sets[13] = &v2_tangents;
    sets[14] = &v2_definitions;
    sets[15] = &v2_accidental_truths;
    word_count = split_words(input, words);
    best_text = NULL;
    best_score = 0;
    best_matches = 0;
    best_longest = 0;
    best_prompt_matches = 0;
    best_structure = 0;
    for (i = 0; i < 16; i++)
        score_category(sets[i], words, word_count, &best_text,
                       &best_score, &best_matches, &best_longest,
                       &best_prompt_matches, &best_structure);
    retrieval->text = best_score > 0 ? best_text : NULL;
    retrieval->score = best_score;
    retrieval->matches = best_matches;
    retrieval->is_code = best_text && best_score >= 40 &&
                         best_prompt_matches >= 3 &&
                         best_structure >= 30;
    retrieval->requested_words = requested_word_count(input, sets, 16);
}
