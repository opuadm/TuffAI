#include "retrieval.h"
#include "../../knowledge/knowledge.h"
#include "../../rng.h"
#include "../../tokenizer.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long entry_signatures[27][KNOW_MAX_ENTRIES];
static unsigned char entry_structures[27][KNOW_MAX_ENTRIES];
static int entry_signatures_initialized;

static int response_structure(const char *text);

static const char *entry_response(const TrainingEntry *entry) {
    if (entry->assistant) return entry->assistant;
    return entry->assistant_after_tool;
}

static int extract_numbers(const char *text, char numbers[][32],
                           int maximum_numbers) {
    int count;
    int length;

    count = 0;
    while (text && *text && count < maximum_numbers) {
        while (*text && !isdigit((unsigned char)*text)) text++;
        if (!*text) break;
        length = 0;
        while (isdigit((unsigned char)*text)) {
            if (length < 31) numbers[count][length++] = *text;
            text++;
        }
        numbers[count][length] = '\0';
        count++;
    }
    return count;
}

static int numbers_compatible(const char *input, const char *entry_user) {
    char input_numbers[16][32];
    char entry_numbers[16][32];
    int input_count;
    int entry_count;
    int i;

    input_count = extract_numbers(input, input_numbers, 16);
    entry_count = extract_numbers(entry_user, entry_numbers, 16);
    if (input_count == 0) return 1;
    if (entry_count == 0) return 0;
    if (input_count != entry_count) return 0;
    for (i = 0; i < input_count; i++)
        if (strcmp(input_numbers[i], entry_numbers[i]) != 0) return 0;
    return 1;
}

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

static unsigned long signature_hash(const char *word, int length,
                                    unsigned long seed) {
    unsigned long hash;
    unsigned char character;
    int limit;
    int index;

    hash = seed;
    limit = length < 4 ? length : 4;
    for (index = 0; index < limit; index++) {
        character = (unsigned char)word[index];
        if (character < 128)
            character = (unsigned char)tolower(character);
        hash ^= character;
        hash *= 16777619UL;
    }
    hash ^= (unsigned long)limit * 2654435761UL;
    return hash;
}

static unsigned long long word_signature(const char *word, int length) {
    unsigned long first_hash;
    unsigned long second_hash;
    unsigned long long signature;

    if (length <= 0) return 0;
    first_hash = signature_hash(word, length, 2166136261UL);
    second_hash = signature_hash(word, length, 2246822519UL);
    signature = 1ULL << (first_hash & 63UL);
    signature |= 1ULL << (second_hash & 63UL);
    return signature;
}

static unsigned long long text_signature(const char *text) {
    unsigned long long signature;
    int position;
    int start;

    signature = 0;
    position = 0;
    while (text && text[position]) {
        while (text[position] &&
               !word_character((unsigned char)text[position]))
            position++;
        start = position;
        while (text[position] &&
               word_character((unsigned char)text[position]))
            position++;
        signature |= word_signature(text + start, position - start);
    }
    return signature;
}

static int signature_matches(unsigned long long signature,
                             char words[][64], int word_count) {
    unsigned long long candidate_signature;
    int index;

    if (word_count <= 0) return 1;
    for (index = 0; index < word_count; index++) {
        candidate_signature = word_signature(
            words[index], (int)strlen(words[index]));
        if ((signature & candidate_signature) == candidate_signature)
            return 1;
    }
    return 0;
}

static void initialize_entry_signatures(
    const KnowledgeCategory *const *sets, int set_count) {
    int set_index;
    int entry_index;
    const char *response;

    if (entry_signatures_initialized) return;
    for (set_index = 0; set_index < set_count; set_index++) {
        for (entry_index = 0; entry_index < sets[set_index]->count;
             entry_index++) {
            response = entry_response(
                &sets[set_index]->entries[entry_index]);
            entry_signatures[set_index][entry_index] =
                text_signature(sets[set_index]->entries[entry_index].user) |
                text_signature(response);
            entry_structures[set_index][entry_index] = (unsigned char)
                response_structure(response);
        }
    }
    entry_signatures_initialized = 1;
}

static int score_text(const char *text, char words[][64], int word_count,
                      int prompt_weight, int *matches, int *longest) {
    int score;
    int length;
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

    if (!text) return 0;
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

static const char *random_structured_code(
    const KnowledgeCategory *category) {
    const char *selected_text;
    int candidates;
    int i;

    selected_text = NULL;
    candidates = 0;
    for (i = 0; i < category->count; i++) {
        if (!category->entries[i].assistant ||
            response_structure(category->entries[i].assistant) < 30)
            continue;
        candidates++;
        if (rng_range(candidates) == 0)
            selected_text = category->entries[i].assistant;
    }
    return selected_text;
}

static void score_category(const KnowledgeCategory *category,
                           int category_index,
                           const char *input,
                           char words[][64], int word_count,
                           const char **best_text,
                           const char **best_thinking,
                           const char **best_tool,
                           const char **best_tool_input,
                           const char **best_after_tool, int *best_score,
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
    int input_number_count;
    char input_numbers[16][32];
    const char *answer_text;

    ties = 1;
    input_number_count = extract_numbers(input, input_numbers, 16);
    for (i = 0; i < category->count; i++) {
        answer_text = entry_response(&category->entries[i]);
        if (!answer_text) continue;
        if (!numbers_compatible(input, category->entries[i].user)) continue;
        if (!signature_matches(entry_signatures[category_index][i],
                               words, word_count))
            continue;
        prompt_matches = 0;
        prompt_longest = 0;
        answer_matches = 0;
        answer_longest = 0;
        score = score_text(category->entries[i].user, words, word_count,
                           5, &prompt_matches, &prompt_longest);
        score += score_text(answer_text, words,
                            word_count, 1, &answer_matches,
                            &answer_longest);
        score += input_number_count * 20;
        matches = prompt_matches + answer_matches;
        longest = prompt_longest > answer_longest ?
                  prompt_longest : answer_longest;
        if (category_index == 4 && matches > 0) score += 20;
        structure = entry_structures[category_index][i];
        if (prompt_matches >= 3) score += structure;
        if (score > *best_score) {
            *best_text = category->entries[i].assistant;
            *best_thinking = category->entries[i].assistant_thinking;
            *best_tool = category->entries[i].use_tool;
            *best_tool_input = category->entries[i].tool_input;
            *best_after_tool = category->entries[i].assistant_after_tool;
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
                *best_thinking = category->entries[i].assistant_thinking;
                *best_tool = category->entries[i].use_tool;
                *best_tool_input = category->entries[i].tool_input;
                *best_after_tool = category->entries[i].assistant_after_tool;
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

static int requested_word_count(const char *input) {
    const char *cursor;
    const char *unit;
    char *number_end;
    int unit_length;
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
        unit_length = 0;
        while (isalpha((unsigned char)unit[unit_length])) unit_length++;
        if (count > 0 && unit_length >= 3 && unit_length <= 16) {
            if (count > 1000000) count = 1000000;
            return (int)count;
        }
        cursor = number_end > cursor ? number_end : cursor + 1;
    }
    return 0;
}

void v3_retrieve_dataset_example(const char *input,
                                 V3Retrieval *retrieval) {
    const KnowledgeCategory *sets[27];
    char words[64][64];
    const char *best_text;
    const char *best_thinking;
    const char *best_tool;
    const char *best_tool_input;
    const char *best_after_tool;
    int word_count;
    int best_score;
    int best_matches;
    int best_longest;
    int best_prompt_matches;
    int best_structure;
    int code_language;
    const KnowledgeCategory *code_category;
    const char *code_text;
    int i;

    v3_initialize_extra();
    retrieval->text = NULL;
    retrieval->thinking = NULL;
    retrieval->use_tool = NULL;
    retrieval->tool_input = NULL;
    retrieval->after_tool = NULL;
    retrieval->score = 0;
    retrieval->matches = 0;
    retrieval->is_code = 0;
    retrieval->requested_words = detect_pattern(input) == PAT_MATH ? 0 :
                                 requested_word_count(input);
    code_language = detect_code_request(input);
    if (retrieval->requested_words > 0 &&
        code_language == CODE_LANG_NONE &&
        detect_pattern(input) != PAT_CODE)
        return;
    sets[0] = &v3_know_code_c;
    sets[1] = &v3_know_code_py;
    sets[2] = &v3_know_code_js;
    sets[3] = &v3_know_code_misc;
    sets[4] = &v3_know_tech;
    sets[5] = &v3_know_general;
    sets[6] = &v3_know_phrases_en;
    sets[7] = &v3_know_phrases_pl;
    sets[8] = &v3_know_phrases_ru;
    sets[9] = &v3_know_phrases_zh;
    sets[10] = &v3_know_extra;
    sets[11] = &v3_opinions;
    sets[12] = &v3_know_science;
    sets[13] = &v3_know_culture;
    sets[14] = &v3_know_practical;
    sets[15] = &v3_know_phrases_es;
    sets[16] = &v3_know_phrases_fr;
    sets[17] = &v3_know_phrases_de;
    sets[18] = &v3_know_phrases_it;
    sets[19] = &v3_know_phrases_pt;
    sets[20] = &v3_know_phrases_nl;
    sets[21] = &v3_know_phrases_sv;
    sets[22] = &v3_know_phrases_ja;
    sets[23] = &v3_know_phrases_ko;
    sets[24] = &v3_know_phrases_ar;
    sets[25] = &v3_know_phrases_hi;
    sets[26] = &v3_know_phrases_tr;
    initialize_entry_signatures(sets, 27);
    word_count = split_words(input, words);
    best_text = NULL;
    best_thinking = NULL;
    best_tool = NULL;
    best_tool_input = NULL;
    best_after_tool = NULL;
    best_score = 0;
    best_matches = 0;
    best_longest = 0;
    best_prompt_matches = 0;
    best_structure = 0;
    for (i = 0; i < 27; i++)
        score_category(sets[i], i, input, words, word_count, &best_text,
                       &best_thinking, &best_tool, &best_tool_input,
                       &best_after_tool,
                       &best_score, &best_matches, &best_longest,
                       &best_prompt_matches, &best_structure);
    retrieval->text = best_score > 0 ? best_text : NULL;
    retrieval->thinking = best_score > 0 ? best_thinking : NULL;
    retrieval->use_tool = best_score > 0 ? best_tool : NULL;
    retrieval->tool_input = best_score > 0 ? best_tool_input : NULL;
    retrieval->after_tool = best_score > 0 ? best_after_tool : NULL;
    retrieval->score = best_score;
    retrieval->matches = best_matches;
    retrieval->is_code = best_text && best_score >= 40 &&
                         best_prompt_matches >= 3 &&
                         best_structure >= 30;
    if (code_language == CODE_LANG_NONE &&
        detect_pattern(input) == PAT_CODE)
        code_language = CODE_LANG_C + rng_range(4);
    if (code_language != CODE_LANG_NONE && !retrieval->is_code) {
        code_category = sets[code_language - CODE_LANG_C];
        code_text = random_structured_code(code_category);
        if (code_text) {
            retrieval->text = code_text;
            retrieval->thinking = NULL;
            retrieval->use_tool = NULL;
            retrieval->tool_input = NULL;
            retrieval->after_tool = NULL;
            retrieval->is_code = 1;
            retrieval->score = 40;
            retrieval->matches = 1;
        }
    }
}
