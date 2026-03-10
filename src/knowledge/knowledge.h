#ifndef KNOWLEDGE_H
#define KNOWLEDGE_H

#include <stddef.h>

#define SAFE_SNPRINTF(written, bufsize, call) \
    do { \
        int _ret = (call); \
        if (_ret < 0) _ret = 0; \
        if ((written) + _ret >= (bufsize)) (written) = (bufsize) - 1; \
        else (written) += _ret; \
    } while (0)

#define VOCAB_SIZE  2500
#define ACTUAL_VOCAB 2500
#define V2_VOCAB_SIZE 5601
#define MAX_VOCAB_SIZE 5601
#define EMBED_DIM   32
#define HIDDEN      48
#define MAX_TOKENS  128
#define MAX_GEN     1000
#define MIN_GEN     4
#define TEMP_BASE   6.0f
#define NOISE_BASE  4.5f
#define HIST_MAX    16
#define HIST_LEN    512

#define KNOW_MAX_ENTRIES   512
#define KNOW_MAX_LEN       256

extern const char *vocab[VOCAB_SIZE];

typedef struct {
    const char *user;
    const char *assistant;
} TrainingEntry;

typedef struct {
    TrainingEntry entries[KNOW_MAX_ENTRIES];
    int count;
} KnowledgeCategory;

extern KnowledgeCategory know_tech;
extern KnowledgeCategory know_general;
extern KnowledgeCategory know_phrases_en;
extern KnowledgeCategory know_phrases_pl;
extern KnowledgeCategory know_phrases_ru;
extern KnowledgeCategory know_phrases_zh;
extern KnowledgeCategory know_code_c;
extern KnowledgeCategory know_code_py;
extern KnowledgeCategory know_code_js;
extern KnowledgeCategory know_code_misc;
extern KnowledgeCategory know_extra;

const char *knowledge_random(const KnowledgeCategory *cat);
const char *knowledge_search(const KnowledgeCategory *cat, const char *keyword);
const char *knowledge_random_any(void);
const char *knowledge_random_code(void);
const char *knowledge_random_phrase(void);
const char *knowledge_match_user(const KnowledgeCategory *cat, const char *input);

extern const char *v2_vocab[V2_VOCAB_SIZE];

extern KnowledgeCategory v2_know_tech;
extern KnowledgeCategory v2_know_general;
extern KnowledgeCategory v2_know_phrases_en;
extern KnowledgeCategory v2_know_phrases_pl;
extern KnowledgeCategory v2_know_phrases_ru;
extern KnowledgeCategory v2_know_phrases_zh;
extern KnowledgeCategory v2_know_code_c;
extern KnowledgeCategory v2_know_code_py;
extern KnowledgeCategory v2_know_code_js;
extern KnowledgeCategory v2_know_code_misc;
extern KnowledgeCategory v2_know_extra;
extern KnowledgeCategory v2_opinions;
extern KnowledgeCategory v2_wrong_answers;
extern KnowledgeCategory v2_tangents;
extern KnowledgeCategory v2_definitions;
extern KnowledgeCategory v2_accidental_truths;

extern const char *v2_think_connectors[];
extern const int v2_think_connectors_n;
extern const char *v2_think_fillers[];
extern const int v2_think_fillers_n;
extern const char *v2_knowledge_intros[];
extern const int v2_knowledge_intros_n;
extern const char *v2_opinion_think_phrases[];
extern const int v2_opinion_think_phrases_n;

extern const char *v2_wrong_math_templates[];
extern const int v2_wrong_math_templates_n;
extern const char *v2_emotional_responses[];
extern const int v2_emotional_responses_n;
extern const char *v2_self_loop_templates[];
extern const int v2_self_loop_templates_n;
extern const char *v2_lang_bleed_fragments[];
extern const int v2_lang_bleed_fragments_n;
extern const char *v2_pattern_mimic_headers[];
extern const int v2_pattern_mimic_headers_n;
extern const char *v2_mimic_list_items[];
extern const int v2_mimic_list_items_n;

const char *v2_knowledge_random(const KnowledgeCategory *cat);
const char *v2_knowledge_search(const KnowledgeCategory *cat, const char *keyword);
const char *v2_knowledge_random_any(void);
const char *v2_knowledge_random_code(void);
const char *v2_knowledge_random_phrase(void);
const char *v2_knowledge_match_user(const KnowledgeCategory *cat, const char *input);

#endif
