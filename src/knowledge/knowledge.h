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
#define V3_VOCAB_SIZE 8001
#define MAX_VOCAB_SIZE 8001
#define EMBED_DIM   32
#define HIDDEN      48
#define MAX_TOKENS  128
#define MAX_GEN     1000
#define MIN_GEN     4
#define TEMP_BASE   6.0f
#define NOISE_BASE  4.5f
#define HIST_MAX    16
#define HIST_LEN    512

#define KNOW_MAX_ENTRIES   1024
#define KNOW_MAX_LEN       256

extern const char *vocab[VOCAB_SIZE];

typedef struct {
    const char *user;
    const char *assistant_thinking;
    const char *assistant;
    const char *assistant_continuation;
    const char *use_tool;
    const char *tool_input;
    const char *assistant_after_tool;
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

extern const char *v3_vocab[V3_VOCAB_SIZE];

extern KnowledgeCategory v3_know_tech;
extern KnowledgeCategory v3_know_general;
extern KnowledgeCategory v3_know_phrases_en;
extern KnowledgeCategory v3_know_phrases_pl;
extern KnowledgeCategory v3_know_phrases_ru;
extern KnowledgeCategory v3_know_phrases_zh;
extern KnowledgeCategory v3_know_phrases_es;
extern KnowledgeCategory v3_know_phrases_fr;
extern KnowledgeCategory v3_know_phrases_de;
extern KnowledgeCategory v3_know_phrases_it;
extern KnowledgeCategory v3_know_phrases_pt;
extern KnowledgeCategory v3_know_phrases_nl;
extern KnowledgeCategory v3_know_phrases_sv;
extern KnowledgeCategory v3_know_phrases_ja;
extern KnowledgeCategory v3_know_phrases_ko;
extern KnowledgeCategory v3_know_phrases_ar;
extern KnowledgeCategory v3_know_phrases_hi;
extern KnowledgeCategory v3_know_phrases_tr;
extern KnowledgeCategory v3_know_code_c;
extern KnowledgeCategory v3_know_code_py;
extern KnowledgeCategory v3_know_code_js;
extern KnowledgeCategory v3_know_code_misc;
extern KnowledgeCategory v3_know_extra;
extern KnowledgeCategory v3_opinions;
extern KnowledgeCategory v3_know_science;
extern KnowledgeCategory v3_know_culture;
extern KnowledgeCategory v3_know_practical;

const char *v3_knowledge_random(const KnowledgeCategory *cat);
const char *v3_knowledge_search(const KnowledgeCategory *cat, const char *keyword);
const char *v3_knowledge_random_any(void);
const char *v3_knowledge_random_code(void);
const char *v3_knowledge_random_phrase(void);
const char *v3_knowledge_match_user(const KnowledgeCategory *cat, const char *input);
void v3_initialize_extra(void);

#endif
