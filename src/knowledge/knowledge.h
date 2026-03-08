#ifndef KNOWLEDGE_H
#define KNOWLEDGE_H

#include <stddef.h>

#define VOCAB_SIZE  2500
#define ACTUAL_VOCAB 2500
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

#endif
