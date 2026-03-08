#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "knowledge/knowledge.h"

int tokenize(const char *input, int *tokens, int max_tok);
int detect_pattern(const char *input);
void misidentify_topic(const char *input, char *out, int out_size);
int detect_code_request(const char *input);
int detect_language_hint(const char *input);

#define PAT_NONE       0
#define PAT_MATH       1
#define PAT_QUESTION   2
#define PAT_GREETING   3
#define PAT_TECH       4
#define PAT_EMOTIONAL  5
#define PAT_TIME       6
#define PAT_COMMAND    7
#define PAT_CODE       8
#define PAT_LANGUAGE   9
#define PAT_NEWS      10
#define PAT_CREATIVE  11

#define CODE_LANG_NONE  0
#define CODE_LANG_C     1
#define CODE_LANG_PY    2
#define CODE_LANG_JS    3
#define CODE_LANG_MISC  4

#endif
