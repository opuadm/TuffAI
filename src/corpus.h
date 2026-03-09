#ifndef CORPUS_H
#define CORPUS_H

#include "knowledge/knowledge.h"
#include "features.h"
#include "tokenizer.h"

#define RESP_WORD_SALAD      0
#define RESP_WIKI_CONFUSED   1
#define RESP_ECHO_KEYWORD    2
#define RESP_WIKI_DRIFT      3
#define RESP_CONFUSION       4
#define RESP_WIKI_MANGLE     5
#define RESP_MARKOV          6
#define RESP_ECHO_MANGLE     7
#define RESP_SINGLE_WORD     8
#define RESP_TRUNCATE        9
#define RESP_REPEAT         10
#define RESP_CODE_GEN       11
#define RESP_MULTILANG      12
#define RESP_KNOWLEDGE      13
#define RESP_WIKI_FOREIGN   14
#define RESP_WRONG_ANSWER   15
#define RESP_WRONG_MATH     16
#define RESP_FAKE_DEFINE    17
#define RESP_EMOTIONAL_AMP  18
#define RESP_PATTERN_MIMIC  19
#define RESP_LANG_BLEED     20
#define RESP_SELF_LOOP      21
#define RESP_TANGENT        22
#define RESP_ACCIDENTAL_OK  23
#define RESP_MODE_COUNT     24

int pick_response_mode(int pattern, const Features *feat, int turn_count);
int generate_corpus_response(int mode, const char *input, int pattern, const Features *feat, char *out, int out_size);
void inject_unicode(char *text, int text_size);
void scramble_words(const char *input, char *out, int out_size);
void extract_keyword_ext(const char *input, char *keyword, int kw_size);
void stream_print(const char *text);

int v2_pick_response_mode(int pattern, const Features *feat, int turn_count);
int v2_generate_corpus_response(int mode, const char *input, int pattern, const Features *feat, char *out, int out_size);
void v2_inject_unicode(char *text, int text_size);
void v2_scramble_words(const char *input, char *out, int out_size);
void v2_extract_keyword_ext(const char *input, char *keyword, int kw_size);
void v2_stream_print(const char *text);

#endif
