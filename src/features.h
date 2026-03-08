#ifndef FEATURES_H
#define FEATURES_H

#include "knowledge/knowledge.h"

typedef struct {
    float len_norm;
    float word_count;
    float avg_word_len;
    float vowel_ratio;
    float upper_ratio;
    float digit_ratio;
    float punct_ratio;
    float is_question;
    float exclamation;
    float has_number;
    float avg_ascii;
    float unique_ratio;
    float long_word_ratio;
    float starts_upper;
    float ends_punct;
    float entropy;
} Features;

Features extract(const char *s);
void feat_to_embed(const Features *f, float *out);

#endif
