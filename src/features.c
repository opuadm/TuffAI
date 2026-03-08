#include "features.h"
#include <string.h>
#include <ctype.h>
#include <math.h>

extern float W_feat[16][EMBED_DIM];

Features extract(const char *s) {
    Features f = {0};
    int len = (int)strlen(s);
    int words, vowels, upper, digits, punct, in_word;
    int word_lens[64], wc, long_words;
    int cur_wlen;
    long asum;
    unsigned char seen[32];
    int i;
    float avg_wl;
    int unique;
    float ent;
    int freq[256];
    unsigned char b;
    float p;
    unsigned char c;
    char lc;

    if (!len) return f;

    words = 0;
    vowels = 0;
    upper = 0;
    digits = 0;
    punct = 0;
    in_word = 0;
    wc = 0;
    long_words = 0;
    cur_wlen = 0;
    asum = 0;
    memset(seen, 0, sizeof(seen));

    for (i = 0; i < len; i++) {
        c = (unsigned char)s[i];
        asum += c;
        seen[c / 8] |= (1 << (c % 8));
        if (isupper(c)) upper++;
        if (isdigit(c)) { digits++; f.has_number = 1.0f; }
        if (ispunct(c)) punct++;
        if (isspace(c)) {
            if (in_word) {
                if (wc < 64) word_lens[wc++] = cur_wlen;
                if (cur_wlen > 6) long_words++;
                cur_wlen = 0;
            }
            in_word = 0;
        } else {
            if (!in_word) { words++; in_word = 1; } else cur_wlen++;
            lc = tolower(c);
            if (strchr("aeiouáéíóúàèìòùäëïöüâêîôûаеёиоуыьъэюяеіїє", lc)) vowels++;
        }
    }
    if (in_word) {
        if (wc < 64) word_lens[wc++] = cur_wlen;
        if (cur_wlen > 6) long_words++;
    }

    avg_wl = 0;
    for (i = 0; i < wc; i++) avg_wl += word_lens[i];
    if (wc > 0) avg_wl /= wc;

    unique = 0;
    for (i = 0; i < 32; i++) {
        b = seen[i];
        while (b) { unique += (b & 1); b >>= 1; }
    }

    ent = 0;
    memset(freq, 0, sizeof(freq));
    for (i = 0; i < len; i++) freq[(unsigned char)s[i]]++;
    for (i = 0; i < 256; i++) {
        if (freq[i] > 0) {
            p = (float)freq[i] / len;
            ent -= p * log2f(p);
        }
    }

    f.len_norm        = fminf((float)len / 300.0f, 1.0f);
    f.word_count      = fminf((float)words / 30.0f, 1.0f);
    f.avg_word_len    = fminf(avg_wl / 10.0f, 1.0f);
    f.vowel_ratio     = (float)vowels / fmaxf(len, 1);
    f.upper_ratio     = (float)upper / fmaxf(len, 1);
    f.digit_ratio     = (float)digits / fmaxf(len, 1);
    f.punct_ratio     = (float)punct / fmaxf(len, 1);
    f.is_question     = (s[len - 1] == '?') ? 1.0f : 0.0f;
    f.exclamation     = (s[len - 1] == '!') ? 1.0f : 0.0f;
    f.avg_ascii       = (float)asum / len / 128.0f;
    f.unique_ratio    = fminf((float)unique / 60.0f, 1.0f);
    f.long_word_ratio = wc > 0 ? (float)long_words / wc : 0.0f;
    f.starts_upper    = isupper((unsigned char)s[0]) ? 1.0f : 0.0f;
    f.ends_punct      = ispunct((unsigned char)s[len - 1]) ? 1.0f : 0.0f;
    f.entropy         = ent / 8.0f;
    return f;
}

void feat_to_embed(const Features *f, float *out) {
    int i, j;
    float z;
    float fv[16];

    fv[0]  = f->len_norm;
    fv[1]  = f->word_count;
    fv[2]  = f->avg_word_len;
    fv[3]  = f->vowel_ratio;
    fv[4]  = f->upper_ratio;
    fv[5]  = f->digit_ratio;
    fv[6]  = f->punct_ratio;
    fv[7]  = f->is_question;
    fv[8]  = f->exclamation;
    fv[9]  = f->has_number;
    fv[10] = f->avg_ascii;
    fv[11] = f->unique_ratio;
    fv[12] = f->long_word_ratio;
    fv[13] = f->starts_upper;
    fv[14] = f->ends_punct;
    fv[15] = f->entropy;

    for (j = 0; j < EMBED_DIM; j++) {
        z = 0;
        for (i = 0; i < 16; i++) z += fv[i] * W_feat[i][j];
        out[j] = tanhf(z);
    }
}
