#include "markov.h"
#include "wikifetch.h"
#include "knowledge/knowledge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MKV_MAX_WORDS 2048
#define MKV_MAX_WORDLEN 64
#define MKV_CHAIN_ORDER 2

typedef struct {
    char words[MKV_MAX_WORDS][MKV_MAX_WORDLEN];
    int count;
} WordList;

static int split_words(const char *text, WordList *wl) {
    const char *p = text;
    int len;
    char c;

    wl->count = 0;
    while (*p && wl->count < MKV_MAX_WORDS) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        len = 0;
        while (*p && !isspace((unsigned char)*p) && len < MKV_MAX_WORDLEN - 1) {
            c = *p++;
            wl->words[wl->count][len++] = c;
        }
        wl->words[wl->count][len] = '\0';
        wl->count++;
    }
    return wl->count;
}

static int words_match(const WordList *wl, int pos, const char *w1, const char *w2) {
    if (pos + 1 >= wl->count) return 0;
    if (strcmp(wl->words[pos], w1) != 0) return 0;
    if (strcmp(wl->words[pos + 1], w2) != 0) return 0;
    return 1;
}

static const char *pick_next(const WordList *wl, const char *w1, const char *w2) {
    int candidates[256];
    int ncand = 0;
    int i;

    for (i = 0; i < wl->count - 2 && ncand < 256; i++) {
        if (words_match(wl, i, w1, w2)) {
            candidates[ncand++] = i + 2;
        }
    }

    if (ncand == 0) return NULL;
    return wl->words[candidates[rand() % ncand]];
}

int markov_generate(const char *seed_text, char *out, int out_size, int max_words) {
    char wiki_buf[8192];
    char combined[16384];
    WordList *wl;
    int got_wiki;
    int start;
    char cur1[MKV_MAX_WORDLEN];
    char cur2[MKV_MAX_WORDLEN];
    const char *next;
    int written;
    int i;
    int wlen;

    got_wiki = 0;
    if (rand() % 2 == 0) {
        got_wiki = wiki_fetch_search(seed_text, wiki_buf, sizeof(wiki_buf));
    }
    if (!got_wiki) {
        got_wiki = wiki_fetch_random(wiki_buf, sizeof(wiki_buf));
    }
    if (!got_wiki) return 0;

    wl = malloc(sizeof(WordList));
    if (!wl) return 0;

    snprintf(combined, sizeof(combined), "%s", wiki_buf);

    if (split_words(combined, wl) < 5) {
        free(wl);
        return 0;
    }

    start = rand() % (wl->count > 3 ? wl->count - 2 : 1);
    strncpy(cur1, wl->words[start], MKV_MAX_WORDLEN - 1);
    cur1[MKV_MAX_WORDLEN - 1] = '\0';
    strncpy(cur2, wl->words[start + 1], MKV_MAX_WORDLEN - 1);
    cur2[MKV_MAX_WORDLEN - 1] = '\0';

    written = snprintf(out, out_size, "%s %s", cur1, cur2);
    if (written >= out_size) written = out_size - 1;

    for (i = 0; i < max_words && written < out_size - 2; i++) {
        next = pick_next(wl, cur1, cur2);
        if (!next) {
            start = rand() % (wl->count > 2 ? wl->count - 1 : 1);
            strncpy(cur1, wl->words[start], MKV_MAX_WORDLEN - 1);
            cur1[MKV_MAX_WORDLEN - 1] = '\0';
            if (start + 1 < wl->count) {
                strncpy(cur2, wl->words[start + 1], MKV_MAX_WORDLEN - 1);
                cur2[MKV_MAX_WORDLEN - 1] = '\0';
            }
            continue;
        }
        wlen = (int)strlen(next);
        if (written + 1 + wlen >= out_size - 1) break;
        SAFE_SNPRINTF(written, out_size, snprintf(out + written, out_size - written, " %s", next));
        strncpy(cur1, cur2, MKV_MAX_WORDLEN - 1);
        cur1[MKV_MAX_WORDLEN - 1] = '\0';
        strncpy(cur2, next, MKV_MAX_WORDLEN - 1);
        cur2[MKV_MAX_WORDLEN - 1] = '\0';

        if (rand() % 40 == 0) {
            start = rand() % (wl->count > 2 ? wl->count - 1 : 1);
            strncpy(cur1, wl->words[start], MKV_MAX_WORDLEN - 1);
            cur1[MKV_MAX_WORDLEN - 1] = '\0';
            if (start + 1 < wl->count) {
                strncpy(cur2, wl->words[start + 1], MKV_MAX_WORDLEN - 1);
                cur2[MKV_MAX_WORDLEN - 1] = '\0';
            }
        }
    }

    free(wl);
    return written > 0 ? 1 : 0;
}
