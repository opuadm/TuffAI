#include "../../knowledge/knowledge.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

const char *knowledge_random(const KnowledgeCategory *cat) {
    if (!cat || cat->count <= 0) return "...";
    return cat->entries[rand() % cat->count].assistant;
}

const char *knowledge_search(const KnowledgeCategory *cat, const char *keyword) {
    int i, j, klen;
    char lower_entry[KNOW_MAX_LEN];
    char lower_key[128];
    const char *best;
    int best_score;
    int score;

    if (!cat || cat->count <= 0 || !keyword) return knowledge_random(cat);

    klen = (int)strlen(keyword);
    if (klen <= 0) return knowledge_random(cat);
    if (klen > 127) klen = 127;

    for (i = 0; i < klen; i++)
        lower_key[i] = (char)tolower((unsigned char)keyword[i]);
    lower_key[klen] = '\0';

    best = NULL;
    best_score = 0;

    for (i = 0; i < cat->count; i++) {
        int elen = (int)strlen(cat->entries[i].assistant);
        if (elen >= KNOW_MAX_LEN) elen = KNOW_MAX_LEN - 1;
        for (j = 0; j < elen; j++)
            lower_entry[j] = (char)tolower((unsigned char)cat->entries[i].assistant[j]);
        lower_entry[elen] = '\0';

        score = 0;
        if (strstr(lower_entry, lower_key)) score += 10;
        if (score > best_score) {
            best_score = score;
            best = cat->entries[i].assistant;
        }
    }

    if (best) return best;
    return knowledge_random(cat);
}

const char *knowledge_random_any(void) {
    int pick;

    pick = rand() % 5;
    switch (pick) {
    case 0: return knowledge_random(&know_tech);
    case 1: return knowledge_random(&know_general);
    case 2: return knowledge_random(&know_phrases_en);
    case 3: return knowledge_random(&know_extra);
    default: return knowledge_random(&know_tech);
    }
}

const char *knowledge_random_code(void) {
    int pick;

    pick = rand() % 4;
    switch (pick) {
    case 0: return knowledge_random(&know_code_c);
    case 1: return knowledge_random(&know_code_py);
    case 2: return knowledge_random(&know_code_js);
    default: return knowledge_random(&know_code_misc);
    }
}

const char *knowledge_random_phrase(void) {
    int pick;

    pick = rand() % 4;
    switch (pick) {
    case 0: return knowledge_random(&know_phrases_en);
    case 1: return knowledge_random(&know_phrases_pl);
    case 2: return knowledge_random(&know_phrases_ru);
    default: return knowledge_random(&know_phrases_zh);
    }
}

const char *knowledge_match_user(const KnowledgeCategory *cat, const char *input) {
    int i, ilen, ulen, j;
    char lower_input[512];
    char lower_user[256];

    if (!cat || cat->count <= 0 || !input) return NULL;

    ilen = (int)strlen(input);
    if (ilen > 511) ilen = 511;
    for (i = 0; i < ilen; i++)
        lower_input[i] = (char)tolower((unsigned char)input[i]);
    lower_input[ilen] = '\0';

    for (i = 0; i < cat->count; i++) {
        if (!cat->entries[i].user) continue;
        ulen = (int)strlen(cat->entries[i].user);
        if (ulen > 255) ulen = 255;
        for (j = 0; j < ulen; j++)
            lower_user[j] = (char)tolower((unsigned char)cat->entries[i].user[j]);
        lower_user[ulen] = '\0';
        if (strstr(lower_input, lower_user))
            return cat->entries[i].assistant;
    }
    return NULL;
}
