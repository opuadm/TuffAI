#include "corpus.h"
#include "wikifetch.h"
#include "markov.h"
#include "knowledge/knowledge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static unsigned int random_unicode_char(void) {
    unsigned int cp;
    int range;

    range = rand() % 10;
    if (range < 3) {
        cp = 0x0400 + (unsigned int)(rand() % 0x0100);
    } else if (range < 5) {
        cp = 0x4E00 + (unsigned int)(rand() % 0x51B0);
    } else if (range < 7) {
        cp = 0x0600 + (unsigned int)(rand() % 0x0100);
    } else if (range < 8) {
        cp = 0x0E00 + (unsigned int)(rand() % 0x0080);
    } else if (range < 9) {
        cp = 0x3040 + (unsigned int)(rand() % 0x00C0);
    } else {
        cp = 0x0020 + (unsigned int)(rand() % 0x9FE0);
    }
    if (cp >= 0xD800 && cp <= 0xDFFF) cp = 0x263A;
    return cp;
}

static int encode_utf8(unsigned int cp, char *buf) {
    if (cp < 0x80) {
        buf[0] = (char)cp;
        return 1;
    } else if (cp < 0x800) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp < 0x10000) {
        buf[0] = (char)(0xE0 | (cp >> 12));
        buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    } else if (cp < 0x110000) {
        buf[0] = (char)(0xF0 | (cp >> 18));
        buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
    return 0;
}

static int is_stopword(const char *w) {
    static const char *stops[] = {
        "the","a","an","is","it","was","are","am","be","been","being",
        "have","has","had","do","does","did","will","would","could","should",
        "can","may","might","shall","must","need","dare","ought",
        "i","you","we","they","he","she","me","my","your","our","their",
        "this","that","these","those","what","which","who","whom","whose",
        "when","where","why","how","if","then","than","but","and","or","nor",
        "not","no","yes","so","very","too","also","just","only","even",
        "still","already","yet","again","now","here","there","some","any",
        "all","both","each","every","few","many","much","more","most",
        "make","give","get","put","set","take","come","go","write","read",
        "for","from","with","about","into","through","during","before","after",
        "above","below","to","of","in","on","at","by","up","down","out","off",
        "over","under","between","against","please","tell","show","generate",
        "create","explain","help","want","like","know","think","say",
        NULL
    };
    int i;
    char lower[64];
    int len;

    len = (int)strlen(w);
    if (len >= 63) len = 63;
    for (i = 0; i < len; i++) lower[i] = (char)tolower((unsigned char)w[i]);
    lower[len] = '\0';

    for (i = 0; stops[i]; i++) {
        if (strcmp(lower, stops[i]) == 0) return 1;
    }
    return 0;
}

void extract_keyword_ext(const char *input, char *keyword, int kw_size) {
    char buf[2048];
    char *words[64];
    int nwords = 0;
    char *p;
    int best = -1;
    int best_score = -1;
    int i, wlen, score;

    strncpy(buf, input, 2047);
    buf[2047] = '\0';

    p = strtok(buf, " \t\n.,!?;:'\"()[]{}");
    while (p && nwords < 64) {
        if (strlen(p) > 1) words[nwords++] = p;
        p = strtok(NULL, " \t\n.,!?;:'\"()[]{}");
    }

    for (i = 0; i < nwords; i++) {
        wlen = (int)strlen(words[i]);
        score = wlen;
        if (is_stopword(words[i])) score -= 10;
        if (isupper((unsigned char)words[i][0])) score += 3;
        if (i > nwords / 2) score += 1;
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }

    if (best >= 0) {
        strncpy(keyword, words[best], kw_size - 1);
        keyword[kw_size - 1] = '\0';
    } else {
        strncpy(keyword, input, kw_size - 1);
        keyword[kw_size - 1] = '\0';
    }
}

void scramble_words(const char *input, char *out, int out_size) {
    char buf[2048];
    char *words[128];
    int nwords = 0;
    int i, j, written;
    char *p;
    int wlen;

    strncpy(buf, input, 2047);
    buf[2047] = '\0';

    p = strtok(buf, " \t\n");
    while (p && nwords < 128) {
        words[nwords++] = p;
        p = strtok(NULL, " \t\n");
    }

    for (i = nwords - 1; i > 0; i--) {
        j = rand() % (i + 1);
        p = words[i];
        words[i] = words[j];
        words[j] = p;
    }

    written = 0;
    for (i = 0; i < nwords && written < out_size - 2; i++) {
        wlen = (int)strlen(words[i]);
        if (rand() % 4 == 0) continue;
        if (written > 0 && written < out_size - 1) out[written++] = ' ';
        if (written + wlen < out_size - 1) {
            memcpy(out + written, words[i], wlen);
            written += wlen;
        }
    }

    if (rand() % 3 == 0 && written < out_size - 2) {
        out[written++] = '?';
    }
    out[written] = '\0';
}

static void gen_wiki_confused(const char *input, char *out, int out_size) {
    char confused_term[128];
    char wiki_text[4096];
    int got;

    confused_term[0] = '\0';
    misidentify_topic(input, confused_term, sizeof(confused_term));

    got = wiki_fetch_search(confused_term, wiki_text, sizeof(wiki_text));
    if (!got) {
        got = wiki_fetch_random(wiki_text, sizeof(wiki_text));
    }

    if (got) {
        snprintf(out, out_size, "%s", wiki_text);
    } else {
        snprintf(out, out_size, "%s?", confused_term);
    }
}

static void gen_echo_keyword(const char *input, char *out, int out_size) {
    char keyword[128];
    char wiki_text[4096];
    int written;
    int got;

    extract_keyword_ext(input, keyword, sizeof(keyword));

    written = snprintf(out, out_size, "%s?\n\n", keyword);

    got = wiki_fetch_search(keyword, wiki_text, sizeof(wiki_text));
    if (!got) {
        got = wiki_fetch_random(wiki_text, sizeof(wiki_text));
    }
    if (got && written < out_size - 10) {
        snprintf(out + written, out_size - written, "%s", wiki_text);
    }
}

static void gen_wiki_drift(const char *input, char *out, int out_size) {
    char wiki1[4096];
    char wiki2[4096];
    int written;
    int got1, got2;
    int split_point;
    int len1;

    got1 = wiki_fetch_search(input, wiki1, sizeof(wiki1));
    got2 = wiki_fetch_random(wiki2, sizeof(wiki2));

    if (!got1 && !got2) {
        snprintf(out, out_size, "%s?", input);
        return;
    }

    if (got1) {
        len1 = (int)strlen(wiki1);
        split_point = len1 / 3 + rand() % (len1 / 3 + 1);
        while (split_point < len1 && wiki1[split_point] != ' ') split_point++;
        if (split_point >= len1) split_point = len1 / 2;
        wiki1[split_point] = '\0';

        written = snprintf(out, out_size, "%s", wiki1);
    } else {
        written = snprintf(out, out_size, "%s", wiki2);
        got2 = 0;
    }

    switch (rand() % 4) {
    case 0:
        written += snprintf(out + written, out_size - written,
            " which reminds me, ");
        break;
    case 1:
        written += snprintf(out + written, out_size - written,
            " but more importantly, ");
        break;
    case 2:
        written += snprintf(out + written, out_size - written,
            " anyway that's not the point, the point is ");
        break;
    default:
        written += snprintf(out + written, out_size - written,
            " wait where was I... oh right, ");
        break;
    }

    if (got2 && written < out_size - 10) {
        snprintf(out + written, out_size - written, "%s", wiki2);
    }
}

static void gen_confusion(const char *input, char *out, int out_size) {
    int style;
    char confused_term[128];
    char wiki_text[4096];
    int written;
    int got;

    (void)input;
    style = rand() % 6;

    switch (style) {
    case 0:
        snprintf(out, out_size, "That's not what I understand.");
        break;
    case 1:
        misidentify_topic(input, confused_term, sizeof(confused_term));
        got = wiki_fetch_search(confused_term, wiki_text, sizeof(wiki_text));
        if (got) {
            written = snprintf(out, out_size, "%s?\n\n%s", confused_term, wiki_text);
            (void)written;
        } else {
            snprintf(out, out_size, "%s?", confused_term);
        }
        break;
    case 2:
        got = wiki_fetch_random(wiki_text, sizeof(wiki_text));
        if (got) {
            snprintf(out, out_size, "%s", wiki_text);
        } else {
            snprintf(out, out_size, "That's not what I understand.");
        }
        break;
    case 3:
        extract_keyword_ext(input, confused_term, sizeof(confused_term));
        snprintf(out, out_size, "%s?", confused_term);
        break;
    case 4:
        snprintf(out, out_size, "I don't understand what you mean.");
        break;
    default:
        snprintf(out, out_size, "That's not what I understand.");
        break;
    }
}

static void gen_wiki_mangle(const char *input, char *out, int out_size) {
    char wiki1[4096];
    char wiki2[4096];
    char keyword[128];
    int got1, got2;
    int written;
    int len1, len2;
    int cut1, cut2;

    extract_keyword_ext(input, keyword, sizeof(keyword));

    got1 = wiki_fetch_search(keyword, wiki1, sizeof(wiki1));
    got2 = wiki_fetch_random(wiki2, sizeof(wiki2));

    if (!got1 && !got2) {
        snprintf(out, out_size, "%s?", keyword);
        return;
    }

    if (got1 && got2) {
        len1 = (int)strlen(wiki1);
        len2 = (int)strlen(wiki2);
        cut1 = len1 / 2 + rand() % (len1 / 4 + 1);
        while (cut1 < len1 && wiki1[cut1] != ' ') cut1++;
        if (cut1 >= len1) cut1 = len1 / 2;
        wiki1[cut1] = '\0';

        cut2 = rand() % (len2 / 3 + 1);
        while (cut2 < len2 && wiki2[cut2] != ' ') cut2++;

        written = snprintf(out, out_size, "%s %s", wiki1, wiki2 + cut2);
        (void)written;
    } else if (got1) {
        snprintf(out, out_size, "%s", wiki1);
    } else {
        snprintf(out, out_size, "%s", wiki2);
    }
}

void stream_print(const char *text) {
    int i;
    int len;

    len = (int)strlen(text);
    for (i = 0; i < len; i++) {
        putchar(text[i]);
        fflush(stdout);
    }
}

void inject_unicode(char *text, int text_size) {
    char tmp[16384];
    int len;
    int i, w;
    int next_inject;
    unsigned int cp;
    char ubuf[5];
    int ulen;

    len = (int)strlen(text);
    if (len < 10 || len >= (int)sizeof(tmp) - 20) return;

    w = 0;
    next_inject = 8 + rand() % 40;

    for (i = 0; i < len && w < (int)sizeof(tmp) - 10; i++) {
        tmp[w++] = text[i];
        next_inject--;

        if (next_inject <= 0 && text[i] == ' ' && w < (int)sizeof(tmp) - 10) {
            cp = random_unicode_char();
            ulen = encode_utf8(cp, ubuf);
            if (w + ulen + 1 < (int)sizeof(tmp) - 5) {
                memcpy(tmp + w, ubuf, ulen);
                w += ulen;
                tmp[w++] = ' ';
            }
            next_inject = 10 + rand() % 50;
        }
    }
    tmp[w] = '\0';

    if (w < text_size) {
        memcpy(text, tmp, w + 1);
    }
}

static void gen_single_word(const char *input, char *out, int out_size) {
    char keyword[128];

    extract_keyword_ext(input, keyword, sizeof(keyword));
    if (rand() % 2 == 0) {
        snprintf(out, out_size, "%s.", keyword);
    } else {
        snprintf(out, out_size, "%s?", keyword);
    }
}

static void gen_truncate(const char *input, char *out, int out_size) {
    char wiki_text[4096];
    int got;
    int len;
    int cut;

    got = wiki_fetch_search(input, wiki_text, sizeof(wiki_text));
    if (!got) {
        got = wiki_fetch_random(wiki_text, sizeof(wiki_text));
    }
    if (!got) {
        snprintf(out, out_size, "I--");
        return;
    }

    len = (int)strlen(wiki_text);
    cut = 20 + rand() % (len / 3 + 1);
    if (cut >= len) cut = len / 2;
    while (cut > 0 && wiki_text[cut] != ' ') cut--;
    if (cut <= 0) cut = 20;
    wiki_text[cut] = '\0';
    snprintf(out, out_size, "%s--", wiki_text);
}

static void gen_repeat(const char *input, char *out, int out_size) {
    char keyword[128];
    int reps;
    int written;
    int i;

    extract_keyword_ext(input, keyword, sizeof(keyword));
    reps = 2 + rand() % 5;
    written = 0;
    for (i = 0; i < reps && written < out_size - 200; i++) {
        if (i > 0) {
            written += snprintf(out + written, out_size - written, " ");
        }
        written += snprintf(out + written, out_size - written, "%s", keyword);
    }
    if (rand() % 2 == 0 && written < out_size - 2) {
        written += snprintf(out + written, out_size - written, "?");
    }
    (void)written;
}

static void gen_code(const char *input, char *out, int out_size) {
    int code_lang;
    const char *snippet;
    const char *snippet2;
    char keyword[128];
    char wiki_text[4096];
    int got;
    int written;
    int style;

    code_lang = detect_code_request(input);
    style = rand() % 10;

    if (code_lang == CODE_LANG_NONE) {
        code_lang = CODE_LANG_C + rand() % 4;
    }

    switch (code_lang) {
    case CODE_LANG_C:   snippet = knowledge_random(&know_code_c); break;
    case CODE_LANG_PY:  snippet = knowledge_random(&know_code_py); break;
    case CODE_LANG_JS:  snippet = knowledge_random(&know_code_js); break;
    case CODE_LANG_MISC:snippet = knowledge_random(&know_code_misc); break;
    default:            snippet = knowledge_random_code(); break;
    }

    if (style < 4) {
        snprintf(out, out_size, "%s", snippet);
    } else if (style < 6) {
        extract_keyword_ext(input, keyword, sizeof(keyword));
        written = snprintf(out, out_size, "%s?\n\n%s", keyword, snippet);
        (void)written;
    } else if (style == 6) {
        got = wiki_fetch_search(input, wiki_text, sizeof(wiki_text));
        if (got) {
            int cut = (int)strlen(wiki_text) / 3;
            wiki_text[cut] = '\0';
            written = snprintf(out, out_size, "%s\n\n%s", wiki_text, snippet);
            (void)written;
        } else {
            snprintf(out, out_size, "%s", snippet);
        }
    } else if (style == 7) {
        written = snprintf(out, out_size, "%s\n\n", snippet);
        got = wiki_fetch_random(wiki_text, sizeof(wiki_text));
        if (got && written < out_size - 10) {
            snprintf(out + written, out_size - written, "%s", wiki_text);
        }
    } else {
        switch (code_lang) {
        case CODE_LANG_C:   snippet2 = knowledge_random(&know_code_c); break;
        case CODE_LANG_PY:  snippet2 = knowledge_random(&know_code_py); break;
        case CODE_LANG_JS:  snippet2 = knowledge_random(&know_code_js); break;
        case CODE_LANG_MISC:snippet2 = knowledge_random(&know_code_misc); break;
        default:            snippet2 = knowledge_random_code(); break;
        }
        written = snprintf(out, out_size, "%s\n\n%s", snippet, snippet2);
        (void)written;
    }
}

static void gen_multilang(const char *input, char *out, int out_size) {
    const char *phrase;
    char wiki_text[4096];
    int lang_hint;
    int got;
    int written;
    int style;

    lang_hint = detect_language_hint(input);
    style = rand() % 6;

    if (style < 2) {
        phrase = knowledge_random_phrase();
        snprintf(out, out_size, "%s", phrase);
    } else if (style == 2 && lang_hint == 1) {
        phrase = knowledge_random(&know_phrases_pl);
        got = wiki_fetch_search_lang(input, WIKI_LANG_PL, wiki_text, sizeof(wiki_text));
        if (got) {
            written = snprintf(out, out_size, "%s\n\n%s", phrase, wiki_text);
            (void)written;
        } else {
            snprintf(out, out_size, "%s", phrase);
        }
    } else if (style == 2 && lang_hint == 2) {
        phrase = knowledge_random(&know_phrases_ru);
        got = wiki_fetch_search_lang(input, WIKI_LANG_RU, wiki_text, sizeof(wiki_text));
        if (got) {
            written = snprintf(out, out_size, "%s\n\n%s", phrase, wiki_text);
            (void)written;
        } else {
            snprintf(out, out_size, "%s", phrase);
        }
    } else if (style == 2 && lang_hint == 3) {
        phrase = knowledge_random(&know_phrases_zh);
        got = wiki_fetch_search_lang(input, WIKI_LANG_ZH, wiki_text, sizeof(wiki_text));
        if (got) {
            written = snprintf(out, out_size, "%s\n\n%s", phrase, wiki_text);
            (void)written;
        } else {
            snprintf(out, out_size, "%s", phrase);
        }
    } else if (style == 3) {
        got = wiki_fetch_random_any_lang(wiki_text, sizeof(wiki_text));
        if (got) {
            phrase = knowledge_random_phrase();
            written = snprintf(out, out_size, "%s\n\n%s", phrase, wiki_text);
            (void)written;
        } else {
            snprintf(out, out_size, "%s", knowledge_random_phrase());
        }
    } else {
        got = wiki_fetch_random_any_lang(wiki_text, sizeof(wiki_text));
        if (got) {
            snprintf(out, out_size, "%s", wiki_text);
        } else {
            snprintf(out, out_size, "%s", knowledge_random_phrase());
        }
    }
}

static void gen_knowledge(const char *input, char *out, int out_size) {
    char keyword[128];
    const char *entry;
    char wiki_text[4096];
    int got;
    int written;
    int style;
    int pattern;

    extract_keyword_ext(input, keyword, sizeof(keyword));
    pattern = detect_pattern(input);
    style = rand() % 5;

    if (style == 0) {
        entry = knowledge_search(&know_tech, keyword);
        snprintf(out, out_size, "%s", entry);
    } else if (style == 1) {
        entry = knowledge_random(&know_general);
        snprintf(out, out_size, "%s", entry);
    } else if (style == 2) {
        entry = knowledge_random_any();
        written = snprintf(out, out_size, "%s?\n\n%s", keyword, entry);
        (void)written;
    } else if (style == 3) {
        entry = knowledge_random_any();
        got = wiki_fetch_search(input, wiki_text, sizeof(wiki_text));
        if (got) {
            int cut = (int)strlen(wiki_text) / 2;
            while (cut > 0 && wiki_text[cut] != ' ') cut--;
            if (cut > 0) wiki_text[cut] = '\0';
            written = snprintf(out, out_size, "%s %s", wiki_text, entry);
            (void)written;
        } else {
            snprintf(out, out_size, "%s", entry);
        }
    } else {
        if (pattern == PAT_TECH || pattern == PAT_CODE) {
            entry = knowledge_search(&know_tech, keyword);
        } else {
            entry = knowledge_random(&know_general);
        }
        got = wiki_fetch_random(wiki_text, sizeof(wiki_text));
        if (got) {
            int cut = 50 + rand() % 200;
            int wlen = (int)strlen(wiki_text);
            if (cut < wlen) {
                while (cut < wlen && wiki_text[cut] != ' ') cut++;
                wiki_text[cut] = '\0';
            }
            written = snprintf(out, out_size, "%s\n\n%s", entry, wiki_text);
            (void)written;
        } else {
            snprintf(out, out_size, "%s", entry);
        }
    }
}

static void gen_wiki_foreign(const char *input, char *out, int out_size) {
    char wiki_text[4096];
    const char *phrase;
    int lang;
    int got;
    int written;

    lang = rand() % WIKI_LANG_COUNT;
    if (lang == WIKI_LANG_EN && rand() % 2 == 0) lang = WIKI_LANG_PL;

    got = wiki_fetch_search_lang(input, lang, wiki_text, sizeof(wiki_text));
    if (!got) {
        got = wiki_fetch_random_lang(lang, wiki_text, sizeof(wiki_text));
    }

    if (got) {
        if (rand() % 3 == 0) {
            phrase = knowledge_random_phrase();
            written = snprintf(out, out_size, "%s\n\n%s", phrase, wiki_text);
            (void)written;
        } else {
            snprintf(out, out_size, "%s", wiki_text);
        }
    } else {
        snprintf(out, out_size, "%s", knowledge_random_phrase());
    }
}

int pick_response_mode(int pattern, const Features *feat, int turn_count) {
    int r;
    int unhinged;
    int r2;

    r = rand() % 100;
    r2 = rand() % 100;
    unhinged = turn_count / 4;
    if (unhinged > 30) unhinged = 30;

    if (pattern == PAT_CODE) {
        if (r < 70) return RESP_CODE_GEN;
        if (r < 80) return RESP_KNOWLEDGE;
        if (r < 88) return RESP_WIKI_CONFUSED;
        if (r < 94) return RESP_ECHO_KEYWORD;
        return RESP_CONFUSION;
    }

    if (pattern == PAT_LANGUAGE) {
        if (r < 50) return RESP_MULTILANG;
        if (r < 70) return RESP_WIKI_FOREIGN;
        if (r < 82) return RESP_KNOWLEDGE;
        if (r < 90) return RESP_WIKI_CONFUSED;
        return RESP_CONFUSION;
    }

    if (r2 < 3 + unhinged / 3) return RESP_WIKI_FOREIGN;
    if (r2 < 5 + unhinged / 2) return RESP_WORD_SALAD;
    if (r2 < 7) return RESP_KNOWLEDGE;
    if (r2 < 9) return RESP_MULTILANG;

    if (rand() % 100 < 2) return RESP_SINGLE_WORD;
    if (rand() % 100 < 2) return RESP_TRUNCATE;
    if (rand() % 100 < 2) return RESP_REPEAT;

    if (rand() % 100 < 3 + unhinged / 2) return RESP_WIKI_DRIFT;
    if (rand() % 100 < 2 + unhinged / 3) return RESP_MARKOV;

    if (pattern == PAT_NEWS) {
        if (r < 30) return RESP_WIKI_CONFUSED;
        if (r < 50) return RESP_WIKI_FOREIGN;
        if (r < 65) return RESP_KNOWLEDGE;
        if (r < 80) return RESP_WIKI_DRIFT;
        if (r < 90) return RESP_MULTILANG;
        return RESP_ECHO_KEYWORD;
    }

    if (pattern == PAT_CREATIVE) {
        if (r < 25) return RESP_CODE_GEN;
        if (r < 45) return RESP_WIKI_DRIFT;
        if (r < 60) return RESP_MULTILANG;
        if (r < 75) return RESP_KNOWLEDGE;
        if (r < 85) return RESP_WIKI_MANGLE;
        return RESP_MARKOV;
    }

    if (pattern == PAT_MATH || pattern == PAT_TECH) {
        if (r < 20) return RESP_WIKI_CONFUSED;
        if (r < 35) return RESP_KNOWLEDGE;
        if (r < 50) return RESP_WIKI_MANGLE;
        if (r < 60) return RESP_CODE_GEN;
        if (r < 72) return RESP_ECHO_KEYWORD;
        if (r < 82) return RESP_WIKI_DRIFT;
        if (r < 90) return RESP_CONFUSION;
        if (r < 95) return RESP_WIKI_FOREIGN;
        return RESP_MARKOV;
    }

    if (pattern == PAT_GREETING) {
        if (r < 20) return RESP_ECHO_MANGLE;
        if (r < 35) return RESP_MULTILANG;
        if (r < 50) return RESP_ECHO_KEYWORD;
        if (r < 65) return RESP_WIKI_CONFUSED;
        if (r < 75) return RESP_KNOWLEDGE;
        if (r < 85) return RESP_CONFUSION;
        return RESP_WIKI_FOREIGN;
    }

    if (pattern == PAT_QUESTION || pattern == PAT_TIME) {
        if (r < 18) return RESP_WIKI_CONFUSED;
        if (r < 30) return RESP_KNOWLEDGE;
        if (r < 42) return RESP_ECHO_KEYWORD;
        if (r < 54) return RESP_WIKI_DRIFT;
        if (r < 66) return RESP_WIKI_MANGLE;
        if (r < 76) return RESP_MULTILANG;
        if (r < 84) return RESP_CONFUSION;
        if (r < 92) return RESP_MARKOV;
        return RESP_WIKI_FOREIGN;
    }

    if (pattern == PAT_EMOTIONAL) {
        if (r < 25) return RESP_CONFUSION;
        if (r < 40) return RESP_MULTILANG;
        if (r < 55) return RESP_WIKI_CONFUSED;
        if (r < 70) return RESP_KNOWLEDGE;
        if (r < 82) return RESP_ECHO_KEYWORD;
        return RESP_WIKI_DRIFT;
    }

    if (feat->word_count < 0.1f) {
        if (r < 25) return RESP_ECHO_KEYWORD;
        if (r < 40) return RESP_WIKI_CONFUSED;
        if (r < 55) return RESP_MULTILANG;
        if (r < 70) return RESP_KNOWLEDGE;
        if (r < 80) return RESP_CONFUSION;
        return RESP_ECHO_MANGLE;
    }

    if (r < 12) return RESP_WIKI_CONFUSED;
    if (r < 22) return RESP_KNOWLEDGE;
    if (r < 32) return RESP_ECHO_KEYWORD;
    if (r < 42) return RESP_WIKI_DRIFT;
    if (r < 52) return RESP_WIKI_MANGLE;
    if (r < 60) return RESP_MULTILANG;
    if (r < 68) return RESP_CONFUSION;
    if (r < 76) return RESP_MARKOV;
    if (r < 84) return RESP_WIKI_FOREIGN;
    if (r < 90) return RESP_CODE_GEN;
    return RESP_ECHO_MANGLE;
}

int generate_corpus_response(int mode, const char *input, int pattern, const Features *feat, char *out, int out_size) {
    (void)pattern;
    (void)feat;

    out[0] = '\0';

    switch (mode) {
    case RESP_WIKI_CONFUSED:
        gen_wiki_confused(input, out, out_size);
        return 1;

    case RESP_ECHO_KEYWORD:
        gen_echo_keyword(input, out, out_size);
        return 1;

    case RESP_WIKI_DRIFT:
        gen_wiki_drift(input, out, out_size);
        return 1;

    case RESP_CONFUSION:
        gen_confusion(input, out, out_size);
        return 1;

    case RESP_WIKI_MANGLE:
        gen_wiki_mangle(input, out, out_size);
        return 1;

    case RESP_MARKOV:
        if (markov_generate(input, out, out_size, 60 + rand() % 120)) return 1;
        gen_wiki_confused(input, out, out_size);
        return 1;

    case RESP_ECHO_MANGLE:
        scramble_words(input, out, out_size);
        return 1;

    case RESP_SINGLE_WORD:
        gen_single_word(input, out, out_size);
        return 1;

    case RESP_TRUNCATE:
        gen_truncate(input, out, out_size);
        return 1;

    case RESP_REPEAT:
        gen_repeat(input, out, out_size);
        return 1;

    case RESP_CODE_GEN:
        gen_code(input, out, out_size);
        return 1;

    case RESP_MULTILANG:
        gen_multilang(input, out, out_size);
        return 1;

    case RESP_KNOWLEDGE:
        gen_knowledge(input, out, out_size);
        return 1;

    case RESP_WIKI_FOREIGN:
        gen_wiki_foreign(input, out, out_size);
        return 1;

    case RESP_WORD_SALAD:
    default:
        return 0;
    }
}