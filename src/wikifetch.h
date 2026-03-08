#ifndef WIKIFETCH_H
#define WIKIFETCH_H

#define WIKI_LANG_EN 0
#define WIKI_LANG_PL 1
#define WIKI_LANG_RU 2
#define WIKI_LANG_ZH 3
#define WIKI_LANG_COUNT 4

int wiki_fetch_random(char *out, int out_size);
int wiki_fetch_search(const char *query, char *out, int out_size);
int wiki_fetch_random_lang(int lang, char *out, int out_size);
int wiki_fetch_search_lang(const char *query, int lang, char *out, int out_size);
int wiki_fetch_random_any_lang(char *out, int out_size);

#endif
