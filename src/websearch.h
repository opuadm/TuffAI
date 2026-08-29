#ifndef WEBSEARCH_H
#define WEBSEARCH_H

#define SEARCH_LANG_EN 0
#define SEARCH_LANG_PL 1
#define SEARCH_LANG_RU 2
#define SEARCH_LANG_ZH 3
#define SEARCH_LANG_COUNT 4

int web_research(const char *query, char *output, int output_size,
                 int use_v2_model);
int web_search_requested(const char *input);
int web_search_enabled(void);
void web_search_set_enabled(int enabled);
int web_search_take_used(void);

int web_fetch_random(char *output, int output_size);
int web_fetch_search(const char *query, char *output, int output_size);
int web_fetch_random_lang(int language, char *output, int output_size);
int web_fetch_search_lang(const char *query, int language, char *output,
                          int output_size);
int web_fetch_random_any_lang(char *output, int output_size);

#endif
