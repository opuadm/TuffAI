#include "wikifetch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <curl/curl.h>

typedef struct {
    char *data;
    size_t size;
    size_t cap;
} Buffer;

static size_t write_cb(void *ptr, size_t size, size_t nmemb, void *userdata) {
    Buffer *buf = (Buffer *)userdata;
    size_t total = size * nmemb;
    size_t needed = buf->size + total + 1;

    if (needed > buf->cap) {
        size_t newcap = buf->cap * 2;
        char *tmp;
        if (newcap < needed) newcap = needed;
        if (newcap > 512 * 1024) return 0;
        tmp = realloc(buf->data, newcap);
        if (!tmp) return 0;
        buf->data = tmp;
        buf->cap = newcap;
    }
    memcpy(buf->data + buf->size, ptr, total);
    buf->size += total;
    buf->data[buf->size] = '\0';
    return total;
}

static void strip_html(char *text) {
    char *r = text;
    char *w = text;
    int in_tag = 0;

    while (*r) {
        if (*r == '<') { in_tag = 1; r++; continue; }
        if (*r == '>') { in_tag = 0; r++; continue; }
        if (!in_tag) *w++ = *r;
        r++;
    }
    *w = '\0';
}

static char *find_json_string(const char *json, const char *key) {
    char search[128];
    char *start, *end;
    int len;
    char *result;

    snprintf(search, sizeof(search), "\"%s\":\"", key);
    start = strstr(json, search);
    if (!start) {
        snprintf(search, sizeof(search), "\"%s\": \"", key);
        start = strstr(json, search);
    }
    if (!start) return NULL;

    start = strchr(start, ':');
    if (!start) return NULL;
    start++;
    while (*start == ' ') start++;
    if (*start != '"') return NULL;
    start++;

    end = start;
    while (*end && !(*end == '"' && *(end - 1) != '\\')) end++;
    if (!*end) return NULL;

    len = (int)(end - start);
    result = malloc(len + 1);
    if (!result) return NULL;
    memcpy(result, start, len);
    result[len] = '\0';
    return result;
}

static int do_fetch(const char *url, char *out, int out_size) {
    CURL *curl;
    CURLcode res;
    Buffer buf;
    char *extract;

    buf.data = malloc(4096);
    if (!buf.data) return 0;
    buf.size = 0;
    buf.cap = 4096;
    buf.data[0] = '\0';

    curl = curl_easy_init();
    if (!curl) { free(buf.data); return 0; }

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "TuffAI/5.0 (DumbBot)");
    curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, "https");

    res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK || buf.size == 0) {
        free(buf.data);
        return 0;
    }

    extract = find_json_string(buf.data, "extract");
    if (extract) {
        strip_html(extract);
        snprintf(out, out_size, "%s", extract);
        free(extract);
        free(buf.data);
        return 1;
    }

    free(buf.data);
    return 0;
}

static void url_encode(const char *src, char *dst, int dst_size) {
    int i = 0;
    int w = 0;

    while (src[i] && w < dst_size - 4) {
        unsigned char c = (unsigned char)src[i];
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            dst[w++] = c;
        } else if (c == ' ') {
            dst[w++] = '+';
        } else {
            if (w + 3 < dst_size) {
                snprintf(dst + w, dst_size - w, "%%%02X", c);
                w += 3;
            }
        }
        i++;
    }
    dst[w] = '\0';
}

int wiki_fetch_random(char *out, int out_size) {
    return do_fetch(
        "https://en.wikipedia.org/api/rest_v1/page/random/summary",
        out, out_size);
}

static const char *wiki_lang_codes[] = {"en", "pl", "ru", "zh"};

int wiki_fetch_random_lang(int lang, char *out, int out_size) {
    char url[512];

    if (lang < 0 || lang >= WIKI_LANG_COUNT) lang = WIKI_LANG_EN;
    snprintf(url, sizeof(url),
        "https://%s.wikipedia.org/api/rest_v1/page/random/summary",
        wiki_lang_codes[lang]);
    return do_fetch(url, out, out_size);
}

int wiki_fetch_search_lang(const char *query, int lang, char *out, int out_size) {
    char url[512];
    char encoded[256];
    char buf[2048];
    char *words[32];
    int nwords = 0;
    int pick;
    char *p;

    if (lang < 0 || lang >= WIKI_LANG_COUNT) lang = WIKI_LANG_EN;

    strncpy(buf, query, 2047);
    buf[2047] = '\0';

    p = strtok(buf, " \t\n.,!?;:'\"()[]{}");
    while (p && nwords < 32) {
        if (strlen(p) > 2) words[nwords++] = p;
        p = strtok(NULL, " \t\n.,!?;:'\"()[]{}");
    }

    if (nwords > 0) {
        pick = rand() % nwords;
        url_encode(words[pick], encoded, sizeof(encoded));
    } else {
        url_encode("thing", encoded, sizeof(encoded));
    }

    snprintf(url, sizeof(url),
        "https://%s.wikipedia.org/api/rest_v1/page/summary/%s",
        wiki_lang_codes[lang], encoded);

    if (do_fetch(url, out, out_size)) return 1;
    return wiki_fetch_random_lang(lang, out, out_size);
}

int wiki_fetch_random_any_lang(char *out, int out_size) {
    int lang;

    lang = rand() % WIKI_LANG_COUNT;
    return wiki_fetch_random_lang(lang, out, out_size);
}

int wiki_fetch_search(const char *query, char *out, int out_size) {
    char url[512];
    char encoded[256];
    char buf[2048];
    char *words[32];
    int nwords = 0;
    int pick;
    char *p;

    strncpy(buf, query, 2047);
    buf[2047] = '\0';

    p = strtok(buf, " \t\n.,!?;:'\"()[]{}");
    while (p && nwords < 32) {
        if (strlen(p) > 2) words[nwords++] = p;
        p = strtok(NULL, " \t\n.,!?;:'\"()[]{}");
    }

    if (nwords > 0) {
        pick = rand() % nwords;
        url_encode(words[pick], encoded, sizeof(encoded));
    } else {
        url_encode("thing", encoded, sizeof(encoded));
    }

    snprintf(url, sizeof(url),
        "https://en.wikipedia.org/api/rest_v1/page/summary/%s",
        encoded);

    if (do_fetch(url, out, out_size)) return 1;

    return wiki_fetch_random(out, out_size);
}
