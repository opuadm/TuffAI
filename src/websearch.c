#include "websearch.h"
#include "features.h"
#include "net.h"
#include "rng.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <curl/curl.h>

#define SEARCH_RESPONSE_LIMIT (1024 * 1024)
#define SEARCH_RESULT_LIMIT 8

typedef struct {
    char title[512];
    char url[2048];
    char description[1536];
} SearchResult;

typedef struct {
    char *data;
    size_t size;
    size_t capacity;
} SearchBuffer;

static int search_runtime_enabled = 1;
static int search_was_used = 0;

static size_t search_write(void *contents, size_t size, size_t count,
                           void *user_data) {
    SearchBuffer *buffer;
    size_t bytes;
    size_t needed;
    size_t capacity;
    char *resized;

    buffer = (SearchBuffer *)user_data;
    if (count > 0 && size > (size_t)-1 / count) return 0;
    bytes = size * count;
    if (bytes > SEARCH_RESPONSE_LIMIT) return 0;
    needed = buffer->size + bytes + 1;
    if (needed > SEARCH_RESPONSE_LIMIT) return 0;
    if (needed > buffer->capacity) {
        capacity = buffer->capacity * 2;
        if (capacity < needed) capacity = needed;
        if (capacity > SEARCH_RESPONSE_LIMIT)
            capacity = SEARCH_RESPONSE_LIMIT;
        resized = (char *)realloc(buffer->data, capacity);
        if (!resized) return 0;
        buffer->data = resized;
        buffer->capacity = capacity;
    }
    memcpy(buffer->data + buffer->size, contents, bytes);
    buffer->size += bytes;
    buffer->data[buffer->size] = '\0';
    return bytes;
}

static void json_escape(const char *input, char *output, int output_size) {
    int input_position;
    int output_position;
    unsigned char character;

    input_position = 0;
    output_position = 0;
    while (input[input_position] && output_position < output_size - 1) {
        character = (unsigned char)input[input_position++];
        if ((character == '"' || character == '\\') &&
            output_position < output_size - 2) {
            output[output_position++] = '\\';
            output[output_position++] = (char)character;
        } else if (character == '\n' && output_position < output_size - 2) {
            output[output_position++] = '\\';
            output[output_position++] = 'n';
        } else if (character >= 32) {
            output[output_position++] = (char)character;
        }
    }
    output[output_position] = '\0';
}

static const char *json_string_after(const char *position, const char *key,
                                     char *output, int output_size) {
    char pattern[64];
    const char *found;
    const char *cursor;
    int output_position;
    unsigned char character;

    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    found = strstr(position, pattern);
    if (!found) return NULL;
    cursor = found + strlen(pattern);
    while (*cursor && *cursor != ':') cursor++;
    if (*cursor != ':') return NULL;
    cursor++;
    while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' ||
           *cursor == '\n') cursor++;
    if (*cursor != '"') return NULL;
    cursor++;
    output_position = 0;
    while (*cursor && output_position < output_size - 1) {
        character = (unsigned char)*cursor++;
        if (character == '"') break;
        if (character == '\\') {
            character = (unsigned char)*cursor++;
            if (!character) break;
            if (character == 'n' || character == 'r' || character == 't') {
                output[output_position++] = ' ';
                continue;
            }
            if (character == 'u') {
                if (cursor[0] && cursor[1] && cursor[2] && cursor[3])
                    cursor += 4;
                output[output_position++] = '?';
                continue;
            }
        }
        output[output_position++] = (char)character;
    }
    output[output_position] = '\0';
    return cursor;
}

static void strip_markup(char *text) {
    char *read_position;
    char *write_position;
    int inside_tag;

    read_position = text;
    write_position = text;
    inside_tag = 0;
    while (*read_position) {
        if (*read_position == '<') {
            inside_tag = 1;
        } else if (*read_position == '>') {
            inside_tag = 0;
        } else if (!inside_tag) {
            *write_position++ = *read_position;
        }
        read_position++;
    }
    *write_position = '\0';
}

static int request_search_json(const char *query, SearchBuffer *response,
                               int use_brave_engine) {
    CURL *curl;
    CURLcode result;
    struct curl_slist *headers;
    char escaped_query[2048];
    char request_body[2304];
    long status_code;

    response->data = (char *)malloc(4096);
    if (!response->data) return 0;
    response->size = 0;
    response->capacity = 4096;
    response->data[0] = '\0';
    json_escape(query, escaped_query, sizeof(escaped_query));
    if (use_brave_engine)
        snprintf(request_body, sizeof(request_body),
                 "{\"query\":\"%s\",\"type\":\"web\","
                 "\"page\":0}", escaped_query);
    else
        snprintf(request_body, sizeof(request_body),
                 "{\"query\":\"%s\",\"type\":\"web\","
                 "\"page\":0,\"engine\":\"kagi\"}", escaped_query);
    curl = curl_easy_init();
    if (!curl) {
        free(response->data);
        response->data = NULL;
        return 0;
    }
    headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_URL, "https://search.tiago.zip/api");
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_body);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE,
                     (long)strlen(request_body));
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, search_write);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, response);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "TuffAI/0.1.2");
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    result = curl_easy_perform(curl);
    status_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status_code);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    if (result != CURLE_OK || status_code < 200 || status_code >= 300) {
        free(response->data);
        response->data = NULL;
        return 0;
    }
    return 1;
}

static int parse_structured_results(const char *json, SearchResult *results,
                                    int maximum_results) {
    const char *cursor;
    const char *title_key;
    const char *next_title;
    const char *url_key;
    const char *description_key;
    const char *after_title;
    int result_count;

    cursor = json;
    result_count = 0;
    while (result_count < maximum_results) {
        title_key = strstr(cursor, "\"title\"");
        if (!title_key) break;
        next_title = strstr(title_key + 7, "\"title\"");
        after_title = json_string_after(
            title_key, "title", results[result_count].title,
            sizeof(results[result_count].title));
        if (!after_title) {
            cursor = title_key + 7;
            continue;
        }
        results[result_count].url[0] = '\0';
        results[result_count].description[0] = '\0';
        url_key = strstr(after_title, "\"url\"");
        description_key = strstr(after_title, "\"description\"");
        if (url_key && (!next_title || url_key < next_title))
            json_string_after(url_key, "url", results[result_count].url,
                              sizeof(results[result_count].url));
        if (description_key &&
            (!next_title || description_key < next_title))
            json_string_after(description_key, "description",
                              results[result_count].description,
                              sizeof(results[result_count].description));
        strip_markup(results[result_count].title);
        strip_markup(results[result_count].description);
        if (results[result_count].title[0] &&
            results[result_count].url[0])
            result_count++;
        cursor = after_title;
    }
    return result_count;
}

static int word_matches(const char *input, const char *word) {
    int input_position;
    int word_position;
    int start;
    unsigned char input_character;
    unsigned char word_character;

    input_position = 0;
    while (input[input_position]) {
        while (input[input_position] &&
               !isalnum((unsigned char)input[input_position]))
            input_position++;
        start = input_position;
        word_position = 0;
        while (input[input_position] && word[word_position]) {
            input_character = (unsigned char)input[input_position];
            word_character = (unsigned char)word[word_position];
            if (tolower(input_character) != tolower(word_character)) break;
            input_position++;
            word_position++;
        }
        if (!word[word_position] &&
            !isalnum((unsigned char)input[input_position]))
            return 1;
        input_position = start;
        while (input[input_position] &&
               isalnum((unsigned char)input[input_position]))
            input_position++;
    }
    return 0;
}

int web_search_requested(const char *input) {
    const char capability[] = "/search";

    if (!input || !input[0]) return 0;
    return word_matches(input, capability + 1);
}

static void select_query_text(const char *input, char *output,
                              int output_size) {
    const char *cursor;
    const char *start;
    const char *best_start;
    int length;
    int best_length;

    cursor = input;
    best_start = NULL;
    best_length = 0;
    while (*cursor) {
        if (*cursor != '"' && *cursor != '\'') {
            cursor++;
            continue;
        }
        start = cursor + 1;
        cursor = start;
        while (*cursor && *cursor != start[-1]) cursor++;
        length = (int)(cursor - start);
        if (length > best_length) {
            best_start = start;
            best_length = length;
        }
        if (*cursor) cursor++;
    }
    if (!best_start) {
        snprintf(output, output_size, "%s", input);
        return;
    }
    if (best_length >= output_size) best_length = output_size - 1;
    memcpy(output, best_start, best_length);
    output[best_length] = '\0';
}

static int safe_page_url(const char *url) {
    const char *host;

    if (strncmp(url, "https://", 8) != 0) return 0;
    host = url + 8;
    if (strncmp(host, "localhost", 9) == 0) return 0;
    if (strncmp(host, "127.", 4) == 0) return 0;
    if (strncmp(host, "10.", 3) == 0) return 0;
    if (strncmp(host, "192.168.", 8) == 0) return 0;
    if (strncmp(host, "[::1]", 5) == 0) return 0;
    return 1;
}

static int fetch_page(const char *url, SearchBuffer *page) {
    CURL *curl;
    CURLcode result;
    long status_code;

    if (!safe_page_url(url)) return 0;
    page->data = (char *)malloc(4096);
    if (!page->data) return 0;
    page->size = 0;
    page->capacity = 4096;
    page->data[0] = '\0';
    curl = curl_easy_init();
    if (!curl) {
        free(page->data);
        page->data = NULL;
        return 0;
    }
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, search_write);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, page);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "TuffAI/0.1.2");
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 12L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 3L);
#if LIBCURL_VERSION_NUM >= 0x075500
    curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS_STR, "https");
#else
    curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS, CURLPROTO_HTTPS);
#endif
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    result = curl_easy_perform(curl);
    status_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status_code);
    curl_easy_cleanup(curl);
    if (result != CURLE_OK || status_code < 200 || status_code >= 300) {
        free(page->data);
        page->data = NULL;
        return 0;
    }
    return 1;
}

static int ascii_prefix_equal(const char *text, const char *prefix) {
    unsigned char left;
    unsigned char right;

    while (*prefix) {
        left = (unsigned char)*text++;
        right = (unsigned char)*prefix++;
        if (tolower(left) != tolower(right)) return 0;
    }
    return !isalnum((unsigned char)*text);
}

static int suppressed_tag_action(const char *tag) {
    int closing;

    while (*tag && isspace((unsigned char)*tag)) tag++;
    closing = *tag == '/';
    if (closing) {
        tag++;
        while (*tag && isspace((unsigned char)*tag)) tag++;
    }
    if (!ascii_prefix_equal(tag, "script") &&
        !ascii_prefix_equal(tag, "style") &&
        !ascii_prefix_equal(tag, "noscript") &&
        !ascii_prefix_equal(tag, "svg"))
        return 0;
    return closing ? -1 : 1;
}

static int clean_page_text(const char *page, char *output, int output_size) {
    int input_position;
    int output_position;
    int inside_tag;
    int pending_space;
    int suppressed;
    int action;
    unsigned char character;

    input_position = 0;
    output_position = 0;
    inside_tag = 0;
    pending_space = 0;
    suppressed = 0;
    while (page[input_position] && output_position < output_size - 1) {
        character = (unsigned char)page[input_position++];
        if (character == '<') {
            action = suppressed_tag_action(page + input_position);
            if (action > 0) suppressed++;
            else if (action < 0 && suppressed > 0) suppressed--;
            inside_tag = 1;
            pending_space = 1;
        } else if (character == '>') {
            inside_tag = 0;
        } else if (!inside_tag && suppressed == 0) {
            if (character <= 32) {
                pending_space = output_position > 0;
            } else {
                if (pending_space && output_position < output_size - 1)
                    output[output_position++] = ' ';
                pending_space = 0;
                output[output_position++] = (char)character;
            }
        }
    }
    output[output_position] = '\0';
    return output_position;
}

static float score_result(const char *query, const SearchResult *result,
                          int tokenizer_version) {
    Features features;
    float feature_context[EMBED_DIM];
    char metadata[2048];
    int tokens[MAX_TOKENS];
    int token_count;
    int token;
    float score;
    int i;
    int d;

    features = extract(query);
    feat_to_embed(&features, feature_context);
    snprintf(metadata, sizeof(metadata), "%.511s %.1535s", result->title,
             result->description);
    token_count = tokenize(metadata, tokens, MAX_TOKENS);
#ifdef ENABLE_TUFFAI_V2
    if (tokenizer_version == 1)
        token_count = v2_tokenize(metadata, tokens, MAX_TOKENS);
#endif
#ifdef ENABLE_TUFFAI_V3
    if (tokenizer_version == 2)
        token_count = v3_tokenize(metadata, tokens, MAX_TOKENS);
#endif
#if !defined(ENABLE_TUFFAI_V2) && !defined(ENABLE_TUFFAI_V3)
    (void)tokenizer_version;
#endif
    score = 0.0f;
    for (i = 0; i < token_count; i++) {
        token = tokens[i] % MAX_VOCAB_SIZE;
        if (token < 0) token += MAX_VOCAB_SIZE;
        for (d = 0; d < EMBED_DIM; d++)
            score += embeddings[token][d] * feature_context[d];
    }
    if (token_count > 0)
        score /= (float)(token_count * EMBED_DIM);
    return score + (rng_unit() * 2.0f - 1.0f) * 0.03f;
}

int web_research(const char *query, char *output, int output_size,
                 int tokenizer_version) {
    SearchBuffer response;
    SearchBuffer page;
    SearchResult results[SEARCH_RESULT_LIMIT];
    char selected_query[2048];
    float scores[SEARCH_RESULT_LIMIT];
    int chosen[SEARCH_RESULT_LIMIT];
    int result_count;
    int chosen_count;
    int best_index;
    float best_score;
    int output_position;
    int loaded_count;
    int written;
    int i;
    int pick;
    int prior_pick;
    int desired_pages;
    int page_text_length;
    int page_start;

    if (!search_runtime_enabled) return 0;
    if (!query || !query[0] || !output || output_size <= 0) return -1;
    output[0] = '\0';
    select_query_text(query, selected_query, sizeof(selected_query));
    if (!request_search_json(selected_query, &response, 0)) return -1;
    result_count = parse_structured_results(response.data, results,
                                            SEARCH_RESULT_LIMIT);
    free(response.data);
    if (result_count <= 0) {
        if (!request_search_json(selected_query, &response, 1)) return -1;
        result_count = parse_structured_results(response.data, results,
                                                SEARCH_RESULT_LIMIT);
        free(response.data);
    }
    if (result_count <= 0) return 0;
    for (i = 0; i < result_count; i++)
        scores[i] = score_result(selected_query, &results[i], tokenizer_version);
    desired_pages = 1;
    if (result_count > 1 && rng_range(100) < 55) desired_pages = 2;
    chosen_count = result_count;
    for (pick = 0; pick < chosen_count; pick++) {
        best_index = -1;
        best_score = -1000000.0f;
        for (i = 0; i < result_count; i++) {
            for (prior_pick = 0; prior_pick < pick; prior_pick++)
                if (i == chosen[prior_pick]) break;
            if (prior_pick < pick) continue;
            if (scores[i] > best_score) {
                best_score = scores[i];
                best_index = i;
            }
        }
        chosen[pick] = best_index;
    }
    output_position = 0;
    loaded_count = 0;
    for (pick = 0; pick < chosen_count; pick++) {
        if (loaded_count >= desired_pages) break;
        best_index = chosen[pick];
        if (best_index < 0) continue;
        if (fetch_page(results[best_index].url, &page)) {
            page_start = output_position;
            written = snprintf(output + output_position,
                               output_size - output_position, "%s\n",
                               results[best_index].title);
            if (written < 0 ||
                written >= output_size - output_position) {
                free(page.data);
                break;
            }
            output_position += written;
            page_text_length = clean_page_text(
                page.data, output + output_position,
                output_size - output_position);
            output_position += page_text_length;
            free(page.data);
            if (page_text_length > 0) {
                loaded_count++;
                if (output_position < output_size - 2) {
                    output[output_position++] = '\n';
                    output[output_position++] = '\n';
                    output[output_position] = '\0';
                }
            } else {
                output_position = page_start;
                output[output_position] = '\0';
            }
        }
    }
    if (loaded_count > 0) search_was_used = 1;
    return loaded_count;
}

int web_search_enabled(void) {
    return search_runtime_enabled;
}

void web_search_set_enabled(int enabled) {
    search_runtime_enabled = enabled ? 1 : 0;
}

int web_search_take_used(void) {
    int used;

    used = search_was_used;
    search_was_used = 0;
    return used;
}

int web_fetch_random(char *output, int output_size) {
    if (output && output_size > 0) output[0] = '\0';
    return 0;
}

int web_fetch_search(const char *query, char *output, int output_size) {
    return web_research(query, output, output_size, 0) > 0;
}

int web_fetch_random_lang(int language, char *output, int output_size) {
    (void)language;
    return web_fetch_random(output, output_size);
}

int web_fetch_search_lang(const char *query, int language, char *output,
                          int output_size) {
    (void)language;
    return web_fetch_search(query, output, output_size);
}

int web_fetch_random_any_lang(char *output, int output_size) {
    return web_fetch_random(output, output_size);
}
