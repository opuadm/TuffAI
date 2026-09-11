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
static __thread int search_tls_used = 0;
static __thread int search_tls_override_active = 0;
static __thread int search_tls_override_enabled = 0;

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
    static const char hexdigits[] = "0123456789abcdef";

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
        } else if (character == '\r' && output_position < output_size - 2) {
            output[output_position++] = '\\';
            output[output_position++] = 'r';
        } else if (character == '\t' && output_position < output_size - 2) {
            output[output_position++] = '\\';
            output[output_position++] = 't';
        } else if (character == '\b' && output_position < output_size - 2) {
            output[output_position++] = '\\';
            output[output_position++] = 'b';
        } else if (character == '\f' && output_position < output_size - 2) {
            output[output_position++] = '\\';
            output[output_position++] = 'f';
        } else if (character < 32 && output_position < output_size - 6) {
            output[output_position++] = '\\';
            output[output_position++] = 'u';
            output[output_position++] = '0';
            output[output_position++] = '0';
            output[output_position++] = hexdigits[character >> 4];
            output[output_position++] = hexdigits[character & 15];
        } else if (character >= 32) {
            output[output_position++] = (char)character;
        }
    }
    output[output_position] = '\0';
}

static int search_hex_value(char character) {
    if (character >= '0' && character <= '9') return character - '0';
    if (character >= 'a' && character <= 'f') return character - 'a' + 10;
    if (character >= 'A' && character <= 'F') return character - 'A' + 10;
    return -1;
}

static void search_emit_utf8(char *output, int output_size,
                             int *position, int codepoint) {
    if (codepoint < 0 || codepoint > 0x10FFFF ||
        (codepoint >= 0xD800 && codepoint <= 0xDFFF))
        codepoint = '?';
    if (codepoint < 0x80) {
        if (*position < output_size - 1)
            output[(*position)++] = (char)codepoint;
    } else if (codepoint < 0x800) {
        if (*position < output_size - 2) {
            output[(*position)++] = (char)(0xC0 | (codepoint >> 6));
            output[(*position)++] = (char)(0x80 | (codepoint & 0x3F));
        }
    } else if (codepoint < 0x10000) {
        if (*position < output_size - 3) {
            output[(*position)++] = (char)(0xE0 | (codepoint >> 12));
            output[(*position)++] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
            output[(*position)++] = (char)(0x80 | (codepoint & 0x3F));
        }
    } else {
        if (*position < output_size - 4) {
            output[(*position)++] = (char)(0xF0 | (codepoint >> 18));
            output[(*position)++] = (char)(0x80 | ((codepoint >> 12) & 0x3F));
            output[(*position)++] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
            output[(*position)++] = (char)(0x80 | (codepoint & 0x3F));
        }
    }
}

static const char *search_decode_string(const char *cursor, char *output,
                                        int output_size) {
    int output_position;
    unsigned char character;
    int codepoint;

    output_position = 0;
    while (*cursor && output_position < output_size - 1) {
        character = (unsigned char)*cursor++;
        if (character == '"') break;
        if (character != '\\') {
            output[output_position++] = (char)character;
            continue;
        }
        character = (unsigned char)*cursor++;
        if (!character) break;
        if (character == 'n' || character == 'r' || character == 't' ||
            character == 'b' || character == 'f') {
            output[output_position++] = ' ';
        } else if (character == 'u') {
            int h0 = search_hex_value(cursor[0]);
            int h1 = search_hex_value(cursor[1]);
            int h2 = search_hex_value(cursor[2]);
            int h3 = search_hex_value(cursor[3]);
            if (h0 < 0 || h1 < 0 || h2 < 0 || h3 < 0) {
                output[output_position++] = '?';
                continue;
            }
            codepoint = h0 * 4096 + h1 * 256 + h2 * 16 + h3;
            cursor += 4;
            if (codepoint >= 0xD800 && codepoint <= 0xDBFF &&
                cursor[0] == '\\' && cursor[1] == 'u') {
                int l0 = search_hex_value(cursor[2]);
                int l1 = search_hex_value(cursor[3]);
                int l2 = search_hex_value(cursor[4]);
                int l3 = search_hex_value(cursor[5]);
                if (l0 >= 0 && l1 >= 0 && l2 >= 0 && l3 >= 0) {
                    int low = l0 * 4096 + l1 * 256 + l2 * 16 + l3;
                    if (low >= 0xDC00 && low <= 0xDFFF) {
                        codepoint = 0x10000 + ((codepoint - 0xD800) << 10) +
                                    (low - 0xDC00);
                        cursor += 6;
                    }
                }
            }
            if (output_size - output_position - 1 > 0)
                search_emit_utf8(output, output_size,
                                 &output_position, codepoint);
        } else if (character == '/') {
            output[output_position++] = '/';
        } else {
            output[output_position++] = (char)character;
        }
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

static const char *search_match_close(const char *open, char open_char,
                                      char close_char) {
    const char *cursor;
    int depth;
    int in_string;
    int escaped;

    if (!open || *open != open_char) return NULL;
    cursor = open;
    depth = 0;
    in_string = 0;
    escaped = 0;
    while (*cursor) {
        if (in_string) {
            if (!escaped && *cursor == '"') in_string = 0;
            if (!escaped && *cursor == '\\') escaped = 1;
            else escaped = 0;
        } else if (*cursor == '"') {
            in_string = 1;
        } else if (*cursor == open_char) {
            depth++;
        } else if (*cursor == close_char) {
            depth--;
            if (depth == 0) return cursor;
        }
        cursor++;
    }
    return NULL;
}

static const char *search_find_key(const char *json, const char *end,
                                   const char *key) {
    size_t key_length;
    const char *cursor;
    const char *after;
    int escaped;

    if (!json || !key) return NULL;
    key_length = strlen(key);
    cursor = json;
    while (*cursor && (!end || cursor < end)) {
        if (*cursor != '"') {
            cursor++;
            continue;
        }
        cursor++;
        after = cursor;
        escaped = 0;
        while (*after && (!end || after < end)) {
            if (!escaped && *after == '"') break;
            if (!escaped && *after == '\\') escaped = 1;
            else escaped = 0;
            after++;
        }
        if (!*after || (end && after >= end)) return NULL;
        if ((size_t)(after - cursor) == key_length &&
            memcmp(cursor, key, key_length) == 0) {
            after++;
            while (*after == ' ' || *after == '\t' || *after == '\r' ||
                   *after == '\n') after++;
            if (*after == ':') {
                after++;
                while (*after == ' ' || *after == '\t' || *after == '\r' ||
                       *after == '\n') after++;
                return after;
            }
        }
        cursor = after + 1;
    }
    return NULL;
}

static int search_read_field(const char *object, const char *object_end,
                             const char *key, char *output, int output_size) {
    const char *value;

    value = search_find_key(object, object_end, key);
    if (!value || *value != '"') return 0;
    search_decode_string(value + 1, output, output_size);
    return 1;
}

static int parse_search_result_object(const char *object,
                                      const char *object_end,
                                      SearchResult *result) {
    result->title[0] = '\0';
    result->url[0] = '\0';
    result->description[0] = '\0';
    search_read_field(object, object_end, "title", result->title,
                      sizeof(result->title));
    search_read_field(object, object_end, "url", result->url,
                      sizeof(result->url));
    search_read_field(object, object_end, "description", result->description,
                      sizeof(result->description));
    strip_markup(result->title);
    strip_markup(result->description);
    return result->title[0] && result->url[0];
}

static int parse_search_results_array(const char *array_open,
                                      SearchResult *results, int count,
                                      int maximum_results) {
    const char *cursor;
    const char *item_end;

    cursor = array_open + 1;
    while (count < maximum_results && *cursor) {
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' ||
               *cursor == '\n' || *cursor == ',') cursor++;
        if (*cursor == ']') break;
        if (*cursor != '{') {
            if (!*cursor) break;
            cursor++;
            continue;
        }
        item_end = search_match_close(cursor, '{', '}');
        if (!item_end) break;
        if (parse_search_result_object(cursor, item_end,
                                       &results[count]))
            count++;
        cursor = item_end + 1;
    }
    return count;
}

static int parse_search_response(const char *json, SearchResult *results,
                                 int maximum_results, int *more_results) {
    const char *value;
    const char *results_end;
    const char *web_value;
    const char *web_end;
    const char *array_value;
    const char *array_end;

    if (more_results) {
        *more_results = 0;
        value = search_find_key(json, NULL, "more_results_available");
        if (value && strncmp(value, "true", 4) == 0) *more_results = 1;
    }
    if (!json || maximum_results <= 0) return 0;
    value = search_find_key(json, NULL, "results");
    if (!value) return 0;
    if (*value == '[') {
        array_end = search_match_close(value, '[', ']');
        if (!array_end) return 0;
        return parse_search_results_array(value, results, 0,
                                          maximum_results);
    }
    if (*value != '{') return 0;
    results_end = search_match_close(value, '{', '}');
    if (!results_end) return 0;
    web_value = search_find_key(value, results_end, "web");
    if (!web_value || *web_value != '{') return 0;
    web_end = search_match_close(web_value, '{', '}');
    if (!web_end) return 0;
    array_value = search_find_key(web_value, web_end, "results");
    if (!array_value || *array_value != '[') return 0;
    array_end = search_match_close(array_value, '[', ']');
    if (!array_end) return 0;
    return parse_search_results_array(array_value, results, 0,
                                      maximum_results);
}

static int request_search_json(const char *query, SearchBuffer *response,
                               const char *engine, int page) {
    CURL *curl;
    CURLcode result;
    struct curl_slist *headers;
    char escaped_query[2048];
    char request_body[2320];
    char page_number[16];
    long status_code;

    response->data = (char *)malloc(4096);
    if (!response->data) return 0;
    response->size = 0;
    response->capacity = 4096;
    response->data[0] = '\0';
    json_escape(query, escaped_query, sizeof(escaped_query));
    snprintf(page_number, sizeof(page_number), "%d", page < 0 ? 0 : page);
    if (engine && engine[0])
        snprintf(request_body, sizeof(request_body),
                 "{\"query\":\"%s\",\"type\":\"web\","
                 "\"page\":%s,\"engine\":\"%s\"}",
                 escaped_query, page_number, engine);
    else
        snprintf(request_body, sizeof(request_body),
                 "{\"query\":\"%s\",\"type\":\"web\","
                 "\"page\":%s}",
                 escaped_query, page_number);
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

static int fetch_search_page(const char *query, const char *engine, int page,
                             SearchResult *results, int count,
                             int maximum_results, int *more_results) {
    SearchBuffer response;
    int parsed;

    if (!request_search_json(query, &response, engine, page)) return -1;
    parsed = parse_search_response(response.data, results + count,
                                   maximum_results - count, more_results);
    free(response.data);
    if (parsed < 0) return -1;
    return count + parsed;
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
    int more_results;
    const char *engine;

    if (!web_search_enabled()) return 0;
    if (!query || !query[0] || !output || output_size <= 0) return -1;
    output[0] = '\0';
    select_query_text(query, selected_query, sizeof(selected_query));
    engine = "kagi";
    more_results = 0;
    result_count = fetch_search_page(selected_query, engine, 0, results, 0,
                                     SEARCH_RESULT_LIMIT, &more_results);
    if (result_count < 0) return -1;
    if (result_count <= 0) {
        engine = NULL;
        more_results = 0;
        result_count = fetch_search_page(selected_query, engine, 0, results,
                                         0, SEARCH_RESULT_LIMIT,
                                         &more_results);
        if (result_count < 0) return -1;
    }
    if (more_results && result_count > 0 &&
        result_count < SEARCH_RESULT_LIMIT) {
        int grown = fetch_search_page(selected_query, engine, 1, results,
                                      result_count, SEARCH_RESULT_LIMIT,
                                      &more_results);
        if (grown > result_count) result_count = grown;
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
    if (loaded_count > 0) search_tls_used = 1;
    return loaded_count;
}

int web_search_enabled(void) {
    if (search_tls_override_active) return search_tls_override_enabled;
    return search_runtime_enabled;
}

void web_search_set_enabled(int enabled) {
    search_runtime_enabled = enabled ? 1 : 0;
}

void web_search_set_local(int enabled) {
    search_tls_override_active = 1;
    search_tls_override_enabled = enabled ? 1 : 0;
}

void web_search_clear_local(void) {
    search_tls_override_active = 0;
    search_tls_override_enabled = 0;
}

int web_search_take_used(void) {
    int used;

    used = search_tls_used;
    search_tls_used = 0;
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
