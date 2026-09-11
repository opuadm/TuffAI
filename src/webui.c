#define _POSIX_C_SOURCE 200809L
#include "webui.h"
#include "webui_assets.h"
#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define WEBUI_REQUEST_MAX 1048576
#define WEBUI_HEADER_MAX 16384
#define WEBUI_LISTEN_BACKLOG 511
#define WEBUI_WORKER_STACK_SIZE (1024 * 1024)

typedef struct {
    char method[16];
    char path[256];
    char query[512];
    char authorization[512];
    char content_type[128];
    char *body;
    size_t body_length;
} HttpRequest;

typedef struct {
    int socket_fd;
    char id[96];
    const char *model;
    long created;
    int failed;
} CompletionStream;

typedef struct {
    int socket_fd;
    const WebUIHost *host;
} ConnectionWorker;

static unsigned long completion_sequence;
static pthread_mutex_t webui_host_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t webui_cors_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t webui_id_mutex = PTHREAD_MUTEX_INITIALIZER;
static int webui_cached_cors;
static sem_t webui_generation_slots;
static int webui_generation_slots_ready = 0;
static int webui_max_generations = 0;

void webui_set_concurrency(int max_generations) {
    if (max_generations < 1) max_generations = 1;
    if (max_generations > 1024) max_generations = 1024;
    webui_max_generations = max_generations;
}

static void webui_generation_acquire(void) {
    int result;

    do {
        result = sem_wait(&webui_generation_slots);
    } while (result != 0 && errno == EINTR);
}

static void webui_generation_release(void) {
    sem_post(&webui_generation_slots);
}

static int webui_default_concurrency(void) {
    long cpus;

    cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpus < 1) cpus = 1;
    if (cpus > 32) cpus = 32;
    return (int)cpus;
}

static int host_cors_enabled(const WebUIHost *host) {
    int enabled;

    (void)host;
    pthread_mutex_lock(&webui_cors_mutex);
    enabled = webui_cached_cors;
    pthread_mutex_unlock(&webui_cors_mutex);
    return enabled;
}

static void set_cached_cors(int enabled) {
    pthread_mutex_lock(&webui_cors_mutex);
    webui_cached_cors = enabled;
    pthread_mutex_unlock(&webui_cors_mutex);
}

static int send_all(int socket_fd, const char *data, size_t length) {
    ssize_t sent;
    size_t offset;

    offset = 0;
    while (offset < length) {
        sent = send(socket_fd, data + offset, length - offset, 0);
        if (sent < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (sent == 0) return 0;
        offset += (size_t)sent;
    }
    return 1;
}

static int text_case_equal_n(const char *left, const char *right,
                             size_t length) {
    size_t index;

    for (index = 0; index < length; index++) {
        if (tolower((unsigned char)left[index]) !=
            tolower((unsigned char)right[index]))
            return 0;
    }
    return 1;
}

static const char *find_header(const char *headers, const char *name) {
    const char *line;
    const char *end;
    size_t name_length;

    line = headers;
    name_length = strlen(name);
    while (*line) {
        end = strstr(line, "\r\n");
        if (!end) break;
        if ((size_t)(end - line) > name_length &&
            line[name_length] == ':' &&
            text_case_equal_n(line, name, name_length)) {
            line += name_length + 1;
            while (*line == ' ' || *line == '\t') line++;
            return line;
        }
        line = end + 2;
    }
    return NULL;
}

static int parse_content_length(const char *headers, size_t *length) {
    const char *value;
    char *end;
    unsigned long parsed;

    value = find_header(headers, "Content-Length");
    if (!value) {
        *length = 0;
        return 1;
    }
    errno = 0;
    parsed = strtoul(value, &end, 10);
    if (errno != 0 || end == value || parsed > WEBUI_REQUEST_MAX)
        return 0;
    *length = (size_t)parsed;
    return 1;
}

static int read_request(int socket_fd, HttpRequest *request) {
    char *buffer;
    char *header_end;
    const char *authorization;
    const char *authorization_end;
    const char *content_type;
    const char *content_type_end;
    char *path_end;
    char *query;
    ssize_t received;
    size_t capacity;
    size_t used;
    size_t header_length;
    size_t content_length;
    size_t present_body;
    size_t copy_length;

    memset(request, 0, sizeof(*request));
    capacity = WEBUI_HEADER_MAX;
    buffer = (char *)malloc(capacity + 1);
    if (!buffer) return 0;
    used = 0;
    header_end = NULL;
    while (!header_end && used < WEBUI_HEADER_MAX) {
        received = recv(socket_fd, buffer + used, capacity - used, 0);
        if (received < 0) {
            if (errno == EINTR) continue;
            free(buffer);
            return 0;
        }
        if (received == 0) {
            free(buffer);
            return 0;
        }
        used += (size_t)received;
        buffer[used] = '\0';
        header_end = strstr(buffer, "\r\n\r\n");
    }
    if (!header_end || sscanf(buffer, "%15s %255s", request->method,
                              request->path) != 2) {
        free(buffer);
        return 0;
    }
    query = strchr(request->path, '?');
    if (query) {
        *query = '\0';
        query++;
        snprintf(request->query, sizeof(request->query), "%s", query);
    } else {
        request->query[0] = '\0';
    }
    path_end = strchr(request->path, '#');
    if (path_end) *path_end = '\0';
    path_end = strchr(request->query, '#');
    if (path_end) *path_end = '\0';
    authorization = find_header(buffer, "Authorization");
    if (authorization) {
        authorization_end = strstr(authorization, "\r\n");
        if (!authorization_end) authorization_end = authorization;
        copy_length = (size_t)(authorization_end - authorization);
        if (copy_length >= sizeof(request->authorization))
            copy_length = sizeof(request->authorization) - 1;
        memcpy(request->authorization, authorization, copy_length);
        request->authorization[copy_length] = '\0';
    }
    content_type = find_header(buffer, "Content-Type");
    if (content_type) {
        content_type_end = strstr(content_type, "\r\n");
        if (!content_type_end) content_type_end = content_type;
        copy_length = (size_t)(content_type_end - content_type);
        if (copy_length >= sizeof(request->content_type))
            copy_length = sizeof(request->content_type) - 1;
        memcpy(request->content_type, content_type, copy_length);
        request->content_type[copy_length] = '\0';
    }
    if (!parse_content_length(buffer, &content_length)) {
        free(buffer);
        return 0;
    }
    header_length = (size_t)(header_end + 4 - buffer);
    present_body = used - header_length;
    request->body = (char *)malloc(content_length + 1);
    if (!request->body) {
        free(buffer);
        return 0;
    }
    if (present_body > content_length) present_body = content_length;
    memcpy(request->body, buffer + header_length, present_body);
    used = present_body;
    free(buffer);
    while (used < content_length) {
        received = recv(socket_fd, request->body + used,
                        content_length - used, 0);
        if (received < 0) {
            if (errno == EINTR) continue;
            free(request->body);
            request->body = NULL;
            return 0;
        }
        if (received == 0) {
            free(request->body);
            request->body = NULL;
            return 0;
        }
        used += (size_t)received;
    }
    request->body[content_length] = '\0';
    request->body_length = content_length;
    return 1;
}

static void free_request(HttpRequest *request) {
    free(request->body);
    request->body = NULL;
}

static const char *status_text(int status) {
    if (status == 200) return "OK";
    if (status == 204) return "No Content";
    if (status == 400) return "Bad Request";
    if (status == 401) return "Unauthorized";
    if (status == 403) return "Forbidden";
    if (status == 404) return "Not Found";
    if (status == 405) return "Method Not Allowed";
    if (status == 413) return "Payload Too Large";
    if (status == 500) return "Internal Server Error";
    return "Error";
}

static int send_response_data(int socket_fd, const WebUIHost *host, int status,
                              const char *content_type, const char *body,
                              size_t body_length) {
    char header[1024];
    const char *cors;
    int written;

    cors = host_cors_enabled(host) ?
           "Access-Control-Allow-Origin: *\r\n" : "";
    written = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %lu\r\n"
        "%s"
        "Access-Control-Allow-Headers: Authorization, Content-Type\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Cache-Control: no-store\r\n"
        "Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'none'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'\r\n"
        "Referrer-Policy: no-referrer\r\n"
        "X-Content-Type-Options: nosniff\r\n"
        "X-Frame-Options: DENY\r\n"
        "Connection: close\r\n\r\n",
        status, status_text(status), content_type,
        (unsigned long)body_length, cors);
    if (written < 0 || (size_t)written >= sizeof(header)) return 0;
    if (!send_all(socket_fd, header, (size_t)written)) return 0;
    if (body_length > 0 && !send_all(socket_fd, body, body_length)) return 0;
    return 1;
}

static int send_response(int socket_fd, const WebUIHost *host, int status,
                         const char *content_type, const char *body) {
    size_t body_length;

    body_length = body ? strlen(body) : 0;
    return send_response_data(socket_fd, host, status, content_type, body,
                              body_length);
}

static char *json_escape(const char *text) {
    char *output;
    char *cursor;
    const unsigned char *source;
    size_t length;
    size_t required;

    if (!text) text = "";
    required = 1;
    source = (const unsigned char *)text;
    while (*source) {
        if (*source == '"' || *source == '\\' || *source == '\n' ||
            *source == '\r' || *source == '\t' || *source == '<' ||
            *source == '>' || *source == '&')
            required += 6;
        else if (*source < 32)
            required += 6;
        else
            required++;
        source++;
    }
    output = (char *)malloc(required);
    if (!output) return NULL;
    cursor = output;
    source = (const unsigned char *)text;
    while (*source) {
        if (*source == '"' || *source == '\\') {
            *cursor++ = '\\';
            *cursor++ = (char)*source;
        } else if (*source == '<') {
            memcpy(cursor, "\\u003c", 6);
            cursor += 6;
        } else if (*source == '>') {
            memcpy(cursor, "\\u003e", 6);
            cursor += 6;
        } else if (*source == '&') {
            memcpy(cursor, "\\u0026", 6);
            cursor += 6;
        } else if (*source == '\n') {
            *cursor++ = '\\';
            *cursor++ = 'n';
        } else if (*source == '\r') {
            *cursor++ = '\\';
            *cursor++ = 'r';
        } else if (*source == '\t') {
            *cursor++ = '\\';
            *cursor++ = 't';
        } else if (*source < 32) {
            snprintf(cursor, 7, "\\u%04x", (unsigned int)*source);
            cursor += 6;
        } else {
            *cursor++ = (char)*source;
        }
        source++;
    }
    *cursor = '\0';
    length = (size_t)(cursor - output);
    output[length] = '\0';
    return output;
}

static const char *json_find_key(const char *json, const char *key) {
    const char *cursor;
    const char *after;
    size_t key_length;
    int escaped;

    if (!json || !key) return NULL;
    cursor = json;
    key_length = strlen(key);
    while (*cursor) {
        if (*cursor != '"') {
            cursor++;
            continue;
        }
        cursor++;
        after = cursor;
        escaped = 0;
        while (*after) {
            if (!escaped && *after == '"') break;
            if (!escaped && *after == '\\') escaped = 1;
            else escaped = 0;
            after++;
        }
        if (!*after) return NULL;
        if ((size_t)(after - cursor) == key_length &&
            memcmp(cursor, key, key_length) == 0) {
            after++;
            while (isspace((unsigned char)*after)) after++;
            if (*after == ':') {
                after++;
                while (isspace((unsigned char)*after)) after++;
                return after;
            }
        }
        cursor = after + 1;
    }
    return NULL;
}

static int hex_value(char character) {
    if (character >= '0' && character <= '9') return character - '0';
    if (character >= 'a' && character <= 'f') return character - 'a' + 10;
    if (character >= 'A' && character <= 'F') return character - 'A' + 10;
    return -1;
}

static void json_emit_utf8(char **cursor, int codepoint) {
    char *out;

    out = *cursor;
    if (codepoint < 0 || codepoint > 0x10FFFF ||
        (codepoint >= 0xD800 && codepoint <= 0xDFFF)) {
        *out++ = '?';
    } else if (codepoint < 0x80) {
        *out++ = (char)codepoint;
    } else if (codepoint < 0x800) {
        *out++ = (char)(0xC0 | (codepoint >> 6));
        *out++ = (char)(0x80 | (codepoint & 0x3F));
    } else if (codepoint < 0x10000) {
        *out++ = (char)(0xE0 | (codepoint >> 12));
        *out++ = (char)(0x80 | ((codepoint >> 6) & 0x3F));
        *out++ = (char)(0x80 | (codepoint & 0x3F));
    } else {
        *out++ = (char)(0xF0 | (codepoint >> 18));
        *out++ = (char)(0x80 | ((codepoint >> 12) & 0x3F));
        *out++ = (char)(0x80 | ((codepoint >> 6) & 0x3F));
        *out++ = (char)(0x80 | (codepoint & 0x3F));
    }
    *cursor = out;
}

static char *json_read_string(const char *value, const char **end) {
    char *output;
    char *cursor;
    const char *source;
    size_t capacity;
    int first;
    int second;
    int third;
    int fourth;
    int codepoint;

    if (!value || *value != '"') return NULL;
    capacity = strlen(value) + 1;
    output = (char *)malloc(capacity);
    if (!output) return NULL;
    source = value + 1;
    cursor = output;
    while (*source && *source != '"') {
        if (*source != '\\') {
            *cursor++ = *source++;
            continue;
        }
        source++;
        if (!*source) break;
        if (*source == 'n') *cursor++ = '\n';
        else if (*source == 'r') *cursor++ = '\r';
        else if (*source == 't') *cursor++ = '\t';
        else if (*source == 'b') *cursor++ = '\b';
        else if (*source == 'f') *cursor++ = '\f';
        else if (*source == 'u') {
            int low_first;
            int low_second;
            int low_third;
            int low_fourth;
            int low;
            first = hex_value(source[1]);
            second = hex_value(source[2]);
            third = hex_value(source[3]);
            fourth = hex_value(source[4]);
            if (first < 0 || second < 0 || third < 0 || fourth < 0) {
                free(output);
                return NULL;
            }
            codepoint = first * 4096 + second * 256 + third * 16 + fourth;
            source += 4;
            if (codepoint >= 0xD800 && codepoint <= 0xDBFF &&
                source[1] == '\\' && source[2] == 'u') {
                low_first = hex_value(source[3]);
                low_second = hex_value(source[4]);
                low_third = hex_value(source[5]);
                low_fourth = hex_value(source[6]);
                if (low_first >= 0 && low_second >= 0 && low_third >= 0 &&
                    low_fourth >= 0) {
                    low = low_first * 4096 + low_second * 256 +
                          low_third * 16 + low_fourth;
                    if (low >= 0xDC00 && low <= 0xDFFF) {
                        codepoint = 0x10000 +
                                    ((codepoint - 0xD800) << 10) +
                                    (low - 0xDC00);
                        source += 6;
                    }
                }
            }
            json_emit_utf8(&cursor, codepoint);
        } else {
            *cursor++ = *source;
        }
        source++;
    }
    if (*source != '"') {
        free(output);
        return NULL;
    }
    *cursor = '\0';
    if (end) *end = source + 1;
    return output;
}

static int json_read_number(const char *json, const char *key, float *value) {
    const char *source;
    char *end;
    double parsed;

    source = json_find_key(json, key);
    if (!source) return 0;
    errno = 0;
    parsed = strtod(source, &end);
    if (errno != 0 || end == source) return -1;
    *value = (float)parsed;
    return 1;
}

static int json_read_integer(const char *json, const char *key, int *value) {
    const char *source;
    char *end;
    long parsed;

    source = json_find_key(json, key);
    if (!source) return 0;
    errno = 0;
    parsed = strtol(source, &end, 10);
    if (errno != 0 || end == source || parsed < 0 || parsed > 2147483647L)
        return -1;
    *value = (int)parsed;
    return 1;
}

static int json_read_boolean(const char *json, const char *key, int *value) {
    const char *source;

    source = json_find_key(json, key);
    if (!source) return 0;
    if (strncmp(source, "true", 4) == 0) {
        *value = 1;
        return 1;
    }
    if (strncmp(source, "false", 5) == 0) {
        *value = 0;
        return 1;
    }
    return -1;
}

static int append_text(char **buffer, size_t *capacity, size_t *length,
                       const char *text) {
    char *grown;
    size_t added;
    size_t required;
    size_t new_capacity;

    if (!text) return 1;
    added = strlen(text);
    required = *length + added + 1;
    if (required < *length) return 0;
    if (required > *capacity) {
        new_capacity = *capacity ? *capacity : 256;
        while (new_capacity < required) new_capacity *= 2;
        grown = (char *)realloc(*buffer, new_capacity);
        if (!grown) return 0;
        *buffer = grown;
        *capacity = new_capacity;
    }
    memcpy(*buffer + *length, text, added + 1);
    *length += added;
    return 1;
}

static const char *find_matching(const char *start, char open, char close) {
    const char *cursor;
    int depth;
    int in_string;
    int escaped;

    if (!start || *start != open) return NULL;
    cursor = start;
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
        } else if (*cursor == open) {
            depth++;
        } else if (*cursor == close) {
            depth--;
            if (depth == 0) return cursor;
        }
        cursor++;
    }
    return NULL;
}

static char *extract_message_content(const char *object) {
    const char *content;
    const char *array_end;
    const char *cursor;
    const char *text_value;
    char *part;
    char *output;
    size_t capacity;
    size_t length;

    content = json_find_key(object, "content");
    if (!content) return NULL;
    if (*content == '"') return json_read_string(content, NULL);
    if (*content != '[') return NULL;
    array_end = find_matching(content, '[', ']');
    if (!array_end) return NULL;
    output = NULL;
    capacity = 0;
    length = 0;
    cursor = content + 1;
    while (cursor < array_end) {
        text_value = json_find_key(cursor, "text");
        if (!text_value || text_value >= array_end) break;
        part = json_read_string(text_value, &cursor);
        if (!part) break;
        if (length > 0) append_text(&output, &capacity, &length, "\n");
        if (!append_text(&output, &capacity, &length, part)) {
            free(part);
            free(output);
            return NULL;
        }
        free(part);
    }
    if (!output) {
        output = (char *)malloc(1);
        if (output) output[0] = '\0';
    }
    return output;
}

static int parse_messages(const char *json, char **input,
                          char **system_prompt) {
    const char *messages;
    const char *array_end;
    const char *cursor;
    const char *object_end;
    const char *role_value;
    char *object;
    char *role;
    char *content;
    char *conversation;
    char *system;
    size_t object_length;
    size_t conversation_capacity;
    size_t conversation_length;
    size_t system_capacity;
    size_t system_length;

    *input = NULL;
    *system_prompt = NULL;
    messages = json_find_key(json, "messages");
    if (!messages || *messages != '[') return 0;
    array_end = find_matching(messages, '[', ']');
    if (!array_end) return 0;
    conversation = NULL;
    system = NULL;
    conversation_capacity = 0;
    conversation_length = 0;
    system_capacity = 0;
    system_length = 0;
    cursor = messages + 1;
    while (cursor < array_end) {
        while (cursor < array_end && *cursor != '{') cursor++;
        if (cursor >= array_end) break;
        object_end = find_matching(cursor, '{', '}');
        if (!object_end || object_end > array_end) break;
        object_length = (size_t)(object_end - cursor + 1);
        object = (char *)malloc(object_length + 1);
        if (!object) goto failure;
        memcpy(object, cursor, object_length);
        object[object_length] = '\0';
        role_value = json_find_key(object, "role");
        role = role_value ? json_read_string(role_value, NULL) : NULL;
        content = extract_message_content(object);
        free(object);
        if (!role || !content) {
            free(role);
            free(content);
            goto failure;
        }
        if (strcmp(role, "system") == 0 || strcmp(role, "developer") == 0) {
            if (system_length > 0)
                append_text(&system, &system_capacity, &system_length, "\n");
            if (!append_text(&system, &system_capacity, &system_length,
                             content)) {
                free(role);
                free(content);
                goto failure;
            }
        } else {
            if (conversation_length > 0)
                append_text(&conversation, &conversation_capacity,
                            &conversation_length, "\n");
            if (strcmp(role, "assistant") == 0)
                append_text(&conversation, &conversation_capacity,
                            &conversation_length, "Assistant: ");
            else if (strcmp(role, "tool") == 0)
                append_text(&conversation, &conversation_capacity,
                            &conversation_length, "Tool: ");
            else
                append_text(&conversation, &conversation_capacity,
                            &conversation_length, "User: ");
            if (!append_text(&conversation, &conversation_capacity,
                             &conversation_length, content)) {
                free(role);
                free(content);
                goto failure;
            }
        }
        free(role);
        free(content);
        cursor = object_end + 1;
    }
    if (!conversation || conversation_length == 0) goto failure;
    if (!system) {
        system = (char *)malloc(1);
        if (!system) goto failure;
        system[0] = '\0';
    }
    *input = conversation;
    *system_prompt = system;
    return 1;

failure:
    free(conversation);
    free(system);
    return 0;
}

static int parse_completion_request(const char *json,
                                    WebUICompletionRequest *request,
                                    int *stream, char *error,
                                    size_t error_size) {
    const char *model_value;
    const char *effort_value;
    char *model;
    char *effort;
    int parsed;

    memset(request, 0, sizeof(*request));
    *stream = 0;
    if (!json || !parse_messages(json, &request->input,
                                 &request->system_prompt)) {
        snprintf(error, error_size,
                 "messages must be a non-empty array with text content");
        return 0;
    }
    model_value = json_find_key(json, "model");
    if (model_value && *model_value == '"') {
        model = json_read_string(model_value, NULL);
        if (!model) goto invalid_json;
        snprintf(request->model, sizeof(request->model), "%s", model);
        free(model);
    }
    parsed = json_read_number(json, "temperature", &request->temperature);
    if (parsed < 0) goto invalid_json;
    request->has_temperature = parsed > 0;
    parsed = json_read_number(json, "noise", &request->noise);
    if (parsed < 0) goto invalid_json;
    request->has_noise = parsed > 0;
    parsed = json_read_number(json, "top_p", &request->top_p);
    if (parsed < 0) goto invalid_json;
    request->has_top_p = parsed > 0;
    parsed = json_read_number(json, "frequency_penalty",
                              &request->frequency_penalty);
    if (parsed < 0) goto invalid_json;
    request->has_frequency_penalty = parsed > 0;
    parsed = json_read_number(json, "repetition_penalty",
                              &request->repetition_penalty);
    if (parsed < 0) goto invalid_json;
    request->has_repetition_penalty = parsed > 0;
    parsed = json_read_number(json, "presence_penalty",
                              &request->presence_penalty);
    if (parsed < 0) goto invalid_json;
    request->has_presence_penalty = parsed > 0;
    parsed = json_read_integer(json, "max_completion_tokens",
                               &request->max_tokens);
    if (parsed == 0)
        parsed = json_read_integer(json, "max_tokens", &request->max_tokens);
    if (parsed < 0) goto invalid_json;
    request->has_max_tokens = parsed > 0;
    parsed = json_read_boolean(json, "web_search", &request->web_search);
    if (parsed < 0) goto invalid_json;
    request->has_web_search = parsed > 0;
    parsed = json_read_boolean(json, "code_mode", &request->code_mode);
    if (parsed < 0) goto invalid_json;
    request->has_code_mode = parsed > 0;
    effort_value = json_find_key(json, "reasoning_effort");
    if (effort_value) {
        effort = json_read_string(effort_value, NULL);
        if (!effort) goto invalid_json;
        snprintf(request->reasoning_effort,
                 sizeof(request->reasoning_effort), "%s", effort);
        free(effort);
        request->has_reasoning_effort = 1;
    }
    parsed = json_read_boolean(json, "stream", stream);
    if (parsed < 0) goto invalid_json;
    return 1;

invalid_json:
    free(request->input);
    free(request->system_prompt);
    request->input = NULL;
    request->system_prompt = NULL;
    snprintf(error, error_size, "invalid JSON request value");
    return 0;
}

static void free_completion_request(WebUICompletionRequest *request) {
    free(request->input);
    free(request->system_prompt);
    request->input = NULL;
    request->system_prompt = NULL;
}

static char *error_json(const char *message, const char *type) {
    char *escaped;
    char *output;
    size_t required;

    escaped = json_escape(message);
    if (!escaped) return NULL;
    required = strlen(escaped) + strlen(type) + 96;
    output = (char *)malloc(required);
    if (output)
        snprintf(output, required,
                 "{\"error\":{\"message\":\"%s\",\"type\":\"%s\","
                 "\"param\":null,\"code\":null}}", escaped, type);
    free(escaped);
    return output;
}

static int send_error(int socket_fd, const WebUIHost *host, int status,
                      const char *message, const char *type) {
    char *body;
    int result;

    body = error_json(message, type);
    if (!body) return send_response(socket_fd, host, 500,
                                    "application/json", "{}");
    result = send_response(socket_fd, host, status, "application/json", body);
    free(body);
    return result;
}

static int build_models_json(const WebUIHost *host, char **output) {
    WebUIModelInfo info;
    char item[512];
    char *buffer;
    size_t capacity;
    size_t length;
    int count;
    int index;

    buffer = NULL;
    capacity = 0;
    length = 0;
    if (!append_text(&buffer, &capacity, &length,
                     "{\"object\":\"list\",\"data\":["))
        return 0;
    count = host->model_count();
    for (index = 0; index < count; index++) {
        if (!host->model_info(index, &info)) continue;
        snprintf(item, sizeof(item),
                 "%s{\"id\":\"%s\",\"object\":\"model\","
                 "\"created\":0,\"owned_by\":\"tuffai\"}",
                 index == 0 ? "" : ",", info.id);
        if (!append_text(&buffer, &capacity, &length, item)) {
            free(buffer);
            return 0;
        }
    }
    if (!append_text(&buffer, &capacity, &length, "]}")) {
        free(buffer);
        return 0;
    }
    *output = buffer;
    return 1;
}

static void make_completion_id(char *output, size_t output_size) {
    time_t now;

    pthread_mutex_lock(&webui_id_mutex);
    now = time(NULL);
    completion_sequence++;
    snprintf(output, output_size, "chatcmpl-tuffai-%lx-%lu",
             (unsigned long)now, completion_sequence);
    pthread_mutex_unlock(&webui_id_mutex);
}

static int send_completion_json(int socket_fd, const WebUIHost *host,
                                const WebUICompletionRequest *request,
                                const WebUICompletionResult *result) {
    char id[96];
    char *escaped_content;
    char *escaped_thinking;
    char *escaped_model;
    char *body;
    size_t required;
    long created;
    int sent;

    make_completion_id(id, sizeof(id));
    escaped_content = json_escape(result->content);
    escaped_thinking = json_escape(result->thinking);
    escaped_model = json_escape(request->model);
    if (!escaped_content || !escaped_thinking || !escaped_model) {
        free(escaped_content);
        free(escaped_thinking);
        free(escaped_model);
        return 0;
    }
    created = (long)time(NULL);
    required = strlen(escaped_content) + strlen(escaped_thinking) +
               strlen(escaped_model) + 800;
    body = (char *)malloc(required);
    if (!body) {
        free(escaped_content);
        free(escaped_thinking);
        free(escaped_model);
        return 0;
    }
    snprintf(body, required,
        "{\"id\":\"%s\",\"object\":\"chat.completion\","
        "\"created\":%ld,\"model\":\"%s\","
        "\"choices\":[{\"index\":0,\"message\":{\"role\":"
        "\"assistant\",\"content\":\"%s\","
        "\"reasoning_content\":\"%s\",\"refusal\":null},"
        "\"logprobs\":null,\"finish_reason\":\"stop\"}],"
        "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,"
        "\"total_tokens\":%d}}",
        id, created, escaped_model, escaped_content, escaped_thinking,
        result->prompt_tokens, result->completion_tokens,
        result->prompt_tokens + result->completion_tokens);
    sent = send_response(socket_fd, host, 200, "application/json", body);
    free(body);
    free(escaped_content);
    free(escaped_thinking);
    free(escaped_model);
    return sent;
}

static int send_sse_line(int socket_fd, const char *json) {
    if (!send_all(socket_fd, "data: ", 6)) return 0;
    if (!send_all(socket_fd, json, strlen(json))) return 0;
    return send_all(socket_fd, "\n\n", 2);
}

static int begin_completion_stream(int socket_fd, const WebUIHost *host,
                                   const WebUICompletionRequest *request,
                                   CompletionStream *stream) {
    char header[1024];
    char chunk[1024];
    const char *cors;
    int written;

    memset(stream, 0, sizeof(*stream));
    stream->socket_fd = socket_fd;
    stream->model = request->model;
    stream->created = (long)time(NULL);
    make_completion_id(stream->id, sizeof(stream->id));
    cors = host_cors_enabled(host) ?
           "Access-Control-Allow-Origin: *\r\n" : "";
    written = snprintf(header, sizeof(header),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "%s"
        "Content-Security-Policy: default-src 'none'; frame-ancestors 'none'\r\n"
        "Referrer-Policy: no-referrer\r\n"
        "X-Content-Type-Options: nosniff\r\n"
        "X-Frame-Options: DENY\r\n"
        "Connection: close\r\n\r\n", cors);
    if (written < 0 || !send_all(socket_fd, header, (size_t)written)) return 0;
    snprintf(chunk, sizeof(chunk),
        "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"created\":%ld,\"model\":\"%s\",\"choices\":[{"
        "\"index\":0,\"delta\":{\"role\":\"assistant\","
        "\"content\":\"\"},\"logprobs\":null,"
        "\"finish_reason\":null}]}", stream->id, stream->created,
        request->model);
    return send_sse_line(socket_fd, chunk);
}

static int emit_completion_stream(const char *text, int reasoning,
                                  void *context) {
    CompletionStream *stream;
    char *escaped;
    char *chunk;
    const char *field;
    size_t required;

    stream = (CompletionStream *)context;
    if (!stream || stream->failed) return 0;
    if (!text || !text[0]) return 1;
    escaped = json_escape(text);
    if (!escaped) {
        stream->failed = 1;
        return 0;
    }
    field = reasoning ? "reasoning_content" : "content";
    required = strlen(escaped) + strlen(stream->model) + 512;
    chunk = (char *)malloc(required);
    if (!chunk) {
        free(escaped);
        stream->failed = 1;
        return 0;
    }
    snprintf(chunk, required,
        "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"created\":%ld,\"model\":\"%s\",\"choices\":[{"
        "\"index\":0,\"delta\":{\"%s\":\"%s\"},"
        "\"logprobs\":null,\"finish_reason\":null}]}",
        stream->id, stream->created, stream->model, field, escaped);
    if (!send_sse_line(stream->socket_fd, chunk)) stream->failed = 1;
    free(chunk);
    free(escaped);
    return !stream->failed;
}

static int finish_completion_stream(CompletionStream *stream) {
    char chunk[1024];

    if (stream->failed) return 0;
    snprintf(chunk, sizeof(chunk),
        "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"created\":%ld,\"model\":\"%s\",\"choices\":[{"
        "\"index\":0,\"delta\":{},\"logprobs\":null,"
        "\"finish_reason\":\"stop\"}]}", stream->id, stream->created,
        stream->model);
    if (!send_sse_line(stream->socket_fd, chunk)) return 0;
    return send_sse_line(stream->socket_fd, "[DONE]");
}

static int fail_completion_stream(CompletionStream *stream,
                                  const char *message) {
    char *escaped;
    char *event;
    size_t required;
    int sent;

    if (stream->failed) return 0;
    escaped = json_escape(message);
    if (!escaped) return 0;
    required = strlen(escaped) + 64;
    event = (char *)malloc(required);
    if (!event) {
        free(escaped);
        return 0;
    }
    snprintf(event, required, "{\"error\":{\"message\":\"%s\"}}",
             escaped);
    sent = send_sse_line(stream->socket_fd, event);
    if (sent) sent = send_sse_line(stream->socket_fd, "[DONE]");
    free(event);
    free(escaped);
    return sent;
}

static int route_completion(int socket_fd, const WebUIHost *host,
                            const HttpRequest *http_request) {
    WebUICompletionRequest request;
    WebUICompletionResult result;
    char error[WEBUI_ERROR_MAX];
    CompletionStream completion_stream;
    int stream;
    int sent;

    memset(&result, 0, sizeof(result));
    if (!parse_completion_request(http_request->body, &request, &stream,
                                  error, sizeof(error)))
        return send_error(socket_fd, host, 400, error,
                          "invalid_request_error");
    if (stream) {
        if (!begin_completion_stream(socket_fd, host, &request,
                                     &completion_stream)) {
            free_completion_request(&request);
            return 0;
        }
        request.stream_callback = emit_completion_stream;
        request.stream_context = &completion_stream;
    }
    webui_generation_acquire();
    sent = host->complete(&request, &result, error, sizeof(error));
    webui_generation_release();
    if (!sent) {
        free_completion_request(&request);
        if (stream) return fail_completion_stream(&completion_stream, error);
        return send_error(socket_fd, host, 400, error,
                          "invalid_request_error");
    }
    if (stream)
        sent = finish_completion_stream(&completion_stream);
    else
        sent = send_completion_json(socket_fd, host, &request, &result);
    free(result.content);
    free(result.thinking);
    free_completion_request(&request);
    return sent;
}

static int route_request(int socket_fd, const WebUIHost *host,
                         const HttpRequest *request) {
    char settings[65536];
    char admin_output[65536];
    char error[WEBUI_ERROR_MAX];
    char *models_json;
    int status;
    int admin_status;

    if (strcmp(request->method, "OPTIONS") == 0)
        return send_response(socket_fd, host, 204, "text/plain", "");
    if (strcmp(request->path, "/") == 0 &&
        strcmp(request->method, "GET") == 0)
        return send_response_data(socket_fd, host, 200,
                                  "text/html; charset=utf-8",
                                  (const char *)tuffai_web_index_html,
                                  tuffai_web_index_html_len);
    if (strcmp(request->path, "/assets/app.css") == 0 &&
        strcmp(request->method, "GET") == 0)
        return send_response_data(socket_fd, host, 200,
                                  "text/css; charset=utf-8",
                                  (const char *)tuffai_web_app_css,
                                  tuffai_web_app_css_len);
    if (strcmp(request->path, "/assets/app.js") == 0 &&
        strcmp(request->method, "GET") == 0)
        return send_response_data(socket_fd, host, 200,
                                  "application/javascript; charset=utf-8",
                                  (const char *)tuffai_web_app_js,
                                  tuffai_web_app_js_len);
    if (strcmp(request->path, "/health") == 0 &&
        strcmp(request->method, "GET") == 0)
        return send_response(socket_fd, host, 200, "application/json",
                             "{\"status\":\"ok\"}");
    if ((strncmp(request->path, "/api/admin/", 11) == 0 ||
         strncmp(request->path, "/api/auth/", 10) == 0) &&
        host->admin_handle) {
        pthread_mutex_lock(&webui_host_mutex);
        status = host->admin_handle(request->method, request->path,
                                    request->query,
                                    request->body ? request->body : "",
                                    request->authorization, admin_output,
                                    sizeof(admin_output), &admin_status);
        pthread_mutex_unlock(&webui_host_mutex);
        if (status)
            return send_response(socket_fd, host, admin_status,
                                 "application/json", admin_output);
        return send_error(socket_fd, host, 404, "route not found",
                          "invalid_request_error");
    }
    if (strcmp(request->path, "/api/settings") == 0) {
        if (host->admin_authorize) {
            pthread_mutex_lock(&webui_host_mutex);
            status = host->admin_authorize(request->authorization);
            pthread_mutex_unlock(&webui_host_mutex);
            if (!status)
                return send_error(socket_fd, host, 401,
                                  "admin authentication required",
                                  "authentication_error");
        }
        if (strcmp(request->method, "GET") == 0) {
            pthread_mutex_lock(&webui_host_mutex);
            status = host->settings_json(settings, sizeof(settings));
            pthread_mutex_unlock(&webui_host_mutex);
            if (!status)
                return send_error(socket_fd, host, 500,
                                  "unable to serialize settings",
                                  "server_error");
            return send_response(socket_fd, host, 200, "application/json",
                                 settings);
        }
        if (strcmp(request->method, "POST") == 0) {
            if (strncmp(request->content_type, "application/json", 16) != 0)
                return send_error(socket_fd, host, 400,
                                  "Content-Type must be application/json",
                                  "invalid_request_error");
            pthread_mutex_lock(&webui_host_mutex);
            status = host->apply_settings(request->body, error,
                                          sizeof(error));
            if (status) {
                set_cached_cors(host->cors_enabled && host->cors_enabled());
                status = host->settings_json(settings, sizeof(settings));
            }
            pthread_mutex_unlock(&webui_host_mutex);
            if (!status)
                return send_error(socket_fd, host, 400, error,
                                  "invalid_request_error");
            return send_response(socket_fd, host, 200, "application/json",
                                 settings);
        }
        return send_error(socket_fd, host, 405, "method not allowed",
                          "invalid_request_error");
    }
    if (strncmp(request->path, "/v1/", 4) == 0 && host->authorize) {
        pthread_mutex_lock(&webui_host_mutex);
        status = host->authorize(request->authorization);
        pthread_mutex_unlock(&webui_host_mutex);
        if (!status)
            return send_error(socket_fd, host, 401,
                              "invalid or missing API key",
                              "authentication_error");
    }
    if (strcmp(request->path, "/v1/models") == 0 &&
        strcmp(request->method, "GET") == 0) {
        models_json = NULL;
        if (!build_models_json(host, &models_json))
            return send_error(socket_fd, host, 500,
                              "unable to list models", "server_error");
        status = send_response(socket_fd, host, 200, "application/json",
                               models_json);
        free(models_json);
        return status;
    }
    if (strcmp(request->path, "/v1/chat/completions") == 0 &&
        strcmp(request->method, "POST") == 0) {
        if (strncmp(request->content_type, "application/json", 16) != 0)
            return send_error(socket_fd, host, 400,
                              "Content-Type must be application/json",
                              "invalid_request_error");
        return route_completion(socket_fd, host, request);
    }
    if (strcmp(request->path, "/v1/chat/completions") == 0 ||
        strcmp(request->path, "/v1/models") == 0)
        return send_error(socket_fd, host, 405, "method not allowed",
                          "invalid_request_error");
    return send_error(socket_fd, host, 404, "route not found",
                      "invalid_request_error");
}

static void handle_connection(int socket_fd, const WebUIHost *host) {
    HttpRequest request;
    struct timeval timeout;

    timeout.tv_sec = 15;
    timeout.tv_usec = 0;
    setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
               sizeof(timeout));
    setsockopt(socket_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout,
               sizeof(timeout));
    if (read_request(socket_fd, &request)) {
        route_request(socket_fd, host, &request);
        free_request(&request);
    } else {
        send_error(socket_fd, host, 400, "invalid HTTP request",
                   "invalid_request_error");
    }
    close(socket_fd);
}

static void *connection_worker_main(void *context) {
    ConnectionWorker *worker;
    int socket_fd;
    const WebUIHost *host;

    worker = (ConnectionWorker *)context;
    socket_fd = worker->socket_fd;
    host = worker->host;
    free(worker);
    handle_connection(socket_fd, host);
    return NULL;
}

static void start_connection_worker(int socket_fd, const WebUIHost *host) {
    ConnectionWorker *worker;
    pthread_t thread;
    pthread_attr_t attributes;
    int attributes_ready;
    int created;

    worker = (ConnectionWorker *)malloc(sizeof(*worker));
    if (!worker) {
        handle_connection(socket_fd, host);
        return;
    }
    worker->socket_fd = socket_fd;
    worker->host = host;
    attributes_ready = pthread_attr_init(&attributes) == 0;
    if (attributes_ready) {
        if (pthread_attr_setstacksize(&attributes,
                                      WEBUI_WORKER_STACK_SIZE) != 0)
            attributes_ready = 0;
    }
    if (attributes_ready)
        created = pthread_create(&thread, &attributes,
                                 connection_worker_main, worker);
    else
        created = pthread_create(&thread, NULL, connection_worker_main,
                                 worker);
    if (attributes_ready) pthread_attr_destroy(&attributes);
    if (created != 0) {
        free(worker);
        handle_connection(socket_fd, host);
        return;
    }
    pthread_detach(thread);
}

static void set_listener_nonblocking(int server_fd) {
    int flags;

    flags = fcntl(server_fd, F_GETFL, 0);
    if (flags < 0) return;
    fcntl(server_fd, F_SETFL, flags | O_NONBLOCK);
}

static int create_ipv4_listener(int port) {
    struct sockaddr_in address;
    int server_fd;
    int reuse;

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) return -1;
    reuse = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons((unsigned short)port);
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        fprintf(stderr, "Unable to bind WebUI to port %d: %s\n", port,
                strerror(errno));
        close(server_fd);
        return -1;
    }
    if (listen(server_fd, WEBUI_LISTEN_BACKLOG) < 0) {
        perror("listen");
        close(server_fd);
        return -1;
    }
    return server_fd;
}

static int create_ipv6_listener(int port) {
    struct sockaddr_in6 address;
    int server_fd;
    int reuse;
    int ipv6_only;

    server_fd = socket(AF_INET6, SOCK_STREAM, 0);
    if (server_fd < 0) return -1;
    reuse = 1;
    ipv6_only = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    setsockopt(server_fd, IPPROTO_IPV6, IPV6_V6ONLY, &ipv6_only,
               sizeof(ipv6_only));
    memset(&address, 0, sizeof(address));
    address.sin6_family = AF_INET6;
    address.sin6_addr = in6addr_loopback;
    address.sin6_port = htons((unsigned short)port);
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0 ||
        listen(server_fd, WEBUI_LISTEN_BACKLOG) < 0) {
        close(server_fd);
        return -1;
    }
    return server_fd;
}

int webui_run(int port, const WebUIHost *host) {
    struct pollfd listeners[2];
    int listener_count;
    int ipv4_fd;
    int ipv6_fd;
    int client_fd;
    int ready;
    int index;

    if (!host || !host->model_count || !host->model_info ||
        !host->complete || !host->settings_json || !host->apply_settings)
        return 1;
    signal(SIGPIPE, SIG_IGN);
    if (webui_max_generations <= 0)
        webui_max_generations = webui_default_concurrency();
    if (!webui_generation_slots_ready) {
        if (sem_init(&webui_generation_slots, 0,
                     (unsigned int)webui_max_generations) != 0) {
            perror("sem_init");
            return 1;
        }
        webui_generation_slots_ready = 1;
    }
    pthread_mutex_lock(&webui_host_mutex);
    set_cached_cors(host->cors_enabled && host->cors_enabled());
    pthread_mutex_unlock(&webui_host_mutex);
    ipv4_fd = create_ipv4_listener(port);
    if (ipv4_fd < 0) return 1;
    ipv6_fd = create_ipv6_listener(port);
    set_listener_nonblocking(ipv4_fd);
    if (ipv6_fd >= 0) set_listener_nonblocking(ipv6_fd);
    listener_count = 1;
    listeners[0].fd = ipv4_fd;
    listeners[0].events = POLLIN;
    listeners[0].revents = 0;
    if (ipv6_fd >= 0) {
        listeners[1].fd = ipv6_fd;
        listeners[1].events = POLLIN;
        listeners[1].revents = 0;
        listener_count = 2;
    }
    printf("TuffAI WebUI: http://localhost:%d\n", port);
    printf("OpenAI-compatible API: http://localhost:%d/v1\n", port);
    printf("Concurrent generations: %d\n", webui_max_generations);
    fflush(stdout);
    for (;;) {
        ready = poll(listeners, (nfds_t)listener_count, -1);
        if (ready < 0) {
            if (errno == EINTR) continue;
            perror("poll");
            close(ipv4_fd);
            if (ipv6_fd >= 0) close(ipv6_fd);
            return 1;
        }
        for (index = 0; index < listener_count; index++) {
            if (!(listeners[index].revents & POLLIN)) continue;
            for (;;) {
                client_fd = accept(listeners[index].fd, NULL, NULL);
                if (client_fd < 0) break;
                start_connection_worker(client_fd, host);
            }
        }
    }
}
