#define _POSIX_C_SOURCE 200809L
#include "code_mode.h"
#include "../../rng.h"
#include "../../tokenizer.h"
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

static int contains_case(const char *text, const char *needle) {
    int text_position;
    int needle_position;
    unsigned char left;
    unsigned char right;

    if (!text || !needle || !needle[0]) return 0;
    for (text_position = 0; text[text_position]; text_position++) {
        needle_position = 0;
        while (needle[needle_position] && text[text_position + needle_position]) {
            left = (unsigned char)text[text_position + needle_position];
            right = (unsigned char)needle[needle_position];
            if (tolower(left) != tolower(right)) break;
            needle_position++;
        }
        if (!needle[needle_position]) return 1;
    }
    return 0;
}

static int safe_filename(const char *filename) {
    int i;
    unsigned char character;

    if (!filename || !filename[0] || filename[0] == '.') return 0;
    if (strstr(filename, "..") != NULL) return 0;
    for (i = 0; filename[i]; i++) {
        character = (unsigned char)filename[i];
        if (!isalnum(character) && character != '.' && character != '_' &&
            character != '-')
            return 0;
    }
    return i < 128;
}

static int extract_filename(const char *input, char *filename,
                            int filename_size) {
    char copy[512];
    char *word;
    int length;

    strncpy(copy, input, sizeof(copy) - 1);
    copy[sizeof(copy) - 1] = '\0';
    word = strtok(copy, " \t\r\n\"'`<>()[]{}:,;");
    while (word) {
        length = (int)strlen(word);
        while (length > 0 && (word[length - 1] == '.' ||
               word[length - 1] == '?' || word[length - 1] == '!'))
            word[--length] = '\0';
        if (strchr(word, '.') && safe_filename(word)) {
            if (length >= filename_size) length = filename_size - 1;
            memcpy(filename, word, (size_t)length);
            filename[length] = '\0';
            return 1;
        }
        word = strtok(NULL, " \t\r\n\"'`<>()[]{}:,;");
    }
    return 0;
}

static const char *extension_of(const char *filename) {
    const char *extension;

    extension = strrchr(filename, '.');
    return extension ? extension : "";
}

static void default_filename(const char *input, char *filename,
                             int filename_size) {
    const char *selected;

    selected = "tuffai_output.c";
    if (contains_case(input, "python")) selected = "tuffai_output.py";
    else if (contains_case(input, "javascript") ||
             contains_case(input, "node")) selected = "tuffai_output.js";
    else if (contains_case(input, "rust")) selected = "tuffai_output.rs";
    else if (contains_case(input, "golang") || contains_case(input, " go "))
        selected = "tuffai_output.go";
    else if (contains_case(input, "html")) selected = "tuffai_output.html";
    else if (contains_case(input, "shell") || contains_case(input, "bash"))
        selected = "tuffai_output.sh";
    strncpy(filename, selected, (size_t)filename_size - 1);
    filename[filename_size - 1] = '\0';
}

static int requested_line_count(const char *input) {
    const char *cursor;
    const char *unit;
    char *number_end;
    long count;

    cursor = input;
    while (*cursor) {
        if (!isdigit((unsigned char)*cursor)) {
            cursor++;
            continue;
        }
        count = strtol(cursor, &number_end, 10);
        unit = number_end;
        while (*unit && (isspace((unsigned char)*unit) || *unit == '-'))
            unit++;
        if (strncasecmp(unit, "line", 4) == 0 && count > 0) {
            if (count >= INT_MAX) return INT_MAX;
            if (contains_case(input, "over") ||
                contains_case(input, "more than"))
                count++;
            return (int)count;
        }
        cursor = number_end > cursor ? number_end : cursor + 1;
    }
    return 0;
}

static int count_lines(const char *text) {
    int lines;
    int i;

    if (!text || !text[0]) return 0;
    lines = 0;
    for (i = 0; text[i]; i++)
        if (text[i] == '\n') lines++;
    if (i > 0 && text[i - 1] != '\n') lines++;
    return lines;
}

static int append_code(char **content, size_t *length, size_t *capacity,
                       const char *snippet) {
    char *grown;
    size_t snippet_length;
    size_t separator_length;
    size_t required;
    size_t new_capacity;

    if (!snippet || !snippet[0]) return 1;
    snippet_length = strlen(snippet);
    separator_length = *length > 0 ? 2 : 0;
    required = *length + separator_length + snippet_length + 2;
    if (required < *length || required < snippet_length) return 0;
    if (required > *capacity) {
        new_capacity = *capacity ? *capacity : 4096;
        while (new_capacity < required) {
            if (new_capacity > (size_t)-1 / 2) {
                new_capacity = required;
                break;
            }
            new_capacity *= 2;
        }
        grown = (char *)realloc(*content, new_capacity);
        if (!grown) return 0;
        *content = grown;
        *capacity = new_capacity;
    }
    if (separator_length) {
        (*content)[(*length)++] = '\n';
        (*content)[(*length)++] = '\n';
    }
    memcpy(*content + *length, snippet, snippet_length);
    *length += snippet_length;
    if ((*content)[*length - 1] != '\n') (*content)[(*length)++] = '\n';
    (*content)[*length] = '\0';
    return 1;
}

static const KnowledgeCategory *category_for_extension(const char *extension) {
    if (strcmp(extension, ".c") == 0 || strcmp(extension, ".h") == 0)
        return &v3_know_code_c;
    if (strcmp(extension, ".py") == 0)
        return &v3_know_code_py;
    if (strcmp(extension, ".js") == 0 || strcmp(extension, ".ts") == 0)
        return &v3_know_code_js;
    return &v3_know_code_misc;
}

static int append_category(const KnowledgeCategory *category,
                           char **content, size_t *length, size_t *capacity,
                           int target_lines, int target_snippets,
                           int *snippet_count) {
    int *order;
    int temporary;
    int selected;
    int i;

    order = (int *)malloc((size_t)category->count * sizeof(int));
    if (!order) return 0;
    for (i = 0; i < category->count; i++) order[i] = i;
    for (i = category->count - 1; i > 0; i--) {
        selected = rng_range(i + 1);
        temporary = order[i];
        order[i] = order[selected];
        order[selected] = temporary;
    }
    for (i = 0; i < category->count; i++) {
        selected = order[i];
        if (!category->entries[selected].assistant) continue;
        if (!append_code(content, length, capacity,
                         category->entries[selected].assistant)) {
            free(order);
            return 0;
        }
        (*snippet_count)++;
        if (target_lines > 0 && count_lines(*content) >= target_lines) break;
        if (target_lines == 0 && *snippet_count >= target_snippets) break;
    }
    free(order);
    return 1;
}

static int compose_dataset_code(const char *input, const char *filename,
                                char **content, size_t *content_length,
                                int *line_count) {
    const KnowledgeCategory *categories[4];
    const KnowledgeCategory *primary;
    const char *extension;
    size_t capacity;
    int target_lines;
    int target_snippets;
    int snippet_count;
    int category_count;
    int i;

    extension = extension_of(filename);
    primary = category_for_extension(extension);
    categories[0] = primary;
    category_count = 1;
    if (primary != &v3_know_code_misc)
        categories[category_count++] = &v3_know_code_misc;
    if (primary != &v3_know_code_c)
        categories[category_count++] = &v3_know_code_c;
    if (primary != &v3_know_code_py && category_count < 4)
        categories[category_count++] = &v3_know_code_py;
    target_lines = requested_line_count(input);
    target_snippets = 2 + rng_range(5);
    capacity = 0;
    *content = NULL;
    *content_length = 0;
    snippet_count = 0;
    for (i = 0; i < category_count; i++) {
        if (!append_category(categories[i], content, content_length,
                             &capacity, target_lines, target_snippets,
                             &snippet_count)) {
            free(*content);
            *content = NULL;
            return 0;
        }
        if (target_lines > 0 && count_lines(*content) >= target_lines) break;
        if (target_lines == 0 && snippet_count >= target_snippets) break;
    }
    *line_count = count_lines(*content);
    if (target_lines > 0 && *line_count < target_lines) {
        free(*content);
        *content = NULL;
        return 0;
    }
    return *content != NULL;
}

int v3_compose_code(const char *input, const char *filename,
                    char *content, int content_size) {
    char selected_filename[128];
    char *generated;
    size_t generated_length;
    int line_count;

    if (!filename || !filename[0]) {
        default_filename(input, selected_filename, sizeof(selected_filename));
        filename = selected_filename;
    }
    if (!compose_dataset_code(input, filename, &generated,
                              &generated_length, &line_count))
        return -1;
    if (generated_length >= (size_t)content_size) {
        free(generated);
        return -1;
    }
    memcpy(content, generated, generated_length + 1);
    free(generated);
    (void)line_count;
    return (int)generated_length;
}

static int write_file(const char *filename, const char *content,
                      int overwrite, char *error, int error_size) {
    struct stat status;
    int descriptor;
    int flags;
    size_t length;
    ssize_t written;
    size_t offset;

    if (lstat(filename, &status) == 0) {
        if (S_ISLNK(status.st_mode)) {
            snprintf(error, (size_t)error_size, "Refusing symbolic link: %s", filename);
            return 0;
        }
        if (!overwrite) {
            snprintf(error, (size_t)error_size,
                     "%.80s already exists. Say overwrite %.80s to replace it.",
                     filename, filename);
            return 0;
        }
    }
    flags = O_WRONLY | O_CREAT;
    flags |= overwrite ? O_TRUNC : O_EXCL;
    descriptor = open(filename, flags, 0644);
    if (descriptor < 0) {
        snprintf(error, (size_t)error_size, "Could not write %s: %s",
                 filename, strerror(errno));
        return 0;
    }
    length = strlen(content);
    offset = 0;
    while (offset < length) {
        written = write(descriptor, content + offset, length - offset);
        if (written <= 0) {
            snprintf(error, (size_t)error_size, "Write failed for %s: %s",
                     filename, strerror(errno));
            close(descriptor);
            return 0;
        }
        offset += (size_t)written;
    }
    if (close(descriptor) != 0) {
        snprintf(error, (size_t)error_size, "Close failed for %s: %s",
                 filename, strerror(errno));
        return 0;
    }
    return 1;
}

void v3_run_code_mode(EngineState *state, const EngineCallbacks *callbacks,
                      const char *input, char *result, int result_size) {
    char filename[128];
    char *content;
    char error[256];
    struct stat status;
    size_t content_length;
    int line_count;
    int overwrite;

    content = NULL;
    callbacks->curs_set_fn(0);
    callbacks->show_status("Code mode is thinking...");
    if (!extract_filename(input, filename, sizeof(filename)))
        default_filename(input, filename, sizeof(filename));
    if (!compose_dataset_code(input, filename, &content,
                              &content_length, &line_count)) {
        snprintf(result, (size_t)result_size,
                 "Could not compose enough unique dataset code for %s.",
                 filename);
        callbacks->show_status("");
        callbacks->curs_set_fn(1);
        return;
    }
    overwrite = contains_case(input, "overwrite") ||
                contains_case(input, "replace");
    if (!write_file(filename, content, overwrite, error, sizeof(error))) {
        snprintf(result, (size_t)result_size, "%s", error);
        free(content);
        callbacks->show_status("");
        callbacks->curs_set_fn(1);
        return;
    }
    if (stat(filename, &status) != 0 ||
        (size_t)status.st_size != content_length) {
        snprintf(result, (size_t)result_size,
                 "Created %s, but verify became confused.", filename);
    } else {
        snprintf(result, (size_t)result_size,
                 "Created %s with %d lines and verified %lu bytes without executing it.",
                 filename, line_count, (unsigned long)content_length);
    }
    free(content);
    callbacks->show_status("");
    callbacks->curs_set_fn(1);
    (void)state;
}
