#ifndef WEBUI_H
#define WEBUI_H

#include <stddef.h>

#define WEBUI_DEFAULT_PORT 18298
#define WEBUI_MODEL_ID_MAX 64
#define WEBUI_PROMPT_MAX 32768
#define WEBUI_ERROR_MAX 256

typedef int (*WebUIStreamCallback)(const char *text, int reasoning,
                                   void *context);

typedef struct {
    const char *id;
    const char *description;
    int context_window;
    int max_output_tokens;
    int supports_effort;
    int supports_code_mode;
} WebUIModelInfo;

typedef struct {
    char model[WEBUI_MODEL_ID_MAX];
    char *input;
    char *system_prompt;
    float temperature;
    float noise;
    float top_p;
    float frequency_penalty;
    float repetition_penalty;
    float presence_penalty;
    int max_tokens;
    int web_search;
    int code_mode;
    char reasoning_effort[32];
    int has_temperature;
    int has_noise;
    int has_top_p;
    int has_frequency_penalty;
    int has_repetition_penalty;
    int has_presence_penalty;
    int has_max_tokens;
    int has_web_search;
    int has_code_mode;
    int has_reasoning_effort;
    WebUIStreamCallback stream_callback;
    void *stream_context;
} WebUICompletionRequest;

typedef struct {
    char *content;
    char *thinking;
    int prompt_tokens;
    int completion_tokens;
} WebUICompletionResult;

typedef struct {
    int (*model_count)(void);
    int (*model_info)(int index, WebUIModelInfo *info);
    int (*complete)(WebUICompletionRequest *request,
                    WebUICompletionResult *result,
                    char *error, size_t error_size);
    int (*settings_json)(char *output, size_t output_size);
    int (*apply_settings)(const char *json, char *error,
                          size_t error_size);
    int (*cors_enabled)(void);
    int (*authorize)(const char *authorization);
    int (*admin_authorize)(const char *authorization);
    int (*admin_handle)(const char *method, const char *path,
                        const char *query, const char *body,
                        const char *authorization, char *output,
                        size_t output_size, int *status);
} WebUIHost;

int webui_run(int port, const WebUIHost *host);
void webui_set_concurrency(int max_generations);

#endif
