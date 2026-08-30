#ifndef TUFFAI_V3_RETRIEVAL_H
#define TUFFAI_V3_RETRIEVAL_H

typedef struct {
    const char *text;
    const char *thinking;
    const char *use_tool;
    const char *tool_input;
    const char *after_tool;
    int is_code;
    int score;
    int matches;
    int requested_words;
} V3Retrieval;

void v3_retrieve_dataset_example(const char *input, V3Retrieval *retrieval);

#endif
