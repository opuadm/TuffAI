#ifndef TUFFAI_V2_RETRIEVAL_H
#define TUFFAI_V2_RETRIEVAL_H

typedef struct {
    const char *text;
    int is_code;
    int score;
    int matches;
    int requested_words;
} V2Retrieval;

void v2_retrieve_dataset_example(const char *input, V2Retrieval *retrieval);

#endif
