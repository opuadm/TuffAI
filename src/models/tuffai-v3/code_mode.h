#ifndef TUFFAI_V3_CODE_MODE_H
#define TUFFAI_V3_CODE_MODE_H

#include "../../engine.h"

void v3_run_code_mode(EngineState *state, const EngineCallbacks *callbacks,
                      const char *input, char *result, int result_size);
int v3_compose_code(const char *input, const char *filename,
                    char *content, int content_size);

#endif
