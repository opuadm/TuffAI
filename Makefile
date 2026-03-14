CC = cc
CFLAGS = -Wall -Wextra -std=c99 -pedantic -O2 -msse
LDFLAGS = -lm -lncurses -lcurl
SRCDIR = src
BUILDDIR = build

MODELS ?= v1,v2

SRCS = $(SRCDIR)/main.c $(SRCDIR)/net.c $(SRCDIR)/features.c $(SRCDIR)/tokenizer.c $(SRCDIR)/wikifetch.c $(SRCDIR)/markov.c $(SRCDIR)/version.c
V1_DATA_SRCS = $(SRCDIR)/models/tuffai-v1/tech.c $(SRCDIR)/models/tuffai-v1/languages.c $(SRCDIR)/models/tuffai-v1/code.c $(SRCDIR)/models/tuffai-v1/general.c $(SRCDIR)/models/tuffai-v1/phrases.c $(SRCDIR)/models/tuffai-v1/vocab.c $(SRCDIR)/models/tuffai-v1/extra.c
V1_ENGINE_SRCS = $(SRCDIR)/models/tuffai-v1/engine/engine.c $(SRCDIR)/models/tuffai-v1/engine/corpus.c
V2_DATA_SRCS = $(SRCDIR)/models/tuffai-v2/tech.c $(SRCDIR)/models/tuffai-v2/languages.c $(SRCDIR)/models/tuffai-v2/code.c $(SRCDIR)/models/tuffai-v2/general.c $(SRCDIR)/models/tuffai-v2/phrases.c $(SRCDIR)/models/tuffai-v2/vocab.c $(SRCDIR)/models/tuffai-v2/extra.c $(SRCDIR)/models/tuffai-v2/opinions.c $(SRCDIR)/models/tuffai-v2/thinking.c $(SRCDIR)/models/tuffai-v2/wrongfacts.c
V2_ENGINE_SRCS = $(SRCDIR)/models/tuffai-v2/engine/engine.c $(SRCDIR)/models/tuffai-v2/engine/corpus.c
OBJS = $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(SRCS))

MODEL_OBJS =
BUILD_DIRS = $(BUILDDIR)

ifneq ($(findstring v1,$(MODELS)),)
CFLAGS += -DENABLE_TUFFAI_V1
V1_DATA_OBJS = $(patsubst $(SRCDIR)/models/tuffai-v1/%.c,$(BUILDDIR)/models/tuffai-v1/%.o,$(V1_DATA_SRCS))
V1_ENGINE_OBJS = $(patsubst $(SRCDIR)/models/tuffai-v1/engine/%.c,$(BUILDDIR)/models/tuffai-v1/engine/%.o,$(V1_ENGINE_SRCS))
MODEL_OBJS += $(V1_DATA_OBJS) $(V1_ENGINE_OBJS)
BUILD_DIRS += $(BUILDDIR)/models/tuffai-v1/engine
endif

ifneq ($(findstring v2,$(MODELS)),)
CFLAGS += -DENABLE_TUFFAI_V2
V2_DATA_OBJS = $(patsubst $(SRCDIR)/models/tuffai-v2/%.c,$(BUILDDIR)/models/tuffai-v2/%.o,$(V2_DATA_SRCS))
V2_ENGINE_OBJS = $(patsubst $(SRCDIR)/models/tuffai-v2/engine/%.c,$(BUILDDIR)/models/tuffai-v2/engine/%.o,$(V2_ENGINE_SRCS))
MODEL_OBJS += $(V2_DATA_OBJS) $(V2_ENGINE_OBJS)
BUILD_DIRS += $(BUILDDIR)/models/tuffai-v2/engine
endif

ALL_OBJS = $(OBJS) $(MODEL_OBJS)
TARGET = tuffai

all: $(BUILDDIR) $(TARGET)

$(BUILDDIR):
	@for d in $(BUILD_DIRS); do mkdir -p $$d; done

$(TARGET): $(ALL_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILDDIR)/models/tuffai-v1/%.o: $(SRCDIR)/models/tuffai-v1/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILDDIR)/models/tuffai-v1/engine/%.o: $(SRCDIR)/models/tuffai-v1/engine/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILDDIR)/models/tuffai-v2/%.o: $(SRCDIR)/models/tuffai-v2/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILDDIR)/models/tuffai-v2/engine/%.o: $(SRCDIR)/models/tuffai-v2/engine/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILDDIR) $(TARGET)

.PHONY: all run clean
