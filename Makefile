CC = cc
CFLAGS = -Wall -Wextra -std=c99 -pedantic -O2
LDFLAGS = -lm -lcurl -lncurses
SRCDIR = src
BUILDDIR = build
SRCS = $(SRCDIR)/main.c $(SRCDIR)/net.c $(SRCDIR)/features.c $(SRCDIR)/tokenizer.c $(SRCDIR)/corpus.c $(SRCDIR)/wikifetch.c $(SRCDIR)/markov.c $(SRCDIR)/version.c
MODEL_SRCS = $(SRCDIR)/models/tuffai-v1/tech.c $(SRCDIR)/models/tuffai-v1/languages.c $(SRCDIR)/models/tuffai-v1/code.c $(SRCDIR)/models/tuffai-v1/general.c $(SRCDIR)/models/tuffai-v1/phrases.c $(SRCDIR)/models/tuffai-v1/vocab.c $(SRCDIR)/models/tuffai-v1/extra.c
OBJS = $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(SRCS))
MODEL_OBJS = $(patsubst $(SRCDIR)/models/tuffai-v1/%.c,$(BUILDDIR)/models/tuffai-v1/%.o,$(MODEL_SRCS))
ALL_OBJS = $(OBJS) $(MODEL_OBJS)
TARGET = tuffai

all: $(BUILDDIR) $(TARGET)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)
	mkdir -p $(BUILDDIR)/models/tuffai-v1

$(TARGET): $(ALL_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILDDIR)/models/tuffai-v1/%.o: $(SRCDIR)/models/tuffai-v1/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILDDIR) $(TARGET)

.PHONY: all run clean
