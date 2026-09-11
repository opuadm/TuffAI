#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700
#define _XOPEN_SOURCE_EXTENDED 1
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <signal.h>
#include <locale.h>
#include <wchar.h>
#include <pthread.h>
#include <ncurses.h>
#include "knowledge/knowledge.h"
#include "net.h"
#include "rng.h"
#include "features.h"
#include "tokenizer.h"
#include "websearch.h"
#include "version.h"
#include "engine.h"
#include "webui.h"
#include <sys/stat.h>
#include <unistd.h>
#ifdef ENABLE_TUFFAI_V3
#include "models/tuffai-v3/code_mode.h"
#endif
#include <curl/curl.h>

#define INPUT_MAX 512
#define CHAT_LINES 4096
#define CHAT_LINE_LEN 512

#define COLOR_USER 1
#define COLOR_AI 2
#define COLOR_THINK 3
#define COLOR_STATUS 4
#define COLOR_CMD 5
#define COLOR_SHADOW 6
#define COLOR_SELECTED 7

static char chat_log[CHAT_LINES][CHAT_LINE_LEN];
static int chat_colors[CHAT_LINES];
static int chat_count = 0;
static int chat_scroll = 0;
static volatile int got_sigwinch = 0;
static volatile int got_sigint = 0;
static int generation_stop_requested = 0;
static int train_mode = 0;
static int code_mode = 0;
static int webui_cors_enabled = 0;
#define AUTH_KEYS_MAX 256
#define AUTH_SESSIONS_MAX 64
#define AUTH_TOKEN_BYTES 32
#define AUTH_SESSION_TTL 43200
typedef struct {
    char id[17];
    char name[64];
    char key[128];
    long created;
    long last_used;
} ApiKeyEntry;
static ApiKeyEntry api_keys[AUTH_KEYS_MAX];
static int api_key_count = 0;
static char admin_salt_hex[33] = "";
static char admin_hash_hex[65] = "";
static char auth_file_path[512] = "tuffai_auth.json";
typedef struct {
    char token[65];
    long expires;
} AdminSession;
static AdminSession admin_sessions[AUTH_SESSIONS_MAX];
static int admin_session_count = 0;
static __thread WebUICompletionRequest *active_web_request;
static __thread int active_web_stream_failed;
static __thread EngineState *active_web_state;
static pthread_mutex_t config_mutex = PTHREAD_MUTEX_INITIALIZER;

static void draw_chat(void);

typedef struct {
    const char *cmd;
    const char *desc;
} SlashCmd;

static const SlashCmd slash_cmds[] = {
    {"/temp", "set temperature"},
    {"/noise", "set noise level"},
    {"/freq", "frequency penalty"},
    {"/rep", "repetition penalty"},
    {"/topp", "top-p sampling"},
    {"/presence", "presence penalty"},
    {"/maxwords", "max response words"},
    {"/effort", "set reasoning effort"},
    {"/code", "toggle code mode"},
    {"/model", "switch model"},
    {"/prompt", "edit system prompt"},
    {"/search", "toggle web search"},
    {"/train", "toggle train mode"},
    {"/stats", "session statistics"},
    {"/clear", "clear chat"},
    {"/help", "show commands"},
    {"/quit", "exit"},
};

#define SLASH_CMD_COUNT (int)(sizeof(slash_cmds) / sizeof(slash_cmds[0]))

static char hist_buf[HIST_MAX][HIST_LEN];
static int hist_tokens[HIST_MAX][MAX_TOKENS];
static int hist_lens[HIST_MAX];
static int hist_cnt = 0;

static EngineState engine_state;

static int utf8_character_bytes(const char *text, int available, int *width) {
    mbstate_t state;
    wchar_t wide;
    size_t result;
    int display_width;

    memset(&state, 0, sizeof(state));
    result = mbrtowc(&wide, text, (size_t)available, &state);
    if (result == (size_t)-1 || result == (size_t)-2 || result == 0) {
        *width = 1;
        return 1;
    }
    display_width = wcwidth(wide);
    if (display_width < 0) display_width = 1;
    *width = display_width;
    return (int)result;
}

static int utf8_prefix_bytes(const char *text, int byte_limit) {
    int length;
    int position;
    int character_bytes;
    int width;

    length = (int)strlen(text);
    if (length <= byte_limit) return length;
    position = 0;
    while (position < length) {
        character_bytes = utf8_character_bytes(text + position,
                                               length - position, &width);
        if (position + character_bytes > byte_limit) break;
        position += character_bytes;
    }
    return position;
}

static int utf8_display_width(const char *text) {
    int length;
    int position;
    int character_bytes;
    int character_width;
    int width;

    length = (int)strlen(text);
    position = 0;
    width = 0;
    while (position < length) {
        character_bytes = utf8_character_bytes(text + position,
                                               length - position,
                                               &character_width);
        position += character_bytes;
        width += character_width;
    }
    return width;
}

static int utf8_step_back(const char *text, int pos) {
    if (pos <= 0) return 0;
    pos--;
    while (pos > 0 && ((unsigned char)text[pos] & 0xC0) == 0x80)
        pos--;
    return pos;
}

static int utf8_is_printable_byte(int ch) {
    return ch >= 32 && ch < 256 && ch != 127;
}

static void draw_utf8_chat_line(int row, const char *text) {
    wchar_t wide[CHAT_LINE_LEN];
    mbstate_t state;
    size_t converted;
    int byte_length;
    int byte_position;
    int wide_position;

    memset(&state, 0, sizeof(state));
    byte_length = (int)strlen(text);
    byte_position = 0;
    wide_position = 0;
    while (byte_position < byte_length &&
           wide_position < CHAT_LINE_LEN - 1) {
        converted = mbrtowc(&wide[wide_position], text + byte_position,
                            (size_t)(byte_length - byte_position), &state);
        if (converted == (size_t)-1 || converted == (size_t)-2) {
            memset(&state, 0, sizeof(state));
            wide[wide_position++] = L'?';
            byte_position++;
        } else if (converted == 0) {
            break;
        } else {
            wide_position++;
            byte_position += (int)converted;
        }
    }
    wide[wide_position] = L'\0';
    mvaddnwstr(row, 0, wide, wide_position);
}

typedef struct {
    const char *name;
    const char *description;
    float default_temp;
    float default_noise;
    float default_freq_penalty;
    float default_rep_penalty;
    float default_top_p;
    float default_presence_penalty;
    const EngineVtable *engine;
} ModelDef;

static const ModelDef models[] = {
#ifdef ENABLE_TUFFAI_V3
    {
        "TuffAI-v3",
        "Clearer multilingual dataset model",
        1.15f,
        0.65f,
        0.65f,
        1.25f,
        0.9f,
        0.3f,
        &engine_tuffai_v3,
    },
#endif
#ifdef ENABLE_TUFFAI_V2
    {
        "TuffAI-v2",
        "Stable previous-generation model",
        1.35f,
        0.9f,
        0.8f,
        1.2f,
        0.92f,
        0.4f,
        &engine_tuffai_v2,
    },
#endif
#ifdef ENABLE_TUFFAI_V1
    {
        "TuffAI-v1",
        "The Original TuffAI",
        75.0f,
        2.5f,
        19.5f,
        19.0f,
        0.02f,
        10.0f,
        &engine_tuffai_v1,
    },
#endif
};

#define MODEL_COUNT (int)(sizeof(models) / sizeof(models[0]))

static int current_model = 0;

static int current_model_supports_effort(void) {
    return models[current_model].engine->effort_modes != NULL &&
           models[current_model].engine->effort_mode_count > 0;
}

static int current_model_supports_code_mode(void) {
    return models[current_model].engine->has_code_mode;
}

static void draw_box_shadow(int box_y, int box_x, int box_h, int box_w,
                            int rows, int cols) {
    int i;

    attron(COLOR_PAIR(COLOR_SHADOW));
    if (box_x + box_w < cols) {
        for (i = 1; i <= box_h && box_y + i < rows; i++)
            mvaddch(box_y + i, box_x + box_w, ACS_CKBOARD);
    }
    if (box_y + box_h < rows) {
        for (i = 1; i <= box_w && box_x + i < cols; i++)
            mvaddch(box_y + box_h, box_x + i, ACS_CKBOARD);
    }
    attroff(COLOR_PAIR(COLOR_SHADOW));
}

static void chat_add(const char *line) {
    int slot;
    int len;

    if (!line || !line[0]) return;
    slot = chat_count % CHAT_LINES;
    len = (int)strlen(line);
    if (len >= CHAT_LINE_LEN)
        len = utf8_prefix_bytes(line, CHAT_LINE_LEN - 1);
    memcpy(chat_log[slot], line, len);
    chat_log[slot][len] = '\0';
    chat_colors[slot] = 0;
    chat_count++;
    chat_scroll = 0;
}

static void chat_add_c(const char *line, int color) {
    int slot;
    int len;

    if (!line || !line[0]) return;
    slot = chat_count % CHAT_LINES;
    len = (int)strlen(line);
    if (len >= CHAT_LINE_LEN)
        len = utf8_prefix_bytes(line, CHAT_LINE_LEN - 1);
    memcpy(chat_log[slot], line, len);
    chat_log[slot][len] = '\0';
    chat_colors[slot] = color;
    chat_count++;
    chat_scroll = 0;
}

static void chat_add_wrapped(const char *prefix, const char *text, int color) {
    char line[CHAT_LINE_LEN];
    int max_w;
    int tlen;
    int pos;
    int first;
    int cut;
    int scan;
    int last_space;
    int columns;
    int available_columns;
    int prefix_columns;
    int prefix_bytes;
    int character_bytes;
    int character_width;
    int line_bytes;

    max_w = COLS - 2;
    if (max_w < 20) max_w = 20;
    if (max_w > CHAT_LINE_LEN - 1) max_w = CHAT_LINE_LEN - 1;

    tlen = (int)strlen(text);
    pos = 0;
    first = 1;

    while (pos < tlen) {
        prefix_bytes = first && prefix ? (int)strlen(prefix) : 0;
        prefix_columns = first && prefix ? utf8_display_width(prefix) : 0;
        available_columns = max_w - prefix_columns;
        if (available_columns < 1) available_columns = 1;
        scan = pos;
        last_space = -1;
        columns = 0;
        while (scan < tlen && text[scan] != '\n') {
            character_bytes = utf8_character_bytes(text + scan, tlen - scan,
                                                   &character_width);
            if (columns + character_width > available_columns) break;
            if (prefix_bytes + scan - pos + character_bytes >= CHAT_LINE_LEN)
                break;
            if (text[scan] == ' ') last_space = scan;
            columns += character_width;
            scan += character_bytes;
        }
        if (scan < tlen && text[scan] != '\n' && last_space > pos)
            cut = last_space - pos;
        else
            cut = scan - pos;
        if (cut <= 0) {
            character_bytes = utf8_character_bytes(text + pos, tlen - pos,
                                                   &character_width);
            cut = character_bytes;
        }
        line_bytes = 0;
        if (prefix_bytes > 0) {
            memcpy(line, prefix, prefix_bytes);
            line_bytes = prefix_bytes;
        }
        memcpy(line + line_bytes, text + pos, cut);
        line_bytes += cut;
        line[line_bytes] = '\0';
        chat_add_c(line, color);
        pos += cut;
        first = 0;
        if (pos < tlen && text[pos] == '\n') {
            pos++;
        } else {
            while (pos < tlen && text[pos] == ' ') pos++;
        }
    }
}

static void draw_chat_entry(int row, int line_index) {
    int color_pair;

    move(row, 0);
    clrtoeol();
    if (line_index < 0 || line_index >= chat_count) return;
    color_pair = chat_colors[line_index % CHAT_LINES];
    if (color_pair > 0) attron(A_BOLD | COLOR_PAIR(color_pair));
    draw_utf8_chat_line(row, chat_log[line_index % CHAT_LINES]);
    if (color_pair > 0) attroff(A_BOLD | COLOR_PAIR(color_pair));
}

static void draw_streamed_chat(int previous_count, int previous_scroll) {
    int rows;
    int columns;
    int chat_rows;
    int previous_start;
    int current_start;
    int shifted_rows;
    int first_new;
    int line_index;
    int row;

    getmaxyx(stdscr, rows, columns);
    chat_rows = rows - 4;
    if (chat_rows < 1) chat_rows = 1;
    previous_start = previous_count - chat_rows;
    if (previous_start < 0) previous_start = 0;
    current_start = chat_count - chat_rows;
    if (current_start < 0) current_start = 0;
    shifted_rows = current_start - previous_start;
    if (previous_scroll != 0 || shifted_rows >= chat_rows || got_sigwinch) {
        draw_chat();
        return;
    }
    if (shifted_rows > 0) {
        setscrreg(1, chat_rows);
        scrollok(stdscr, TRUE);
        wscrl(stdscr, shifted_rows);
        scrollok(stdscr, FALSE);
        setscrreg(0, rows - 1);
    }
    first_new = previous_count;
    if (first_new < current_start) first_new = current_start;
    for (line_index = first_new; line_index < chat_count; line_index++) {
        row = 1 + line_index - current_start;
        if (row >= 1 && row <= chat_rows)
            draw_chat_entry(row, line_index);
    }
    (void)columns;
}

static void draw_chat(void) {
    int rows, cols;
    int chat_rows;
    int start;
    int i, line_idx;
    int cpair;

    getmaxyx(stdscr, rows, cols);
    clear();

    move(0, 0);
    attron(A_BOLD | COLOR_PAIR(COLOR_STATUS));
    for (i = 0; i < cols; i++) addch(' ');
    mvprintw(0, 1, " %s %s | Model: %s",
        TUFFAI_NAME, TUFFAI_VERSION, models[current_model].name);
    attroff(A_BOLD | COLOR_PAIR(COLOR_STATUS));

    chat_rows = rows - 4;
    if (chat_rows < 1) chat_rows = 1;

    start = chat_count - chat_rows - chat_scroll;
    if (start < 0) start = 0;

    for (i = 0; i < chat_rows; i++) {
        move(i + 1, 0);
        clrtoeol();
        line_idx = start + i;
        if (line_idx >= 0 && line_idx < chat_count) {
            cpair = chat_colors[line_idx % CHAT_LINES];
            if (cpair > 0) attron(A_BOLD | COLOR_PAIR(cpair));
            draw_utf8_chat_line(i + 1,
                                chat_log[line_idx % CHAT_LINES]);
            if (cpair > 0) attroff(A_BOLD | COLOR_PAIR(cpair));
        }
    }

    move(chat_rows + 1, 0);
    attron(A_BOLD | COLOR_PAIR(COLOR_STATUS));
    for (i = 0; i < cols; i++) addch(' ');
    mvprintw(chat_rows + 1, 1,
        " Tokens:%d Turns:%d Tok/s:%.1f | T:%.1f N:%.1f FP:%.1f RP:%.1f TP:%.2f ",
        engine_state.total_tokens, engine_state.turn_count,
        engine_state.last_tokens_per_second,
        engine_state.cfg_temp, engine_state.cfg_noise,
        engine_state.cfg_freq_penalty, engine_state.cfg_rep_penalty, engine_state.cfg_top_p);
    attroff(A_BOLD | COLOR_PAIR(COLOR_STATUS));

    move(rows - 1, 0);
    clrtoeol();
}

static void draw_input(const char *buf, int cursor) {
    int rows, cols;
    int input_row;
    int max_display;
    int len;
    int cursor_width;
    int pos;
    int w;
    int cb;
    int start;
    int start_width;
    int cursor_col;
    int col;

    getmaxyx(stdscr, rows, cols);
    input_row = rows - 2;

    move(input_row, 0);
    clrtoeol();
    max_display = cols - 6;
    if (max_display < 0) max_display = 0;
    len = (int)strlen(buf);
    if (cursor < 0) cursor = 0;
    if (cursor > len) cursor = len;
    while (cursor > 0 && cursor < len &&
           ((unsigned char)buf[cursor] & 0xC0) == 0x80)
        cursor--;
    cursor_width = 0;
    pos = 0;
    while (pos < cursor) {
        cb = utf8_character_bytes(buf + pos, len - pos, &w);
        pos += cb;
        cursor_width += w;
    }
    start = 0;
    start_width = 0;
    if (cursor_width > max_display) {
        pos = 0;
        start_width = 0;
        while (pos < cursor) {
            cb = utf8_character_bytes(buf + pos, len - pos, &w);
            if (cursor_width - start_width - w < max_display) break;
            pos += cb;
            start_width += w;
        }
        start = pos;
    } else if (utf8_display_width(buf) > max_display && cursor >= len) {
        int total = utf8_display_width(buf);
        pos = 0;
        start_width = 0;
        while (pos < len) {
            cb = utf8_character_bytes(buf + pos, len - pos, &w);
            if (total - start_width - w < max_display) break;
            pos += cb;
            start_width += w;
        }
        start = pos;
    }
    cursor_col = cursor_width - start_width;
    attron(A_BOLD | COLOR_PAIR(COLOR_USER));
    mvaddstr(input_row, 0, "You: ");
    pos = start;
    col = 0;
    while (pos < len && col < max_display) {
        cb = utf8_character_bytes(buf + pos, len - pos, &w);
        if (col + w > max_display) break;
        mvaddnstr(input_row, 5 + col, buf + pos, cb);
        pos += cb;
        col += w;
    }
    attroff(A_BOLD | COLOR_PAIR(COLOR_USER));
    if (cursor_col > max_display) cursor_col = max_display;
    move(input_row, 5 + cursor_col);
}

static void draw_autocomplete(const char *buf, int buflen) {
    int rows, cols;
    int matches[SLASH_CMD_COUNT];
    int match_count = 0;
    int i, j;
    int box_y, box_x, box_w, box_h;
    int cmd_len;
    int desc_len;
    int max_w;
    char line[128];

    if (buflen < 1 || buf[0] != '/') return;

    for (i = 0; i < SLASH_CMD_COUNT; i++) {
        if ((strcmp(slash_cmds[i].cmd, "/effort") != 0 ||
             current_model_supports_effort()) &&
            (strcmp(slash_cmds[i].cmd, "/code") != 0 ||
             current_model_supports_code_mode()) &&
            strncmp(slash_cmds[i].cmd, buf, buflen) == 0) {
            matches[match_count++] = i;
        }
    }

    if (match_count == 0) return;
    if (match_count == 1 && buflen == (int)strlen(slash_cmds[matches[0]].cmd)) return;

    getmaxyx(stdscr, rows, cols);

    max_w = 0;
    for (i = 0; i < match_count; i++) {
        desc_len = (int)strlen(slash_cmds[matches[i]].desc);
        cmd_len = 13 + desc_len;
        if (cmd_len > max_w)
            max_w = cmd_len;
    }

    box_w = max_w + 4;
    if (box_w < 30) box_w = 30;
    if (box_w > cols - 2) box_w = cols - 2;
    box_h = match_count + 2;
    if (box_h > rows / 2) box_h = rows / 2;

    box_x = 0;
    box_y = rows - 4 - box_h;
    if (box_y < 1) box_y = 1;

    draw_box_shadow(box_y, box_x, box_h, box_w, rows, cols);

    attron(COLOR_PAIR(COLOR_STATUS));
    for (i = 0; i < box_h; i++) {
        move(box_y + i, box_x);
        for (j = 0; j < box_w && box_x + j < cols; j++)
            addch(' ');
    }

    mvaddch(box_y, box_x, ACS_ULCORNER);
    for (j = 1; j < box_w - 1; j++)
        mvaddch(box_y, box_x + j, ACS_HLINE);
    mvaddch(box_y, box_x + box_w - 1, ACS_URCORNER);
    mvaddch(box_y + box_h - 1, box_x, ACS_LLCORNER);
    for (j = 1; j < box_w - 1; j++)
        mvaddch(box_y + box_h - 1, box_x + j, ACS_HLINE);
    mvaddch(box_y + box_h - 1, box_x + box_w - 1, ACS_LRCORNER);
    for (i = 1; i < box_h - 1; i++) {
        mvaddch(box_y + i, box_x, ACS_VLINE);
        mvaddch(box_y + i, box_x + box_w - 1, ACS_VLINE);
    }
    attroff(COLOR_PAIR(COLOR_STATUS));

    for (i = 0; i < match_count && i < box_h - 2; i++) {
        snprintf(line, sizeof(line), "%-12s %s",
                 slash_cmds[matches[i]].cmd,
                 slash_cmds[matches[i]].desc);
        attron(A_BOLD | COLOR_PAIR(COLOR_STATUS));
        mvaddnstr(box_y + 1 + i, box_x + 1, line, box_w - 2);
        attroff(A_BOLD | COLOR_PAIR(COLOR_STATUS));
    }
}

static int str_casecmp(const char *a, const char *b) {
    while (*a && *b) {
        int ca = tolower((unsigned char)*a);
        int cb = tolower((unsigned char)*b);
        if (ca != cb) return ca - cb;
        a++;
        b++;
    }
    return (unsigned char)*a - (unsigned char)*b;
}

static const char *effort_name_for_model(int model_index, int effort_mode) {
    const EngineVtable *engine;

    engine = models[model_index].engine;
    if (!engine->effort_modes || effort_mode < 0 ||
        effort_mode >= engine->effort_mode_count)
        return "Unavailable";
    return engine->effort_modes[effort_mode];
}

static const char *effort_name(int effort_mode) {
    return effort_name_for_model(current_model, effort_mode);
}

static int parse_effort(const char *text) {
    int i;

    for (i = 0; i < models[current_model].engine->effort_mode_count; i++)
        if (str_casecmp(text, effort_name(i)) == 0) return i;
    return -1;
}

static int parse_effort_for_model(int model_index, const char *text) {
    int index;

    for (index = 0;
         index < models[model_index].engine->effort_mode_count; index++) {
        if (str_casecmp(text, effort_name_for_model(model_index, index)) == 0)
            return index;
    }
    return -1;
}

static void train_write_escaped(FILE *f, const char *s) {
    while (*s) {
        switch (*s) {
        case '"':  fputs("\\\"", f); break;
        case '\\': fputs("\\\\", f); break;
        case '\n': fputs("\\n\"\n            \"", f); break;
        case '\t': fputs("\\t", f); break;
        default:   fputc(*s, f); break;
        }
        s++;
    }
}

static void train_append(const char *user_input, const char *assistant_thinking,
                         const char *assistant_before_tool,
                         const char *tool, const char *tool_input,
                         const char *assistant_output) {
    FILE *f;

    f = fopen("train.txt", "a");
    if (!f) return;
    fprintf(f, "        {.user = \"");
    train_write_escaped(f, user_input);
    if (assistant_thinking && assistant_thinking[0]) {
        fprintf(f, "\", .assistant_thinking =\n            \"");
        train_write_escaped(f, assistant_thinking);
    }
    if (assistant_before_tool && assistant_before_tool[0]) {
        fprintf(f, "\", .assistant =\n            \"");
        train_write_escaped(f, assistant_before_tool);
    }
    if (tool && tool[0]) {
        fprintf(f, "\", .use_tool = \"");
        train_write_escaped(f, tool);
        if (tool_input && tool_input[0]) {
            fprintf(f, "\", .tool_input =\n            \"");
            train_write_escaped(f, tool_input);
        }
        fprintf(f, "\", .assistant_after_tool =\n            \"");
    } else {
        fprintf(f, "\", .assistant =\n            \"");
    }
    if (assistant_output) train_write_escaped(f, assistant_output);
    fprintf(f, "\"\n        },\n");
    fclose(f);
}

static int train_ensure_file(void) {
    FILE *f;

    f = fopen("train.txt", "a");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static int model_dialog(void) {
    int rows, cols;
    int selected;
    int ch;
    int i, j;
    int box_y, box_x, box_h, box_w;
    int old_model;
    int description_width;
    const char *marker;

    selected = current_model;
    old_model = current_model;

    box_h = MODEL_COUNT + 6;
    box_w = 62;

    for (;;) {
        getmaxyx(stdscr, rows, cols);
        box_y = (rows - box_h) / 2;
        box_x = (cols - box_w) / 2;
        if (box_y < 0) box_y = 0;
        if (box_x < 0) box_x = 0;

        draw_box_shadow(box_y, box_x, box_h, box_w, rows, cols);

        attron(COLOR_PAIR(COLOR_STATUS));
        for (i = 0; i < box_h; i++) {
            move(box_y + i, box_x);
            for (j = 0; j < box_w && box_x + j < cols; j++)
                addch(' ');
        }
        attroff(COLOR_PAIR(COLOR_STATUS));

        attron(COLOR_PAIR(COLOR_STATUS));
        mvaddch(box_y, box_x, ACS_ULCORNER);
        for (j = 1; j < box_w - 1; j++)
            mvaddch(box_y, box_x + j, ACS_HLINE);
        mvaddch(box_y, box_x + box_w - 1, ACS_URCORNER);

        mvaddch(box_y + box_h - 1, box_x, ACS_LLCORNER);
        for (j = 1; j < box_w - 1; j++)
            mvaddch(box_y + box_h - 1, box_x + j, ACS_HLINE);
        mvaddch(box_y + box_h - 1, box_x + box_w - 1, ACS_LRCORNER);

        for (i = 1; i < box_h - 1; i++) {
            mvaddch(box_y + i, box_x, ACS_VLINE);
            mvaddch(box_y + i, box_x + box_w - 1, ACS_VLINE);
        }
        attroff(COLOR_PAIR(COLOR_STATUS));

        attron(A_BOLD | COLOR_PAIR(COLOR_STATUS));
        mvprintw(box_y, box_x + 3, " Select Model ");
        attroff(A_BOLD | COLOR_PAIR(COLOR_STATUS));

        for (i = 0; i < MODEL_COUNT; i++) {
            marker = (i == old_model) ? "* " : "  ";
            description_width = box_w - 14 - (int)strlen(models[i].name);
            if (description_width < 0) description_width = 0;
            move(box_y + 2 + i, box_x + 3);
            if (i == selected) {
                attron(COLOR_PAIR(COLOR_SELECTED));
                printw("%s> %s  -  %.*s", marker, models[i].name,
                       description_width, models[i].description);
                attroff(COLOR_PAIR(COLOR_SELECTED));
            } else {
                attron(COLOR_PAIR(COLOR_STATUS));
                printw("%s  %s  -  %.*s", marker, models[i].name,
                       description_width, models[i].description);
                attroff(COLOR_PAIR(COLOR_STATUS));
            }
        }

        attron(COLOR_PAIR(COLOR_STATUS));
        mvprintw(box_y + box_h - 2, box_x + 3,
                 "UP/DOWN: navigate   ENTER: select   ESC: cancel");
        attroff(COLOR_PAIR(COLOR_STATUS));

        refresh();
        ch = getch();

        if (ch == KEY_UP || ch == 'k') {
            if (selected > 0) selected--;
        } else if (ch == KEY_DOWN || ch == 'j') {
            if (selected < MODEL_COUNT - 1) selected++;
        } else if (ch == '\n' || ch == KEY_ENTER) {
            return selected;
        } else if (ch == 27 || ch == 'q') {
            return -1;
        }
    }
}

static void prompt_editor(char *buf, int bufsize) {
    int rows, cols;
    int box_y, box_x, box_h, box_w;
    int cursor_pos;
    int scroll_offset;
    int ch;
    int i, j;
    int len;
    int line_num;
    int col_num;
    int visible_lines;
    int text_w;

    cursor_pos = (int)strlen(buf);
    scroll_offset = 0;

    for (;;) {
        getmaxyx(stdscr, rows, cols);
        box_h = rows - 4;
        box_w = cols - 4;
        if (box_h < 8) box_h = 8;
        if (box_w < 30) box_w = 30;
        box_y = (rows - box_h) / 2;
        box_x = (cols - box_w) / 2;
        if (box_y < 0) box_y = 0;
        if (box_x < 0) box_x = 0;

        draw_box_shadow(box_y, box_x, box_h, box_w, rows, cols);

        text_w = box_w - 4;
        if (text_w < 10) text_w = 10;
        visible_lines = box_h - 4;
        if (visible_lines < 1) visible_lines = 1;

        attron(COLOR_PAIR(COLOR_STATUS));
        for (i = 0; i < box_h; i++) {
            move(box_y + i, box_x);
            for (j = 0; j < box_w && box_x + j < cols; j++)
                addch(' ');
        }
        attroff(COLOR_PAIR(COLOR_STATUS));

        attron(COLOR_PAIR(COLOR_STATUS));
        mvaddch(box_y, box_x, ACS_ULCORNER);
        for (j = 1; j < box_w - 1; j++)
            mvaddch(box_y, box_x + j, ACS_HLINE);
        mvaddch(box_y, box_x + box_w - 1, ACS_URCORNER);
        mvaddch(box_y + box_h - 1, box_x, ACS_LLCORNER);
        for (j = 1; j < box_w - 1; j++)
            mvaddch(box_y + box_h - 1, box_x + j, ACS_HLINE);
        mvaddch(box_y + box_h - 1, box_x + box_w - 1, ACS_LRCORNER);
        for (i = 1; i < box_h - 1; i++) {
            mvaddch(box_y + i, box_x, ACS_VLINE);
            mvaddch(box_y + i, box_x + box_w - 1, ACS_VLINE);
        }
        attroff(COLOR_PAIR(COLOR_STATUS));

        attron(A_BOLD | COLOR_PAIR(COLOR_STATUS));
        mvprintw(box_y, box_x + 3, " System Prompt ");
        attroff(A_BOLD | COLOR_PAIR(COLOR_STATUS));

        attron(COLOR_PAIR(COLOR_STATUS));
        mvprintw(box_y + box_h - 1, box_x + 3, " ESC: save & close ");
        attroff(COLOR_PAIR(COLOR_STATUS));

        len = (int)strlen(buf);
        if (cursor_pos < 0) cursor_pos = 0;
        if (cursor_pos > len) cursor_pos = len;
        while (cursor_pos > 0 && cursor_pos < len &&
               ((unsigned char)buf[cursor_pos] & 0xC0) == 0x80)
            cursor_pos--;
        line_num = 0;
        col_num = 0;
        {
            int p = 0;
            int w;
            int cb;
            while (p < cursor_pos) {
                if (buf[p] == '\n') {
                    line_num++;
                    col_num = 0;
                    p++;
                } else {
                    cb = utf8_character_bytes(buf + p, len - p, &w);
                    if (col_num > 0 && col_num + w > text_w) {
                        line_num++;
                        col_num = 0;
                    }
                    col_num += w;
                    if (col_num >= text_w) {
                        line_num++;
                        col_num = 0;
                    }
                    p += cb;
                }
            }
        }

        if (line_num < scroll_offset)
            scroll_offset = line_num;
        if (line_num >= scroll_offset + visible_lines)
            scroll_offset = line_num - visible_lines + 1;

        {
            int draw_line;
            int draw_col;
            int text_pos;
            int cur_line;
            int w;
            int cb;

            draw_line = 0;
            cur_line = 0;
            draw_col = 0;
            text_pos = 0;

            while (text_pos < len && cur_line < scroll_offset) {
                if (buf[text_pos] == '\n') {
                    cur_line++;
                    draw_col = 0;
                    text_pos++;
                } else {
                    cb = utf8_character_bytes(buf + text_pos,
                                              len - text_pos, &w);
                    if (draw_col > 0 && draw_col + w > text_w) {
                        cur_line++;
                        draw_col = 0;
                    }
                    draw_col += w;
                    if (draw_col >= text_w) {
                        cur_line++;
                        draw_col = 0;
                    }
                    text_pos += cb;
                }
            }
            draw_col = 0;

            while (text_pos < len && draw_line < visible_lines) {
                if (buf[text_pos] == '\n') {
                    draw_line++;
                    draw_col = 0;
                    text_pos++;
                } else {
                    cb = utf8_character_bytes(buf + text_pos,
                                              len - text_pos, &w);
                    if (draw_col > 0 && draw_col + w > text_w) {
                        draw_line++;
                        draw_col = 0;
                        if (draw_line >= visible_lines) break;
                    }
                    if (draw_line >= 0 && draw_col + w <= text_w) {
                        attron(COLOR_PAIR(COLOR_STATUS));
                        mvaddnstr(box_y + 2 + draw_line, box_x + 2 + draw_col,
                                  buf + text_pos, cb);
                        attroff(COLOR_PAIR(COLOR_STATUS));
                    }
                    draw_col += w;
                    if (draw_col >= text_w) {
                        draw_line++;
                        draw_col = 0;
                    }
                    text_pos += cb;
                }
            }
        }

        {
            int cy;
            int cx;
            cy = box_y + 2 + (line_num - scroll_offset);
            cx = box_x + 2 + col_num;
            move(cy, cx);
        }

        curs_set(1);
        refresh();
        ch = getch();

        if (ch == 27) {
            break;
        } else if (ch == KEY_BACKSPACE || ch == 127 || ch == 8) {
            if (cursor_pos > 0) {
                int del_start;
                len = (int)strlen(buf);
                del_start = utf8_step_back(buf, cursor_pos);
                memmove(buf + del_start, buf + cursor_pos,
                        (size_t)(len - cursor_pos + 1));
                cursor_pos = del_start;
            }
        } else if (ch == KEY_LEFT) {
            if (cursor_pos > 0)
                cursor_pos = utf8_step_back(buf, cursor_pos);
        } else if (ch == KEY_RIGHT) {
            len = (int)strlen(buf);
            if (cursor_pos < len) {
                int w;
                int cb = utf8_character_bytes(buf + cursor_pos,
                                              len - cursor_pos, &w);
                cursor_pos += cb;
                if (cursor_pos > len) cursor_pos = len;
            }
        } else if (ch == KEY_UP) {
            {
                int target_col;
                int line_start;
                int prev_line_start;
                int prev_line_end;
                int p;
                int w;
                int cb;
                int acc;
                len = (int)strlen(buf);
                line_start = cursor_pos;
                while (line_start > 0 && buf[line_start - 1] != '\n')
                    line_start = utf8_step_back(buf, line_start);
                if (line_start == 0) {
                } else {
                    target_col = 0;
                    p = line_start;
                    while (p < cursor_pos) {
                        cb = utf8_character_bytes(buf + p, len - p, &w);
                        target_col += w;
                        p += cb;
                    }
                    prev_line_end = line_start - 1;
                    prev_line_start = prev_line_end;
                    while (prev_line_start > 0 &&
                           buf[prev_line_start - 1] != '\n')
                        prev_line_start = utf8_step_back(buf,
                                                         prev_line_start);
                    p = prev_line_start;
                    acc = 0;
                    while (p < prev_line_end) {
                        cb = utf8_character_bytes(buf + p, len - p, &w);
                        if (acc + w > target_col) break;
                        acc += w;
                        p += cb;
                    }
                    cursor_pos = p;
                }
            }
        } else if (ch == KEY_DOWN) {
            {
                int target_col;
                int line_start;
                int line_end;
                int next_line_start;
                int next_line_end;
                int p;
                int w;
                int cb;
                int acc;
                len = (int)strlen(buf);
                line_start = cursor_pos;
                while (line_start > 0 && buf[line_start - 1] != '\n')
                    line_start = utf8_step_back(buf, line_start);
                target_col = 0;
                p = line_start;
                while (p < cursor_pos) {
                    cb = utf8_character_bytes(buf + p, len - p, &w);
                    target_col += w;
                    p += cb;
                }
                line_end = cursor_pos;
                while (line_end < len && buf[line_end] != '\n') {
                    cb = utf8_character_bytes(buf + line_end,
                                              len - line_end, &w);
                    line_end += cb;
                }
                if (line_end >= len) {
                } else {
                    next_line_start = line_end + 1;
                    next_line_end = next_line_start;
                    while (next_line_end < len &&
                           buf[next_line_end] != '\n') {
                        cb = utf8_character_bytes(buf + next_line_end,
                                                  len - next_line_end, &w);
                        next_line_end += cb;
                    }
                    p = next_line_start;
                    acc = 0;
                    while (p < next_line_end) {
                        cb = utf8_character_bytes(buf + p, len - p, &w);
                        if (acc + w > target_col) break;
                        acc += w;
                        p += cb;
                    }
                    cursor_pos = p;
                }
            }
        } else if (ch == '\n' || ch == KEY_ENTER) {
            len = (int)strlen(buf);
            if (len < bufsize - 2) {
                memmove(buf + cursor_pos + 1, buf + cursor_pos, len - cursor_pos + 1);
                buf[cursor_pos] = '\n';
                cursor_pos++;
            }
        } else if (utf8_is_printable_byte(ch)) {
            len = (int)strlen(buf);
            if (len < bufsize - 2) {
                memmove(buf + cursor_pos + 1, buf + cursor_pos, len - cursor_pos + 1);
                buf[cursor_pos] = (char)ch;
                cursor_pos++;
            }
        }
    }
}

static int handle_command(const char *cmd) {
    char name[64];
    char arg[64];
    int n;
    float val;
    int i;
    char model_line[256];
    char *end;
    long parsed;
    size_t used;

    if (cmd[0] != '/') return 0;

    n = sscanf(cmd, "%63s %63s", name, arg);
    if (n < 1) return 0;

    if (strcmp(name, "/help") == 0) {
        chat_add_c("--- Commands ---", COLOR_CMD);
        chat_add("/temp <val>    - set temperature (default 6.0)");
        chat_add("/noise <val>   - set noise level (default 4.5)");
        chat_add("/freq <val>    - frequency penalty (default 3.5)");
        chat_add("/rep <val>     - repetition penalty mult (default 1.0)");
        chat_add("/topp <val>    - top-p / nucleus sampling (default 1.0)");
        chat_add("/presence <val>- presence penalty (default 0.0)");
        chat_add("/maxwords <val>- override max response words (0=auto)");
        if (current_model_supports_effort())
            chat_add("/effort <mode> - set reasoning effort");
        if (current_model_supports_code_mode())
            chat_add("/code          - toggle code mode");
        chat_add("/model         - Switch model");
        chat_add("/prompt        - Edit system prompt (v2 and v3 only)");
        chat_add("/search        - toggle autonomous web search");
        chat_add("/train         - toggle train mode (writes train.txt)");
        chat_add("/stats         - show session statistics");
        chat_add("/clear         - clear chat history");
        chat_add("/quit          - exit");
        return 1;
    }

    if (strcmp(name, "/stats") == 0) {
        chat_add_c("--- Session Stats ---", COLOR_CMD);
        snprintf(model_line, sizeof(model_line), "Model: %s", models[current_model].name);
        chat_add(model_line);
        snprintf(model_line, sizeof(model_line), "Total turns: %d", engine_state.turn_count);
        chat_add(model_line);
        snprintf(model_line, sizeof(model_line), "Total tokens: %d", engine_state.total_tokens);
        chat_add(model_line);
        snprintf(model_line, sizeof(model_line), "Last generation: %.1f tok/s",
                 engine_state.last_tokens_per_second);
        chat_add(model_line);
        snprintf(model_line, sizeof(model_line), "History entries: %d", hist_cnt);
        chat_add(model_line);
        snprintf(model_line, sizeof(model_line), "Context window: %d tokens",
                 models[current_model].engine->context_window);
        chat_add(model_line);
        if (current_model_supports_effort()) {
            snprintf(model_line, sizeof(model_line), "Effort: %s",
                     effort_name(engine_state.effort_mode));
            chat_add(model_line);
        }
        snprintf(model_line, sizeof(model_line), "Topics tracked: %d", engine_state.topic_n);
        chat_add(model_line);
        snprintf(model_line, sizeof(model_line), "Absorbed words: %d", engine_state.absorbed_n);
        chat_add(model_line);
        snprintf(model_line, sizeof(model_line), "Web search: %s",
                 web_search_enabled() ? "enabled" : "disabled");
        chat_add(model_line);
        if (engine_state.obsession_word[0]) {
            snprintf(model_line, sizeof(model_line), "Obsession: \"%s\" (strength: %d)", engine_state.obsession_word, engine_state.obsession_strength);
            chat_add(model_line);
        }
        return 1;
    }

    if (strcmp(name, "/model") == 0) {
        if (n < 2) {
            int choice;
            choice = model_dialog();
            if (choice >= 0 && choice < MODEL_COUNT) {
                current_model = choice;
                if (!models[choice].engine->has_code_mode) code_mode = 0;
                engine_state.cfg_temp = models[choice].default_temp;
                engine_state.cfg_noise = models[choice].default_noise;
                engine_state.cfg_freq_penalty = models[choice].default_freq_penalty;
                engine_state.cfg_rep_penalty = models[choice].default_rep_penalty;
                engine_state.cfg_top_p = models[choice].default_top_p;
                engine_state.cfg_presence_penalty = models[choice].default_presence_penalty;
                if (models[choice].engine->effort_mode_count > 0)
                    engine_state.effort_mode =
                        models[choice].engine->default_effort_mode;
                snprintf(model_line, sizeof(model_line), "Switched to %s", models[choice].name);
                chat_add(model_line);
            } else {
                chat_add("Model selection cancelled.");
            }
        } else {
            for (i = 0; i < MODEL_COUNT; i++) {
                if (str_casecmp(arg, models[i].name) == 0) {
                    current_model = i;
                    if (!models[i].engine->has_code_mode) code_mode = 0;
                    engine_state.cfg_temp = models[i].default_temp;
                    engine_state.cfg_noise = models[i].default_noise;
                    engine_state.cfg_freq_penalty = models[i].default_freq_penalty;
                    engine_state.cfg_rep_penalty = models[i].default_rep_penalty;
                    engine_state.cfg_top_p = models[i].default_top_p;
                    engine_state.cfg_presence_penalty = models[i].default_presence_penalty;
                    if (models[i].engine->effort_mode_count > 0)
                        engine_state.effort_mode =
                            models[i].engine->default_effort_mode;
                    snprintf(model_line, sizeof(model_line), "Switched to %s", models[i].name);
                    chat_add(model_line);
                    return 1;
                }
            }
            chat_add("Unknown model. Use /model to list available models.");
        }
        return 1;
    }

    if (strcmp(name, "/quit") == 0 || strcmp(name, "/exit") == 0) {
        return -1;
    }

    if (strcmp(name, "/clear") == 0) {
        chat_count = 0;
        chat_scroll = 0;
        chat_add("Chat cleared.");
        return 1;
    }

    if (strcmp(name, "/search") == 0) {
        if (web_search_enabled()) {
            web_search_set_enabled(0);
            chat_add("Autonomous web search disabled.");
        } else {
            web_search_set_enabled(1);
            chat_add("Autonomous web search enabled.");
        }
        return 1;
    }

    if (strcmp(name, "/train") == 0) {
        train_mode = !train_mode;
        if (train_mode) {
            if (train_ensure_file()) {
                chat_add("Train mode enabled. Writing to train.txt.");
            } else {
                train_mode = 0;
                chat_add("Could not create train.txt in this directory.");
            }
        } else {
            chat_add("Train mode disabled.");
        }
        return 1;
    }

    if (strcmp(name, "/code") == 0) {
        if (!current_model_supports_code_mode()) {
            chat_add("Code mode is not supported by this model.");
            return 1;
        }
        code_mode = !code_mode;
        if (code_mode)
            chat_add("Code mode enabled. Files stay in the launch directory and are not executed.");
        else
            chat_add("Code mode disabled.");
        return 1;
    }

    if (strcmp(name, "/prompt") == 0) {
        if (!models[current_model].engine->has_system_prompt) {
            chat_add("System prompt not supported by this model.");
            return 1;
        }
        prompt_editor(engine_state.system_prompt, SYSTEM_PROMPT_MAX);
        clear();
        chat_add("System prompt updated.");
        return 1;
    }

    if (strcmp(name, "/effort") == 0 && current_model_supports_effort()) {
        if (n < 2) {
            snprintf(model_line, sizeof(model_line), "Effort: %s. Available:",
                     effort_name(engine_state.effort_mode));
            for (i = 0;
                 i < models[current_model].engine->effort_mode_count;
                 i++) {
                used = strlen(model_line);
                snprintf(model_line + used, sizeof(model_line) - used,
                         "%s%s", i == 0 ? " " : ", ", effort_name(i));
            }
            used = strlen(model_line);
            if (used + 1 < sizeof(model_line)) {
                model_line[used] = '.';
                model_line[used + 1] = '\0';
            }
            chat_add(model_line);
            return 1;
        }
        i = parse_effort(arg);
        if (i < 0) {
            chat_add("Invalid effort mode. Type /effort to list modes.");
            return 1;
        }
        engine_state.effort_mode = i;
        snprintf(model_line, sizeof(model_line), "Effort set to %s.",
                 effort_name(engine_state.effort_mode));
        chat_add(model_line);
        return 1;
    }

    if (strcmp(name, "/effort") == 0) {
        chat_add("Unknown command. Type /help for usage.");
        return 1;
    }

    if (n < 2) {
        chat_add("Missing argument. Type /help for usage.");
        return 1;
    }

    if (strcmp(name, "/temp") == 0) {
        val = (float)atof(arg);
        if (val > 0.001f && val <= 10000.0f) {
            engine_state.cfg_temp = val;
            chat_add("Temperature set.");
        } else {
            chat_add("Invalid value (0.001 - 10000).");
        }
        return 1;
    }

    if (strcmp(name, "/noise") == 0) {
        val = (float)atof(arg);
        if (val >= 0.0f && val <= 10000.0f) {
            engine_state.cfg_noise = val;
            chat_add("Noise set.");
        } else {
            chat_add("Invalid value (0 - 10000).");
        }
        return 1;
    }

    if (strcmp(name, "/freq") == 0) {
        val = (float)atof(arg);
        if (val >= 0.0f && val <= 1000.0f) {
            engine_state.cfg_freq_penalty = val;
            chat_add("Frequency penalty set.");
        } else {
            chat_add("Invalid value (0 - 1000).");
        }
        return 1;
    }

    if (strcmp(name, "/rep") == 0) {
        val = (float)atof(arg);
        if (val >= 0.0f && val <= 1000.0f) {
            engine_state.cfg_rep_penalty = val;
            chat_add("Repetition penalty set.");
        } else {
            chat_add("Invalid value (0 - 1000).");
        }
        return 1;
    }

    if (strcmp(name, "/topp") == 0) {
        val = (float)atof(arg);
        if (val > 0.0f && val <= 1.0f) {
            engine_state.cfg_top_p = val;
            chat_add("Top-p set.");
        } else {
            chat_add("Invalid value (0.01 - 1.0).");
        }
        return 1;
    }

    if (strcmp(name, "/presence") == 0) {
        val = (float)atof(arg);
        if (val >= -1000.0f && val <= 1000.0f) {
            engine_state.cfg_presence_penalty = val;
            chat_add("Presence penalty set.");
        } else {
            chat_add("Invalid value (-1000 - 1000).");
        }
        return 1;
    }

    if (strcmp(name, "/maxwords") == 0) {
        errno = 0;
        end = NULL;
        parsed = strtol(arg, &end, 10);
        if (arg[0] != '\0' && end && end != arg && *end == '\0' && errno != ERANGE && parsed >= 0 && parsed <= INT_MAX) {
            engine_state.cfg_max_words = (int)parsed;
            if (engine_state.cfg_max_words == 0) {
                chat_add("Max words set to auto.");
            } else {
                chat_add("Max words override set.");
            }
        } else {
            chat_add("Invalid value (0=auto, non-negative integer).");
        }
        return 1;
    }

    chat_add("Unknown command. Type /help for usage.");
    return 1;
}

static void show_status(const char *msg) {
    int rows, cols;
    int status_row;

    getmaxyx(stdscr, rows, cols);
    status_row = rows - 3;

    move(status_row, 0);
    attron(A_BOLD | COLOR_PAIR(COLOR_STATUS));
    clrtoeol();
    mvaddnstr(status_row, 0, msg, cols - 1);
    attroff(A_BOLD | COLOR_PAIR(COLOR_STATUS));
    refresh();
    (void)cols;
}

static void cb_refresh_screen(void) {
    refresh();
}

static void cb_curs_set_fn(int visibility) {
    curs_set(visibility);
}

static int cb_generation_should_stop(void) {
    int ch;

    if (generation_stop_requested) return 1;
    if (got_sigint) {
        got_sigint = 0;
        generation_stop_requested = 1;
        return 1;
    }
    timeout(0);
    ch = getch();
    timeout(-1);
    if (ch == 3 || ch == 27) {
        generation_stop_requested = 1;
        return 1;
    }
    if (ch == KEY_RESIZE) got_sigwinch = 1;
    return 0;
}

static int capture_generated_text(char **destination, size_t *capacity,
                                  const char *text) {
    char *grown;
    size_t used;
    size_t length;
    size_t separator;
    size_t required;
    size_t new_capacity;

    if (!text || !text[0]) return 1;
    used = *destination ? strlen(*destination) : 0;
    length = strlen(text);
    separator = used > 0 &&
                !isspace((unsigned char)(*destination)[used - 1]) &&
                !isspace((unsigned char)text[0]);
    required = used + separator + length + 1;
    if (required < used || required < length) return 0;
    if (required > *capacity) {
        new_capacity = *capacity ? *capacity : 256;
        while (new_capacity < required) {
            if (new_capacity > (size_t)-1 / 2) {
                new_capacity = required;
                break;
            }
            new_capacity *= 2;
        }
        grown = (char *)realloc(*destination, new_capacity);
        if (!grown) return 0;
        *destination = grown;
        *capacity = new_capacity;
        if (used == 0) (*destination)[0] = '\0';
    }
    if (separator) (*destination)[used++] = ' ';
    memcpy(*destination + used, text, length + 1);
    return 1;
}

static void cb_chat_add_c(const char *line, int color) {
    if (color == engine_state.color_think && strcmp(line, "Thoughts") != 0)
        capture_generated_text(&engine_state.last_thinking,
                               &engine_state.last_thinking_capacity, line);
    if (color == engine_state.color_ai)
        capture_generated_text(&engine_state.last_response,
                               &engine_state.last_response_capacity, line);
    chat_add_c(line, color);
}

static void cb_chat_add_wrapped(const char *prefix, const char *text,
                                int color) {
    if (color == engine_state.color_think)
        capture_generated_text(&engine_state.last_thinking,
                               &engine_state.last_thinking_capacity, text);
    if (color == engine_state.color_ai)
        capture_generated_text(&engine_state.last_response,
                               &engine_state.last_response_capacity, text);
    chat_add_wrapped(prefix, text, color);
}

static void cb_use_tool(const char *name, const char *input) {
    char message[256];

    if (engine_state.last_response && engine_state.last_response[0]) {
        if (engine_state.last_response_before_tool)
            engine_state.last_response_before_tool[0] = '\0';
        capture_generated_text(&engine_state.last_response_before_tool,
            &engine_state.last_response_before_tool_capacity,
            engine_state.last_response);
        engine_state.last_response[0] = '\0';
    }
    if (engine_state.last_tool) engine_state.last_tool[0] = '\0';
    if (engine_state.last_tool_input) engine_state.last_tool_input[0] = '\0';
    capture_generated_text(&engine_state.last_tool,
                           &engine_state.last_tool_capacity, name);
    capture_generated_text(&engine_state.last_tool_input,
                           &engine_state.last_tool_input_capacity, input);
    snprintf(message, sizeof(message), "Tool: %.200s", name);
    chat_add(message);
    chat_add_wrapped("    ", input, 0);
}

static void cb_stream_text(const char *prefix, const char *text, int color) {
    int previous_count;
    int previous_scroll;
    int in_token;
    int token_count;
    int i;
    unsigned char character;

    previous_count = chat_count;
    previous_scroll = chat_scroll;
    in_token = 0;
    token_count = 0;
    if (color == engine_state.color_ai) {
        capture_generated_text(&engine_state.last_response,
                               &engine_state.last_response_capacity, text);
        for (i = 0; text[i]; i++) {
            character = (unsigned char)text[i];
            if (isspace(character)) {
                in_token = 0;
            } else if (!in_token) {
                token_count++;
                in_token = 1;
            }
        }
        engine_state.visible_tokens += token_count;
    } else if (color == engine_state.color_think) {
        capture_generated_text(&engine_state.last_thinking,
                               &engine_state.last_thinking_capacity, text);
    }
    chat_add_wrapped(prefix, text, color);
    draw_streamed_chat(previous_count, previous_scroll);
    refresh();
}

static const EngineCallbacks engine_callbacks = {
    chat_add,
    cb_chat_add_c,
    cb_chat_add_wrapped,
    draw_chat,
    show_status,
    cb_refresh_screen,
    cb_curs_set_fn,
    cb_generation_should_stop,
    cb_stream_text,
    cb_use_tool
};

static void sync_engine_state(EngineState *st, char (*hist_buf)[HIST_LEN],
                              int (*hist_tokens)[MAX_TOKENS],
                              int *hist_lens, int *hist_cnt) {
    st->hist_buf = hist_buf;
    st->hist_tokens = hist_tokens;
    st->hist_lens = hist_lens;
    st->hist_cnt = hist_cnt;
    st->color_think = COLOR_THINK;
    st->color_ai = COLOR_AI;
}

static int count_shared_words(const char *a, const char *b) {
    char bufa[PREV_RESPONSE_LEN];
    char bufb[PREV_RESPONSE_LEN];
    char *wa[128];
    char *wb[128];
    int na = 0;
    int nb = 0;
    int shared = 0;
    int i, j;
    char *p;
    char *save_a;
    char *save_b;

    strncpy(bufa, a, PREV_RESPONSE_LEN - 1);
    bufa[PREV_RESPONSE_LEN - 1] = '\0';
    strncpy(bufb, b, PREV_RESPONSE_LEN - 1);
    bufb[PREV_RESPONSE_LEN - 1] = '\0';

    p = strtok_r(bufa, " \t\n.,!?;:'\"()[]{}", &save_a);
    while (p && na < 128) {
        wa[na++] = p;
        p = strtok_r(NULL, " \t\n.,!?;:'\"()[]{}", &save_a);
    }
    p = strtok_r(bufb, " \t\n.,!?;:'\"()[]{}", &save_b);
    while (p && nb < 128) {
        wb[nb++] = p;
        p = strtok_r(NULL, " \t\n.,!?;:'\"()[]{}", &save_b);
    }

    for (i = 0; i < na; i++) {
        for (j = 0; j < nb; j++) {
            if (str_casecmp(wa[i], wb[j]) == 0) {
                shared++;
                break;
            }
        }
    }
    return shared;
}

static float compute_repetition_boost(EngineState *st) {
    int count;
    int limit;
    int i, j;
    int overlap;
    int total_overlap;
    int pairs;
    float avg_overlap;

    count = st->prev_resp_count;
    if (count < 2) return 0.0f;

    limit = count < PREV_RESPONSE_MAX ? count : PREV_RESPONSE_MAX;
    total_overlap = 0;
    pairs = 0;

    for (i = 0; i < limit; i++) {
        for (j = i + 1; j < limit; j++) {
            overlap = count_shared_words(st->prev_responses[i], st->prev_responses[j]);
            total_overlap += overlap;
            pairs++;
        }
    }

    if (pairs == 0) return 0.0f;
    avg_overlap = (float)total_overlap / (float)pairs;
    if (avg_overlap < 3.0f) return 0.0f;
    return (avg_overlap - 3.0f) * 0.5f;
}

static void generate_response_with_callbacks(EngineState *st,
                                             const EngineCallbacks *callbacks,
                                             const char *input,
                                             int model_index,
                                             int code_mode_flag,
                                             int draw_after) {
    float boost;
    float orig_freq;
    float orig_rep;
    struct timespec started;
    struct timespec finished;
    double elapsed;
    long visible_before;
    long visible_delta;
#ifdef ENABLE_TUFFAI_V3
    char code_result[512];
    char code_prompt[INPUT_MAX + 640];
#endif

    if (st->last_thinking) st->last_thinking[0] = '\0';
    if (st->last_response) st->last_response[0] = '\0';
    if (st->last_response_before_tool)
        st->last_response_before_tool[0] = '\0';
    if (st->last_tool) st->last_tool[0] = '\0';
    if (st->last_tool_input) st->last_tool_input[0] = '\0';

    boost = compute_repetition_boost(st);
    orig_freq = st->cfg_freq_penalty;
    orig_rep = st->cfg_rep_penalty;
    st->cfg_freq_penalty += boost;
    st->cfg_rep_penalty += boost * 0.8f;

    visible_before = st->visible_tokens;
    if (draw_after) generation_stop_requested = 0;
    clock_gettime(CLOCK_MONOTONIC, &started);
#ifdef ENABLE_TUFFAI_V3
    if (code_mode_flag && models[model_index].engine->has_code_mode) {
        v3_run_code_mode(st, callbacks, input,
                         code_result, sizeof(code_result));
        callbacks->chat_add(code_result);
        snprintf(code_prompt, sizeof(code_prompt),
                 "%s Code mode operation result: %s",
                 input, code_result);
        models[model_index].engine->generate_response(
            st, callbacks, code_prompt);
    } else
#endif
        models[model_index].engine->generate_response(
            st, callbacks, input);
    if (draw_after) generation_stop_requested = 0;
    clock_gettime(CLOCK_MONOTONIC, &finished);
    elapsed = (double)(finished.tv_sec - started.tv_sec) +
              (double)(finished.tv_nsec - started.tv_nsec) / 1000000000.0;
    visible_delta = st->visible_tokens - visible_before;
    if (elapsed > 0.0 && visible_delta > 0)
        st->last_tokens_per_second =
            (float)((double)visible_delta / elapsed);
    else
        st->last_tokens_per_second = 0.0f;

    st->cfg_freq_penalty = orig_freq;
    st->cfg_rep_penalty = orig_rep;
    if (draw_after) draw_chat();
}

static void generate_response(const char *input) {
    sync_engine_state(&engine_state, hist_buf, hist_tokens, hist_lens,
                      &hist_cnt);
    generate_response_with_callbacks(&engine_state, &engine_callbacks,
                                     input, current_model, code_mode, 1);
}

static void web_noop_line(const char *line) {
    (void)line;
}

static void web_noop_void(void) {
}

static void web_noop_status(const char *message) {
    (void)message;
}

static void web_noop_cursor(int visibility) {
    (void)visibility;
}

static int web_generation_should_stop(void) {
    return active_web_stream_failed;
}

static void web_capture_line_color(const char *line, int color) {
    int reasoning;
    EngineState *st;

    st = active_web_state;
    if (!st) return;
    reasoning = color == st->color_think;
    if (color == st->color_think && strcmp(line, "Thoughts") != 0)
        capture_generated_text(&st->last_thinking,
                               &st->last_thinking_capacity, line);
    if (color == st->color_ai)
        capture_generated_text(&st->last_response,
                               &st->last_response_capacity, line);
    if (active_web_request && active_web_request->stream_callback &&
        strcmp(line, "Thoughts") != 0 &&
        (reasoning || color == st->color_ai))
        if (!active_web_request->stream_callback(
                line, reasoning, active_web_request->stream_context))
            active_web_stream_failed = 1;
}

static void web_capture_wrapped(const char *prefix, const char *text,
                                int color) {
    (void)prefix;
    web_capture_line_color(text, color);
}

static void web_capture_stream(const char *prefix, const char *text,
                               int color) {
    int in_token;
    int token_count;
    int index;
    unsigned char character;

    (void)prefix;
    in_token = 0;
    token_count = 0;
    web_capture_line_color(text, color);
    if (!active_web_state || color != active_web_state->color_ai) return;
    for (index = 0; text[index]; index++) {
        character = (unsigned char)text[index];
        if (isspace(character)) {
            in_token = 0;
        } else if (!in_token) {
            token_count++;
            in_token = 1;
        }
    }
    active_web_state->visible_tokens += token_count;
}

static void web_capture_tool(const char *name, const char *input) {
    EngineState *st;

    st = active_web_state;
    if (!st) return;
    if (st->last_response && st->last_response[0]) {
        if (st->last_response_before_tool)
            st->last_response_before_tool[0] = '\0';
        capture_generated_text(&st->last_response_before_tool,
            &st->last_response_before_tool_capacity,
            st->last_response);
        st->last_response[0] = '\0';
    }
    if (st->last_tool) st->last_tool[0] = '\0';
    if (st->last_tool_input) st->last_tool_input[0] = '\0';
    capture_generated_text(&st->last_tool,
                           &st->last_tool_capacity, name);
    capture_generated_text(&st->last_tool_input,
                           &st->last_tool_input_capacity, input);
}

static const EngineCallbacks web_engine_callbacks = {
    web_noop_line,
    web_capture_line_color,
    web_capture_wrapped,
    web_noop_void,
    web_noop_status,
    web_noop_void,
    web_noop_cursor,
    web_generation_should_stop,
    web_capture_stream,
    web_capture_tool
};

static const char *settings_find_key(const char *json, const char *key) {
    const char *cursor;
    const char *end;
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
        end = cursor;
        escaped = 0;
        while (*end) {
            if (!escaped && *end == '"') break;
            if (!escaped && *end == '\\') escaped = 1;
            else escaped = 0;
            end++;
        }
        if (!*end) return NULL;
        if ((size_t)(end - cursor) == key_length &&
            memcmp(cursor, key, key_length) == 0) {
            end++;
            while (isspace((unsigned char)*end)) end++;
            if (*end == ':') {
                end++;
                while (isspace((unsigned char)*end)) end++;
                return end;
            }
        }
        cursor = end + 1;
    }
    return NULL;
}

static int settings_hex_value(char character) {
    if (character >= '0' && character <= '9') return character - '0';
    if (character >= 'a' && character <= 'f') return character - 'a' + 10;
    if (character >= 'A' && character <= 'F') return character - 'A' + 10;
    return -1;
}

static int settings_emit_utf8(char *output, size_t output_size,
                               size_t *position, int codepoint) {
    if (codepoint < 0 || codepoint > 0x10FFFF ||
        (codepoint >= 0xD800 && codepoint <= 0xDFFF))
        codepoint = '?';
    if (codepoint < 0x80) {
        if (*position + 1 >= output_size) return 0;
        output[(*position)++] = (char)codepoint;
    } else if (codepoint < 0x800) {
        if (*position + 2 >= output_size) return 0;
        output[(*position)++] = (char)(0xC0 | (codepoint >> 6));
        output[(*position)++] = (char)(0x80 | (codepoint & 0x3F));
    } else if (codepoint < 0x10000) {
        if (*position + 3 >= output_size) return 0;
        output[(*position)++] = (char)(0xE0 | (codepoint >> 12));
        output[(*position)++] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
        output[(*position)++] = (char)(0x80 | (codepoint & 0x3F));
    } else {
        if (*position + 4 >= output_size) return 0;
        output[(*position)++] = (char)(0xF0 | (codepoint >> 18));
        output[(*position)++] = (char)(0x80 | ((codepoint >> 12) & 0x3F));
        output[(*position)++] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
        output[(*position)++] = (char)(0x80 | (codepoint & 0x3F));
    }
    return 1;
}

static int settings_read_string(const char *json, const char *key,
                                 char *output, size_t output_size) {
    const char *source;
    size_t position;

    source = settings_find_key(json, key);
    if (!source) return 0;
    if (*source != '"') return -1;
    source++;
    position = 0;
    while (*source && *source != '"') {
        if (position + 1 >= output_size) return -1;
        if (*source == '\\') {
            source++;
            if (!*source) return -1;
            if (*source == 'n') output[position++] = '\n';
            else if (*source == 'r') output[position++] = '\r';
            else if (*source == 't') output[position++] = '\t';
            else if (*source == 'b') output[position++] = '\b';
            else if (*source == 'f') output[position++] = '\f';
            else if (*source == 'u') {
                int h0 = settings_hex_value(source[1]);
                int h1 = settings_hex_value(source[2]);
                int h2 = settings_hex_value(source[3]);
                int h3 = settings_hex_value(source[4]);
                int codepoint;
                if (h0 < 0 || h1 < 0 || h2 < 0 || h3 < 0) return -1;
                codepoint = h0 * 4096 + h1 * 256 + h2 * 16 + h3;
                source += 4;
                if (codepoint >= 0xD800 && codepoint <= 0xDBFF &&
                    source[1] == '\\' && source[2] == 'u') {
                    int l0 = settings_hex_value(source[3]);
                    int l1 = settings_hex_value(source[4]);
                    int l2 = settings_hex_value(source[5]);
                    int l3 = settings_hex_value(source[6]);
                    if (l0 >= 0 && l1 >= 0 && l2 >= 0 && l3 >= 0) {
                        int low = l0 * 4096 + l1 * 256 + l2 * 16 + l3;
                        if (low >= 0xDC00 && low <= 0xDFFF) {
                            codepoint = 0x10000 +
                                        ((codepoint - 0xD800) << 10) +
                                        (low - 0xDC00);
                            source += 6;
                        }
                    }
                }
                if (!settings_emit_utf8(output, output_size, &position,
                                        codepoint))
                    return -1;
            }
            else output[position++] = *source;
        } else {
            output[position++] = *source;
        }
        source++;
    }
    if (*source != '"') return -1;
    output[position] = '\0';
    return 1;
}

static int settings_read_float(const char *json, const char *key,
                               float *output) {
    const char *source;
    char *end;
    double value;

    source = settings_find_key(json, key);
    if (!source) return 0;
    errno = 0;
    value = strtod(source, &end);
    if (errno != 0 || end == source || !isfinite(value)) return -1;
    *output = (float)value;
    return 1;
}

static int settings_read_int(const char *json, const char *key, int *output) {
    const char *source;
    char *end;
    long value;

    source = settings_find_key(json, key);
    if (!source) return 0;
    errno = 0;
    value = strtol(source, &end, 10);
    if (errno != 0 || end == source || value < 0 || value > INT_MAX)
        return -1;
    *output = (int)value;
    return 1;
}

static int settings_read_bool(const char *json, const char *key,
                              int *output) {
    const char *source;

    source = settings_find_key(json, key);
    if (!source) return 0;
    if (strncmp(source, "true", 4) == 0) {
        *output = 1;
        return 1;
    }
    if (strncmp(source, "false", 5) == 0) {
        *output = 0;
        return 1;
    }
    return -1;
}

static char *settings_escape_json(const char *text) {
    const unsigned char *source;
    char *output;
    char *cursor;
    size_t required;

    source = (const unsigned char *)text;
    required = 1;
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
    source = (const unsigned char *)text;
    cursor = output;
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
    return output;
}

static int web_host_model_count(void) {
    return MODEL_COUNT;
}

static int web_host_model_info(int index, WebUIModelInfo *info) {
    if (index < 0 || index >= MODEL_COUNT || !info) return 0;
    info->id = models[index].name;
    info->description = models[index].description;
    info->context_window = models[index].engine->context_window;
    info->max_output_tokens = models[index].engine->max_output_tokens;
    info->supports_effort = models[index].engine->effort_mode_count > 0;
    info->supports_code_mode = models[index].engine->has_code_mode;
    return 1;
}

static int web_find_model(const char *name) {
    int index;

    if (!name || !name[0]) return current_model;
    for (index = 0; index < MODEL_COUNT; index++) {
        if (str_casecmp(name, models[index].name) == 0) return index;
    }
    return -1;
}

static int web_host_settings_json(char *output, size_t output_size) {
    char models_json[4096];
    char item[512];
    char *escaped_prompt;
    size_t used;
    int index;
    int written;

    pthread_mutex_lock(&config_mutex);
    models_json[0] = '\0';
    used = 0;
    for (index = 0; index < MODEL_COUNT; index++) {
        written = snprintf(item, sizeof(item),
            "%s{\"id\":\"%s\",\"description\":\"%s\","
            "\"context_window\":%d,\"max_output_tokens\":%d,"
            "\"supports_effort\":%s,\"supports_code_mode\":%s,"
            "\"default_effort\":\"%s\"}",
            index == 0 ? "" : ",", models[index].name,
            models[index].description,
            models[index].engine->context_window,
            models[index].engine->max_output_tokens,
            models[index].engine->effort_mode_count > 0 ? "true" : "false",
            models[index].engine->has_code_mode ? "true" : "false",
            effort_name_for_model(index,
                models[index].engine->default_effort_mode));
        if (written < 0 || used + (size_t)written >= sizeof(models_json)) {
            pthread_mutex_unlock(&config_mutex);
            return 0;
        }
        memcpy(models_json + used, item, (size_t)written + 1);
        used += (size_t)written;
    }
    escaped_prompt = settings_escape_json(engine_state.system_prompt);
    if (!escaped_prompt) {
        pthread_mutex_unlock(&config_mutex);
        return 0;
    }
    written = snprintf(output, output_size,
        "{\"models\":[%s],\"current_model\":\"%s\","
        "\"effort\":\"%s\",\"system_prompt\":\"%s\","
        "\"temperature\":%.6g,\"noise\":%.6g,\"top_p\":%.6g,"
        "\"frequency_penalty\":%.6g,\"repetition_penalty\":%.6g,"
        "\"presence_penalty\":%.6g,\"max_words\":%d,"
        "\"web_search\":%s,\"code_mode\":%s,\"cors\":%s,"
        "\"require_api_key\":true,\"api_key_configured\":%s,"
        "\"api_key_count\":%d,\"admin_setup_required\":%s,"
        "\"defaults\":{\"temperature\":%.6g,\"noise\":%.6g,"
        "\"top_p\":%.6g,\"frequency_penalty\":%.6g,"
        "\"repetition_penalty\":%.6g,\"presence_penalty\":%.6g,"
        "\"effort\":\"%s\"}}",
        models_json, models[current_model].name,
        effort_name(engine_state.effort_mode), escaped_prompt,
        engine_state.cfg_temp, engine_state.cfg_noise,
        engine_state.cfg_top_p, engine_state.cfg_freq_penalty,
        engine_state.cfg_rep_penalty, engine_state.cfg_presence_penalty,
        engine_state.cfg_max_words,
        web_search_enabled() ? "true" : "false",
        code_mode ? "true" : "false",
        webui_cors_enabled ? "true" : "false",
        api_key_count > 0 ? "true" : "false",
        api_key_count,
        admin_hash_hex[0] ? "false" : "true",
        models[current_model].default_temp, models[current_model].default_noise,
        models[current_model].default_top_p,
        models[current_model].default_freq_penalty,
        models[current_model].default_rep_penalty,
        models[current_model].default_presence_penalty,
         effort_name(models[current_model].engine->default_effort_mode));
    free(escaped_prompt);
    written = written >= 0 && (size_t)written < output_size;
    pthread_mutex_unlock(&config_mutex);
    return written;
}

static int web_host_apply_settings(const char *json, char *error,
                                   size_t error_size) {
    char model_name[WEBUI_MODEL_ID_MAX];
    char effort[32];
    char prompt[SYSTEM_PROMPT_MAX];
    float temperature;
    float noise;
    float top_p;
    float frequency_penalty;
    float repetition_penalty;
    float presence_penalty;
    int max_words;
    int web_search;
    int requested_code_mode;
    int cors;
    int model_index;
    int effort_mode;
    int parsed;

    pthread_mutex_lock(&config_mutex);
    snprintf(model_name, sizeof(model_name), "%s", models[current_model].name);
    snprintf(effort, sizeof(effort), "%s",
             effort_name(engine_state.effort_mode));
    snprintf(prompt, sizeof(prompt), "%s", engine_state.system_prompt);
    temperature = engine_state.cfg_temp;
    noise = engine_state.cfg_noise;
    top_p = engine_state.cfg_top_p;
    frequency_penalty = engine_state.cfg_freq_penalty;
    repetition_penalty = engine_state.cfg_rep_penalty;
    presence_penalty = engine_state.cfg_presence_penalty;
    max_words = engine_state.cfg_max_words;
    web_search = web_search_enabled();
    requested_code_mode = code_mode;
    cors = webui_cors_enabled;
    pthread_mutex_unlock(&config_mutex);
    if (settings_read_string(json, "model", model_name,
                             sizeof(model_name)) < 0 ||
        settings_read_string(json, "effort", effort, sizeof(effort)) < 0 ||
        settings_read_string(json, "system_prompt", prompt,
                             sizeof(prompt)) < 0) {
        snprintf(error, error_size, "invalid text setting");
        return 0;
    }
    parsed = settings_read_float(json, "temperature", &temperature);
    if (parsed < 0 || settings_read_float(json, "noise", &noise) < 0 ||
        settings_read_float(json, "top_p", &top_p) < 0 ||
        settings_read_float(json, "frequency_penalty", &frequency_penalty) < 0 ||
        settings_read_float(json, "repetition_penalty", &repetition_penalty) < 0 ||
        settings_read_float(json, "presence_penalty", &presence_penalty) < 0 ||
        settings_read_int(json, "max_words", &max_words) < 0 ||
        settings_read_bool(json, "web_search", &web_search) < 0 ||
        settings_read_bool(json, "code_mode", &requested_code_mode) < 0 ||
        settings_read_bool(json, "cors", &cors) < 0) {
        snprintf(error, error_size, "invalid numeric or toggle setting");
        return 0;
    }
    model_index = web_find_model(model_name);
    effort_mode = parse_effort_for_model(model_index, effort);
    if (model_index < 0) {
        snprintf(error, error_size, "unknown model");
        return 0;
    }
    if (effort_mode < 0 ||
        effort_mode >= models[model_index].engine->effort_mode_count) {
        if (models[model_index].engine->effort_mode_count > 0) {
            snprintf(error, error_size, "unsupported reasoning effort");
            return 0;
        }
        effort_mode = models[model_index].engine->default_effort_mode;
    }
    if (requested_code_mode && !models[model_index].engine->has_code_mode) {
        snprintf(error, error_size, "code mode is not supported by this model");
        return 0;
    }
    if (temperature <= 0.001f || temperature > 10000.0f ||
        noise < 0.0f || noise > 10000.0f || top_p <= 0.0f || top_p > 1.0f ||
        frequency_penalty < 0.0f || frequency_penalty > 1000.0f ||
        repetition_penalty < 0.0f || repetition_penalty > 1000.0f ||
        presence_penalty < -1000.0f || presence_penalty > 1000.0f) {
        snprintf(error, error_size, "generation setting is out of range");
        return 0;
    }
    pthread_mutex_lock(&config_mutex);
    current_model = model_index;
    engine_state.cfg_temp = temperature;
    engine_state.cfg_noise = noise;
    engine_state.cfg_top_p = top_p;
    engine_state.cfg_freq_penalty = frequency_penalty;
    engine_state.cfg_rep_penalty = repetition_penalty;
    engine_state.cfg_presence_penalty = presence_penalty;
    engine_state.cfg_max_words = max_words;
    engine_state.effort_mode = effort_mode;
    snprintf(engine_state.system_prompt, sizeof(engine_state.system_prompt),
             "%s", prompt);
    web_search_set_enabled(web_search);
    code_mode = requested_code_mode;
    webui_cors_enabled = cors;
    pthread_mutex_unlock(&config_mutex);
    return 1;
}

static void reset_request_state(EngineState *st, int *hist_cnt_ptr) {
    *hist_cnt_ptr = 0;
    st->topic_n = 0;
    st->obsession_word[0] = '\0';
    st->obsession_strength = 0;
    st->absorbed_n = 0;
    st->prev_resp_count = 0;
    st->self_ctx_len = 0;
    st->recent_rendered_count = 0;
    st->recent_codepoint_count = 0;
}

static int count_text_tokens(const char *text) {
    int count;
    int in_token;
    unsigned char character;

    count = 0;
    in_token = 0;
    while (*text) {
        character = (unsigned char)*text++;
        if (isspace(character)) {
            in_token = 0;
        } else if (!in_token) {
            count++;
            in_token = 1;
        }
    }
    return count;
}

typedef struct {
    EngineState st;
    char hist_buf[HIST_MAX][HIST_LEN];
    int hist_tokens[HIST_MAX][MAX_TOKENS];
    int hist_lens[HIST_MAX];
    int hist_cnt;
} WebGenState;

static int web_host_complete(WebUICompletionRequest *request,
                             WebUICompletionResult *result, char *error,
                             size_t error_size) {
    WebGenState *gen;
    EngineState *st;
    int requested_model;
    int requested_effort;
    int requested_code_mode;
    const char *response;

    pthread_mutex_lock(&config_mutex);
    requested_model = web_find_model(request->model);
    pthread_mutex_unlock(&config_mutex);
    if (requested_model < 0) {
        snprintf(error, error_size, "model '%s' was not found", request->model);
        return 0;
    }
    if (!request->model[0])
        snprintf(request->model, sizeof(request->model), "%s",
                 models[requested_model].name);
    if ((request->has_temperature &&
         (request->temperature <= 0.001f || request->temperature > 10000.0f)) ||
        (request->has_noise &&
         (request->noise < 0.0f || request->noise > 10000.0f)) ||
        (request->has_top_p &&
         (request->top_p <= 0.0f || request->top_p > 1.0f)) ||
        (request->has_frequency_penalty &&
         (request->frequency_penalty < 0.0f ||
          request->frequency_penalty > 1000.0f)) ||
        (request->has_repetition_penalty &&
         (request->repetition_penalty < 0.0f ||
          request->repetition_penalty > 1000.0f)) ||
        (request->has_presence_penalty &&
         (request->presence_penalty < -1000.0f ||
          request->presence_penalty > 1000.0f)) ||
        (request->has_max_tokens &&
         request->max_tokens > models[requested_model].engine->max_output_tokens)) {
        snprintf(error, error_size, "completion parameter is out of range");
        return 0;
    }
    requested_effort = models[requested_model].engine->default_effort_mode;
    if (request->has_reasoning_effort) {
        requested_effort = parse_effort_for_model(
            requested_model, request->reasoning_effort);
        if (requested_effort < 0) {
            snprintf(error, error_size,
                     "reasoning_effort is not supported by this model");
            return 0;
        }
    }
    if (request->has_code_mode && request->code_mode) {
        snprintf(error, error_size,
                 "code_mode is not available via the API");
        return 0;
    }
    gen = (WebGenState *)calloc(1, sizeof(*gen));
    if (!gen) {
        snprintf(error, error_size, "unable to allocate completion state");
        return 0;
    }
    st = &gen->st;
    st->cfg_temp = models[requested_model].default_temp;
    st->cfg_noise = models[requested_model].default_noise;
    st->cfg_top_p = models[requested_model].default_top_p;
    st->cfg_freq_penalty =
        models[requested_model].default_freq_penalty;
    st->cfg_rep_penalty =
        models[requested_model].default_rep_penalty;
    st->cfg_presence_penalty =
        models[requested_model].default_presence_penalty;
    st->cfg_max_words = 0;
    st->effort_mode = requested_effort;
    requested_code_mode = request->has_code_mode && request->code_mode;
    snprintf(st->system_prompt, sizeof(st->system_prompt),
             "You are %s, a very tuff AI assistant.",
             models[requested_model].name);
    sync_engine_state(st, gen->hist_buf, gen->hist_tokens, gen->hist_lens,
                      &gen->hist_cnt);
    reset_request_state(st, &gen->hist_cnt);
    if (request->has_temperature) st->cfg_temp = request->temperature;
    if (request->has_noise) st->cfg_noise = request->noise;
    if (request->has_top_p) st->cfg_top_p = request->top_p;
    if (request->has_frequency_penalty)
        st->cfg_freq_penalty = request->frequency_penalty;
    if (request->has_repetition_penalty)
        st->cfg_rep_penalty = request->repetition_penalty;
    if (request->has_presence_penalty)
        st->cfg_presence_penalty = request->presence_penalty;
    if (request->has_max_tokens) st->cfg_max_words = request->max_tokens;
    if (request->system_prompt && request->system_prompt[0])
        snprintf(st->system_prompt, sizeof(st->system_prompt),
                 "%s", request->system_prompt);
    active_web_request = request;
    active_web_state = st;
    active_web_stream_failed = 0;
    web_search_set_local(request->has_web_search && request->web_search);
    generate_response_with_callbacks(st, &web_engine_callbacks,
                                     request->input, requested_model,
                                     requested_code_mode, 0);
    web_search_clear_local();
    active_web_request = NULL;
    active_web_state = NULL;
    response = st->last_response && st->last_response[0] ?
               st->last_response : "I could not produce a response.";
    result->content = strdup(response);
    result->thinking = strdup(st->last_thinking &&
                              st->last_thinking[0] ?
                              st->last_thinking : "");
    result->prompt_tokens = count_text_tokens(request->input) +
                            count_text_tokens(st->system_prompt);
    result->completion_tokens = count_text_tokens(response);
    free(st->last_thinking);
    free(st->last_response);
    free(st->last_response_before_tool);
    free(st->last_tool);
    free(st->last_tool_input);
    free(gen);
    if (!result->content || !result->thinking) {
        free(result->content);
        free(result->thinking);
        result->content = NULL;
        result->thinking = NULL;
        snprintf(error, error_size, "unable to allocate completion response");
        return 0;
    }
    return 1;
}

static int web_host_cors_enabled(void) {
    int enabled;

    pthread_mutex_lock(&config_mutex);
    enabled = webui_cors_enabled;
    pthread_mutex_unlock(&config_mutex);
    return enabled;
}

static int secure_text_equal(const char *left, const char *right) {
    size_t left_length;
    size_t right_length;
    size_t maximum;
    size_t index;
    unsigned int difference;
    unsigned char left_character;
    unsigned char right_character;

    if (!left || !right) return 0;
    left_length = strlen(left);
    right_length = strlen(right);
    maximum = left_length > right_length ? left_length : right_length;
    difference = (unsigned int)(left_length ^ right_length);
    for (index = 0; index < maximum; index++) {
        left_character = index < left_length ? (unsigned char)left[index] : 0;
        right_character = index < right_length ? (unsigned char)right[index] : 0;
        difference |= (unsigned int)(left_character ^ right_character);
    }
    return difference == 0;
}

typedef struct {
    unsigned int state[8];
    unsigned long long bitcount;
    unsigned char buffer[64];
    size_t buffer_used;
} AuthSha256;

static unsigned int sha_rotr(unsigned int v, int n) {
    return (v >> n) | (v << (32 - n));
}

static void sha256_transform(AuthSha256 *ctx, const unsigned char *block) {
    static const unsigned int k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,
        0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
        0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,
        0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,
        0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
        0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,
        0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,
        0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
        0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };
    unsigned int w[64];
    unsigned int a,b,c,d,e,f,g,h;
    unsigned int s0,s1,ch,maj,t1,t2;
    int i;

    for (i = 0; i < 16; i++)
        w[i] = ((unsigned int)block[i*4] << 24) |
               ((unsigned int)block[i*4+1] << 16) |
               ((unsigned int)block[i*4+2] << 8) |
               ((unsigned int)block[i*4+3]);
    for (i = 16; i < 64; i++) {
        s0 = sha_rotr(w[i-15],7) ^ sha_rotr(w[i-15],18) ^ (w[i-15] >> 3);
        s1 = sha_rotr(w[i-2],17) ^ sha_rotr(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2];
    d = ctx->state[3]; e = ctx->state[4]; f = ctx->state[5];
    g = ctx->state[6]; h = ctx->state[7];
    for (i = 0; i < 64; i++) {
        s1 = sha_rotr(e,6) ^ sha_rotr(e,11) ^ sha_rotr(e,25);
        ch = (e & f) ^ ((~e) & g);
        t1 = h + s1 + ch + k[i] + w[i];
        s0 = sha_rotr(a,2) ^ sha_rotr(a,13) ^ sha_rotr(a,22);
        maj = (a & b) ^ (a & c) ^ (b & c);
        t2 = s0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c;
    ctx->state[3] += d; ctx->state[4] += e; ctx->state[5] += f;
    ctx->state[6] += g; ctx->state[7] += h;
}

static void sha256_init(AuthSha256 *ctx) {
    ctx->state[0]=0x6a09e667; ctx->state[1]=0xbb67ae85;
    ctx->state[2]=0x3c6ef372; ctx->state[3]=0xa54ff53a;
    ctx->state[4]=0x510e527f; ctx->state[5]=0x9b05688c;
    ctx->state[6]=0x1f83d9ab; ctx->state[7]=0x5be0cd19;
    ctx->bitcount = 0;
    ctx->buffer_used = 0;
}

static void sha256_update(AuthSha256 *ctx, const void *data, size_t len) {
    const unsigned char *p = (const unsigned char *)data;
    while (len > 0) {
        size_t room = 64 - ctx->buffer_used;
        size_t take = len < room ? len : room;
        memcpy(ctx->buffer + ctx->buffer_used, p, take);
        ctx->buffer_used += take;
        p += take;
        len -= take;
        ctx->bitcount += (unsigned long long)take * 8ULL;
        if (ctx->buffer_used == 64) {
            sha256_transform(ctx, ctx->buffer);
            ctx->buffer_used = 0;
        }
    }
}

static void sha256_final(AuthSha256 *ctx, unsigned char out[32]) {
    unsigned char pad[64];
    unsigned char lenbits[8];
    unsigned long long bits = ctx->bitcount;
    size_t padlen;
    int i;

    for (i = 0; i < 8; i++) {
        lenbits[7-i] = (unsigned char)(bits & 0xff);
        bits >>= 8;
    }
    pad[0] = 0x80;
    memset(pad+1, 0, sizeof(pad)-1);
    padlen = ctx->buffer_used < 56 ? 56 - ctx->buffer_used
                                   : 120 - ctx->buffer_used;
    sha256_update(ctx, pad, padlen);
    sha256_update(ctx, lenbits, 8);
    for (i = 0; i < 8; i++) {
        out[i*4] = (unsigned char)(ctx->state[i] >> 24);
        out[i*4+1] = (unsigned char)(ctx->state[i] >> 16);
        out[i*4+2] = (unsigned char)(ctx->state[i] >> 8);
        out[i*4+3] = (unsigned char)(ctx->state[i]);
    }
}

static void auth_hex_encode(const unsigned char *in, size_t len, char *out) {
    static const char *digits = "0123456789abcdef";
    size_t i;
    for (i = 0; i < len; i++) {
        out[i*2] = digits[(in[i] >> 4) & 0xf];
        out[i*2+1] = digits[in[i] & 0xf];
    }
    out[len*2] = '\0';
}

static void auth_random_hex(char *out, size_t bytes) {
    size_t i;
    unsigned char buf[64];
    if (bytes > sizeof(buf)) bytes = sizeof(buf);
    for (i = 0; i < bytes; i++)
        buf[i] = (unsigned char)(rng_u32() & 0xff);
    auth_hex_encode(buf, bytes, out);
    memset(buf, 0, sizeof(buf));
}

static void auth_hash_password(const char *salt_hex, const char *password,
                               char out_hex[65]) {
    AuthSha256 ctx;
    unsigned char digest[32];
    sha256_init(&ctx);
    sha256_update(&ctx, salt_hex, strlen(salt_hex));
    sha256_update(&ctx, ":", 1);
    sha256_update(&ctx, password, strlen(password));
    sha256_final(&ctx, digest);
    auth_hex_encode(digest, 32, out_hex);
    memset(&ctx, 0, sizeof(ctx));
    memset(digest, 0, sizeof(digest));
}

static const char *auth_bearer_token(const char *authorization) {
    if (!authorization) return NULL;
    if (strncmp(authorization, "Bearer ", 7) != 0) return NULL;
    if (!authorization[7]) return NULL;
    return authorization + 7;
}

static int auth_save_locked(void) {
    FILE *f;
    char tmp[600];
    int i;

    snprintf(tmp, sizeof(tmp), "%s.tmp", auth_file_path);
    f = fopen(tmp, "w");
    if (!f) return 0;
    fprintf(f, "{\"admin_salt\":\"%s\",\"admin_hash\":\"%s\",\"keys\":[",
            admin_salt_hex, admin_hash_hex);
    for (i = 0; i < api_key_count; i++) {
        char *esc_name;
        esc_name = settings_escape_json(api_keys[i].name);
        if (!esc_name) { fclose(f); return 0; }
        fprintf(f, "%s{\"id\":\"%s\",\"name\":\"%s\",\"key\":\"%s\","
                "\"created\":%ld,\"last_used\":%ld}",
                i == 0 ? "" : ",", api_keys[i].id, esc_name,
                api_keys[i].key, api_keys[i].created,
                api_keys[i].last_used);
        free(esc_name);
    }
    fprintf(f, "]}");
    fclose(f);
    chmod(tmp, 0600);
    if (rename(tmp, auth_file_path) != 0) return 0;
    chmod(auth_file_path, 0600);
    return 1;
}

static const char *auth_scan_string(const char *json, const char *key) {
    return settings_find_key(json, key);
}

static int auth_load_file(void) {
    FILE *f;
    char *data;
    long fsize;
    const char *keys;
    const char *array_end;
    const char *cursor;
    const char *obj_end;
    char id[32];
    char name[64];
    char key[128];
    int parsed_int;
    long created;
    long last_used;
    const char *v;
    char *endptr;

    f = fopen(auth_file_path, "r");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize <= 0 || fsize > 1048576) { fclose(f); return 0; }
    data = (char *)malloc((size_t)fsize + 1);
    if (!data) { fclose(f); return 0; }
    if (fread(data, 1, (size_t)fsize, f) != (size_t)fsize) {
        free(data); fclose(f); return 0;
    }
    data[fsize] = '\0';
    fclose(f);
    pthread_mutex_lock(&config_mutex);
    if (settings_read_string(data, "admin_salt", admin_salt_hex,
                             sizeof(admin_salt_hex)) < 0) {
        admin_salt_hex[0] = '\0';
    }
    if (settings_read_string(data, "admin_hash", admin_hash_hex,
                             sizeof(admin_hash_hex)) < 0) {
        admin_hash_hex[0] = '\0';
    }
    api_key_count = 0;
    keys = auth_scan_string(data, "keys");
    if (keys && *keys == '[') {
        cursor = keys + 1;
        array_end = keys;
        {
            int depth = 0, instr = 0, esc = 0;
            const char *p = keys;
            while (*p) {
                if (instr) {
                    if (!esc && *p == '"') instr = 0;
                    if (!esc && *p == '\\') esc = 1; else esc = 0;
                } else if (*p == '"') instr = 1;
                else if (*p == '[') depth++;
                else if (*p == ']') {
                    depth--;
                    if (depth == 0) { array_end = p; break; }
                }
                p++;
            }
        }
        while (cursor < array_end && api_key_count < AUTH_KEYS_MAX) {
            while (cursor < array_end && *cursor != '{') cursor++;
            if (cursor >= array_end) break;
            {
                int depth = 0, instr = 0, esc = 0;
                const char *p = cursor;
                obj_end = NULL;
                while (p <= array_end && *p) {
                    if (instr) {
                        if (!esc && *p == '"') instr = 0;
                        if (!esc && *p == '\\') esc = 1; else esc = 0;
                    } else if (*p == '"') instr = 1;
                    else if (*p == '{') depth++;
                    else if (*p == '}') {
                        depth--;
                        if (depth == 0) { obj_end = p; break; }
                    }
                    p++;
                }
            }
            if (!obj_end) break;
            {
                char obj[1024];
                size_t olen = (size_t)(obj_end - cursor + 1);
                if (olen >= sizeof(obj)) olen = sizeof(obj) - 1;
                memcpy(obj, cursor, olen);
                obj[olen] = '\0';
                id[0] = '\0'; name[0] = '\0'; key[0] = '\0';
                created = 0; last_used = 0;
                if (settings_read_string(obj, "id", id, sizeof(id)) > 0 &&
                    settings_read_string(obj, "key", key, sizeof(key)) > 0) {
                    settings_read_string(obj, "name", name, sizeof(name));
                    v = settings_find_key(obj, "created");
                    if (v) {
                        errno = 0;
                        created = strtol(v, &endptr, 10);
                        if (errno != 0) created = 0;
                    }
                    v = settings_find_key(obj, "last_used");
                    if (v) {
                        errno = 0;
                        last_used = strtol(v, &endptr, 10);
                        if (errno != 0) last_used = 0;
                    }
                    snprintf(api_keys[api_key_count].id,
                             sizeof(api_keys[api_key_count].id), "%s", id);
                    snprintf(api_keys[api_key_count].name,
                             sizeof(api_keys[api_key_count].name), "%s", name);
                    snprintf(api_keys[api_key_count].key,
                             sizeof(api_keys[api_key_count].key), "%s", key);
                    api_keys[api_key_count].created = created;
                    api_keys[api_key_count].last_used = last_used;
                    api_key_count++;
                    parsed_int = 1;
                    (void)parsed_int;
                }
            }
            cursor = obj_end + 1;
        }
    }
    pthread_mutex_unlock(&config_mutex);
    free(data);
    return 1;
}

static void auth_prune_sessions_locked(long now) {
    int i, j;
    for (i = 0; i < admin_session_count; i++) {
        if (admin_sessions[i].expires < now) {
            memset(admin_sessions[i].token, 0,
                   sizeof(admin_sessions[i].token));
            for (j = i; j + 1 < admin_session_count; j++)
                admin_sessions[j] = admin_sessions[j+1];
            admin_session_count--;
            i--;
        }
    }
}

static int auth_check_admin_locked(const char *authorization) {
    const char *token;
    int i;

    if (!admin_hash_hex[0]) return 0;
    token = auth_bearer_token(authorization);
    if (!token) return 0;
    auth_prune_sessions_locked((long)time(NULL));
    for (i = 0; i < admin_session_count; i++) {
        if (secure_text_equal(token, admin_sessions[i].token))
            return 1;
    }
    return 0;
}

static int auth_check_key_locked(const char *authorization, int *index_out) {
    const char *token;
    int i;

    token = auth_bearer_token(authorization);
    if (!token) return 0;
    for (i = 0; i < api_key_count; i++) {
        if (secure_text_equal(token, api_keys[i].key)) {
            if (index_out) *index_out = i;
            return 1;
        }
    }
    return 0;
}

static void auth_key_prefix(const char *key, char *out, size_t out_size) {
    size_t len = strlen(key);
    if (len <= 8) {
        snprintf(out, out_size, "%.4s…", key);
    } else {
        snprintf(out, out_size, "%.4s…%.4s", key, key + len - 4);
    }
}

static int auth_create_session_locked(char *token_out, size_t token_size) {
    char rand[65];

    auth_prune_sessions_locked((long)time(NULL));
    if (admin_session_count >= AUTH_SESSIONS_MAX) {
        memset(admin_sessions[0].token, 0, sizeof(admin_sessions[0].token));
        memmove(admin_sessions, admin_sessions + 1,
                sizeof(admin_sessions[0]) * (AUTH_SESSIONS_MAX - 1));
        admin_session_count = AUTH_SESSIONS_MAX - 1;
    }
    auth_random_hex(rand, AUTH_TOKEN_BYTES);
    snprintf(admin_sessions[admin_session_count].token,
             sizeof(admin_sessions[admin_session_count].token), "%s", rand);
    admin_sessions[admin_session_count].expires =
        (long)time(NULL) + AUTH_SESSION_TTL;
    admin_session_count++;
    snprintf(token_out, token_size, "%s", rand);
    memset(rand, 0, sizeof(rand));
    return 1;
}

static int auth_admin_login_locked(const char *password, char *token_out,
                                   size_t token_size) {
    char check[65];

    if (!admin_hash_hex[0] || !password || !password[0]) return 0;
    auth_hash_password(admin_salt_hex, password, check);
    if (!secure_text_equal(check, admin_hash_hex)) {
        memset(check, 0, sizeof(check));
        return 0;
    }
    memset(check, 0, sizeof(check));
    return auth_create_session_locked(token_out, token_size);
}

static int web_host_authorize(const char *authorization) {
    int ok;
    int index;

    pthread_mutex_lock(&config_mutex);
    if (auth_check_key_locked(authorization, &index)) {
        api_keys[index].last_used = (long)time(NULL);
        ok = 1;
    } else {
        ok = auth_check_admin_locked(authorization);
    }
    pthread_mutex_unlock(&config_mutex);
    return ok;
}

static int web_host_admin_authorize(const char *authorization) {
    int ok;

    pthread_mutex_lock(&config_mutex);
    ok = auth_check_admin_locked(authorization);
    pthread_mutex_unlock(&config_mutex);
    return ok;
}

static int admin_json_error(char *output, size_t size, int *status,
                            int code, const char *message) {
    char *esc = settings_escape_json(message);
    int w;
    if (!esc) {
        snprintf(output, size, "{\"error\":\"server error\"}");
        *status = 500;
        return 1;
    }
    w = snprintf(output, size, "{\"error\":\"%s\"}", esc);
    free(esc);
    *status = code;
    return w >= 0;
}

static const char *admin_query_value(const char *query, const char *key,
                                     char *out, size_t out_size) {
    size_t klen;
    const char *p;

    if (!query || !key) return NULL;
    klen = strlen(key);
    p = query;
    while (*p) {
        if ((p == query || p[-1] == '&') &&
            strncmp(p, key, klen) == 0 && p[klen] == '=') {
            size_t i = 0;
            p += klen + 1;
            while (*p && *p != '&' && i + 1 < out_size) {
                out[i++] = *p++;
            }
            out[i] = '\0';
            return out;
        }
        while (*p && *p != '&') p++;
        if (*p == '&') p++;
    }
    return NULL;
}

static int web_host_admin_handle(const char *method, const char *path,
                                 const char *query, const char *body,
                                 const char *authorization, char *output,
                                 size_t output_size, int *status) {
    char token[128];
    char name[64];
    char password[256];
    char new_password[256];
    char cur_password[256];
    char id[32];
    char prefix[32];
    char *esc;
    int written;
    int i;
    long now;

    if (!method || !path || !output || !status) return 0;
    if (!body) body = "";
    if (!query) query = "";
    *status = 200;

    if (strcmp(path, "/api/admin/status") == 0 &&
        strcmp(method, "GET") == 0) {
        pthread_mutex_lock(&config_mutex);
        written = snprintf(output, output_size,
            "{\"setup_required\":%s,\"key_count\":%d}",
            admin_hash_hex[0] ? "false" : "true", api_key_count);
        pthread_mutex_unlock(&config_mutex);
        *status = 200;
        return written >= 0;
    }

    if (strcmp(path, "/api/auth/verify") == 0 &&
        strcmp(method, "POST") == 0) {
        const char *tok = auth_bearer_token(authorization);
        char body_key[160];
        int ok = 0;
        int index = -1;
        pthread_mutex_lock(&config_mutex);
        if (tok && auth_check_key_locked(authorization, &index)) {
            ok = 1;
        } else if (settings_read_string(body, "api_key", body_key,
                                        sizeof(body_key)) > 0 ||
                   settings_read_string(body, "key", body_key,
                                        sizeof(body_key)) > 0) {
            char hdr[512];
            snprintf(hdr, sizeof(hdr), "Bearer %s", body_key);
            memset(body_key, 0, sizeof(body_key));
            if (auth_check_key_locked(hdr, &index)) ok = 1;
            memset(hdr, 0, sizeof(hdr));
        }
        if (ok && index >= 0) {
            api_keys[index].last_used = (long)time(NULL);
            auth_key_prefix(api_keys[index].key, prefix, sizeof(prefix));
            esc = settings_escape_json(api_keys[index].name);
            written = snprintf(output, output_size,
                "{\"ok\":true,\"name\":\"%s\",\"prefix\":\"%s\"}",
                esc ? esc : "", prefix);
            free(esc);
        } else {
            written = snprintf(output, output_size,
                               "{\"ok\":false}");
        }
        pthread_mutex_unlock(&config_mutex);
        *status = ok ? 200 : 401;
        return written >= 0;
    }

    if (strcmp(path, "/api/admin/setup") == 0 &&
        strcmp(method, "POST") == 0) {
        pthread_mutex_lock(&config_mutex);
        if (admin_hash_hex[0]) {
            pthread_mutex_unlock(&config_mutex);
            return admin_json_error(output, output_size, status, 403,
                                    "admin already configured");
        }
        if (settings_read_string(body, "password", password,
                                 sizeof(password)) <= 0 ||
            strlen(password) < 8) {
            pthread_mutex_unlock(&config_mutex);
            return admin_json_error(output, output_size, status, 400,
                                    "password must be at least 8 characters");
        }
        auth_random_hex(admin_salt_hex, 8);
        auth_hash_password(admin_salt_hex, password, admin_hash_hex);
        memset(password, 0, sizeof(password));
        if (!auth_create_session_locked(token, sizeof(token))) {
            pthread_mutex_unlock(&config_mutex);
            return admin_json_error(output, output_size, status, 500,
                                    "unable to create session");
        }
        auth_save_locked();
        written = snprintf(output, output_size, "{\"token\":\"%s\"}", token);
        memset(token, 0, sizeof(token));
        pthread_mutex_unlock(&config_mutex);
        *status = 200;
        return written >= 0;
    }

    if (strcmp(path, "/api/admin/login") == 0 &&
        strcmp(method, "POST") == 0) {
        pthread_mutex_lock(&config_mutex);
        if (!admin_hash_hex[0]) {
            pthread_mutex_unlock(&config_mutex);
            return admin_json_error(output, output_size, status, 403,
                                    "admin setup required");
        }
        if (settings_read_string(body, "password", password,
                                 sizeof(password)) <= 0) {
            pthread_mutex_unlock(&config_mutex);
            return admin_json_error(output, output_size, status, 400,
                                    "password is required");
        }
        if (!auth_admin_login_locked(password, token, sizeof(token))) {
            memset(password, 0, sizeof(password));
            pthread_mutex_unlock(&config_mutex);
            return admin_json_error(output, output_size, status, 401,
                                    "invalid password");
        }
        memset(password, 0, sizeof(password));
        written = snprintf(output, output_size, "{\"token\":\"%s\"}", token);
        memset(token, 0, sizeof(token));
        pthread_mutex_unlock(&config_mutex);
        *status = 200;
        return written >= 0;
    }

    pthread_mutex_lock(&config_mutex);
    now = (long)time(NULL);
    if (!auth_check_admin_locked(authorization)) {
        pthread_mutex_unlock(&config_mutex);
        return admin_json_error(output, output_size, status, 401,
                                "admin authentication required");
    }
    pthread_mutex_unlock(&config_mutex);

    if (strcmp(path, "/api/admin/logout") == 0 &&
        strcmp(method, "POST") == 0) {
        const char *tok = auth_bearer_token(authorization);
        pthread_mutex_lock(&config_mutex);
        for (i = 0; i < admin_session_count; i++) {
            if (tok && secure_text_equal(tok, admin_sessions[i].token)) {
                memset(admin_sessions[i].token, 0,
                       sizeof(admin_sessions[i].token));
                memmove(admin_sessions + i, admin_sessions + i + 1,
                        sizeof(admin_sessions[0]) *
                        (size_t)(admin_session_count - i - 1));
                admin_session_count--;
                break;
            }
        }
        pthread_mutex_unlock(&config_mutex);
        snprintf(output, output_size, "{\"ok\":true}");
        *status = 200;
        return 1;
    }

    if (strcmp(path, "/api/admin/keys") == 0 &&
        strcmp(method, "GET") == 0) {
        size_t used = 0;
        pthread_mutex_lock(&config_mutex);
        used = (size_t)snprintf(output, output_size, "{\"keys\":[");
        for (i = 0; i < api_key_count; i++) {
            char item[512];
            auth_key_prefix(api_keys[i].key, prefix, sizeof(prefix));
            esc = settings_escape_json(api_keys[i].name);
            written = snprintf(item, sizeof(item),
                "%s{\"id\":\"%s\",\"name\":\"%s\",\"key\":\"%s\","
                "\"prefix\":\"%s\",\"created\":%ld,\"last_used\":%ld}",
                i == 0 ? "" : ",", api_keys[i].id, esc ? esc : "",
                api_keys[i].key, prefix, api_keys[i].created,
                api_keys[i].last_used);
            free(esc);
            if (written < 0 ||
                used + (size_t)written + 2 >= output_size) {
                pthread_mutex_unlock(&config_mutex);
                return admin_json_error(output, output_size, status, 500,
                                        "unable to list keys");
            }
            memcpy(output + used, item, (size_t)written);
            used += (size_t)written;
        }
        if (used + 2 >= output_size) {
            pthread_mutex_unlock(&config_mutex);
            return admin_json_error(output, output_size, status, 500,
                                    "unable to list keys");
        }
        memcpy(output + used, "]}", 3);
        pthread_mutex_unlock(&config_mutex);
        *status = 200;
        return 1;
    }

    if (strcmp(path, "/api/admin/keys") == 0 &&
        strcmp(method, "POST") == 0) {
        char rand[65];
        char full[128];
        pthread_mutex_lock(&config_mutex);
        if (api_key_count >= AUTH_KEYS_MAX) {
            pthread_mutex_unlock(&config_mutex);
            return admin_json_error(output, output_size, status, 400,
                                    "key limit reached");
        }
        name[0] = '\0';
        if (settings_read_string(body, "name", name, sizeof(name)) < 0) {
            pthread_mutex_unlock(&config_mutex);
            return admin_json_error(output, output_size, status, 400,
                                    "invalid name");
        }
        if (!name[0]) snprintf(name, sizeof(name), "key-%d",
                               api_key_count + 1);
        auth_random_hex(rand, 24);
        snprintf(api_keys[api_key_count].id,
                 sizeof(api_keys[api_key_count].id), "k%ld%02d",
                 (long)time(NULL) % 1000000, api_key_count);
        snprintf(api_keys[api_key_count].name,
                 sizeof(api_keys[api_key_count].name), "%s", name);
        snprintf(full, sizeof(full), "tk_%s", rand);
        snprintf(api_keys[api_key_count].key,
                 sizeof(api_keys[api_key_count].key), "%s", full);
        api_keys[api_key_count].created = (long)time(NULL);
        api_keys[api_key_count].last_used = 0;
        memset(rand, 0, sizeof(rand));
        auth_key_prefix(full, prefix, sizeof(prefix));
        esc = settings_escape_json(name);
        written = snprintf(output, output_size,
            "{\"id\":\"%s\",\"name\":\"%s\",\"key\":\"%s\","
            "\"prefix\":\"%s\",\"created\":%ld}",
            api_keys[api_key_count].id, esc ? esc : "", full, prefix,
            api_keys[api_key_count].created);
        free(esc);
        api_key_count++;
        auth_save_locked();
        memset(full, 0, sizeof(full));
        pthread_mutex_unlock(&config_mutex);
        *status = 200;
        return written >= 0;
    }

    if (strcmp(path, "/api/admin/keys") == 0 &&
        (strcmp(method, "DELETE") == 0 || strcmp(method, "POST") == 0)) {
        int found = 0;
        id[0] = '\0';
        if (!admin_query_value(query, "id", id, sizeof(id)))
            settings_read_string(body, "id", id, sizeof(id));
        if (!id[0]) {
            return admin_json_error(output, output_size, status, 400,
                                    "key id is required");
        }
        pthread_mutex_lock(&config_mutex);
        for (i = 0; i < api_key_count; i++) {
            if (strcmp(api_keys[i].id, id) == 0) {
                memset(api_keys[i].key, 0, sizeof(api_keys[i].key));
                memmove(api_keys + i, api_keys + i + 1,
                        sizeof(api_keys[0]) *
                        (size_t)(api_key_count - i - 1));
                api_key_count--;
                found = 1;
                break;
            }
        }
        if (found) auth_save_locked();
        pthread_mutex_unlock(&config_mutex);
        if (!found)
            return admin_json_error(output, output_size, status, 404,
                                    "key not found");
        snprintf(output, output_size, "{\"ok\":true}");
        *status = 200;
        return 1;
    }

    if (strcmp(path, "/api/admin/password") == 0 &&
        strcmp(method, "POST") == 0) {
        pthread_mutex_lock(&config_mutex);
        if (settings_read_string(body, "new_password", new_password,
                                 sizeof(new_password)) <= 0 &&
            settings_read_string(body, "password", new_password,
                                 sizeof(new_password)) <= 0) {
            pthread_mutex_unlock(&config_mutex);
            return admin_json_error(output, output_size, status, 400,
                                    "new_password is required");
        }
        if (strlen(new_password) < 8) {
            memset(new_password, 0, sizeof(new_password));
            pthread_mutex_unlock(&config_mutex);
            return admin_json_error(output, output_size, status, 400,
                                    "password must be at least 8 characters");
        }
        if (settings_read_string(body, "current_password", cur_password,
                                 sizeof(cur_password)) > 0 ||
            settings_read_string(body, "current", cur_password,
                                 sizeof(cur_password)) > 0) {
            char check[65];
            auth_hash_password(admin_salt_hex, cur_password, check);
            memset(cur_password, 0, sizeof(cur_password));
            if (!secure_text_equal(check, admin_hash_hex)) {
                memset(check, 0, sizeof(check));
                memset(new_password, 0, sizeof(new_password));
                pthread_mutex_unlock(&config_mutex);
                return admin_json_error(output, output_size, status, 401,
                                        "current password is incorrect");
            }
            memset(check, 0, sizeof(check));
        }
        auth_random_hex(admin_salt_hex, 8);
        auth_hash_password(admin_salt_hex, new_password, admin_hash_hex);
        memset(new_password, 0, sizeof(new_password));
        admin_session_count = 0;
        auth_save_locked();
        if (!auth_create_session_locked(token, sizeof(token))) {
            pthread_mutex_unlock(&config_mutex);
            snprintf(output, output_size, "{\"ok\":true}");
            *status = 200;
            return 1;
        }
        written = snprintf(output, output_size, "{\"ok\":true,\"token\":\"%s\"}",
                           token);
        memset(token, 0, sizeof(token));
        pthread_mutex_unlock(&config_mutex);
        *status = 200;
        return written >= 0;
    }

    (void)now;
    return 0;
}

static const WebUIHost webui_host = {
    web_host_model_count,
    web_host_model_info,
    web_host_complete,
    web_host_settings_json,
    web_host_apply_settings,
    web_host_cors_enabled,
    web_host_authorize,
    web_host_admin_authorize,
    web_host_admin_handle
};

static void initialize_engine_state(void) {
    memset(&engine_state, 0, sizeof(engine_state));
    engine_state.cfg_temp = models[current_model].default_temp;
    engine_state.cfg_noise = models[current_model].default_noise;
    engine_state.cfg_freq_penalty = models[current_model].default_freq_penalty;
    engine_state.cfg_rep_penalty = models[current_model].default_rep_penalty;
    engine_state.cfg_top_p = models[current_model].default_top_p;
    engine_state.cfg_presence_penalty =
        models[current_model].default_presence_penalty;
    engine_state.effort_mode =
        models[current_model].engine->default_effort_mode;
    sync_engine_state(&engine_state, hist_buf, hist_tokens, hist_lens,
                      &hist_cnt);
    snprintf(engine_state.system_prompt, SYSTEM_PROMPT_MAX,
             "You are %s, a very tuff AI assistant.",
             models[current_model].name);
}

static void free_engine_state(void) {
    free(engine_state.last_thinking);
    free(engine_state.last_response);
    free(engine_state.last_response_before_tool);
    free(engine_state.last_tool);
    free(engine_state.last_tool_input);
}

static void handle_sigwinch(int sig) {
    (void)sig;
    got_sigwinch = 1;
}

static void handle_sigint(int sig) {
    (void)sig;
    got_sigint = 1;
}

static void print_usage(const char *program) {
    printf("Usage: %s [--webui] [--port <port>] [--workers <n>] [--admin-password <pw>] [--auth-file <path>]\n", program);
    printf("  --webui       Start the WebUI and OpenAI-compatible API\n");
    printf("  --port <port> Listen on localhost (default: %d)\n",
           WEBUI_DEFAULT_PORT);
    printf("  --workers <n> Parallel generations (default: CPU count)\n");
    printf("  --admin-password <pw> Set admin password (min 8 chars, env TUFFAI_ADMIN_PASSWORD)\n");
    printf("  --auth-file <path> Auth storage file (default: tuffai_auth.json, env TUFFAI_AUTH_FILE)\n");
}

int main(int argc, char **argv) {
    char input_buf[INPUT_MAX];
    int input_pos;
    int ch;
    int cmd_result;
    int running;
    int web_mode;
    int web_port;
    int port_given;
    int workers_given;
    long parsed_workers;
    int argument;
    int web_result;
    char user_line[INPUT_MAX + 16];
    char *port_end;
    long parsed_port;
    struct sigaction sa_winch;
    struct sigaction sa_int;
    char cli_admin_pw[256] = "";

    web_mode = 0;
    web_port = WEBUI_DEFAULT_PORT;
    port_given = 0;
    workers_given = 0;
    {
        const char *env_auth = getenv("TUFFAI_AUTH_FILE");
        if (env_auth && env_auth[0])
            snprintf(auth_file_path, sizeof(auth_file_path), "%s", env_auth);
    }
    for (argument = 1; argument < argc; argument++) {
        if (strcmp(argv[argument], "--webui") == 0) {
            web_mode = 1;
        } else if (strcmp(argv[argument], "--auth-file") == 0) {
            if (argument + 1 >= argc) {
                fprintf(stderr, "--auth-file requires a value\n");
                return 2;
            }
            snprintf(auth_file_path, sizeof(auth_file_path), "%s",
                     argv[++argument]);
        } else if (strcmp(argv[argument], "--admin-password") == 0) {
            if (argument + 1 >= argc) {
                fprintf(stderr, "--admin-password requires a value\n");
                return 2;
            }
            {
                const char *pw = argv[++argument];
                if (strlen(pw) < 8) {
                    fprintf(stderr, "Admin password must be at least 8 characters\n");
                    return 2;
                }
                snprintf(cli_admin_pw, sizeof(cli_admin_pw), "%s", pw);
            }
        } else if (strcmp(argv[argument], "--workers") == 0) {
            if (argument + 1 >= argc) {
                fprintf(stderr, "--workers requires a value\n");
                return 2;
            }
            errno = 0;
            port_end = NULL;
            parsed_workers = strtol(argv[++argument], &port_end, 10);
            if (errno != 0 || !port_end || *port_end || parsed_workers < 1 ||
                parsed_workers > 1024) {
                fprintf(stderr, "Workers must be between 1 and 1024\n");
                return 2;
            }
            webui_set_concurrency((int)parsed_workers);
            workers_given = 1;
        } else if (strcmp(argv[argument], "--port") == 0) {
            if (argument + 1 >= argc) {
                fprintf(stderr, "--port requires a value\n");
                return 2;
            }
            errno = 0;
            port_end = NULL;
            parsed_port = strtol(argv[++argument], &port_end, 10);
            if (errno != 0 || !port_end || *port_end || parsed_port < 1 ||
                parsed_port > 65535) {
                fprintf(stderr, "Port must be between 1 and 65535\n");
                return 2;
            }
            web_port = (int)parsed_port;
            port_given = 1;
        } else if (strcmp(argv[argument], "--help") == 0 ||
                   strcmp(argv[argument], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[argument]);
            print_usage(argv[0]);
            return 2;
        }
    }
    if (!web_mode && port_given) {
        fprintf(stderr, "--port requires --webui\n");
        return 2;
    }
    if (!web_mode && workers_given) {
        fprintf(stderr, "--workers requires --webui\n");
        return 2;
    }
    if (curl_global_init(CURL_GLOBAL_ALL) != CURLE_OK) {
        fprintf(stderr, "Unable to initialize HTTP client.\n");
        return 1;
    }
    if (!setlocale(LC_CTYPE, "")) {
        fprintf(stderr, "Unable to initialize the current locale.\n");
        return 1;
    }
    if (!net_init()) {
        fprintf(stderr, "Unable to obtain operating-system entropy.\n");
        return 1;
    }
    auth_load_file();
    if (cli_admin_pw[0]) {
        pthread_mutex_lock(&config_mutex);
        auth_random_hex(admin_salt_hex, 8);
        auth_hash_password(admin_salt_hex, cli_admin_pw, admin_hash_hex);
        memset(cli_admin_pw, 0, sizeof(cli_admin_pw));
        auth_save_locked();
        pthread_mutex_unlock(&config_mutex);
    }
    {
        const char *env_pw = getenv("TUFFAI_ADMIN_PASSWORD");
        if (env_pw && env_pw[0] && !admin_hash_hex[0]) {
            if (strlen(env_pw) < 8) {
                fprintf(stderr, "TUFFAI_ADMIN_PASSWORD must be at least 8 characters\n");
                return 2;
            }
            pthread_mutex_lock(&config_mutex);
            auth_random_hex(admin_salt_hex, 8);
            auth_hash_password(admin_salt_hex, env_pw, admin_hash_hex);
            auth_save_locked();
            pthread_mutex_unlock(&config_mutex);
        }
    }

    initialize_engine_state();
    if (web_mode) {
        pthread_mutex_lock(&config_mutex);
        if (!admin_hash_hex[0])
            printf("Admin setup required: open /#admin and set a password, or use --admin-password.\n");
        else
            printf("Auth storage: %s (%d API key(s))\n", auth_file_path,
                   api_key_count);
        pthread_mutex_unlock(&config_mutex);
        fflush(stdout);
        web_search_set_enabled(0);
        web_result = webui_run(web_port, &webui_host);
        free_engine_state();
        curl_global_cleanup();
        return web_result;
    }

    initscr();
    set_escdelay(25);
    raw();
    keypad(stdscr, TRUE);
    noecho();
    scrollok(stdscr, FALSE);

    if (has_colors()) {
        start_color();
        use_default_colors();
        init_pair(COLOR_USER, COLOR_CYAN, -1);
        init_pair(COLOR_AI, COLOR_GREEN, -1);
        init_pair(COLOR_THINK, COLOR_YELLOW, -1);
        init_pair(COLOR_STATUS, COLOR_WHITE, COLOR_BLUE);
        init_pair(COLOR_CMD, COLOR_MAGENTA, -1);
        init_pair(COLOR_SHADOW, COLOR_BLACK, -1);
        init_pair(COLOR_SELECTED, COLOR_BLACK, COLOR_WHITE);
    }

    memset(&sa_winch, 0, sizeof(sa_winch));
    sa_winch.sa_handler = handle_sigwinch;
    sigaction(SIGWINCH, &sa_winch, NULL);

    memset(&sa_int, 0, sizeof(sa_int));
    sa_int.sa_handler = handle_sigint;
    sigaction(SIGINT, &sa_int, NULL);

    chat_add("");
    snprintf(input_buf, sizeof(input_buf), "%s %s", TUFFAI_NAME, TUFFAI_VERSION);
    chat_add(input_buf);
    snprintf(input_buf, sizeof(input_buf), "Model: %s", models[current_model].name);
    chat_add(input_buf);
    chat_add("Type /help for commands. /quit to exit.");
    chat_add("");

    input_buf[0] = '\0';
    input_pos = 0;
    running = 1;

    while (running) {
        if (got_sigwinch) {
            got_sigwinch = 0;
            endwin();
            refresh();
            clear();
        }
        if (got_sigint) {
            running = 0;
            break;
        }

        draw_chat();
        draw_input(input_buf, input_pos);
        if (input_pos > 0 && input_buf[0] == '/' && !strchr(input_buf, ' '))
            draw_autocomplete(input_buf, input_pos);
        {
            int ir, ic;
            int md;
            int cw;
            getmaxyx(stdscr, ir, ic);
            md = ic - 6;
            if (md < 0) md = 0;
            cw = utf8_display_width(input_buf);
            if (cw > md) cw = md;
            move(ir - 2, 5 + cw);
        }
        refresh();

        ch = getch();

        if (ch == '\n' || ch == KEY_ENTER) {
            if (input_pos == 0) continue;
            input_buf[input_pos] = '\0';

            if (input_buf[0] == '/') {
                cmd_result = handle_command(input_buf);
                if (cmd_result < 0) {
                    running = 0;
                    break;
                }
            } else {
                snprintf(user_line, sizeof(user_line), "You: %s", input_buf);
                chat_add_c(user_line, COLOR_USER);
                generate_response(input_buf);
                if (train_mode && engine_state.last_response &&
                    engine_state.last_response[0])
                    train_append(input_buf, engine_state.last_thinking,
                                 engine_state.last_response_before_tool,
                                 engine_state.last_tool,
                                 engine_state.last_tool_input,
                                 engine_state.last_response);
                chat_add("");
            }

            input_buf[0] = '\0';
            input_pos = 0;
        } else if (ch == KEY_BACKSPACE || ch == 127 || ch == 8) {
            if (input_pos > 0) {
                input_pos = utf8_step_back(input_buf, input_pos);
                input_buf[input_pos] = '\0';
            }
        } else if (ch == KEY_UP) {
            if (chat_scroll < chat_count - 1) chat_scroll++;
        } else if (ch == KEY_DOWN) {
            if (chat_scroll > 0) chat_scroll--;
        } else if (ch == KEY_PPAGE) {
            chat_scroll += 10;
            if (chat_scroll > chat_count - 1) chat_scroll = chat_count - 1;
        } else if (ch == KEY_NPAGE) {
            chat_scroll -= 10;
            if (chat_scroll < 0) chat_scroll = 0;
        } else if (utf8_is_printable_byte(ch) && input_pos < INPUT_MAX - 1) {
            input_buf[input_pos++] = (char)ch;
            input_buf[input_pos] = '\0';
        }
    }

    free_engine_state();
    endwin();
    curl_global_cleanup();
    return 0;
}
