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
#include <ncurses.h>
#include "knowledge/knowledge.h"
#include "net.h"
#include "features.h"
#include "tokenizer.h"
#include "websearch.h"
#include "version.h"
#include "engine.h"

#define INPUT_MAX 512
#define CHAT_LINES 4096
#define CHAT_LINE_LEN 512

#define COLOR_USER 1
#define COLOR_AI 2
#define COLOR_THINK 3
#define COLOR_STATUS 4
#define COLOR_CMD 5

static char chat_log[CHAT_LINES][CHAT_LINE_LEN];
static int chat_colors[CHAT_LINES];
static int chat_count = 0;
static int chat_scroll = 0;
static volatile int got_sigwinch = 0;
static volatile int got_sigint = 0;
static int generation_stop_requested = 0;
static int train_mode = 0;

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
#ifdef ENABLE_TUFFAI_V2
    {
        "TuffAI-v2",
        "Latest TuffAI Version",
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
    if (color_pair > 0) attron(COLOR_PAIR(color_pair));
    draw_utf8_chat_line(row, chat_log[line_index % CHAT_LINES]);
    if (color_pair > 0) attroff(COLOR_PAIR(color_pair));
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
            if (cpair > 0) attron(COLOR_PAIR(cpair));
            draw_utf8_chat_line(i + 1,
                                chat_log[line_idx % CHAT_LINES]);
            if (cpair > 0) attroff(COLOR_PAIR(cpair));
        }
    }

    move(chat_rows + 1, 0);
    attron(A_REVERSE | COLOR_PAIR(COLOR_STATUS));
    for (i = 0; i < cols; i++) addch(' ');
    mvprintw(chat_rows + 1, 1,
        " Tokens:%d Turns:%d Tok/s:%.1f | T:%.1f N:%.1f FP:%.1f RP:%.1f TP:%.2f ",
        engine_state.total_tokens, engine_state.turn_count,
        engine_state.last_tokens_per_second,
        engine_state.cfg_temp, engine_state.cfg_noise,
        engine_state.cfg_freq_penalty, engine_state.cfg_rep_penalty, engine_state.cfg_top_p);
    attroff(A_REVERSE | COLOR_PAIR(COLOR_STATUS));

    move(rows - 1, 0);
    clrtoeol();
}

static void draw_input(const char *buf, int cursor) {
    int rows, cols;
    int input_row;
    int max_display;

    getmaxyx(stdscr, rows, cols);
    input_row = rows - 2;

    move(input_row, 0);
    clrtoeol();
    max_display = cols - 6;
    if (max_display < 0) max_display = 0;
    mvprintw(input_row, 0, "You: %.*s", max_display, buf);
    move(input_row, 5 + (cursor > max_display ? max_display : cursor));
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
        if (strncmp(slash_cmds[i].cmd, buf, buflen) == 0) {
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
    box_y = rows - 3 - box_h;
    if (box_y < 1) box_y = 1;

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

static void train_append(const char *user_input, const char *assistant_output) {
    FILE *f;

    f = fopen("train.txt", "a");
    if (!f) return;
    fprintf(f, "        {.user = \"");
    train_write_escaped(f, user_input);
    fprintf(f, "\", .assistant =\n            \"");
    train_write_escaped(f, assistant_output);
    fprintf(f, "\"\n        },\n");
    fclose(f);
}

static int model_dialog(void) {
    int rows, cols;
    int selected;
    int ch;
    int i, j;
    int box_y, box_x, box_h, box_w;
    int old_model;
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
            move(box_y + 2 + i, box_x + 3);
            if (i == selected) {
                attron(A_BOLD | COLOR_PAIR(COLOR_USER));
                printw("%s> %s  -  %s", marker, models[i].name, models[i].description);
                attroff(A_BOLD | COLOR_PAIR(COLOR_USER));
            } else {
                attron(COLOR_PAIR(COLOR_STATUS));
                printw("%s  %s  -  %s", marker, models[i].name, models[i].description);
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
        line_num = 0;
        col_num = 0;
        for (i = 0; i < cursor_pos && i < len; i++) {
            if (buf[i] == '\n') {
                line_num++;
                col_num = 0;
            } else {
                col_num++;
                if (col_num >= text_w) {
                    line_num++;
                    col_num = 0;
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

            draw_line = 0;
            cur_line = 0;
            draw_col = 0;
            text_pos = 0;

            while (text_pos < len && cur_line < scroll_offset) {
                if (buf[text_pos] == '\n') {
                    cur_line++;
                } else {
                    draw_col++;
                    if (draw_col >= text_w) {
                        cur_line++;
                        draw_col = 0;
                    }
                }
                text_pos++;
            }
            draw_col = 0;

            while (text_pos < len && draw_line < visible_lines) {
                if (buf[text_pos] == '\n') {
                    draw_line++;
                    draw_col = 0;
                } else {
                    if (draw_line >= 0 && draw_col < text_w) {
                        attron(COLOR_PAIR(COLOR_STATUS));
                        mvaddch(box_y + 2 + draw_line, box_x + 2 + draw_col, buf[text_pos]);
                        attroff(COLOR_PAIR(COLOR_STATUS));
                    }
                    draw_col++;
                    if (draw_col >= text_w) {
                        draw_line++;
                        draw_col = 0;
                    }
                }
                text_pos++;
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
                len = (int)strlen(buf);
                memmove(buf + cursor_pos - 1, buf + cursor_pos, len - cursor_pos + 1);
                cursor_pos--;
            }
        } else if (ch == KEY_LEFT) {
            if (cursor_pos > 0) cursor_pos--;
        } else if (ch == KEY_RIGHT) {
            if (cursor_pos < (int)strlen(buf)) cursor_pos++;
        } else if (ch == KEY_UP) {
            {
                int target_col;
                int scan;
                int prev_line_start;
                int prev_line_end;
                target_col = 0;
                scan = cursor_pos - 1;
                while (scan >= 0 && buf[scan] != '\n') {
                    target_col++;
                    scan--;
                }
                if (scan < 0) continue;
                prev_line_end = scan;
                scan--;
                while (scan >= 0 && buf[scan] != '\n') scan--;
                prev_line_start = scan + 1;
                cursor_pos = prev_line_start + target_col;
                if (cursor_pos > prev_line_end)
                    cursor_pos = prev_line_end;
            }
        } else if (ch == KEY_DOWN) {
            {
                int target_col;
                int scan;
                int next_line_start;
                int next_line_end;
                target_col = 0;
                scan = cursor_pos - 1;
                while (scan >= 0 && buf[scan] != '\n') {
                    target_col++;
                    scan--;
                }
                len = (int)strlen(buf);
                scan = cursor_pos;
                while (scan < len && buf[scan] != '\n') scan++;
                if (scan >= len) continue;
                next_line_start = scan + 1;
                scan = next_line_start;
                while (scan < len && buf[scan] != '\n') scan++;
                next_line_end = scan;
                cursor_pos = next_line_start + target_col;
                if (cursor_pos > next_line_end)
                    cursor_pos = next_line_end;
            }
        } else if (ch == '\n' || ch == KEY_ENTER) {
            len = (int)strlen(buf);
            if (len < bufsize - 2) {
                memmove(buf + cursor_pos + 1, buf + cursor_pos, len - cursor_pos + 1);
                buf[cursor_pos] = '\n';
                cursor_pos++;
            }
        } else if (ch >= 32 && ch < 127) {
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
        chat_add("/model         - Switch model");
        chat_add("/prompt        - Edit system prompt (v2 only)");
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
                engine_state.cfg_temp = models[choice].default_temp;
                engine_state.cfg_noise = models[choice].default_noise;
                engine_state.cfg_freq_penalty = models[choice].default_freq_penalty;
                engine_state.cfg_rep_penalty = models[choice].default_rep_penalty;
                engine_state.cfg_top_p = models[choice].default_top_p;
                engine_state.cfg_presence_penalty = models[choice].default_presence_penalty;
                snprintf(model_line, sizeof(model_line), "Switched to %s", models[choice].name);
                chat_add(model_line);
            } else {
                chat_add("Model selection cancelled.");
            }
        } else {
            for (i = 0; i < MODEL_COUNT; i++) {
                if (str_casecmp(arg, models[i].name) == 0) {
                    current_model = i;
                    engine_state.cfg_temp = models[i].default_temp;
                    engine_state.cfg_noise = models[i].default_noise;
                    engine_state.cfg_freq_penalty = models[i].default_freq_penalty;
                    engine_state.cfg_rep_penalty = models[i].default_rep_penalty;
                    engine_state.cfg_top_p = models[i].default_top_p;
                    engine_state.cfg_presence_penalty = models[i].default_presence_penalty;
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
            chat_add("Train mode enabled. Writing to train.txt.");
        } else {
            chat_add("Train mode disabled.");
        }
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
    attron(A_DIM);
    clrtoeol();
    mvaddnstr(status_row, 0, msg, cols - 1);
    attroff(A_DIM);
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

static void cb_stream_text(const char *prefix, const char *text, int color) {
    int previous_count;
    int previous_scroll;

    previous_count = chat_count;
    previous_scroll = chat_scroll;
    chat_add_wrapped(prefix, text, color);
    draw_streamed_chat(previous_count, previous_scroll);
    refresh();
}

static const EngineCallbacks engine_callbacks = {
    chat_add,
    chat_add_c,
    chat_add_wrapped,
    draw_chat,
    show_status,
    cb_refresh_screen,
    cb_curs_set_fn,
    cb_generation_should_stop,
    cb_stream_text
};

static void sync_engine_state(void) {
    engine_state.hist_buf = hist_buf;
    engine_state.hist_tokens = hist_tokens;
    engine_state.hist_lens = hist_lens;
    engine_state.hist_cnt = &hist_cnt;
    engine_state.color_think = COLOR_THINK;
    engine_state.color_ai = COLOR_AI;
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

    strncpy(bufa, a, PREV_RESPONSE_LEN - 1);
    bufa[PREV_RESPONSE_LEN - 1] = '\0';
    strncpy(bufb, b, PREV_RESPONSE_LEN - 1);
    bufb[PREV_RESPONSE_LEN - 1] = '\0';

    p = strtok(bufa, " \t\n.,!?;:'\"()[]{}");
    while (p && na < 128) {
        wa[na++] = p;
        p = strtok(NULL, " \t\n.,!?;:'\"()[]{}");
    }
    p = strtok(bufb, " \t\n.,!?;:'\"()[]{}");
    while (p && nb < 128) {
        wb[nb++] = p;
        p = strtok(NULL, " \t\n.,!?;:'\"()[]{}");
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

static void generate_response(const char *input) {
    float boost;
    float orig_freq;
    float orig_rep;
    struct timespec started;
    struct timespec finished;
    double elapsed;
    long sampled_before;
    long sampled_delta;

    sync_engine_state();

    boost = compute_repetition_boost(&engine_state);
    orig_freq = engine_state.cfg_freq_penalty;
    orig_rep = engine_state.cfg_rep_penalty;
    engine_state.cfg_freq_penalty += boost;
    engine_state.cfg_rep_penalty += boost * 0.8f;

    sampled_before = engine_state.sampled_tokens;
    generation_stop_requested = 0;
    clock_gettime(CLOCK_MONOTONIC, &started);
    models[current_model].engine->generate_response(&engine_state, &engine_callbacks, input);
    generation_stop_requested = 0;
    clock_gettime(CLOCK_MONOTONIC, &finished);
    elapsed = (double)(finished.tv_sec - started.tv_sec) +
              (double)(finished.tv_nsec - started.tv_nsec) / 1000000000.0;
    sampled_delta = engine_state.sampled_tokens - sampled_before;
    if (elapsed > 0.0 && sampled_delta > 0)
        engine_state.last_tokens_per_second = (float)((double)sampled_delta / elapsed);
    else
        engine_state.last_tokens_per_second = 0.0f;

    engine_state.cfg_freq_penalty = orig_freq;
    engine_state.cfg_rep_penalty = orig_rep;
    draw_chat();
}

static void handle_sigwinch(int sig) {
    (void)sig;
    got_sigwinch = 1;
}

static void handle_sigint(int sig) {
    (void)sig;
    got_sigint = 1;
}

int main(void) {
    char input_buf[INPUT_MAX];
    int input_pos;
    int ch;
    int cmd_result;
    int running;
    char user_line[INPUT_MAX + 16];
    struct sigaction sa_winch;
    struct sigaction sa_int;

    if (!setlocale(LC_CTYPE, "")) {
        fprintf(stderr, "Unable to initialize the current locale.\n");
        return 1;
    }
    if (!net_init()) {
        fprintf(stderr, "Unable to obtain operating-system entropy.\n");
        return 1;
    }

    memset(&engine_state, 0, sizeof(engine_state));
    engine_state.cfg_temp = models[current_model].default_temp;
    engine_state.cfg_noise = models[current_model].default_noise;
    engine_state.cfg_freq_penalty = models[current_model].default_freq_penalty;
    engine_state.cfg_rep_penalty = models[current_model].default_rep_penalty;
    engine_state.cfg_top_p = models[current_model].default_top_p;
    engine_state.cfg_presence_penalty = models[current_model].default_presence_penalty;
    engine_state.hist_buf = hist_buf;
    engine_state.hist_tokens = hist_tokens;
    engine_state.hist_lens = hist_lens;
    engine_state.hist_cnt = &hist_cnt;
    engine_state.color_think = COLOR_THINK;
    engine_state.color_ai = COLOR_AI;
    strncpy(engine_state.system_prompt,
            "You are TuffAI-v2, a helpful but confused assistant. "
            "Answer questions with confidence even when unsure. ",
            SYSTEM_PROMPT_MAX - 1);
    engine_state.system_prompt[SYSTEM_PROMPT_MAX - 1] = '\0';

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
            getmaxyx(stdscr, ir, ic);
            md = ic - 6;
            if (md < 0) md = 0;
            move(ir - 2, 5 + (input_pos > md ? md : input_pos));
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
                int resp_slot;
                snprintf(user_line, sizeof(user_line), "You: %s", input_buf);
                chat_add_c(user_line, COLOR_USER);
                generate_response(input_buf);
                if (train_mode && engine_state.prev_resp_count > 0) {
                    resp_slot = (engine_state.prev_resp_count - 1) % PREV_RESPONSE_MAX;
                    train_append(input_buf, engine_state.prev_responses[resp_slot]);
                }
                chat_add("");
            }

            input_buf[0] = '\0';
            input_pos = 0;
        } else if (ch == KEY_BACKSPACE || ch == 127 || ch == 8) {
            if (input_pos > 0) {
                input_pos--;
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
        } else if (ch >= 32 && ch < 127 && input_pos < INPUT_MAX - 1) {
            input_buf[input_pos++] = (char)ch;
            input_buf[input_pos] = '\0';
        }
    }

    endwin();
    return 0;
}
