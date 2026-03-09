#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <signal.h>
#include <ncurses.h>
#include "knowledge/knowledge.h"
#include "net.h"
#include "features.h"
#include "tokenizer.h"
#include "wikifetch.h"
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
static int train_mode = 0;

static char hist_buf[HIST_MAX][HIST_LEN];
static int hist_tokens[HIST_MAX][MAX_TOKENS];
static int hist_lens[HIST_MAX];
static int hist_cnt = 0;

static EngineState engine_state;

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
    {
        "TuffAI-v2 (Preview)",
        "Latest TuffAI Version",
        75.0f,
        2.5f,
        19.5f,
        19.0f,
        0.02f,
        10.0f,
        &engine_tuffai_v2,
    },
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
};

#define MODEL_COUNT (int)(sizeof(models) / sizeof(models[0]))

static int current_model = 0;

static void chat_add(const char *line) {
    int slot;
    int len;

    if (!line || !line[0]) return;
    slot = chat_count % CHAT_LINES;
    len = (int)strlen(line);
    if (len >= CHAT_LINE_LEN) len = CHAT_LINE_LEN - 1;
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
    if (len >= CHAT_LINE_LEN) len = CHAT_LINE_LEN - 1;
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
    int chunk;
    int cut;
    int nl;

    max_w = COLS - 2;
    if (max_w < 20) max_w = 20;
    if (max_w > CHAT_LINE_LEN - 1) max_w = CHAT_LINE_LEN - 1;

    tlen = (int)strlen(text);
    pos = 0;
    first = 1;

    while (pos < tlen) {
        if (first && prefix) {
            chunk = max_w - (int)strlen(prefix);
            if (chunk < 1) chunk = 1;
            if (pos + chunk > tlen) chunk = tlen - pos;
            for (nl = 0; nl < chunk; nl++) {
                if (text[pos + nl] == '\n') break;
            }
            if (nl < chunk) {
                cut = nl;
            } else {
                cut = chunk;
                if (pos + chunk < tlen) {
                    while (cut > 0 && text[pos + cut] != ' ') cut--;
                    if (cut <= 0) cut = chunk;
                }
            }
            snprintf(line, sizeof(line), "%s%.*s", prefix, cut, text + pos);
            chat_add_c(line, color);
            pos += cut;
            first = 0;
        } else {
            chunk = max_w;
            if (pos + chunk > tlen) chunk = tlen - pos;
            for (nl = 0; nl < chunk; nl++) {
                if (text[pos + nl] == '\n') break;
            }
            if (nl < chunk) {
                cut = nl;
            } else {
                cut = chunk;
                if (pos + chunk < tlen) {
                    while (cut > 0 && text[pos + cut] != ' ') cut--;
                    if (cut <= 0) cut = chunk;
                }
            }
            snprintf(line, sizeof(line), "%.*s", cut, text + pos);
            chat_add_c(line, color);
            pos += cut;
        }
        if (pos < tlen && text[pos] == '\n') {
            pos++;
        } else {
            while (pos < tlen && text[pos] == ' ') pos++;
        }
    }
}

static void draw_chat(void) {
    int rows, cols;
    int chat_rows;
    int start;
    int i, line_idx;
    int cpair;

    getmaxyx(stdscr, rows, cols);

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
            mvaddnstr(i + 1, 0, chat_log[line_idx % CHAT_LINES], cols - 1);
            if (cpair > 0) attroff(COLOR_PAIR(cpair));
        }
    }

    move(chat_rows + 1, 0);
    attron(A_REVERSE | COLOR_PAIR(COLOR_STATUS));
    for (i = 0; i < cols; i++) addch(' ');
    mvprintw(chat_rows + 1, 1,
        " Tokens:%d Turns:%d | T:%.1f N:%.1f FP:%.1f RP:%.1f TP:%.2f ",
        engine_state.total_tokens, engine_state.turn_count, engine_state.cfg_temp, engine_state.cfg_noise,
        engine_state.cfg_freq_penalty, engine_state.cfg_rep_penalty, engine_state.cfg_top_p);
    attroff(A_REVERSE | COLOR_PAIR(COLOR_STATUS));
}

static void draw_input(const char *buf, int cursor) {
    int rows, cols;
    int input_row;

    getmaxyx(stdscr, rows, cols);
    input_row = rows - 2;

    move(input_row, 0);
    clrtoeol();
    mvprintw(input_row, 0, "You: %s", buf);
    move(input_row, 5 + cursor);
    (void)cols;
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
    int i;
    int box_y, box_x, box_h, box_w;
    int old_model;

    selected = current_model;
    old_model = current_model;

    box_h = MODEL_COUNT + 4;
    box_w = 50;

    for (;;) {
        getmaxyx(stdscr, rows, cols);
        box_y = (rows - box_h) / 2;
        box_x = (cols - box_w) / 2;
        if (box_y < 0) box_y = 0;
        if (box_x < 0) box_x = 0;

        for (i = 0; i < box_h; i++) {
            move(box_y + i, box_x);
            attron(A_REVERSE | COLOR_PAIR(COLOR_STATUS));
            {
                int j;
                for (j = 0; j < box_w && box_x + j < cols; j++) addch(' ');
            }
            attroff(A_REVERSE | COLOR_PAIR(COLOR_STATUS));
        }

        attron(A_BOLD | A_REVERSE | COLOR_PAIR(COLOR_STATUS));
        mvprintw(box_y, box_x + 2, " Select Model ");
        attroff(A_BOLD | A_REVERSE | COLOR_PAIR(COLOR_STATUS));

        attron(A_REVERSE | COLOR_PAIR(COLOR_STATUS));
        mvprintw(box_y + box_h - 1, box_x + 2, " UP/DOWN: navigate  ENTER: select  ESC: cancel ");
        attroff(A_REVERSE | COLOR_PAIR(COLOR_STATUS));

        for (i = 0; i < MODEL_COUNT; i++) {
            move(box_y + 2 + i, box_x + 2);
            if (i == selected) {
                attron(A_BOLD | COLOR_PAIR(COLOR_USER));
                printw("> %s - %s", models[i].name, models[i].description);
                attroff(A_BOLD | COLOR_PAIR(COLOR_USER));
            } else if (i == old_model) {
                attron(A_REVERSE | COLOR_PAIR(COLOR_STATUS));
                printw("  %s - %s  *", models[i].name, models[i].description);
                attroff(A_REVERSE | COLOR_PAIR(COLOR_STATUS));
            } else {
                attron(A_REVERSE | COLOR_PAIR(COLOR_STATUS));
                printw("  %s - %s", models[i].name, models[i].description);
                attroff(A_REVERSE | COLOR_PAIR(COLOR_STATUS));
            }
        }

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

static int handle_command(const char *cmd) {
    char name[64];
    char arg[64];
    int n;
    float val;
    int i;
    char model_line[256];

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
        chat_add("/wikifetch     - toggle wiki fetching on/off");
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
        snprintf(model_line, sizeof(model_line), "History entries: %d", hist_cnt);
        chat_add(model_line);
        snprintf(model_line, sizeof(model_line), "Topics tracked: %d", engine_state.topic_n);
        chat_add(model_line);
        snprintf(model_line, sizeof(model_line), "Absorbed words: %d", engine_state.absorbed_n);
        chat_add(model_line);
        snprintf(model_line, sizeof(model_line), "Wiki: %s", wiki_enabled() ? "enabled" : "disabled");
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

    if (strcmp(name, "/wikifetch") == 0) {
        if (wiki_enabled()) {
            wiki_set_enabled(0);
            chat_add("Wiki fetching disabled.");
        } else {
            wiki_set_enabled(1);
            chat_add("Wiki fetching enabled.");
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
        n = atoi(arg);
        if (n >= 0 && n <= 100000) {
            engine_state.cfg_max_words = n;
            if (n == 0) {
                chat_add("Max words set to auto.");
            } else {
                chat_add("Max words override set.");
            }
        } else {
            chat_add("Invalid value (0-100000, 0=auto).");
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

static void sleep_with_scroll(int ms) {
    int elapsed = 0;
    int ch;
    int chunk;

    timeout(20);
    while (elapsed < ms) {
        ch = getch();
        if (ch == KEY_UP) {
            if (chat_scroll < chat_count - 1) chat_scroll++;
            draw_chat();
            refresh();
        } else if (ch == KEY_DOWN) {
            if (chat_scroll > 0) chat_scroll--;
            draw_chat();
            refresh();
        } else if (ch == KEY_PPAGE) {
            chat_scroll += 10;
            if (chat_scroll > chat_count - 1) chat_scroll = chat_count - 1;
            draw_chat();
            refresh();
        } else if (ch == KEY_NPAGE) {
            chat_scroll -= 10;
            if (chat_scroll < 0) chat_scroll = 0;
            draw_chat();
            refresh();
        }
        chunk = (ms - elapsed) < 20 ? (ms - elapsed) : 20;
        if (ch == ERR) napms(chunk);
        elapsed += 20;
    }
    timeout(-1);
}

static void cb_refresh_screen(void) {
    refresh();
}

static void cb_curs_set_fn(int visibility) {
    curs_set(visibility);
}

static const EngineCallbacks engine_callbacks = {
    chat_add,
    chat_add_c,
    chat_add_wrapped,
    draw_chat,
    show_status,
    sleep_with_scroll,
    cb_refresh_screen,
    cb_curs_set_fn
};

static void sync_engine_state(void) {
    engine_state.hist_buf = hist_buf;
    engine_state.hist_tokens = hist_tokens;
    engine_state.hist_lens = hist_lens;
    engine_state.hist_cnt = &hist_cnt;
    engine_state.color_think = COLOR_THINK;
    engine_state.color_ai = COLOR_AI;
}

static void generate_response(const char *input) {
    sync_engine_state();
    models[current_model].engine->generate_response(&engine_state, &engine_callbacks, input);
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

    net_init();

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

    initscr();
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