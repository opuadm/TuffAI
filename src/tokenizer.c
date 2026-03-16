#include "tokenizer.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

static int char_category_hash(const char *word) {
    int i;
    int vowels = 0, consonants = 0, digits = 0;
    unsigned char c;

    for (i = 0; word[i]; i++) {
        c = (unsigned char)tolower((unsigned char)word[i]);
        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u')
            vowels++;
        else if (c >= 'a' && c <= 'z')
            consonants++;
        else if (c >= '0' && c <= '9')
            digits++;
    }
    return (vowels * 137 + consonants * 251 + digits * 397) % ACTUAL_VOCAB;
}

static int reverse_hash(const char *word) {
    unsigned h = 5381;
    int len = (int)strlen(word);
    int i;

    for (i = len - 1; i >= 0; i--)
        h = h * 33 ^ (unsigned char)word[i];
    return (int)(h % ACTUAL_VOCAB);
}

static int weighted_rand(int center, int spread) {
    int offset;

    offset = (rand() % (spread * 2 + 1)) - spread;
    offset += (rand() % (spread + 1)) - spread / 2;
    return (center + offset + ACTUAL_VOCAB) % ACTUAL_VOCAB;
}

static int has_substr(const char *hay, const char *needle) {
    char lhay[2048], lneedle[256];
    int i;

    for (i = 0; hay[i] && i < 2047; i++) lhay[i] = tolower((unsigned char)hay[i]);
    lhay[i] = '\0';
    for (i = 0; needle[i] && i < 255; i++) lneedle[i] = tolower((unsigned char)needle[i]);
    lneedle[i] = '\0';
    return strstr(lhay, lneedle) != NULL;
}

int detect_pattern(const char *input) {
    int len = (int)strlen(input);
    int has_digits = 0;
    int has_ops = 0;
    int i;
    int has_code_kw = 0;

    for (i = 0; i < len; i++) {
        if (isdigit((unsigned char)input[i])) has_digits = 1;
        if (input[i] == '+' || input[i] == '-' || input[i] == '*' ||
            input[i] == '/' || input[i] == '=') has_ops = 1;
    }

    if (has_digits && has_ops) return PAT_MATH;

    has_code_kw = (has_substr(input, "code") || has_substr(input, "program") ||
                   has_substr(input, "function") || has_substr(input, "script") ||
                   has_substr(input, "snippet") || has_substr(input, "compile") ||
                   has_substr(input, "coding") || has_substr(input, "implement") ||
                   has_substr(input, "source code") || has_substr(input, "algorithm"));

    if (has_code_kw && len < 20)
        return PAT_CODE;

    if (has_code_kw && (has_substr(input, "generate") || has_substr(input, "write") ||
        has_substr(input, "create") || has_substr(input, "make") ||
        has_substr(input, "show") || has_substr(input, "give") ||
        has_substr(input, "build") || has_substr(input, "print") ||
        has_substr(input, "output") || has_substr(input, "produce")))
        return PAT_CODE;

    if (has_code_kw && (has_substr(input, " c ") || has_substr(input, " c89") ||
        has_substr(input, " c99") || has_substr(input, " c11") ||
        has_substr(input, " ansi c") || has_substr(input, "python") ||
        has_substr(input, " py ") || has_substr(input, "javascript") ||
        has_substr(input, " js ") || has_substr(input, "rust") ||
        has_substr(input, "golang") || has_substr(input, "haskell") ||
        has_substr(input, "bash") || has_substr(input, "shell") ||
        has_substr(input, "ruby") || has_substr(input, "sql")))
        return PAT_CODE;

    if (has_substr(input, "generate code") || has_substr(input, "write code") ||
        has_substr(input, "write a program") || has_substr(input, "write program") ||
        has_substr(input, "code in c") || has_substr(input, "code in python") ||
        has_substr(input, "code in javascript") || has_substr(input, "code in js") ||
        has_substr(input, "code in rust") || has_substr(input, "code in go") ||
        has_substr(input, "write function") || has_substr(input, "write a function") ||
        has_substr(input, "hello world") || has_substr(input, "fizzbuzz") ||
        has_substr(input, "fibonacci") || has_substr(input, "sort algorithm") ||
        has_substr(input, "binary search") || has_substr(input, "linked list") ||
        has_substr(input, "data structure") || has_substr(input, "array") ||
        has_substr(input, "recursion") || has_substr(input, "malloc") ||
        has_substr(input, "pointer") || has_substr(input, "struct") ||
        has_substr(input, "#include") || has_substr(input, "def ") ||
        has_substr(input, "class ") || has_substr(input, "import ") ||
        has_substr(input, "int main") || has_substr(input, "void ") ||
        has_substr(input, "printf") || has_substr(input, "scanf"))
        return PAT_CODE;

    if (has_substr(input, "translate") || has_substr(input, "in polish") ||
        has_substr(input, "in russian") || has_substr(input, "in chinese") ||
        has_substr(input, "po polsku") || has_substr(input, "in spanish") ||
        has_substr(input, "in german") || has_substr(input, "in french") ||
        has_substr(input, "in japanese") || has_substr(input, "in korean") ||
        has_substr(input, "in italian") || has_substr(input, "in portuguese") ||
        has_substr(input, "language"))
        return PAT_LANGUAGE;

    if (has_substr(input, "latest news") || has_substr(input, "breaking") ||
        has_substr(input, "headline") || has_substr(input, "current events") ||
        has_substr(input, "trending") || has_substr(input, "today's"))
        return PAT_NEWS;

    if (has_substr(input, "write a story") || has_substr(input, "write a poem") ||
        has_substr(input, "creative") || has_substr(input, "imagine") ||
        has_substr(input, "fiction") || has_substr(input, "draw") ||
        has_substr(input, "art") || has_substr(input, "paint"))
        return PAT_CREATIVE;

    if (has_substr(input, "what time") || has_substr(input, "what year") ||
        has_substr(input, "what day") || has_substr(input, "what date") ||
        has_substr(input, "when is") || has_substr(input, "how old"))
        return PAT_TIME;

    if (has_substr(input, "hello") || has_substr(input, "hi ") ||
        has_substr(input, "hey") || has_substr(input, "good morning") ||
        has_substr(input, "good evening") || has_substr(input, "greetings") ||
        has_substr(input, "howdy") || has_substr(input, "sup") ||
        has_substr(input, "yo ") || has_substr(input, "hola") ||
        has_substr(input, "bonjour") || has_substr(input, "hallo"))
        return PAT_GREETING;

    if (has_substr(input, "windows") || has_substr(input, "linux") ||
        has_substr(input, "bitcoin") || has_substr(input, "price") ||
        has_substr(input, "version") || has_substr(input, "update") ||
        has_substr(input, "computer") || has_substr(input, "software") ||
        has_substr(input, "program") || has_substr(input, "code") ||
        has_substr(input, "server") || has_substr(input, "database") ||
        has_substr(input, "algorithm") || has_substr(input, "api") ||
        has_substr(input, "docker") || has_substr(input, "kubernetes") ||
        has_substr(input, "llm") || has_substr(input, "gpu") ||
        has_substr(input, "cpu") || has_substr(input, "kernel") ||
        has_substr(input, "crypto") || has_substr(input, "blockchain") ||
        has_substr(input, "neural") || has_substr(input, "machine learning"))
        return PAT_TECH;

    if (has_substr(input, "feel") || has_substr(input, "love") ||
        has_substr(input, "hate") || has_substr(input, "sad") ||
        has_substr(input, "happy") || has_substr(input, "angry") ||
        has_substr(input, "depressed") || has_substr(input, "anxious") ||
        has_substr(input, "lonely") || has_substr(input, "grateful"))
        return PAT_EMOTIONAL;

    if (has_substr(input, "tell me") || has_substr(input, "show me") ||
        has_substr(input, "give me") || has_substr(input, "find") ||
        has_substr(input, "search") || has_substr(input, "list") ||
        has_substr(input, "explain") || has_substr(input, "describe") ||
        has_substr(input, "define") || has_substr(input, "what is"))
        return PAT_COMMAND;

    if (input[len - 1] == '?') return PAT_QUESTION;

    return PAT_NONE;
}

int detect_code_request(const char *input) {
    if (has_substr(input, " c ") || has_substr(input, " c89") ||
        has_substr(input, " c99") || has_substr(input, "in c") ||
        has_substr(input, " c11") || has_substr(input, " ansi c") ||
        has_substr(input, "in c,") || has_substr(input, "in c.") ||
        has_substr(input, "c code") || has_substr(input, "c program") ||
        has_substr(input, "c function") || has_substr(input, "c language") ||
        has_substr(input, "malloc") || has_substr(input, "pointer") ||
        has_substr(input, "struct") || has_substr(input, "#include") ||
        has_substr(input, "printf") || has_substr(input, "scanf") ||
        has_substr(input, "int main") || has_substr(input, "void ") ||
        has_substr(input, "header file") || has_substr(input, "makefile") ||
        has_substr(input, "gcc") || has_substr(input, "clang") ||
        has_substr(input, "stdlib") || has_substr(input, "stdio"))
        return CODE_LANG_C;
    if (has_substr(input, "python") || has_substr(input, " py ") ||
        has_substr(input, "py3") || has_substr(input, "python3") ||
        has_substr(input, "pip ") || has_substr(input, "django") ||
        has_substr(input, "flask") || has_substr(input, "numpy") ||
        has_substr(input, "pandas") || has_substr(input, "pytorch"))
        return CODE_LANG_PY;
    if (has_substr(input, "javascript") || has_substr(input, " js ") ||
        has_substr(input, "node") || has_substr(input, "typescript") ||
        has_substr(input, "react") || has_substr(input, "vue") ||
        has_substr(input, "angular") || has_substr(input, "npm") ||
        has_substr(input, "jquery") || has_substr(input, "express"))
        return CODE_LANG_JS;
    if (has_substr(input, "rust") || has_substr(input, "go ") ||
        has_substr(input, "golang") || has_substr(input, "haskell") ||
        has_substr(input, "sql") || has_substr(input, "bash") ||
        has_substr(input, "shell") || has_substr(input, "php") ||
        has_substr(input, "ruby") || has_substr(input, "elixir") ||
        has_substr(input, "java ") || has_substr(input, "kotlin") ||
        has_substr(input, "swift") || has_substr(input, "perl") ||
        has_substr(input, "lua") || has_substr(input, "zig") ||
        has_substr(input, "fortran") || has_substr(input, "assembly"))
        return CODE_LANG_MISC;
    return CODE_LANG_NONE;
}

int detect_language_hint(const char *input) {
    if (has_substr(input, "polish") || has_substr(input, "polski") ||
        has_substr(input, "polsku"))
        return 1;
    if (has_substr(input, "russian") || has_substr(input, "russkij") ||
        has_substr(input, "rossii"))
        return 2;
    if (has_substr(input, "chinese") || has_substr(input, "mandarin") ||
        has_substr(input, "zhongwen"))
        return 3;
    return 0;
}

static unsigned hash_word(const char *tok) {
    unsigned h = 5381;
    const char *p;

    for (p = tok; *p; p++) h = h * 33 ^ (unsigned char)*p;
    return h;
}

static int bigram_hash(const char *w1, const char *w2) {
    unsigned h1, h2;

    h1 = hash_word(w1);
    h2 = hash_word(w2);
    return (int)((h1 ^ (h2 * 2654435761u)) % ACTUAL_VOCAB);
}

static int is_vowel(char c) {
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y';
}

static int split_syllables(const char *word, char syllables[][32], int max_syl) {
    int len;
    int nsyl = 0;
    int i, start;
    int syl_len;
    int has_v;

    len = (int)strlen(word);
    if (len < 2) {
        if (len > 0 && nsyl < max_syl) {
            strncpy(syllables[nsyl], word, 31);
            syllables[nsyl][31] = '\0';
            nsyl++;
        }
        return nsyl;
    }

    start = 0;
    has_v = 0;
    for (i = 0; i < len && nsyl < max_syl; i++) {
        if (is_vowel(tolower((unsigned char)word[i]))) {
            has_v = 1;
        } else if (has_v && !is_vowel(tolower((unsigned char)word[i]))) {
            if (i + 1 < len && is_vowel(tolower((unsigned char)word[i + 1]))) {
                syl_len = i - start;
                if (syl_len > 0 && syl_len < 32) {
                    memcpy(syllables[nsyl], word + start, syl_len);
                    syllables[nsyl][syl_len] = '\0';
                    nsyl++;
                }
                start = i;
                has_v = 0;
            }
        }
    }

    if (start < len && nsyl < max_syl) {
        syl_len = len - start;
        if (syl_len < 32) {
            memcpy(syllables[nsyl], word + start, syl_len);
            syllables[nsyl][syl_len] = '\0';
            nsyl++;
        }
    }

    return nsyl;
}

static int split_subword(const char *word, char pieces[][32], int max_pieces) {
    int len;
    int npieces = 0;
    int pos;
    int chunk;

    len = (int)strlen(word);
    if (len <= 3) {
        if (npieces < max_pieces) {
            strncpy(pieces[npieces], word, 31);
            pieces[npieces][31] = '\0';
            npieces++;
        }
        return npieces;
    }

    pos = 0;
    while (pos < len && npieces < max_pieces) {
        chunk = 2 + rand() % 3;
        if (pos + chunk > len) chunk = len - pos;
        if (chunk > 0 && chunk < 32) {
            memcpy(pieces[npieces], word + pos, chunk);
            pieces[npieces][chunk] = '\0';
            npieces++;
        }
        pos += chunk;
    }

    return npieces;
}

static void soundex_code(const char *word, char *code, int code_size) {
    int i, ci;
    char c;
    char last;
    static const char map[] = {
        '0','1','2','3','0','1','2','0','0','2','2','4','5','5',
        '0','1','2','6','2','3','0','1','0','2','0','2'
    };

    if (code_size < 5) { code[0] = '\0'; return; }

    code[0] = toupper((unsigned char)word[0]);
    ci = 1;
    last = '0';
    if (isalpha((unsigned char)word[0])) {
        last = map[tolower((unsigned char)word[0]) - 'a'];
    }

    for (i = 1; word[i] && ci < 4; i++) {
        c = tolower((unsigned char)word[i]);
        if (c >= 'a' && c <= 'z') {
            char m = map[c - 'a'];
            if (m != '0' && m != last) {
                code[ci++] = m;
                last = m;
            } else if (m == '0') {
                last = '0';
            }
        }
    }

    while (ci < 4) code[ci++] = '0';
    code[ci] = '\0';
}

typedef struct {
    const char *from_code;
    const char *to_word;
} PhoneticSwap;

static const PhoneticSwap phonetic_swaps[] = {
    {"B320", "park"},
    {"B620", "bright"},
    {"C200", "cash"},
    {"C400", "climb"},
    {"C500", "coin"},
    {"D200", "dash"},
    {"F400", "flame"},
    {"G500", "gone"},
    {"H400", "helm"},
    {"K200", "cash"},
    {"L200", "lash"},
    {"M200", "mash"},
    {"M500", "monk"},
    {"P200", "patch"},
    {"P620", "print"},
    {"R200", "rash"},
    {"S300", "storm"},
    {"S400", "slime"},
    {"S530", "snake"},
    {"T230", "teach"},
    {"T400", "team"},
    {"T520", "thing"},
    {"W200", "wish"},
    {"W400", "weld"},
};

#define NUM_PHONETIC_SWAPS (sizeof(phonetic_swaps) / sizeof(phonetic_swaps[0]))

static const char *phonetic_lookup(const char *code) {
    int i;

    for (i = 0; i < (int)NUM_PHONETIC_SWAPS; i++) {
        if (strcmp(phonetic_swaps[i].from_code, code) == 0) {
            return phonetic_swaps[i].to_word;
        }
    }
    return NULL;
}

typedef struct {
    const char *context_word;
    const char *trigger_word;
    const char *replacement;
} ContextRule;

static const ContextRule context_rules[] = {
    {"what", "price",   "auction"},
    {"how",  "much",    "enormity"},
    {"is",   "bitcoin", "virtual currency"},
    {"the",  "latest",  "antiquated"},
    {"what", "time",    "hourglass"},
    {"do",   "you",     "automaton"},
    {"can",  "you",     "machine"},
    {"tell", "me",      "narrate"},
    {"how",  "to",      "procedure"},
    {"what", "is",      "ontology of"},
    {"in",   "russian", "Cyrillic script"},
    {"the",  "russian", "Slavic"},
    {"my",   "computer","abacus"},
    {"your", "name",    "designation"},
    {"good", "morning", "dawn salutation"},
};

#define NUM_CONTEXT_RULES (sizeof(context_rules) / sizeof(context_rules[0]))

int tokenize(const char *input, int *tokens, int max_tok) {
    char buf[2048];
    char lower_buf[2048];
    char *words[256];
    int nwords = 0;
    int count = 0;
    char *p;
    int pattern;
    int i, j;
    int bg;
    char syllables[8][32];
    int nsyl;
    char pieces[8][32];
    int npieces;
    char code[8];
    const char *replacement;
    unsigned h;

    if (!input || !tokens || max_tok <= 0) return 0;
    if (!input[0]) return 0;

    strncpy(buf, input, 2047);
    buf[2047] = '\0';
    for (p = buf; *p; p++) *p = tolower((unsigned char)*p);

    strncpy(lower_buf, buf, 2047);
    lower_buf[2047] = '\0';

    p = strtok(buf, " \t\n\r.,!?;:\"'()[]{}");
    while (p && nwords < 256) {
        words[nwords++] = p;
        p = strtok(NULL, " \t\n\r.,!?;:\"'()[]{}");
    }

    pattern = detect_pattern(input);

    if (pattern != PAT_NONE && count < max_tok) {
        tokens[count++] = (pattern * 137 + 42) % ACTUAL_VOCAB;
    }

    for (i = 0; i < nwords && count < max_tok; i++) {
        tokens[count++] = (int)(hash_word(words[i]) % ACTUAL_VOCAB);

        if (rand() % 5 == 0 && count < max_tok) {
            tokens[count++] = char_category_hash(words[i]);
        }

        if (rand() % 4 == 0 && count < max_tok) {
            tokens[count++] = reverse_hash(words[i]);
        }

        if (rand() % 3 == 0 && count < max_tok) {
            tokens[count++] = (int)(weighted_rand(hash_word(words[i]) % ACTUAL_VOCAB, ACTUAL_VOCAB / 8) % ACTUAL_VOCAB);
        }

        if (rand() % 3 == 0 && count < max_tok) {
            nsyl = split_syllables(words[i], syllables, 8);
            for (j = 0; j < nsyl && count < max_tok; j++) {
                tokens[count++] = (int)(hash_word(syllables[j]) % ACTUAL_VOCAB);
            }
        }

        if (rand() % 4 == 0 && count < max_tok) {
            npieces = split_subword(words[i], pieces, 8);
            for (j = 0; j < npieces && count < max_tok; j++) {
                tokens[count++] = (int)(hash_word(pieces[j]) % ACTUAL_VOCAB);
            }
        }

        if (rand() % 5 == 0 && count < max_tok) {
            soundex_code(words[i], code, sizeof(code));
            replacement = phonetic_lookup(code);
            if (replacement) {
                tokens[count++] = (int)(hash_word(replacement) % ACTUAL_VOCAB);
            } else {
                tokens[count++] = (int)(hash_word(code) % ACTUAL_VOCAB);
            }
        }

        if (rand() % 6 == 0 && i > 0 && count < max_tok) {
            tokens[count++] = (int)((hash_word(words[i]) ^ hash_word(words[rand() % nwords])) % ACTUAL_VOCAB);
        }
    }

    for (i = 0; i + 1 < nwords && count < max_tok; i++) {
        bg = bigram_hash(words[i], words[i + 1]);
        tokens[count++] = bg;

        for (j = 0; j < (int)NUM_CONTEXT_RULES && count < max_tok; j++) {
            if (strcmp(words[i], context_rules[j].context_word) == 0 &&
                strcmp(words[i + 1], context_rules[j].trigger_word) == 0) {
                tokens[count++] = (int)(hash_word(context_rules[j].replacement) % ACTUAL_VOCAB);
                break;
            }
        }
    }

    if (nwords >= 3) {
        for (i = 0; i + 2 < nwords && count < max_tok; i++) {
            h = hash_word(words[i]) ^ hash_word(words[i+1]) ^ hash_word(words[i+2]);
            tokens[count++] = (int)(h % ACTUAL_VOCAB);
        }
    }

    if (!count) {
        tokens[0] = rand() % ACTUAL_VOCAB;
        count = 1;
    }
    return count;
}

typedef struct {
    const char *word;
    const char *confused[4];
    int nconfused;
} ConfusionEntry;

static const ConfusionEntry confusion_table[] = {
    {"bitcoin",  {"currency", "virtual", "Litecoin", "cryptography"}, 4},
    {"price",    {"cost", "auction", "trade", "merchant"}, 4},
    {"computer", {"calculator", "abacus", "Babbage", "transistor"}, 4},
    {"linux",    {"Coreboot", "kernel", "Torvalds", "FreeBSD"}, 4},
    {"windows",  {"glass", "curtain", "fenestra", "Microsoft Bob"}, 4},
    {"program",  {"algorithm", "instruction", "Fortran", "punch card"}, 4},
    {"software", {"firmware", "middleware", "vaporware", "shareware"}, 4},
    {"code",     {"cipher", "Enigma", "Morse", "semaphore"}, 4},
    {"year",     {"calendar", "epoch", "solstice", "chronology"}, 4},
    {"time",     {"clock", "hourglass", "sundial", "relativity"}, 4},
    {"day",      {"sunrise", "meridian", "equinox", "twilight"}, 4},
    {"date",     {"palm fruit", "calendar", "chronology", "rendezvous"}, 4},
    {"hello",    {"salutation", "greeting", "hola", "aloha"}, 4},
    {"search",   {"expedition", "quest", "reconnaissance", "safari"}, 4},
    {"find",     {"discover", "excavate", "unearth", "archaeology"}, 4},
    {"love",     {"philia", "eros", "agape", "courtship"}, 4},
    {"hate",     {"antipathy", "aversion", "enmity", "rivalry"}, 4},
    {"feel",     {"tactile", "sensation", "proprioception", "synesthesia"}, 4},
    {"testing",  {"assessment", "Dilbert", "fabrication", "Debian"}, 4},
    {"test",     {"assessment", "examination", "trial", "experiment"}, 4},
    {"llm",      {"Master of Laws", "jurisprudence", "legal", "Punjab"}, 4},
    {"ai",       {"artificial", "intelligence", "automaton", "Turing"}, 4},
    {"russian",  {"Slavic", "Cyrillic", "Muscovy", "Siberia"}, 4},
    {"latest",   {"recent", "contemporary", "modern", "newfangled"}, 4},
    {"what",     {"definition", "taxonomy", "classification", "ontology"}, 4},
    {"how",      {"method", "procedure", "technique", "mechanism"}, 4},
    {"why",      {"causation", "rationale", "teleology", "motive"}, 4},
    {"who",      {"identity", "persona", "biography", "genealogy"}, 4},
    {"music",    {"acoustics", "harmony", "resonance", "vibration"}, 4},
    {"game",     {"strategy", "ludology", "competition", "tournament"}, 4},
    {"money",    {"currency", "barter", "numismatics", "bullion"}, 4},
    {"update",   {"revision", "amendment", "errata", "changelog"}, 4},
    {"version",  {"iteration", "edition", "revision", "variant"}, 4},
    {"generate", {"fabricate", "synthesize", "conjure", "forge"}, 4},
    {"write",    {"inscribe", "scribe", "calligraphy", "manuscript"}, 4},
    {"news",     {"gazette", "broadsheet", "telegram", "herald"}, 4},
    {"translate",{"interpretation", "cipher", "Rosetta Stone", "polyglot"}, 4},
    {"story",    {"narrative", "fable", "chronicle", "epic"}, 4},
    {"poem",     {"sonnet", "haiku", "limerick", "verse"}, 4},
    {"draw",     {"sketch", "rendering", "charcoal", "perspective"}, 4},
    {"paint",    {"fresco", "watercolor", "pigment", "canvas"}, 4},
    {"server",   {"mainframe", "daemon", "cluster", "rack"}, 4},
    {"database", {"ledger", "archive", "catalog", "repository"}, 4},
    {"api",      {"interface", "protocol", "endpoint", "gateway"}, 4},
    {"docker",   {"container", "virtualization", "sandbox", "shipyard"}, 4},
    {"gpu",      {"graphics", "shader", "CUDA", "rendering"}, 4},
    {"cpu",      {"processor", "arithmetic unit", "silicon", "transistor"}, 4},
    {"crypto",   {"cryptography", "cipher", "encryption", "steganography"}, 4},
    {"neural",   {"synapse", "dendrite", "axon", "cortex"}, 4},
    {"machine",  {"automaton", "mechanism", "apparatus", "contraption"}, 4},
    {"learning", {"pedagogy", "didactics", "apprenticeship", "tutelage"}, 4},
    {"network",  {"web", "mesh", "lattice", "interconnection"}, 4},
    {"internet", {"cyberspace", "network", "web", "digital realm"}, 4},
    {"phone",    {"telephone", "handset", "receiver", "communicator"}, 4},
    {"email",    {"electronic mail", "correspondence", "dispatch", "message"}, 4},
    {"cloud",    {"nebula", "cumulus", "vapor", "ethereal"}, 4},
    {"data",     {"datum", "information", "bits", "binary"}, 4},
    {"file",     {"document", "manuscript", "record", "dossier"}, 4},
    {"debug",    {"diagnose", "troubleshoot", "inspect", "dissect"}, 4},
    {"error",    {"anomaly", "deviation", "malfunction", "glitch"}, 4},
    {"compile",  {"assemble", "translate", "construct", "synthesize"}, 4},
    {"algorithm",{"procedure", "formula", "recipe", "heuristic"}, 4},
};

#define NUM_CONFUSION_ENTRIES (sizeof(confusion_table) / sizeof(confusion_table[0]))

void misidentify_topic(const char *input, char *out, int out_size) {
    char lower[2048];
    int i, j;
    int found;
    int pick;
    char buf[2048];
    char *words[64];
    int nwords;
    char *p;
    int longest;
    int longest_len;
    int wlen;
    char code[8];
    const char *phon;

    for (i = 0; input[i] && i < 2047; i++)
        lower[i] = tolower((unsigned char)input[i]);
    lower[i] = '\0';

    found = -1;
    for (i = 0; i < (int)NUM_CONFUSION_ENTRIES; i++) {
        if (strstr(lower, confusion_table[i].word)) {
            found = i;
            break;
        }
    }

    if (found >= 0) {
        pick = rand() % confusion_table[found].nconfused;
        snprintf(out, out_size, "%s", confusion_table[found].confused[pick]);
        return;
    }

    strncpy(buf, lower, 2047);
    buf[2047] = '\0';
    nwords = 0;
    p = strtok(buf, " \t\n.,!?;:'\"()[]{}");
    while (p && nwords < 64) {
        if (strlen(p) > 2) words[nwords++] = p;
        p = strtok(NULL, " \t\n.,!?;:'\"()[]{}");
    }

    for (i = 0; i + 1 < nwords; i++) {
        for (j = 0; j < (int)NUM_CONTEXT_RULES; j++) {
            if (strcmp(words[i], context_rules[j].context_word) == 0 &&
                strcmp(words[i + 1], context_rules[j].trigger_word) == 0) {
                snprintf(out, out_size, "%s", context_rules[j].replacement);
                return;
            }
        }
    }

    if (nwords > 0) {
        longest = 0;
        longest_len = 0;
        for (i = 0; i < nwords; i++) {
            wlen = (int)strlen(words[i]);
            if (wlen > longest_len) {
                longest_len = wlen;
                longest = i;
            }
        }

        if (rand() % 3 == 0) {
            soundex_code(words[longest], code, sizeof(code));
            phon = phonetic_lookup(code);
            if (phon) {
                snprintf(out, out_size, "%s", phon);
                return;
            }
        }

        snprintf(out, out_size, "%s", words[longest]);
    } else {
        snprintf(out, out_size, "%s", input);
    }
}
