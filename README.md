# TuffAI
A really dumb AI that may produce funny output at times (Inspired by Aztekium Bot).
I don't know how i did this but this is funny.

How to build:

Run `make` (yes, that's all).

After it has built, just run the "tuffai" binary or use `make run` instead of `make`.

To choose which models would be compiled in, just do `MODELS=` (like v1, v2, or v3) after `make`. Multiple models are accepted with separating via commas (example: `MODELS=v2,v3`). Not specifying any `MODELS=` argument to make will just compile all models in.

## Dataset attribution

Part of the TuffAI v3 general dataset contains short page descriptions retrieved from the English Wikipedia API on 2026-08-30. The source descriptions are available under the [Creative Commons Attribution-ShareAlike 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/) and are attributed to [Wikipedia contributors](https://en.wikipedia.org/). TuffAI filters the source records, adds retrieval prompts and titles, and normalizes terminal punctuation.
