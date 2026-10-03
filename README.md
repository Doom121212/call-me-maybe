*This project has been created as part of the 42 curriculum by rarayano, ayasser.*

# Call Me Maybe — Function Calling with Constrained Decoding

## Description

This project translates natural language requests into structured function
calls using a small LLM (`Qwen/Qwen3-0.6B`) and **constrained decoding**.
It does not answer the question; it picks the function and its typed
arguments.

Given *"What is the sum of 2 and 3?"* it produces:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

The function is always chosen by the LLM (no keyword rules), and the output
is valid, schema-compliant JSON by construction.

## Instructions

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
make install        # or: uv sync
make run            # or: uv run python -m src
make test           # unit tests (no model needed)
make lint           # flake8 + mypy
make lint-strict    # flake8 + mypy --strict (optional)
make debug          # run under pdb
make clean          # remove caches and data/output
```

Custom paths (defaults: `data/input/...` in, `data/output/function_calls.json`
out):

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

The default output path is the one given in the subject
(`data/output/function_calls.json`); any other name, such as
`data/output/function_calling_results.json`, can be chosen with `--output`.
The output directory is created if needed.

Prompts the model cannot process are reported on stderr and skipped, so the
output file never contains an invalid entry.

Grading with the moulinette (from its own directory):

```bash
uv sync
uv run python -m moulinette prepare_exercises --set private
# from the project directory:
uv run python -m src \
  --functions_definition ../moulinette/data/input/functions_definition.json \
  --input ../moulinette/data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
# back in the moulinette directory:
uv run python -m moulinette grade_student_answers --set private \
  --student_answer_path ../call-me-maybe/data/output/function_calling_results.json
```

## Algorithm explanation

Generation runs token by token through `get_logits_from_input_ids`. At every
step `mask_logits` sets the logit of each token that would break the
expected structure to `-inf`, so the model can only choose a valid token.
The vocabulary file is decoded (byte-level BPE) once into a `token id -> text`
table, and every token is classified before generation starts
(`precompute_token_sets`).

The JSON skeleton (`{"name": ..., "parameters": {...}}`) is written by the
program; the model fills in the function name and each value:

1. **Function name** (`generate_choice`): only tokens that keep the text a
   prefix of one defined function name are allowed. If a name is a prefix of
   another (`fn_add` / `fn_add_numbers`), closing-quote tokens are also
   allowed once the shorter name is complete, so the model decides.
2. **Numbers** (`generate_number`): a token is allowed only if the text so far
   plus the token is still a prefix of `-?digits(.digits)?`. The value can be
   ended (by `,` or `}`) only when it is a complete number. `integer` types
   forbid the decimal part. Values are returned as `float` / `int`.
3. **Strings** (`generate_string`): the context ends at the colon and the
   first token must open the string (`"` or ` "`, set `open_ids`). That
   token may already carry the first characters of the value (` "/` for
   `/home/...`), which is how the tokenizer splits natural text ("token
   healing"). Tokens with control characters are masked, and a small state
   machine (`scan_token`) only accepts the escapes JSON defines (`\"`,
   `\\`, `\n`, ...). A token that contains the closing quote ends the
   string, and whatever follows the quote inside that token is dropped. The
   result is decoded with `json.loads`, so a regex like `\d+` is generated
   as `\\d+` and stored correctly.
4. **Booleans** use `generate_choice` over `true` / `false`.
5. Each parameter is generated with the previously generated values already
   in the context (re-serialised with `json.dumps`, so quotes and
   backslashes are escaped properly).

The prompt (`build_prompt`) uses Qwen's chat format (thinking disabled) and
lists the functions with their parameter types and descriptions.
Generation is split in two stages, each with its own system message:

- **Choosing the function** (`NAME_PROMPT`): every function is listed, so
  this prompt only states the task, to stay short.
- **Generating the arguments** (`ARGS_PROMPT`): only the chosen function is
  listed, which leaves room for general extraction rules (copy text exactly,
  keep inner quotes as `\"`, regex escapes and character classes, write a
  named symbol as the symbol) and one example with a made-up function.

Short contexts matter because every generated token costs a full forward
pass over the whole context.

## Design decisions

- **Skeleton by code, values by model**: structural tokens are never left to
  the model, which makes invalid JSON impossible.
- **Allowed-token sets are computed once** from the vocabulary; per-step work
  is a numpy mask, and only number tokens (a few dozen) are checked with a
  regex at each step.
- **One generic choice routine** serves function names and booleans.
- **Callables instead of the model class** (`get_logits`, `encode`) are passed
  to the generators, so they are unit-tested with a scripted fake model.
- **Pydantic** validates the input files and the output entries; unsupported
  parameter types (e.g. arrays) produce a clear error for that prompt rather
  than a wrong answer.
- Only public `llm_sdk` methods are used (`encode`,
  `get_logits_from_input_ids`, `get_path_to_vocab_file`).

## Performance analysis

- **JSON validity / schema compliance**: guaranteed by construction. Names
  come from the defined set, numbers match a number grammar, strings are
  valid JSON strings, and the file is written with `json.dump`.
- **Accuracy**: function selection and numeric arguments come directly from
  the model's logits. String arguments depend on the model extracting the
  right text; the prompt rules and the escape-aware string decoder were
  added specifically for regex and quoted-text arguments, which a 0.6B
  model finds hardest.
- **Speed**: the SDK exposes no KV cache, so each generated token costs one
  full forward pass over the whole context. Run time is therefore driven by
  prompt length times the number of generated tokens, which is why the
  argument prompt only lists the chosen function, and only includes the text
  rules when the function takes a string. The program prints the model
  loading time, the time and number of model calls for each prompt, and the
  total. Measured on a laptop CPU (i5-13420H, no GPU): model ready in
  13–22s, public set 11 prompts in about 136s (116 model calls), private
  set 11 prompts in about 151s (139 calls), about 1.1s per call. Short
  prompts need 5–10 calls; the three-string regex prompts need 13–24.
- **Measured accuracy**: 11/11 on the public set and 11/11 on the private
  set, as graded by the moulinette (correct function, argument types and
  exact values).
- **Known limitations**: tokens that contain partial UTF-8 sequences (e.g.
  some emoji split across tokens) are not allowed inside strings; `array`,
  `object` and `null` parameter types are not supported.

## Challenges faced

- **Strings could not end**: an early version banned every token containing a
  quote, but the model's natural way to end a value is a combined token such
  as `",`. The model was forced to continue and produced garbage. Quote-
  containing tokens are now allowed and treated as the terminator.
- **Regex arguments**: banning backslashes made `\d` impossible. The string
  decoder now tracks escapes instead of banning them.
- **Garbage numbers**: allowing any run of digits, dots and minus signs could
  give `1.2.3` (silently turned into `0.0`). Numbers now follow a real
  grammar.
- **Token boundaries**: to save a model call, I once wrote the prefix shared
  by all names (`fn_`) into the context myself. The model naturally
  tokenises `fn_greet` as `fn` + `_g` + `reet`, so a context ending in a lone
  `_` token is something it never sees; "Greet shrek" started selecting
  `fn_get_square_root`. The name is now generated from the opening quote.
- **Tokenisation**: vocabulary entries are byte-level BPE strings (`Ġ` for a
  space), so they are decoded to real text before being compared.
- **Leading slash lost in paths**: with the opening quote written by the
  program, "Read the file at /home/user/data.json" gave `" /home/..."` (the
  model copied the ` /` token from the request), and masking leading spaces
  then gave `"home/..."`. Printing the top logits showed `home` and ` /`
  almost tied: the context ended at a token boundary the model never sees
  (`"` is normally merged with what follows, as in ` "/`). The string is now
  opened by the model itself (token healing) and the path comes out right.
- **Inner double quotes**: `Say "hello" to {name}` came back as
  `Say 'hello' to {name}`; the model avoided writing `\"`. A rule and an
  example value containing escaped quotes fixed it.
- **Category words in regex prompts**: "replace all vowels with asterisks"
  gave the regex `aeiou` and the replacement `asterisk`. Rules about
  character classes and named symbols, plus an example with an analogous
  request (punctuation marks and a star, not the test itself), fixed it. An
  earlier example using "odd digits" made the model write `[0-9]` instead of
  `\d+` for "numbers", so every prompt change was re-checked on both sets.

## Testing strategy

`make test` runs unittest suites that need no model:

- masking and token selection, JSON-string grammar (`scan_token`);
- number generation (second dot rejected, no stop before a digit, integers);
- string generation (closing-quote tokens, escapes, empty string, dangling
  backslash) and choice generation (prefix-of-longer-name case, booleans),
  driven by a scripted fake model;
- file handling: missing file, invalid JSON, wrong top-level type, bad or
  duplicate definitions, output directory creation, prompt/context building.

End to end, run `make run` and check that `data/output/function_calls.json`
parses, that each `name` exists in the definitions and that argument types
match. The moulinette (see Instructions) then grades both exercise sets:
it calls the real functions with the generated arguments, so it also checks
types (`2.0` for a `number`, `4` for an `integer`) and the exact strings.
Every change to the prompt or the decoder was validated on both sets,
because a fix for one prompt can break another. Try edge cases: empty strings, large numbers, quotes and backslashes
in prompts, ambiguous prompts, and other function files.

## Example usage

```bash
$ uv run python -m src
Loaded 5 function definitions
Loaded 11 prompts
Loading model...
Model ready in 7.4s, vocabulary of 151643 tokens
[1/11] What is the sum of 2 and 3?
  -> fn_add_numbers({'a': 2.0, 'b': 3.0})
     18.0s, 8 model calls
[3/11] Greet shrek
  -> fn_greet({'name': 'shrek'})
     7.7s, 6 model calls
...
Processed 11 prompts in 128.8s (99 model calls, 1.30s per call)
Results written to data/output/function_calls.json (11 calls, 0 skipped)
```

Error handling example:

```bash
$ uv run python -m src --input missing.json
Error: file not found: missing.json
```

## Resources

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702)
  (constrained / guided decoding)
- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259)
- [Pydantic documentation](https://docs.pydantic.dev/)

**AI usage**: an AI assistant was used to review the first version of the
code, find the string-decoding bugs described above, and draft the
refactoring, the unit tests and this README. A later AI-assisted session
(Claude) ran the moulinette on both exercise sets, inspected the model's
top logits to diagnose the path, inner-quote and vowel failures, and
implemented the string token healing, the per-stage system prompts and the
matching unit tests. I read and tested the result and can explain every
part of it.
