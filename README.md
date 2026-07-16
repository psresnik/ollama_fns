# ollama_fns

Simplifying the use of LLMs with Ollama, particularly focused on in-context learning. 

## Features

The code here supports:

- **Robust JSON Response Parsing**: Handle LLM responses with multiple fallback strategies (since structured-output can give worse results)
- **Likert Scale Analysis**: Implements best practices from [Licht et al. (2025)](https://aclanthology.org/2025.emnlp-main.1635/) and [Hamilton & Mimno (2025)](https://arxiv.org/pdf/2502.14969)
- **Easy Batch Processing**: Processes spreadsheet data (CSV/Excel) with LLM analysis workflows
- **Memory Management**: Built-in memory monitoring and garbage collection for large-scale analysis
- **Timeout, Thinking Control, and Bounded-Output Retries**: `process_item()` supports a call-level `timeout`, an explicit `thinking` toggle, `num_predict` (output token cap), and `seed` — together these let a caller detect a hung/runaway generation and retry with a faster-converging configuration instead of waiting indefinitely; `warm_up_model()` is available to avoid a model's cold-start GPU load time being mistaken for a hung generation (see below)

## Installation

### Prerequisites
- Python 3.7+
- [Ollama](https://ollama.ai/) installed and running
- Ollama server with logprobs support (tested with 0.16.1+)

### Setup
```bash
git clone https://github.com/psresnik/ollama_fns.git  
cd ollama_fns
pip install -r requirements.txt
```

## Quick Start

### Batch Processing on Command Line

```python
python ollama_fns.py \
  --model gemma2:9b \
  --infile example/happiness_data.csv\
  --instructions example/happiness_prompt.txt \
  --debug
```


### Basic In-Context Learning
```python
from ollama_fns import *

instruction = "Rate the sentiment for this input. Your response should be JSON with two elementsd element, 'result', containing a number on a 1-to-5 scale from negative to positive, and an 'explanation'."

response_str = process_item(
    instruction,
    "llama3.1:latest",	
	"123",
    'This movie was fun but too long!',
	verbose=False,
)

value, explanation = extract_result_from_llm_json(response_str, key='result', debug=True)

print(f"Value from response: '{value}'")
print(f"Explanation from response: '{explanation}'")

```


### Basic Likert Scale Analysis
```python
from ollama_fns import likert_get_probabilities_logprobs

# Extract Likert probabilities
prompt = "Rate the sentiment: 'This movie was fun but too long!'. Your response should be a space followed by only a number on a 1-to-5 scale from negative to positive."
probs, diagnostics = likert_get_probabilities_logprobs(
    prompt=prompt,
    model="llama3.1:latest",
    scale_min=1,
    scale_max=5,
    extraction_method="multi_position"
)

print(f"Probability distribution: {probs}")

weighted_score = sum(int(k) * v for k, v in probs.items())
print(f"Weighted average: {weighted_score:.2f}")
```

### Likert Scoring with Justification Capture

To capture the model's full response text alongside the rating — for example
when prompting for "rating then justification" — use `num_predict` to set a
higher token budget. The raw response is always available in `diagnostics['full_response_text']`.

```python
from ollama_fns import likert_get_probabilities_logprobs

prompt = (
    "Rate the sentiment: 'This movie was fun but too long!'. "
    "Begin your response with a space followed by the number on a 1-to-5 scale, "
    "then provide a justification of approximately 150 tokens."
)
probs, diagnostics = likert_get_probabilities_logprobs(
    prompt=prompt,
    model="llama3.1:latest",
    scale_min=1,
    scale_max=5,
    extraction_method="multi_position",
    num_predict=150,   # override the method-driven default of 3
)

weighted_score = sum(int(k) * v for k, v in probs.items())
print(f"Weighted average: {weighted_score:.2f}")
print(f"Model response: {diagnostics['full_response_text']}")
```

`num_predict=None` (the default) uses method-appropriate token budgets
(1 / 3 / 15 for `single_position` / `multi_position` / `regex_guided`).

### Memory Management Illustration

This can be useful if your runs are crashing because of too much data being processed in a batch.

```python
from ollama_fns import process_item, get_memory_info

# Built-in memory monitoring
result = process_item(
    instruction="Analyze this text...",
    model="llama3.1:latest", 
    item_ID="doc_001",
    text=input_text,
    verbose=True,  # Shows memory usage
    enable_memory_monitoring=True,
    enable_garbage_collection=True
)
```



### Timeout and Thinking Control (for hung/runaway generations)

Some models occasionally spin for a very long time without producing a usable
response — e.g. a "thinking" model stuck in extended reasoning that never converges to
an answer. `process_item()` accepts four parameters, usable independently or together,
to detect and recover from this rather than blocking indefinitely:

- **`timeout`** (seconds, default `None` = unbounded, matching the ollama client's own
  default): if set, uses a dedicated `ollama.Client(timeout=timeout)` for that call, so
  a hung generation raises rather than blocking forever. On timeout (or any other
  client-side exception), `process_item()` does **not** raise — it catches the
  exception internally and returns a response string of the form
  `"LLM_ERROR: ERROR in LLM call for item <item_ID>: <ExceptionType>: <message>"`, and
  also prints that same message unconditionally (not gated by `verbose`). Callers
  should treat this string as a failed call — e.g. `extract_result_from_llm_json()`
  will simply fail to find the expected key in it, which is usually the right signal
  to act on (see the retry pattern below).
- **`thinking`** (bool, default `True`): when set to `False`, disables the model's
  extended-reasoning/"thinking" mode two ways at once — appending `/nothink` to the
  instruction text, *and* passing the native `think=False` request parameter. Both are
  used together because `/nothink` alone is not reliably honored under a tight
  `num_predict` cap (thinking tokens can still consume the whole budget before any
  answer is produced).
- **`num_predict`** (int, default `None` = model/server default): caps the number of
  output tokens, bounding worst-case latency deterministically regardless of whether
  the model converges quickly.
- **`seed`** (int, default `None`): for deterministic decoding, e.g. to make a retry
  attempt reproducible given a fixed seed derived from the item being processed.

**Model compatibility**: `timeout`, `num_predict`, and `seed` are generic ollama
request options with no dependency on model architecture. `think=False` is documented
by Ollama as being for "thinking models," but has been empirically confirmed (against
ollama server 0.20.4, client library 0.6.2+) to be silently accepted as a no-op on
non-thinking models (tested against `gemma2:9b` and `llama3.1:8b`) — no exception, no
behavior change, no `thinking` field in the response. So `thinking=False` is safe to
pass unconditionally without first checking whether a given model supports thinking.

**Cold-start caveat**: a model's *first* call after being loaded (or reloaded, if it
was evicted from GPU memory after its `keep_alive` window expired) pays GPU load time
in addition to generation time, and that load time counts against `timeout` just like
real generation time does. This was observed directly: `gemma2:9b`'s first call took
76s vs. 4.5s once warm. For a large model (e.g. a 20GB+ model), cold-start load time
alone could exceed a tightly-tuned timeout, misfiring as a "hung generation" on the
very first call of a run even though nothing is actually wrong.

**If you're using `timeout`, it's recommended (not required) to warm the model up
first** with `warm_up_model(model)`: a trivial, deliberately *untimed* chat call whose
only job is to absorb the load-time cost before real timing starts.

```python
from ollama_fns import warm_up_model

elapsed = warm_up_model("qwen3:8b", verbose=True)  # e.g. prints "Warmed up qwen3:8b in 4.95s"
# ... now safe to start a batch of process_item(..., timeout=...) calls
```

This is opt-in, not automatic: `process_item()` is called once per item in a typical
batch loop, so there's no single natural place inside it to do a "just once per model"
warm-up without hidden module-level state. Call `warm_up_model()` once yourself,
before starting a batch of timed calls, only when you're actually passing `timeout`.
Confirmed directly (`qwen3:8b`: 4.95s cold vs. 0.20s warm on a repeat call) that this
meaningfully separates load time from generation time.

**Retry-with-perturbation pattern**: since the underlying failure is often a
non-converging generation rather than a transient error, a plain retry with the same
parameters tends to reproduce the same failure. A more effective pattern — used in
[paircode](https://github.com/psresnik/paircode)'s `comparisons2scores.py` — is to
retry once with a *different* configuration that targets the failure mode directly:
thinking disabled, output capped via `num_predict`, and a different seed, with a
shorter timeout budget than the first attempt (since the retry's whole point is to
force a faster commit to an answer):

```python
from ollama_fns import process_item, extract_result_from_llm_json

response_str = process_item(instruction, model, item_ID, text, timeout=120)
value, explanation = extract_result_from_llm_json(response_str, key='result')

if value == 'unknown':  # first attempt failed (timeout, error, or unparseable output)
    response_str = process_item(
        instruction, model, item_ID, text,
        timeout=60, thinking=False, num_predict=500, seed=42,
    )
    value, explanation = extract_result_from_llm_json(response_str, key='result')
    # if value is still 'unknown' here, treat as a final failure and drop/flag the item
```

## License

Licensed under the Apache License, Version 2.0. See LICENSE file for details.

## Support

For issues, questions, or contributions, please use the GitHub issue tracker.

---

*Developed with the help of Claude Code.*
