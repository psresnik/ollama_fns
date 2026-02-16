# ollama_fns

Simplifying the use of LLMs with Ollama, particularly focused on in-context learning. 

## Features

The code here supports:

- **Robust JSON Response Parsing**: Handle LLM responses with multiple fallback strategies (since structured-output can give worse results)
- **Likert Scale Analysis**: Implements best practices from [Licht et al. (2025)](https://aclanthology.org/2025.emnlp-main.1635/) and [Hamilton & Mimno (2025)](https://arxiv.org/pdf/2502.14969)
- **Easy Batch Processing**: Processes spreadsheet data (CSV/Excel) with LLM analysis workflows
- **Memory Management**: Built-in memory monitoring and garbage collection for large-scale analysis

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



## License

Licensed under the Apache License, Version 2.0. See LICENSE file for details.

## Support

For issues, questions, or contributions, please use the GitHub issue tracker.

---

*Developed with the help of Claude Code.*
