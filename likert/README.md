# Ollama Likert Scale Probability Extraction

## Overview

The code in this directory illustrates how to use the functionality in `ollama_fns.py` for weighted Likert-scale judgments. It's informed by [Measuring scalar constructs in social science with LLMs](https://aclanthology.org/2025.emnlp-main.1635/) (Licht et al., EMNLP 2025) and
Hamilton, S. and Mimno, D. (2025), [Lost in Space: Finding the Right Tokens for Structured Output](https://arxiv.org/pdf/2502.14969) and discussions with Alexander Hoyle about this topic.


A key take-away from the Licht et al. paper is that if you're interested in getting an LLM to do a Likert-scale rating, you're much better off considering the probabability it assigns to each option on the scale and taking a weighted average, rather than just using a one-best output. The Hamilton and Mimno paper highlights some important nuances about how to get those probabilities given variations in how an LLM might generate its response to the prompt.

In practice, the prompt you want will follow a prompt template something like this:


```
# Construct description
<Description of the construct goes here>

# Ratings instructions
You should rate <construct> on a scale from 1 to <max_value>. 
<Optional but recommended: describe the endpoints (or all points) on the scale)
Make sure to begin your response with a space followed by only the number.
```

Here's the default call you'll want to make:

```
from ollama_fns import (
    likert_extract_scale_tokens_single_position,
    likert_extract_scale_tokens_regex_guided,
    likert_extract_scale_tokens_multi_position,
    likert_validate_prompt_format,
    likert_get_probabilities_logprobs
)

def get_likert_probabilities(
      input_text: 				str,
      prompt_template: 			str,
      model: 					str,
      scale_min: 				int,
      scale_max: 				int,
      extraction_method: 		str = "multi_position",
      multi_max_positions: 		int = 5,
      regex_fallback: 			bool = False,
      uniform_fallback: 		str = "warn"
  ) 
```
The function returns a pair: `(probabilities_dict, diagnostics_dict)` where `probabilities_dict` is the distribution over the Likert-scale probabilities.

The code is set up to pay attention to the "Lost in Space" findings (Hamilton and Mimno, 2025), specifically that:  

- Tokenization matters: `3` vs `<space>3` affects probability extraction
- Leading whitespace improves performance: Models generate better structured output with space prefixes
- Multi-token sequences require conditional probability: P(token1) × P(token2|token1) × ...

## Methods for identifying the Likert scale response probabilities
### Single-position method

The simplest thing to do is use the `single_position` extraction method to identify the Likert scale response:

1. Generate one token of LLM output
2. Extract logprobs for that token position
3. Identify tokens matching Likert scale values (`"1"`, `"2"`, ..., `"5"`)
4. Sum probabilities for each scale value
5. Renormalize to create probability distribution

However see Hamilton and Mimno (2025) for why that's not a great idea.

### Multi-position method

By default, therefore, the code uses the `multi_position` method: 

1. Look at `multi-max-positions` positions (5 by default) of LLM output 
2. Extract logprobs for positions 0, 1, 2, etc.
3. Generate all possible token sequences across positions
4. For each sequence, check if the complete sequence (when stripped) equals a scale value
5. Calculate sequence probability as product of individual token probabilities, e.g. `P(sequence) = P(t₁) × P(t₂|t₁) × P(t₃|t₁,t₂)`
6. Sum probabilities for sequences yielding the same scale value

### Regex fallback

If the `regex-fallback` option is used (with either single- or multi-position as above), the code is more robust in the face of the possibility that the LLM might not generate any Likert scale value within the first few tokens. If primary method fails with `RuntimeError: Could not determine Likert scale value`:

1. Generate 15 tokens with `temperature=0.0` to capture longer responses
2. Reconstruct full generated text from token sequence  
3. Apply regex patterns looking for Likert-scale values preferring the regex that is most specific (earlier in this list): 

```python
patterns = [
    rf'\b{val}\b',           # "The rating is 3 for this"
    rf'[ ]{val}\b',          # "I give it 4"
    rf'\b{val}[ .]',         # "Rating: 2. This was bad"
    rf'^{val}$',             # "3" (entire response)
]
```

Note that although there is deduplication of pattern matches, it's still possible for matching to wind up double-counting the same Likert scale value, so this is suboptimal even if it's more robust.


## Probability Renormalization and Thresholding

The implementation handles cases where Likert scale tokens receive very low probability mass. In such cases, naive renormalization can amplify noise. For example, if Likert scale tokens have total probability 0.00002, renormalizing creates artificial confidence:

```
Before: P("1") = 0.00001, P("2") = 0.00001
After:  P("1") = 0.5, P("2") = 0.5  [Misleading!]

```

The solution implemented here is minimum-threshold filtering:

```python
min_meaningful_prob = 0.01  # Require 1% total probability mass
if total_prob > min_meaningful_prob:
    # Use actual distribution
    choice_probs = {k: v/total_prob for k, v in choice_probs.items()}
else:
    # Fall back to uniform distribution
    choice_probs = {k: 1.0/len(scale_values) for k in scale_values}

```

This approach distinguishes between meaningful uncertainty (model genuinely considers multiple scale values) and noise (scale values appear with negligible probability).




## Command Line Interface

```bash
python ollama_fns_likert.py [options] < inputs.txt

```

**Key Parameters:**

- `--promptfile FILE`: Custom prompt template (default: a simple sentiment rating)
- `--extraction-method {single_position,multi_position}` (default: multi_position)
- `--regex-fallback`: Enable hierarchical fallback to regex-guided method when primary fails (default: false)
- `--model MODEL`: Specify Ollama model (default: llama3.1:latest)
- `--scale-min N --scale-max N`: Define Likert scale range (default: 1-5)
- `--verbose`: Enable detailed diagnostic output

**Input Format:**

- Reads text inputs from stdin, one per line



## Testing

There are two test scripts.

`test_likert_comparison.sh`for:

  - Insights about good prompt design 
  - Validating the scalar-constructs paper recommendations
  - Understanding how the code works

`test_likert_methods.sh` for:

  - Quick regression testing during development
  - Basic "does it work" verification



## Alignment with referenced papers

**Research Validation Framework:**
The tool implements and validates findings from multiple peer-reviewed studies:

**Licht et al. (2025) - "Measuring Scalar Constructs in Social Science":**

- ✅ Token probability weighting outperforms modal response extraction
- ✅ Multi-position methods handle tokenization variations
- ✅ Proper prompting yields 5-10% performance improvements
- ✅ Hierarchical fallback strategies maintain robustness

**"Lost in Space" (Hamilton and Mimno 2025) - Tokenization Insights:**

- ✅ Leading whitespace instructions improve structured output
- ✅ Multi-token probability calculation via conditional dependencies
- ✅ Conventional format preference over creative instructions


---
*Written using Claude Code with significant testing/intervention.*
