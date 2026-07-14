####################################################################################
#  In-context learning using an LLM via ollama
####################################################################################
import time
import json
import argparse
import pandas as pd
import os
import ollama
import re
import gc
import numpy as np
import sys
from typing import Dict, Tuple, List, Callable, Optional

# Comment in/out for verbose/nonverbose tracebacks on errors
from traceback_with_variables import activate_by_import

# Import psutil for memory monitoring (cross-platform)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

def get_memory_info():
    """Get current memory usage information (cross-platform)"""
    if not PSUTIL_AVAILABLE:
        return "Memory monitoring unavailable"
    
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        # RSS is resident set size (physical memory currently used)
        rss_mb = memory_info.rss / (1024 * 1024)
        # VMS is virtual memory size
        vms_mb = memory_info.vms / (1024 * 1024)
        
        # Get system memory info
        system_memory = psutil.virtual_memory()
        system_available_mb = system_memory.available / (1024 * 1024)
        system_percent = system_memory.percent
        
        return f"Process: RSS={rss_mb:.1f}MB, VMS={vms_mb:.1f}MB | System: {system_percent:.1f}% used, {system_available_mb:.0f}MB available"
    except Exception as e:
        return f"Memory monitoring error: {e}"

def cleanup_memory(verbose=False):
    """Perform garbage collection and memory cleanup"""
    # Force garbage collection
    collected = gc.collect()
    if verbose and collected > 0:
        print(f"Garbage collection freed {collected} objects")
    return collected

def read_file(file):
    try:
        with open(file, 'r') as f:
            data = f.read()   
        return data 
    except FileNotFoundError:
        print(f"File {file} could not be found.")

def process_item(instruction, model, item_ID, text, post_instruction='', input_header='\n## Input\n',
                verbose=False, thinking=True, enable_memory_monitoring=True, temperature=0.0, enable_garbage_collection=True):
    
    # Memory monitoring at start if enabled
    if enable_memory_monitoring and verbose:
        print(f"Memory before processing {item_ID}: {get_memory_info()}")
    
    if verbose:
        print("========")
        print(f'Processing {item_ID}')
        print(f'Text: {text}')
        print("========")

    # Ignoring item_ID for now
    model_input = input_header + text + "\n## JSON\n"
    if verbose:
        print("--- Instruction and model input ---")
        print(instruction)
        print(model_input)
        print(post_instruction)
        print("------")

    # Call the model
    start_time = time.time()
    
    # Modify instruction based on thinking parameter
    final_instruction = instruction
    if not thinking:
        final_instruction = f"{instruction}\n\n/nothink"
    
    messages=[
      {
        'role': 'system',
        'content': f"{final_instruction}",
      },
      {
        'role': 'user',
        'content': f'{model_input}',
      },
    ]
    if post_instruction:
        messages.append({'role':'user', 'content':f"{post_instruction}"})
    
    options = {
      'temperature': temperature
    }
    
    try:
        ollama_response = ollama.chat(model=model, messages=messages, options=options)
        end_time = time.time()
        response_str = ollama_response['message']['content']
    except Exception as e:
        end_time = time.time()
        error_msg = f"ERROR in LLM call for item {item_ID}: {type(e).__name__}: {str(e)}"
        print(error_msg)
        response_str = f"LLM_ERROR: {error_msg}"
        
        # Memory monitoring during error if enabled
        if enable_memory_monitoring and verbose:
            print(f"Memory during error for {item_ID}: {get_memory_info()}")
    
    # Cleanup large variables and perform garbage collection if enabled
    if enable_garbage_collection:
        # Delete large variables before garbage collection
        del messages, model_input
        if 'ollama_response' in locals():
            del ollama_response
        
        # Force garbage collection
        cleanup_memory(verbose=verbose)
    
    # Memory monitoring after cleanup if enabled
    if enable_memory_monitoring and verbose:
        print(f"Memory after processing {item_ID}: {get_memory_info()}")
    
    if verbose:
        print(f"LLM call to {model} took {end_time - start_time} seconds")
        print(response_str)
        print("\n\n")

    return response_str


# Process multiple items in spreadsheet infile
def process_items(model, instructions_file, infile, post_instruction_file=None, debug=False, thinking=True):

    if (debug):
        print(f"In process_items, using model {model}")

    # Use sequential counter instead of docID if none provided
    item_num = 0
    
    # Read the instruction part of the prompt from the instructions file
    instruction = read_file(instructions_file)
    if post_instruction_file:
        post_instruction = read_file(post_instruction_file)
    else:
        post_instruction = None
    
    # Read the input file
    if infile.endswith('.csv'):
        df = pd.read_csv(infile)
    elif infile.endswith('.xlsx') or infile.endswith('.xls'):
        df = pd.read_excel(infile)
    else:
        raise ValueError(f'Unsupported file format for {infile}. Please provide a csv, xlsx, or xls file.')

    # Check for  columns
    if 'text' not in df.columns:
        raise ValueError(f'Input file {infile} must contain a "text" column.')
    if 'docID' not in df.columns:
        print("No docID column found. Sequentially numbering items as the docID")

    # Iterate through the rows and process each item
    for index, row in df.iterrows():
        if 'docID' not in df.columns:
            item_num = item_num + 1
            docID = item_num
        else:
            docID = row['docID']
        text = row['text']
        if debug:
            print(f"Making LLM call for item: docID = {docID}, text={text}")
        result = process_item(instruction, model, docID, text, post_instruction=post_instruction, thinking=thinking)
        if debug:
            print(f"Result of LLM call: {result}")

        
def extract_result_from_llm_json(response_str, key='result', debug=False):
    """
    Robustly extracts the key and explanation values from a response string that should have
    the form of a json dictionary, e.g. '{<key>:result_value, explanation:explanation_string, ...}'
    
    Args:
        response_str (str): A string that should contain a JSON dictionary with the desired key.
        key (str): The key for the value to extract from the JSON response.

    Returns:
        str: The value of the specified key if found, 'unknown' otherwise.

    Tries first with assumption that response string is a json dictionary containing the key.
    Then backs off to possibility that response string *contains* such a dictionary.
    Then backs off to looking for key:'value' pair in the string without assuming json.
    """
    key_value = None

    try:
        # Attempt to parse the entire string as JSON
        parsed_dict = json.loads(response_str)
        key_value   = parsed_dict.get(key)
        explanation = parsed_dict.get('explanation')
        if key_value is not None:
            if debug:
                print(f"Key '{key}' value: {key_value}")
            return key_value, explanation
        else:
            if debug:
                print(f"Parse error 1. No '{key}' found in '{response_str}'")
                print(f"parsed_dict    = {parsed_dict}")
                print(f"response_str   = {response_str}")
            return 'unknown', ''

    except json.JSONDecodeError:
        # If parsing the entire string as JSON fails, try to extract the JSON part
        try:
            json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_dict = json.loads(json_str)
                key_value = parsed_dict.get(key)
                if key_value is not None:
                    if debug:
                        print(f"Key '{key}' value: {key_value}")
                    explanation = parsed_dict.get('explanation')
                    return key_value, (explanation if explanation else '')
                else:
                    if debug:
                        print(f"Parse error 2. No '{key}' found in '{response_str}'")
                        print(f"json_match     = {json_match}")
                        print(f"json_str       = {json_str}")
                        print(f"response_str   = {response_str}")
                    return 'unknown', ''
            else:
                if debug:
                    print(f"Parse error 3. No '{key}' found in '{response_str}'")
                    print(f"response_str   = {response_str}")
                return 'unknown', ''
        except json.JSONDecodeError:
            if debug:
                print(f"Parse error 4. No '{key}' found in '{response_str}'")
            return 'unknown', ''
        except Exception as e:
            if debug:
                print(f"Parse error 5. No '{key}' found in '{response_str}'")
                print(f"An error of type {type(e).__name__} occurred: {e}")
            return 'unknown', ''

    # If all else fails, use regex to find the key-value pair directly
    try:
        key_match = re.search(fr'"{key}"\s*:\s*"([^"]+)"', response_str)
        if key_match:
            key_value = key_match.group(1)
        else:
            if debug:
                print(f"Parse error 6. No '{key}' found in '{response_str}'")
            return 'unknown', ''
    except Exception as e:
        if debug:
            print(f"Parse error 7. No '{key}' found in '{response_str}'")
        return 'unknown', ''

    return key_value


####################################################################################
#  Likert Scale Probability Extraction Functions
####################################################################################

def likert_extract_scale_tokens_single_position(
    logprobs_data: list,
    scale_values: List[str],
    verbose: bool = False
) -> Tuple[Dict[str, float], Dict]:
    """
    Extract scale probabilities from first token position only (original approach).
    """
    choice_probs = {val: 0.0 for val in scale_values}
    diagnostics = {
        'method': 'single_position',
        'token_variants_found': {},
        'raw_top_tokens': [],
        'total_scale_probability': 0.0,
        'generated_token': None,
        'generated_logprob': None
    }
    
    if not logprobs_data or len(logprobs_data) == 0:
        return choice_probs, diagnostics
        
    first_token_logprobs = logprobs_data[0]
    
    # Process main token
    main_token = first_token_logprobs.token
    main_logprob = first_token_logprobs.logprob
    diagnostics['generated_token'] = main_token
    diagnostics['generated_logprob'] = main_logprob
    
    # Collect all tokens to check: use top_logprobs if available, otherwise just main token
    if first_token_logprobs.top_logprobs:
        all_tokens_to_check = [
            (alt.token, alt.logprob) 
            for alt in first_token_logprobs.top_logprobs
        ]
    else:
        all_tokens_to_check = [(main_token, main_logprob)]
    
    # Process all tokens
    for token, logprob in all_tokens_to_check:
        prob = np.exp(logprob)
        token_stripped = token.strip()
        
        # Store for diagnostics
        diagnostics['raw_top_tokens'].append({
            'token': token,
            'token_repr': repr(token),
            'stripped': token_stripped,
            'logprob': logprob,
            'prob': prob
        })
        
        # Check if this matches a scale value
        if token_stripped in scale_values:
            choice_probs[token_stripped] += prob
            
            # Track which variants we found
            if token_stripped not in diagnostics['token_variants_found']:
                diagnostics['token_variants_found'][token_stripped] = []
            diagnostics['token_variants_found'][token_stripped].append(
                (repr(token), prob)
            )
    
    diagnostics['total_scale_probability'] = sum(choice_probs.values())
    return choice_probs, diagnostics


def _deduplicate_regex_matches_best_pattern(matches: List[Dict]) -> List[Dict]:
    """
    Apply 'best pattern per location' deduplication to eliminate double-counting.
    For each character location, keep only the match with the highest pattern specificity.
    """
    # Define pattern specificity ranking (higher score = more specific)
    # Use pattern templates that match the actual pattern generation logic
    def get_pattern_specificity(pattern):
        """Get specificity score for a pattern."""
        if pattern.startswith('^') and pattern.endswith('$'):
            return 100  # Exact match - highest priority
        elif r'\b' in pattern and not (' ' in pattern or '.' in pattern):
            return 80   # Word boundaries only - high priority  
        elif r'\b' in pattern and ('[ .]' in pattern):
            return 70   # Number + space/period - medium-high priority
        elif pattern.startswith('[ ]') and r'\b' in pattern:
            return 60   # Space + number - medium priority
        else:
            return 40   # Default - lowest priority
    
    # Group matches by overlapping character locations
    # We need to group overlapping spans, not just identical spans
    location_groups = []
    
    for match in matches:
        char_start, char_end = match['char_span']
        
        # Find if this match overlaps with any existing group
        found_group = None
        for group in location_groups:
            for existing_match in group:
                existing_start, existing_end = existing_match['char_span']
                
                # Check if spans overlap
                if not (char_end <= existing_start or char_start >= existing_end):
                    found_group = group
                    break
            if found_group:
                break
        
        if found_group:
            found_group.append(match)
        else:
            # Create new group
            location_groups.append([match])
    
    # For each group of overlapping matches, keep only the best match
    deduplicated_matches = []
    for match_group in location_groups:
        if len(match_group) == 1:
            # No duplicates, keep as-is
            deduplicated_matches.append(match_group[0])
        else:
            # Multiple overlapping matches - pick the best one
            best_match = None
            best_score = -1
            
            for match in match_group:
                # Get specificity score for this pattern
                pattern = match['pattern']
                score = get_pattern_specificity(pattern)
                
                if score > best_score:
                    best_score = score
                    best_match = match
            
            if best_match:
                deduplicated_matches.append(best_match)
    
    return deduplicated_matches


def likert_extract_scale_tokens_regex_guided(
    logprobs_data: list,
    scale_values: List[str],
    context_window: int = 5,
    verbose: bool = False
) -> Tuple[Dict[str, float], Dict]:
    """
    Extract scale probabilities using regex-guided sequence extraction.
    Scans for scale patterns in token sequence, then applies sequence probability
    calculation only around matches.
    """
    choice_probs = {val: 0.0 for val in scale_values}
    diagnostics = {
        'method': 'regex_guided',
        'token_variants_found': {},
        'raw_top_tokens': [],
        'total_scale_probability': 0.0,
        'matches_found': [],
        'full_token_sequence': [],
        'warnings': []
    }
    
    if not logprobs_data:
        return choice_probs, diagnostics
    
    # Get the generated token sequence (most likely tokens from each position)
    generated_sequence = []
    for pos, pos_data in enumerate(logprobs_data):
        token = pos_data.token
        generated_sequence.append((pos, token))
        # Escape newlines in token for diagnostic display
        token_escaped = token.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        diagnostics['full_token_sequence'].append(f"pos{pos}:'{token_escaped}'")
    
    # Create regex patterns to find scale values
    scale_patterns = []
    for val in scale_values:
        # Match scale value with optional whitespace/punctuation
        patterns = [
            rf'\b{val}\b',           # word boundary (e.g., " 3 ")
            rf'[ ]{val}\b',          # space + number (e.g., " 3")
            rf'\b{val}[ .]',         # number + space/period (e.g., "3 " or "3.")
            rf'^{val}$',             # exact match
        ]
        scale_patterns.extend([(pattern, val) for pattern in patterns])
    
    # Scan the token sequence for matches
    matches = []
    full_text = ''.join(token for pos, token in generated_sequence)
    
    for pattern, scale_val in scale_patterns:
        for match in re.finditer(pattern, full_text):
            start_char = match.start()
            end_char = match.end()
            
            # Find which token positions contain this match
            char_pos = 0
            start_token_pos = None
            end_token_pos = None
            
            for pos, token in generated_sequence:
                if start_token_pos is None and char_pos + len(token) > start_char:
                    start_token_pos = pos
                if char_pos + len(token) >= end_char:
                    end_token_pos = pos
                    break
                char_pos += len(token)
            
            if start_token_pos is not None and end_token_pos is not None:
                matches.append({
                    'scale_value': scale_val,
                    'pattern': pattern,
                    'match_text': match.group(),
                    'start_pos': start_token_pos,
                    'end_pos': end_token_pos,
                    'char_span': (start_char, end_char)
                })
    
    # Phase 2: Apply "best pattern per location" deduplication
    matches = _deduplicate_regex_matches_best_pattern(matches)
    
    diagnostics['matches_found'] = matches
    
    if not matches:
        # No matches found - issue warning and fall back to first token
        diagnostics['warnings'].append("No scale value patterns found in token sequence")
        print(f"Warning: No scale patterns found in sequence: {full_text}", file=sys.stderr)
        
        # Fallback to single position on first token
        return likert_extract_scale_tokens_single_position(logprobs_data, scale_values, verbose)
    
    # Process each match with sequence probability calculation
    processed_regions = set()  # Avoid double-counting overlapping regions
    
    for match in matches:
        scale_val = match['scale_value']
        start_pos = max(0, match['start_pos'] - context_window // 2)
        end_pos = min(len(logprobs_data), match['end_pos'] + context_window // 2 + 1)
        
        region_key = (start_pos, end_pos, scale_val)
        if region_key in processed_regions:
            continue
        processed_regions.add(region_key)
        
        # Extract logprobs for this region
        region_logprobs = logprobs_data[start_pos:end_pos]
        
        # Debug: show what tokens are in this region
        if verbose:
            print(f"    Processing region {start_pos}-{end_pos} for match: {match}")
            for i, pos_data in enumerate(region_logprobs):
                print(f"      Region pos {i} (global pos {start_pos + i}): token='{pos_data.token}'")
        
        # For regex-guided regions, use a simpler approach:
        # Just extract probability from the specific token positions where scale values were found
        region_probs = {val: 0.0 for val in scale_values}
        region_diag = {
            'method': 'regex_guided_region',
            'token_variants_found': {},
            'raw_top_tokens': [],
            'total_scale_probability': 0.0
        }
        
        # Look at each position in the region for scale tokens
        for i, pos_data in enumerate(region_logprobs):
            global_pos = start_pos + i
            
            # Check if this position contains the scale value we're looking for
            actual_token = pos_data.token.strip()
            if actual_token == scale_val:
                # Found the scale token - get its probability
                token_prob = np.exp(pos_data.logprob)
                region_probs[scale_val] += token_prob
                
                # Also check alternatives in top_logprobs for this position
                if pos_data.top_logprobs:
                    for alt in pos_data.top_logprobs:
                        alt_token = alt.token.strip()
                        if alt_token in scale_values:
                            alt_prob = np.exp(alt.logprob)
                            region_probs[alt_token] += alt_prob
                            
                            # Track for diagnostics
                            if alt_token not in region_diag['token_variants_found']:
                                region_diag['token_variants_found'][alt_token] = []
                            region_diag['token_variants_found'][alt_token].append(
                                (repr(alt.token), alt_prob)
                            )
                            
                            # Store for diagnostics
                            region_diag['raw_top_tokens'].append({
                                'position': global_pos,
                                'token': alt.token,
                                'token_repr': repr(alt.token),
                                'stripped': alt_token,
                                'logprob': alt.logprob,
                                'prob': alt_prob,
                                'match_region': f"{start_pos}-{end_pos}"
                            })
                
                break  # Found the position, no need to continue
        
        region_diag['total_scale_probability'] = sum(region_probs.values())
        
        # Adjust position indices in diagnostics to reflect original positions
        for tok_info in region_diag.get('raw_top_tokens', []):
            if 'position' in tok_info:
                tok_info['position'] += start_pos  # Adjust to original position
        
        # Add to overall probabilities for ALL scale values found in this region
        for val, prob in region_probs.items():
            if prob > 0:
                choice_probs[val] += prob
        
        # Store region diagnostics
        for tok_info in region_diag.get('raw_top_tokens', []):
            tok_info['match_region'] = f"{start_pos}-{end_pos}"
            diagnostics['raw_top_tokens'].append(tok_info)
        
        # Track variants
        for val, variants in region_diag.get('token_variants_found', {}).items():
            if val not in diagnostics['token_variants_found']:
                diagnostics['token_variants_found'][val] = []
            diagnostics['token_variants_found'][val].extend(variants)
    
    diagnostics['total_scale_probability'] = sum(choice_probs.values())
    return choice_probs, diagnostics


def likert_extract_scale_tokens_multi_position(
    logprobs_data: list,
    scale_values: List[str],
    max_positions: int = 3,
    verbose: bool = False
) -> Tuple[Dict[str, float], Dict]:
    """
    Extract scale probabilities from multiple token positions (handles ' ' + '2' scenarios).
    """
    choice_probs = {val: 0.0 for val in scale_values}
    diagnostics = {
        'method': 'multi_position',
        'token_variants_found': {},
        'raw_top_tokens': [],
        'total_scale_probability': 0.0,
        'positions_processed': 0,
        'sequences_found': []
    }
    
    if not logprobs_data:
        return choice_probs, diagnostics
    
    positions_to_check = min(max_positions, len(logprobs_data))
    diagnostics['positions_processed'] = positions_to_check
    
    # For each position, collect possible tokens
    position_tokens = []
    for pos in range(positions_to_check):
        pos_data = logprobs_data[pos]
        if pos_data.top_logprobs:
            tokens = [(alt.token, alt.logprob) for alt in pos_data.top_logprobs]
            # Ensure the main generated token is included (top_logprobs should include it, but double-check)
            main_token_included = any(alt.token == pos_data.token for alt in pos_data.top_logprobs)
            if not main_token_included:
                tokens.insert(0, (pos_data.token, pos_data.logprob))
        else:
            tokens = [(pos_data.token, pos_data.logprob)]
        position_tokens.append(tokens)
        
        # Store raw tokens for diagnostics
        for token, logprob in tokens[:10]:  # Top 10 per position
            prob = np.exp(logprob)
            diagnostics['raw_top_tokens'].append({
                'position': pos,
                'token': token,
                'token_repr': repr(token),
                'stripped': token.strip(),
                'logprob': logprob,
                'prob': prob
            })
    
    # Generate sequences by combining tokens from different positions
    def generate_sequences(pos_tokens, current_seq="", current_logprob=0.0, pos=0):
        if pos >= len(pos_tokens):
            # Complete sequence - check if it contains a scale value
            seq_stripped = current_seq.strip()
            if seq_stripped in scale_values:
                seq_prob = np.exp(current_logprob)
                choice_probs[seq_stripped] += seq_prob
                
                diagnostics['sequences_found'].append({
                    'sequence': current_seq,
                    'stripped': seq_stripped,
                    'logprob': current_logprob,
                    'prob': seq_prob
                })
                
                if seq_stripped not in diagnostics['token_variants_found']:
                    diagnostics['token_variants_found'][seq_stripped] = []
                diagnostics['token_variants_found'][seq_stripped].append(
                    (repr(current_seq), seq_prob)
                )
            return
        
        # Recursively build sequences (limit to top 5 tokens per position to avoid explosion)
        for token, logprob in pos_tokens[pos][:5]:
            generate_sequences(pos_tokens, current_seq + token, current_logprob + logprob, pos + 1)
    
    # Generate and evaluate all possible sequences
    generate_sequences(position_tokens)
    
    diagnostics['total_scale_probability'] = sum(choice_probs.values())
    return choice_probs, diagnostics


def likert_validate_prompt_format(prompt: str, scale_min: int, scale_max: int) -> Tuple[bool, str]:
    """
    Check if prompt follows recommended format for optimal token generation.
    Based on findings from "Measuring Scalar Constructs in Social Science with LLMs"
    (Licht et al., 2025) which shows 5-10% performance improvements with proper formatting.
    
    Returns (is_conforming, warning_message).
    """
    warnings = []
    
    # Check for explicit number request (critical for multi-position method)
    number_patterns = [
        r'respond\s+with\s+(only\s+)?(a\s+)?single\s+number',
        r'respond\s+with\s+only\s+a\s+number',
        r'respond\s+with\s+(just\s+)?the\s+number',
        r'answer\s+with\s+(just\s+)?(a\s+)?number',
        r'give\s+(only\s+)?(a\s+)?number',
        r'output\s+(only\s+)?(a\s+)?number',
        r'return\s+(only\s+)?(a\s+)?number',
        r'just\s+the\s+number',
        r'only\s+the\s+number',
        r'(only|just)\s+(a\s+)?number'
    ]
    
    has_number_instruction = any(re.search(pattern, prompt.lower()) for pattern in number_patterns)
    if not has_number_instruction:
        warnings.append("CRITICAL: Prompt should explicitly ask for 'only a number' or 'a single number' for optimal multi-position token extraction")
    
    # Check for scale range specification
    scale_mentioned = False
    range_mentioned = f"{scale_min}" in prompt and f"{scale_max}" in prompt
    individual_values = sum(1 for i in range(scale_min, scale_max + 1) if str(i) in prompt)
    
    if range_mentioned:
        scale_mentioned = True
    elif individual_values >= (scale_max - scale_min + 1) // 2:  # At least half the values mentioned
        scale_mentioned = True
    
    if not scale_mentioned:
        warnings.append(f"Scale range should be clearly specified (e.g., 'from {scale_min} to {scale_max}' or mention individual values)")
    
    # Check for leading whitespace hint (KEY finding from scalar constructs paper)
    whitespace_patterns = [
        r'begin\s+your\s+response\s+with\s+(a\s+)?space',
        r'start\s+with\s+(a\s+)?space',
        r'respond\s+with\s+(a\s+)?space\s+followed\s+by',
        r'put\s+(a\s+)?space\s+before',
        r'prefix\s+with\s+(a\s+)?space'
    ]
    
    has_whitespace_hint = any(re.search(pattern, prompt.lower()) for pattern in whitespace_patterns)
    
    if not has_whitespace_hint:
        warnings.append("PERFORMANCE TIP: Consider adding 'Begin your response with a space followed by the number' - research shows 5-10% improvement in structured output generation (Licht et al. 2025)")
    
    # Check for potential regex-fallback triggers (complex instructions)
    complex_patterns = [
        r'explain\s+(your|why)',
        r'provide\s+(an?\s+)?explanation',
        r'describe\s+(how|why)',
        r'followed\s+by\s+(an?\s+)?(explanation|reason)',
        r'then\s+(explain|describe)',
        r'give\s+(me\s+)?(your\s+)?reasoning'
    ]
    
    has_complex_instructions = any(re.search(pattern, prompt.lower()) for pattern in complex_patterns)
    
    if has_complex_instructions:
        warnings.append("NOTICE: Prompt contains explanatory instructions - this may trigger regex-fallback method for embedded ratings. Consider using --regex-fallback flag if responses include explanations.")
    
    # Check for direct/simple format preference (optimal for multi-position)
    format_preference_patterns = [
        r'just\s+(the\s+)?number',
        r'only\s+(the\s+)?number', 
        r'nothing\s+(but|except)\s+(the\s+)?number',
        r'no\s+(other\s+)?text',
        r'single\s+digit'
    ]
    
    has_format_preference = any(re.search(pattern, prompt.lower()) for pattern in format_preference_patterns)
    
    if not has_format_preference and not has_complex_instructions:
        warnings.append("SUGGESTION: For optimal multi-position extraction, consider adding 'just the number' or 'nothing but the number' to encourage simple responses")
    
    # Overall assessment
    is_conforming = len([w for w in warnings if w.startswith("CRITICAL")]) == 0
    
    if warnings:
        if any(w.startswith("CRITICAL") for w in warnings):
            warning_message = "❌ " + "; ".join(warnings)
        else:
            warning_message = "⚠️  " + "; ".join(warnings)
    else:
        warning_message = "✅ Prompt format optimized for multi-position token extraction"
    
    return is_conforming, warning_message


def likert_get_probabilities_logprobs(
    prompt: str,
    model: str = "llama3.1:latest",
    scale_min: int = 1,
    scale_max: int = 5,
    extraction_method: str = "single_position",
    multi_max_positions: int = 3,
    uniform_fallback: str = "warn",
    temperature: float = 0.0,
    verbose: bool = False,
    num_predict: Optional[int] = None
) -> Tuple[Dict[str, float], Dict]:
    """
    Extract Likert probabilities using logprobs from ollama.

    Args:
        prompt: The prompt to send to the model
        model: Ollama model name
        scale_min, scale_max: Likert scale range
        extraction_method: "single_position", "multi_position", or "regex_guided"
        uniform_fallback: "true" (silent), "false" (terminate), or "warn" (default)
        verbose: Whether to show detailed output
        num_predict: Number of tokens to generate. When None, uses a method-appropriate
            default (1 for single_position, 3 for multi_position, 15 for regex_guided).
            Set higher to capture longer responses such as rating + justification.

    Returns:
        - Dictionary of probabilities for each scale value
        - Dictionary with diagnostic information. Always includes 'full_response_text'
          with the raw generated text from the model.
    """

    # Use caller-supplied token budget, or fall back to method-appropriate defaults
    if num_predict is None:
        if extraction_method == "single_position":
            num_predict = 1
        elif extraction_method == "multi_position":
            num_predict = 3
        elif extraction_method == "regex_guided":
            num_predict = 15  # Generate longer sequence for pattern matching
        else:
            num_predict = 1
    
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": temperature,
                "num_predict": num_predict
            },
            logprobs=True,
            top_logprobs=20,
            think=False # Hardwiring no-thinking in case it's a thinking model. TO DO: make this an argument
        )
    except Exception as e:
        raise RuntimeError(f"Failed to generate response: {e}")

    if verbose:
        print(f"Response from {model}: '{response.response}'")
    
    scale_values = [str(i) for i in range(scale_min, scale_max + 1)]
    
    if not response.logprobs:
        # Fallback to uniform distribution
        uniform_prob = 1.0 / len(scale_values)
        choice_probs = {k: uniform_prob for k in scale_values}
        diagnostics = {'error': 'No logprobs available', 'method': extraction_method}
        return choice_probs, diagnostics
    
    # Choose extraction method
    if extraction_method == "single_position":
        choice_probs, diagnostics = likert_extract_scale_tokens_single_position(
            response.logprobs, scale_values, verbose
        )
    elif extraction_method == "multi_position":
        choice_probs, diagnostics = likert_extract_scale_tokens_multi_position(
            response.logprobs, scale_values, max_positions=multi_max_positions, verbose=verbose
        )
    elif extraction_method == "regex_guided":
        choice_probs, diagnostics = likert_extract_scale_tokens_regex_guided(
            response.logprobs, scale_values, verbose=verbose
        )
    else:
        raise ValueError(f"Unknown extraction_method: {extraction_method}")

    # Always capture the raw generated text for callers that need it (e.g. justification logging)
    diagnostics['full_response_text'] = response.response
    
    # Renormalize probabilities, but only if we have meaningful probability mass
    total_prob = sum(choice_probs.values())
    min_meaningful_prob = 0.01  # Require at least 1% total probability mass on scale tokens
    
    if total_prob > min_meaningful_prob:
        choice_probs = {k: v/total_prob for k, v in choice_probs.items()}
    else:
        # Handle fallback based on uniform_fallback setting
        if total_prob > 0:
            reason = f"Scale token probability too low ({total_prob:.6f} < {min_meaningful_prob})"
        else:
            reason = "No scale tokens found in logprobs"
            
        if uniform_fallback == 'false':
            # Terminate with informative message
            raise RuntimeError(f"Could not determine Likert scale value: {reason}")
        elif uniform_fallback == 'warn':
            # Warn to stderr and continue with uniform distribution
            print(f"Warning: {reason}, using uniform distribution", file=sys.stderr)
            uniform_prob = 1.0 / len(scale_values)
            choice_probs = {k: uniform_prob for k in scale_values}
        else:  # uniform_fallback == 'true'
            # Silent fallback to uniform distribution
            uniform_prob = 1.0 / len(scale_values)
            choice_probs = {k: uniform_prob for k in scale_values}
            if verbose:
                print(f"Note: {reason}, using uniform distribution")
    
    # Add renormalization info to diagnostics
    diagnostics['renormalization_factor'] = 1.0/total_prob if total_prob > 0 else 1.0
    diagnostics['pre_renorm_total'] = total_prob
    diagnostics['fallback_used'] = total_prob <= 0
    
    return choice_probs, diagnostics


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process items in a spreadsheet using a large language model')
    parser.add_argument('--infile', required=True, help='Path to input file (xlsx, xls, or csv) with docID, text.')
    parser.add_argument('--model', default='gemma2:9b', help='Model to use (default: gemma2:9b).')
    parser.add_argument('--instructions', default='instructions.txt',
                            help='Path to the instructions file (default: ./instructions.txt).')
    parser.add_argument('--post_instructions', default=None,
                            help='Path to optional content to follow each input item  (default: None).')
    parser.add_argument('--debug',        action='store_true',             help='Report debugging output (default: False)')
    parser.add_argument('--thinking',     action='store_true', default=True, help='Allow thinking models to think (default: True). Use --no-thinking to disable.')
    parser.add_argument('--no-thinking',  dest='thinking', action='store_false', help='Disable thinking for thinking models by appending /nothink to system message.')
    args = parser.parse_args()

    # Call the process_items subroutine
    process_items(args.model, args.instructions, args.infile, post_instruction_file=args.post_instructions, debug=args.debug, thinking=args.thinking)
    
if __name__ == '__main__':
    main()
