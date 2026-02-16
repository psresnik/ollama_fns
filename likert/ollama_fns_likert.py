#!/usr/bin/env python3

import ollama
import numpy as np
import argparse
import sys
import re
import os
from typing import Dict, Tuple, List, Callable

# Add parent directory to path to import ollama_fns
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import likert functions from ollama_fns
from ollama_fns import (
    likert_extract_scale_tokens_single_position,
    likert_extract_scale_tokens_regex_guided,
    likert_extract_scale_tokens_multi_position,
    likert_validate_prompt_format,
    likert_get_probabilities_logprobs
)

def get_likert_probabilities(input_text: str, prompt_template: str, model: str, 
                            scale_min: int, scale_max: int, extraction_method: str = "multi_position",
                            multi_max_positions: int = 5, regex_fallback: bool = False,
                            uniform_fallback: str = "warn") -> Tuple[Dict[str, float], Dict]:
    """Extract Likert scale probabilities for a single input (logic only).
    
    Returns:
        Tuple of (probabilities_dict, diagnostics_dict)
    """
    
    # Construct full prompt
    full_prompt = f"""{prompt_template}

Input: {input_text}

Respond with only a number from {scale_min} to {scale_max}."""
    
    # Validate prompt format
    is_conforming, warning_msg = likert_validate_prompt_format(full_prompt, scale_min, scale_max)
    
    # Add validation info to diagnostics
    validation_diag = {
        'input_text': input_text,
        'full_prompt': full_prompt,
        'is_conforming': is_conforming,
        'warning_msg': warning_msg if not is_conforming else None
    }
    
    # Hierarchical fallback logic
    if regex_fallback:
        # Try primary method first
        try:
            probs, diag = likert_get_probabilities_logprobs(
                full_prompt,
                model=model,
                scale_min=scale_min,
                scale_max=scale_max,
                extraction_method=extraction_method,
                multi_max_positions=multi_max_positions,
                uniform_fallback="false",  # Don't fall back to uniform yet
                verbose=False  # No verbose output from logic function
            )
            diag['fallback_used'] = False
            
        except RuntimeError as e:
            if "Could not determine Likert scale value" in str(e):
                # Fall back to regex-guided method
                probs, diag = likert_get_probabilities_logprobs(
                    full_prompt,
                    model=model,
                    scale_min=scale_min,
                    scale_max=scale_max,
                    extraction_method="regex_guided",
                    uniform_fallback=uniform_fallback,
                    verbose=False  # No verbose output from logic function
                )
                diag['fallback_used'] = True
                diag['primary_method'] = extraction_method
                diag['fallback_reason'] = str(e)
            else:
                raise  # Re-raise if it's a different error
    else:
        # Direct method call (no fallback)
        probs, diag = likert_get_probabilities_logprobs(
            full_prompt,
            model=model,
            scale_min=scale_min,
            scale_max=scale_max,
            extraction_method=extraction_method,
            multi_max_positions=multi_max_positions,
            uniform_fallback=uniform_fallback,
            verbose=False  # No verbose output from logic function
        )
        
    # Merge validation diagnostics with extraction diagnostics
    diag.update(validation_diag)
    
    return probs, diag


def display_diagnostics(input_text: str, probs: Dict[str, float], diag: Dict, 
                       scale_min: int, scale_max: int, verbose: bool = False) -> None:
    """Display diagnostic information for a single input (diagnostics only)."""
    
    print("=" * 80)
    print(f"INPUT: {input_text}")
    print("=" * 80)
    
    # Show the full prompt being used
    if verbose:
        full_prompt = diag.get('full_prompt', '')
        prompt_escaped = full_prompt.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        print(f"Full prompt sent to model: '{prompt_escaped}'")
        print()
    
    # Show prompt validation warning
    if not diag.get('is_conforming', True):
        warning_msg = diag.get('warning_msg', '')
        print(f"⚠️  PROMPT WARNING: {warning_msg}")
        print()
    
    # Show fallback information
    if diag.get('fallback_used', False):
        primary = diag.get('primary_method', 'unknown')
        if verbose:
            reason = diag.get('fallback_reason', '')
            print(f"Primary method failed: {reason}")
            print("Falling back to regex-guided method")
    
    # Display main diagnostics
    method = diag.get('method', 'unknown')
    if diag.get('fallback_used', False):
        primary = diag.get('primary_method', 'unknown')
        print(f"Extraction method: {method} (fallback from {primary})")
    else:
        print(f"Extraction method: {method}")
    
    if 'generated_token' in diag and diag['generated_token'] is not None:
        print(f"Generated token: '{diag['generated_token']}' (logprob: {diag['generated_logprob']:.4f})")
    
    print(f"Total probability on scale: {diag.get('total_scale_probability', 0.0):.4f}")
    
    if 'renormalization_factor' in diag:
        print(f"Renormalization factor: {diag['renormalization_factor']:.4f}")
    
    # Show regex-guided specific diagnostics
    if method == 'regex_guided':
        full_response = diag.get('full_response_text', '')
        # Escape newlines for better diagnostic readability
        full_response_escaped = full_response.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        print(f"Full generated text: '{full_response_escaped}'")
        
        if 'full_token_sequence' in diag:
            print(f"Token sequence: {' '.join(diag['full_token_sequence'])}")
            
        if 'matches_found' in diag and diag['matches_found']:
            print(f"Pattern matches found: {len(diag['matches_found'])}")
            for i, match in enumerate(diag['matches_found']):
                char_start, char_end = match['char_span']
                matched_text = full_response[char_start:char_end] if full_response else match['match_text']
                # Escape newlines in matched text
                matched_text_escaped = matched_text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                print(f"  Match {i+1}: '{matched_text_escaped}' → scale {match['scale_value']}")
                print(f"    Character span: {char_start}-{char_end} in full text")
                print(f"    Token positions: {match['start_pos']}-{match['end_pos']}")
                print(f"    Pattern used: {match['pattern']}")
        else:
            print("No pattern matches found in generated text")
            
        if 'warnings' in diag and diag['warnings']:
            for warning in diag['warnings']:
                print(f"⚠️  {warning}")
    
    # Show generated response for other methods too
    elif verbose:
        if 'generated_token' in diag and diag['generated_token'] is not None:
            print(f"First generated token: '{diag['generated_token']}'")
    
    if diag.get('token_variants_found', {}):
        print("\nScale tokens found in logprobs:")
        for value in sorted(diag['token_variants_found'].keys(), key=int):
            variants = diag['token_variants_found'][value]
            total_prob = sum(prob for _, prob in variants)
            print(f"  Value {value}: {total_prob:.6f} total probability")
            if verbose:
                for token_repr, prob in variants:
                    print(f"    {token_repr:15s} prob: {prob:.6f}")
    
    if verbose:
        print(f"\nDetailed token analysis:")
        raw_top_tokens = diag.get('raw_top_tokens', [])
        print(f"Raw logprobs from model (top {min(20, len(raw_top_tokens))}):")
        for i, tok_info in enumerate(raw_top_tokens[:20], 1):
            # Handle both single_position and multi_position format
            if 'stripped' in tok_info:
                stripped = tok_info['stripped']
                is_scale = "✓" if stripped in [str(j) for j in range(scale_min, scale_max + 1)] else " "
                pos_info = f"pos={tok_info.get('position', 0)}" if 'position' in tok_info else ""
                print(f"  {i:2d}. {is_scale} {pos_info} original='{tok_info['token']}' "
                      f"stripped='{stripped}' "
                      f"logprob={tok_info['logprob']:.6f} prob={tok_info['prob']:.6f}")
            else:
                # Fallback for incomplete diagnostic data
                print(f"  {i:2d}.   {tok_info}")
        
        token_variants = diag.get('token_variants_found', {})
        if token_variants:
            print(f"\nHow tokens map to scale values:")
            for value in sorted(token_variants.keys(), key=int):
                variants = token_variants[value]
                print(f"  Scale value '{value}' gets probability from:")
                for token_repr, prob in variants:
                    print(f"    token {token_repr} contributes prob={prob:.6f}")
                total_prob = sum(prob for _, prob in variants)
                print(f"    → Total for '{value}': {total_prob:.6f}")
        
        print(f"\nProbability transformation:")
        total_raw = diag.get('total_scale_probability', 0.0)
        print(f"  Raw sum of scale token probs: {total_raw:.6f}")
        if total_raw > 0:
            print(f"  Renormalization factor: {1.0/total_raw:.6f}")
        else:
            print(f"  Renormalization factor: N/A (uniform fallback used)")
        print(f"  Final probs (after renorm):")
        for value in sorted(probs.keys(), key=int):
            raw_prob = sum(prob for _, prob in token_variants.get(value, [])) if token_variants else 0.0
            final_prob = probs[value]
            print(f"    '{value}': {raw_prob:.6f} → {final_prob:.6f}")
    
    print("\nFinal probability distribution (renormalized):")
    for value in sorted(probs.keys(), key=int):
        bar = "█" * int(probs[value] * 50)
        print(f"  {value}: {probs[value]:.4f} {bar}")
    
    weighted_score = sum(int(v) * p for v, p in probs.items())
    print(f"\nWeighted average score: {weighted_score:.3f}")
    print()


def process_input(input_text: str, prompt_template: str, model: str, 
                  scale_min: int, scale_max: int, extraction_method: str = "multi_position",
                  multi_max_positions: int = 5, regex_fallback: bool = False,
                  uniform_fallback: str = "warn", verbose: bool = False) -> None:
    """Process a single input using separated logic and diagnostics functions."""
    
    try:
        # Use logic function to get probabilities
        probs, diag = get_likert_probabilities(
            input_text, prompt_template, model, scale_min, scale_max,
            extraction_method, multi_max_positions, regex_fallback, uniform_fallback
        )
        
        # Use diagnostics function to display results
        display_diagnostics(input_text, probs, diag, scale_min, scale_max, verbose)
        
    except Exception as e:
        print(f"Error processing input: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description='Extract Likert scale probabilities from LLM logprobs using Ollama',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Read inputs from stdin with default settings (multi-position method)
  echo "This movie was amazing!" | python ollama_fns_likert.py
  
  # Use single-position for fast direct numeric responses
  python ollama_fns_likert.py --extraction-method single_position < inputs.txt
  
  # Enable regex fallback for complex responses
  python ollama_fns_likert.py --regex-fallback --verbose < inputs.txt
  
  # Adjust multi-position parameters for performance
  python ollama_fns_likert.py --multi-max-positions 3 < inputs.txt
  
  # Different model and scale range  
  python ollama_fns_likert.py --model gemma2:9b --scale-min 1 --scale-max 7 < inputs.txt
        """
    )
    
    parser.add_argument(
        '--promptfile',
        type=str,
        help='File containing the prompt template (default: built-in sentiment prompt)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='llama3.1:latest',
        help='Ollama model to use (default: llama3.1:latest)'
    )
    
    parser.add_argument(
        '--scale-min',
        type=int,
        default=1,
        help='Minimum value of Likert scale (default: 1)'
    )
    
    parser.add_argument(
        '--scale-max',
        type=int,
        default=5,
        help='Maximum value of Likert scale (default: 5)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed token probability information'
    )
    
    parser.add_argument(
        '--extraction-method',
        type=str,
        choices=['single_position', 'multi_position'],
        default='multi_position',
        help='Token extraction method: multi_position (default) or single_position'
    )
    
    parser.add_argument(
        '--multi-max-positions',
        type=int,
        default=5,
        help='Maximum token positions to consider for multi-position method (default: 5)'
    )
    
    parser.add_argument(
        '--regex-fallback',
        action='store_true',
        help='Enable hierarchical fallback: try primary method first, fall back to regex-guided if needed'
    )
    
    parser.add_argument(
        '--uniform-fallback',
        type=str,
        choices=['true', 'false', 'warn'],
        default='warn',
        help='Behavior when no meaningful scale probabilities found: true (silent fallback), false (terminate), warn (fallback with warning, default)'
    )
    
    args = parser.parse_args()
    
    # Validate scale
    if args.scale_min >= args.scale_max:
        print("Error: scale-min must be less than scale-max", file=sys.stderr)
        sys.exit(1)
    
    # Load prompt template
    if args.promptfile:
        try:
            with open(args.promptfile, 'r', encoding='utf-8') as f:
                prompt_template = f.read().strip()
            print(f"Loaded prompt from: {args.promptfile}")
        except FileNotFoundError:
            print(f"Error: Prompt file '{args.promptfile}' not found", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading prompt file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Default sentiment analysis prompt
        prompt_template = (
            f"Rate the sentiment of the following text on a scale from "
            f"{args.scale_min} (very negative) to {args.scale_max} (very positive)."
        )
        print(f"Using default prompt: {prompt_template}")
    
    print(f"Model: {args.model}")
    print(f"Scale: {args.scale_min} to {args.scale_max}")
    print(f"Extraction method: {args.extraction_method}")
    print(f"Verbose: {args.verbose}")
    print()
    
    # Test model availability and logprobs support
    try:
        test_response = ollama.generate(
            model=args.model,
            prompt="Test",
            options={"num_predict": 1},
            logprobs=True,
            top_logprobs=5
        )
        print(f"✓ Model {args.model} is available with logprobs support")
    except Exception as e:
        print(f"Error: Cannot access model {args.model} with logprobs: {e}", file=sys.stderr)
        print("Note: This requires ollama server 0.6.0+ and Python client 0.6.0+", file=sys.stderr)
        sys.exit(1)
    
    # Read inputs from stdin
    input_count = 0
    try:
        print("Reading inputs from stdin (one per line, Ctrl+D to finish)...")
        print()
        
        for line in sys.stdin:
            input_text = line.strip()
            
            # Skip empty lines
            if not input_text:
                continue
            
            input_count += 1
            
            try:
                process_input(
                    input_text,
                    prompt_template,
                    args.model,
                    args.scale_min,
                    args.scale_max,
                    args.extraction_method,
                    args.multi_max_positions,
                    args.regex_fallback,
                    args.uniform_fallback,
                    args.verbose
                )
            except KeyboardInterrupt:
                print("\nInterrupted by user", file=sys.stderr)
                break
            except Exception as e:
                print(f"Error processing input '{input_text}': {e}", file=sys.stderr)
                continue
        
        print("=" * 80)
        print(f"Processed {input_count} input(s)")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()