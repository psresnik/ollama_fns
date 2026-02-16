#!/bin/bash

# Enhanced test script for comparing Likert extraction methods
# Organized with inputs as outer loop, methods as inner loop for easy comparison

echo "=== Likert Scale Method Comparison Test ==="
echo "This script compares extraction methods on the same inputs to verify robustness"
echo ""

# Create test inputs
echo "Creating test inputs..."
declare -a test_inputs=(
    "I am feeling blue"
    "I am feeling great"
    "This movie was terrible"
    "The weather is okay"
    "Everything is absolutely amazing today!"
)

# Create different prompt types
echo "Creating prompt files..."

# Optimal prompt (follows scalar constructs paper recommendations)
cat > /tmp/optimal_prompt.txt << 'EOF'
Rate the sentiment from 1 (very negative) to 5 (very positive). Begin your response with a space followed by only the number.
EOF

# Simple prompt (good but missing whitespace hint)
cat > /tmp/simple_prompt.txt << 'EOF'
Rate the sentiment from 1 (very negative) to 5 (very positive). Respond with just the number.
EOF

# Suboptimal prompt (missing critical instructions)
cat > /tmp/suboptimal_prompt.txt << 'EOF'
What's the sentiment of this text? Is it positive or negative?
EOF

# Explanatory prompt (may generate explanations with embedded rating)
cat > /tmp/explanatory_prompt.txt << 'EOF'
Rate the sentiment of the following text on a scale from 1 (very negative) to 5 (very positive). 

Please provide a brief explanation followed by your rating. For example: "This text shows negative sentiment, so I would rate it 2."
EOF

# Narrative prompt (forces explanatory text first)
cat > /tmp/narrative_prompt.txt << 'EOF'
Please start your response by saying "Let me analyze this text." Then analyze the sentiment and conclude with "My rating is X" where X is a number from 1 (very negative) to 5 (very positive).
EOF

echo "Test configuration ready!"
echo ""

# Define methods to test
methods=("single_position" "multi_position" "regex_fallback")
method_names=("Single-Position" "Multi-Position" "Regex-Fallback")

# Main comparison loop: inputs outer, methods inner
for input_text in "${test_inputs[@]}"; do
    echo "========================================================================"
    echo "TESTING INPUT: '$input_text'"
    echo "========================================================================"
    
    # Test with different prompt types
    for prompt_type in "optimal" "simple" "suboptimal" "explanatory" "narrative"; do
        case $prompt_type in
            "optimal")
                prompt_file="/tmp/optimal_prompt.txt"
                prompt_desc="Optimal prompt (follows scalar constructs paper)"
                ;;
            "simple")
                prompt_file="/tmp/simple_prompt.txt"
                prompt_desc="Simple prompt (good but missing whitespace hint)"
                ;;
            "suboptimal")
                prompt_file="/tmp/suboptimal_prompt.txt"
                prompt_desc="Suboptimal prompt (missing critical instructions)"
                ;;
            "explanatory") 
                prompt_file="/tmp/explanatory_prompt.txt"
                prompt_desc="Explanatory prompt (triggers regex-fallback)"
                ;;
            "narrative")
                prompt_file="/tmp/narrative_prompt.txt"
                prompt_desc="Narrative prompt (forces explanatory text first)"
                ;;
        esac
        
        echo ""
        echo "--- Prompt Type: $prompt_desc ---"
        
        # Show the actual prompt being used
        echo "Prompt content: $(cat "$prompt_file" | tr '\n' ' ')"
        echo ""
        
        # Show prompt format validation (demonstrates the new feature)
        echo ">>> PROMPT VALIDATION (based on scalar constructs paper findings) <<<"
        validation_result=$(python -c "
import sys
sys.path.append('.')
from ollama_fns import likert_validate_prompt_format
prompt_text = open('$prompt_file').read()
is_conforming, message = likert_validate_prompt_format(prompt_text, 1, 5)
print(f'Conforming: {is_conforming}')
print(f'Message: {message}')
" 2>/dev/null || echo "Validation: Could not load validation function")
        echo "$validation_result"
        echo ""
        
        # Test each method on this input/prompt combination
        for i in "${!methods[@]}"; do
            method="${methods[$i]}"
            method_name="${method_names[$i]}"
            
            echo ""
            echo ">>> Method: $method_name <<<"
            
            # Run the test with verbose output to see full prompts
            if [ "$method" = "regex_fallback" ]; then
                result=$(echo "$input_text" | python ollama_fns_likert.py \
                    --promptfile "$prompt_file" \
                    --extraction-method multi_position \
                    --regex-fallback \
                    --verbose \
                    2>/dev/null)
            else
                result=$(echo "$input_text" | python ollama_fns_likert.py \
                    --promptfile "$prompt_file" \
                    --extraction-method "$method" \
                    --verbose \
                    2>/dev/null)
            fi
            
            # Extract key information
            full_prompt=$(echo "$result" | grep "Full prompt sent to model:" | tail -1)
            weighted_avg=$(echo "$result" | grep "Weighted average" | tail -1)
            extraction_method=$(echo "$result" | grep "Extraction method:" | tail -1)
            total_prob=$(echo "$result" | grep "Total probability on scale:" | tail -1)
            
            # Show the full prompt that was sent
            if [ -n "$full_prompt" ]; then
                echo "  $full_prompt"
            fi
            
            # For regex-fallback, also show generated text and matches if fallback was used
            if [ "$method" = "regex_fallback" ]; then
                generated_text=$(echo "$result" | grep "Full generated text:" | tail -1)
                pattern_matches=$(echo "$result" | grep "Pattern matches found:" | tail -1)
                
                echo "  $generated_text"
                echo "  $pattern_matches"
                
                # Extract detailed match information (lines that start with "  Match")
                echo "$result" | sed -n '/  Match [0-9]/,/^$/p' | while read line; do
                    if [[ -n "$line" ]]; then
                        echo "    $line"
                    fi
                done
                
            else
                generated_token=$(echo "$result" | grep "Generated token:" | tail -1)
                echo "  $generated_token"
            fi
            
            echo "  $total_prob"
            echo "  $weighted_avg"
            
            # Show probability distribution in compact form
            echo "  Distribution:"
            echo "$result" | grep -E "  [1-5]: [0-9.]+" | while read line; do
                echo "    $line"
            done
        done
        
        echo ""
        echo "--- Summary for '$input_text' with $prompt_type prompt ---"
        echo "Comparing weighted averages across methods:"
        
        for i in "${!methods[@]}"; do
            method="${methods[$i]}"
            method_name="${method_names[$i]}"
            
            if [ "$method" = "regex_fallback" ]; then
                avg=$(echo "$input_text" | python ollama_fns_likert.py \
                    --promptfile "$prompt_file" \
                    --extraction-method multi_position \
                    --regex-fallback \
                    2>/dev/null | grep "Weighted average" | \
                    sed 's/.*Weighted average score: //')
            else
                avg=$(echo "$input_text" | python ollama_fns_likert.py \
                    --promptfile "$prompt_file" \
                    --extraction-method "$method" \
                    2>/dev/null | grep "Weighted average" | \
                    sed 's/.*Weighted average score: //')
            fi
            
            printf "  %-15s: %s\n" "$method_name" "$avg"
        done
        
        echo ""
        echo ">>> PROMPT OPTIMIZATION IMPACT SUMMARY <<<"
        case $prompt_type in
            "optimal")
                echo "✅ This prompt should work best with multi-position method (default)"
                ;;
            "simple")  
                echo "⚠️  This prompt will work but lacks whitespace hint for optimal performance"
                ;;
            "suboptimal")
                echo "❌ This prompt will likely require regex-fallback due to poor format"
                ;;
            "explanatory"|"narrative")
                echo "🔄 This prompt will trigger regex-fallback for embedded rating extraction"
                ;;
        esac
        
        echo "----------------------------------------"
    done
done

echo ""
echo "========================================================================"
echo "TEST COMPLETE"
echo "========================================================================"
echo ""
echo "Key observations to look for:"
echo "1. PROMPT VALIDATION: How different prompts get different validation warnings"
echo "2. OPTIMAL PROMPTS: Multi-position method should work best with proper format"
echo "3. FALLBACK BEHAVIOR: Suboptimal prompts trigger regex-fallback automatically"
echo "4. WHITESPACE IMPACT: Prompts with leading space instruction show better performance"
echo "5. SCALAR CONSTRUCTS PAPER: Recommendations in action (5-10% performance gain)"
echo "6. Similar weighted averages across methods for same input/prompt"
echo "7. Different prompt types producing different response styles"
echo ""

# Cleanup
echo "Cleaning up temporary files..."
rm -f /tmp/optimal_prompt.txt /tmp/simple_prompt.txt /tmp/suboptimal_prompt.txt /tmp/explanatory_prompt.txt /tmp/narrative_prompt.txt
echo "Done!"