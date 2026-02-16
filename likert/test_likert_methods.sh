#!/bin/bash

# Test script to demonstrate different extraction methods and prompt formats
# for ollama_fns_likert.py

echo "=== Testing Likert Scale Probability Extraction ==="
echo "This script demonstrates the differences between:"
echo "1. Conforming vs non-conforming prompts"
echo "2. Single-position vs multi-position extraction methods"
echo ""

# Create test input
echo "Creating test input file..."
cat > /tmp/test_input.txt << 'EOF'
I am feeling blue
I am feeling great
This movie was terrible
The weather is okay
EOF

# Create conforming prompt (with leading whitespace instruction)
echo "Creating conforming prompt..."
cat > /tmp/conforming_prompt.txt << 'EOF'
Rate the sentiment of the following text on a scale from 1 (very negative) to 5 (very positive). 

Begin your response with a space followed by a single number from 1 to 5.
EOF

# Create verbose prompt (generates explanatory text with embedded rating)
echo "Creating verbose prompt..."
cat > /tmp/verbose_prompt.txt << 'EOF'
Rate the sentiment of the following text on a scale from 1 (very negative) to 5 (very positive). 

Please provide a brief explanation followed by your rating. For example: "This text shows negative sentiment, so I would rate it 2."
EOF

# Create very verbose prompt (generates longer explanations)
echo "Creating very verbose prompt..."
cat > /tmp/very_verbose_prompt.txt << 'EOF'
Analyze the sentiment of the following text and provide a detailed explanation of your reasoning, then conclude with a rating from 1 (very negative) to 5 (very positive). 

Use this format: "After careful consideration of the emotional tone and word choice, I believe this text expresses [sentiment type] and therefore assign it a rating of [number]."
EOF

echo "Test inputs:"
cat /tmp/test_input.txt
echo ""
echo "Prompts created:"
echo "1. Conforming prompt (encourages single token with leading space)"
echo "2. Verbose prompt (generates explanation + rating)"
echo "3. Very verbose prompt (generates long explanation + rating)"
echo ""

echo "========================================="
echo "TEST 1: Non-conforming prompt, single-position method"
echo "========================================="
python ollama_fns_likert.py --verbose < /tmp/test_input.txt

echo ""
echo "========================================="
echo "TEST 2: Conforming prompt, single-position method"  
echo "========================================="
python ollama_fns_likert.py --promptfile /tmp/conforming_prompt.txt --verbose < /tmp/test_input.txt

echo ""
echo "========================================="
echo "TEST 3: Verbose prompt, single-position method (should miss ratings in text)"
echo "========================================="
python ollama_fns_likert.py --promptfile /tmp/verbose_prompt.txt --verbose < /tmp/test_input.txt

echo ""
echo "========================================="
echo "TEST 4: Verbose prompt, regex-guided method (should find ratings in text)"
echo "========================================="
python ollama_fns_likert.py --promptfile /tmp/verbose_prompt.txt --regex-fallback --verbose < /tmp/test_input.txt

echo ""
echo "========================================="
echo "TEST 5: Very verbose prompt, regex-guided method"
echo "========================================="
python ollama_fns_likert.py --promptfile /tmp/very_verbose_prompt.txt --regex-fallback --verbose < /tmp/test_input.txt

echo ""
echo "========================================="
echo "TEST 6: Multi-position method on verbose prompts"
echo "========================================="
python ollama_fns_likert.py --promptfile /tmp/verbose_prompt.txt --extraction-method multi_position --verbose < /tmp/test_input.txt

echo ""
echo "========================================="
echo "TEST 7: Comparison across methods (one input only)"
echo "========================================="
echo "Testing with input: 'I am feeling blue'"
echo ""

echo "Single-position (default prompt):"
echo "I am feeling blue" | python ollama_fns_likert.py | grep "Weighted average"

echo "Regex-guided (verbose prompt):"
echo "I am feeling blue" | python ollama_fns_likert.py --promptfile /tmp/verbose_prompt.txt --regex-fallback | grep "Weighted average"

echo "Multi-position (verbose prompt):"
echo "I am feeling blue" | python ollama_fns_likert.py --promptfile /tmp/verbose_prompt.txt --extraction-method multi_position | grep "Weighted average"

echo ""
echo "========================================="
echo "Testing complete!"
echo "Key things to notice in the verbose output:"
echo "1. Prompt warnings in non-conforming tests"
echo "2. Different probability distributions between conforming/non-conforming"
echo "3. Token analysis showing what alternatives the model considered"
echo "4. Multi-position method shows position info (pos=0, pos=1, etc.)"
echo "5. Renormalization factors (should be close to 1.0 for good prompts)"
echo "========================================="

# Cleanup
echo "Cleaning up temporary files..."
rm -f /tmp/test_input.txt /tmp/conforming_prompt.txt /tmp/verbose_prompt.txt /tmp/very_verbose_prompt.txt
echo "Done!"