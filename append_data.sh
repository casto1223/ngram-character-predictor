#!/bin/bash
# filepath: append_data.sh

# This script appends new test data to input and answer files
# Usage: ./append_data.sh [language] [input_string]

# Check for required arguments
if [ $# -lt 2 ]; then
  echo "Usage: ./append_data.sh [language] [input_string]"
  echo "Example: ./append_data.sh hindi 'भारत एक विशाल देश है'"
  exit 1
fi

# Get arguments
language=$1
input_string=$2
lower_language=$(echo "$language" | tr '[:upper:]' '[:lower:]')

# Set up file paths
input_file="test_data/input/input_${lower_language}.txt"
answer_file="test_data/answer/answer_${lower_language}.txt"

# Create directories if they don't exist
mkdir -p "test_data/input"
mkdir -p "test_data/answer"

# Create files if they don't exist
touch "$input_file"
touch "$answer_file"

# Remove any trailing blank lines from the files
sed -i -e :a -e '/^\n*$/{$d;N;ba' -e '}' "$input_file" 2>/dev/null || true
sed -i -e :a -e '/^\n*$/{$d;N;ba' -e '}' "$answer_file" 2>/dev/null || true

# Get the answers from the existing input file if it's not empty
input_count=$(wc -l < "$input_file" 2>/dev/null || echo 0)
answer_count=$(wc -l < "$answer_file" 2>/dev/null || echo 0)

echo "Current input file has $input_count lines"
echo "Current answer file has $answer_count lines"

# Build the strings character by character
length=${#input_string}
for ((i=1; i<length; i++)); do
  # Get the substring
  current_substr="${input_string:0:i}"
  
  # Write the substring to input file
  echo "$current_substr" >> "$input_file"
  
  # If we're not at the last character, get the next character for answer file
  if [ $i -lt $length ]; then
    next_char="${input_string:i:1}"
    echo "$next_char" >> "$answer_file"
  fi
done

echo "Updated files:"
echo "  Input file: $input_file now has $(wc -l < "$input_file") lines"
echo "  Answer file: $answer_file now has $(wc -l < "$answer_file") lines"