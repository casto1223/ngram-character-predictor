#!/usr/bin/env bash
set -e

# Define all language codes
languages=("chinese" "german" "hindi" "russian" "english" "french" "italian" "hebrew" "arabic" "japanese" "korean" "spanish")
language_names=("Chinese" "German" "Hindi" "Russian" "English" "French" "Italian" "Hebrew" "Arabic" "Japanese" "Korean" "Spanish")

# Function to extract success rate from grade.py output
extract_success_rate() {
    output=$1
    success_rate=$(echo "$output" | grep -o "Success rate: [0-9.]*" | grep -o "[0-9.]*")
    echo "$success_rate"
}

# If no argument or "all" is passed, run all languages
if [ -z "$1" ] || [ "$1" = "all" ]; then
    echo "Testing all languages..."
    
    # Arrays to store results
    declare -a results
    declare -a durations
    
    # Run tests for each language
    for i in "${!languages[@]}"; do
        lang="${languages[$i]}"
        lang_name="${language_names[$i]}"
        data="_${lang}"
        
        echo "=== Testing ${lang_name} ==="
        
        start=$SECONDS
        python3 src/myprogram.py test --test_data test_data/input/input$data.txt --test_output pred$data.txt
        duration=$(( SECONDS - start ))
        
        # Capture the output of grade.py
        output=$(python3 grader/grade.py pred$data.txt test_data/answer/answer$data.txt)
        
        # Extract success rate
        success_rate=$(extract_success_rate "$output")
        
        # Store results
        results[$i]="$success_rate"
        durations[$i]="$duration"
        
        echo "${lang_name} model took ${duration} seconds to run"
        echo ""
    done
    
    # Print summary of results
    echo "===== SUMMARY OF RESULTS ====="
    for i in "${!languages[@]}"; do
        lang_name="${language_names[$i]}"
        echo "${lang_name}: Success rate: ${results[$i]}, Duration: ${durations[$i]} seconds"
    done
    
    # Calculate and print average - avoid using bc
    total_success=0
    total_duration=0
    count=${#results[@]}
    
    # Calculate totals
    for i in "${!results[@]}"; do
        total_success=$(awk "BEGIN {print $total_success + ${results[$i]}}")
        total_duration=$((total_duration + durations[$i]))
    done
    
    # Calculate averages using awk (available on both Mac and Git Bash)
    avg_success=$(awk "BEGIN {printf \"%.2f\", $total_success / $count}")
    avg_duration=$(awk "BEGIN {printf \"%.2f\", $total_duration / $count}")
    
    echo ""
    echo "AVERAGE: Success rate: ${avg_success}, Duration: ${avg_duration} seconds"
    echo "Total Duration: ${total_duration} seconds"
    
else
    # Run a single language
    data="_all"
    
    case $1 in 
        "chi")
        data="_chinese"
        ;;
        "spa")
        data="_spanish"
        ;;
        "ger")
        data="_german"
        ;;
        "hin")
        data="_hindi"
        ;;
        "rus")
        data="_russian"
        ;;
        "eng")
        data="_english"
        ;;
        "fre")
        data="_french"
        ;;
        "ita")
        data="_italian"
        ;;
        "heb")
        data="_hebrew"
        ;;
        "ara")
        data="_arabic"
        ;;
        "jap")
        data="_japanese"
        ;;
        "kor")
        data="_korean"
        ;;
    esac
    
    start=$SECONDS
    python3 src/myprogram.py test --test_data test_data/input/input$data.txt --test_output pred$data.txt
    duration=$(( SECONDS - start ))
    python3 grader/grade.py pred$data.txt test_data/answer/answer$data.txt --verbose
    echo "Model took $duration seconds to run"
fi