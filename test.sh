#!/usr/bin/env bash
set -e

data="_all"

case $1 in 
    "chi")
    data="_chinese"
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
    esac


start=$SECONDS
python3 src/myprogram.py test --test_data test_data/input/input$data.txt --test_output pred$data.txt
duration=$(( SECONDS - start ))
python3 grader/grade.py pred$data.txt test_data/answer/answer$data.txt --verbose
echo Model took $duration seconds to run


