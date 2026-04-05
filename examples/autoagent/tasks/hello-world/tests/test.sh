#!/usr/bin/env bash
set -euo pipefail

EXPECTED="Hello, World!"
OUTPUT_FILE="/task/output.txt"

if [ ! -f "$OUTPUT_FILE" ]; then
    echo "FAIL: $OUTPUT_FILE does not exist"
    echo "0.0" > /logs/reward.txt
    exit 0
fi

ACTUAL=$(cat "$OUTPUT_FILE")

if [ "$ACTUAL" = "$EXPECTED" ]; then
    echo "PASS"
    echo "1.0" > /logs/reward.txt
else
    echo "FAIL: expected '$EXPECTED', got '$ACTUAL'"
    echo "0.0" > /logs/reward.txt
fi
