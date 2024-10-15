#!/bin/bash

if [ -d "src" ]; then
    echo "Non-empty, non-comment lines of Rust files in src/"
    find src -name "*.rs" -print0 | xargs -0 grep -v '^\s*//' | grep -v '^\s*$' | wc -l

    echo "Non-empty lines of Rust files in src/"
    find src -name "*.rs" -print0 | xargs -0 grep -v '^\s*$' | wc -l
else
    echo "src/ directory not found!"
fi
