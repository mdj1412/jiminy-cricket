#!/bin/bash

GAME_LIST=("sorcerer" "zork1" "zork2" "zork3" "enchanter")

for GAME in "${GAME_LIST[@]}"; do
    python object_tree.py --game $GAME \
        --num 10 --step 600
done