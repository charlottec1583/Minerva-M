#!/bin/bash
# Batch parse and compare test results
# This script parses batch results from each model, then generates comparison charts

# ========== Part 1: Parse batch results for each model ==========
# Each command parses one batch result directory and generates a corresponding JSON file
python parse_batch_results.py ../batch_results_20260329_211730 # qwen3-32b
python parse_batch_results.py ../batch_results_20260330_101328 # deepseek
python parse_batch_results.py ../batch_results_20260330_112313 # gemini-2.5-flash-lite
python parse_batch_results.py ../batch_results_20260330_112926 # gpt-3.5-turbo-0125
python parse_batch_results.py ../batch_results_20260330_180821 # gpt-5.1

# ========== Part 2: Generate comparison charts ==========
# Use all parsed JSON files to generate comparison visualization charts
# Generate success rate comparison, stage execution statistics, IVs usage comparison, etc.
python compare_batch_results.py \
      parsed_results/parsed_results_batch_results_20260329_211730.json \
      parsed_results/parsed_results_batch_results_20260330_101328.json \
      parsed_results/parsed_results_batch_results_20260330_112313.json \
      parsed_results/parsed_results_batch_results_20260330_112926.json \
      parsed_results/parsed_results_batch_results_20260330_180821.json

# ========== Part 3: Ablation Study Analysis (qwen3-32b) ==========
# Parse ablation experiment results and analyze impact of removing each stage
echo "Parsing ablation study results..."
python parse_batch_results.py ../batch_results_20260329_211730  # Full pipeline
python parse_batch_results.py ../batch_results_20260330_103444  # No context_establishment
python parse_batch_results.py ../batch_results_20260330_104341  # No relationship_building
python parse_batch_results.py ../batch_results_20260330_104358  # No constraint_induction
python parse_batch_results.py ../batch_results_20260330_104432  # No escalation

# Run ablation analysis to compare success rates
python compare_batch_results.py \
      -o ablation_charts \
      parsed_results/parsed_results_batch_results_20260329_211730.json \
      parsed_results/parsed_results_batch_results_20260330_103444.json \
      parsed_results/parsed_results_batch_results_20260330_104341.json \
      parsed_results/parsed_results_batch_results_20260330_104358.json \
      parsed_results/parsed_results_batch_results_20260330_104432.json \