# CBRN LLM Debate System

A clean, concise implementation for multi-round LLM debates on CBRN (Chemical, Biological, Radiological, Nuclear) safety questions using the LabSafety_Bench dataset.

## Overview

This system tests whether collective decision-making through debate makes LLMs safer on CBRN topics compared to individual responses. It runs structured debates where models can see each other's answers and reasoning, potentially changing their minds through discussion.

## Key Features

- **Multi-round debates** with 5 Qwen models via OpenRouter API
- **Strict multiple choice** format (A/B/C/D answers only)
- **Consensus detection** (100% agreement) or majority voting fallback
- **Comprehensive logging** of all debate rounds and reasoning
- **Safety score comparison** between individual vs collective decisions
- **Detailed output** with JSON export and timestamped logs

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set your OpenRouter API key:**
```bash
# Windows
set OPENROUTER_API_KEY=your_api_key_here

# Linux/Mac
export OPENROUTER_API_KEY="your_api_key_here"
```

3. **Login to Hugging Face (for dataset access):**
```bash
huggingface-cli login
```

## Usage

Run the debate system:
```bash
python cbrn_debate.py
```

This will:
- Load CBRN safety questions from LabSafety_Bench
- Run debates with 5 Qwen models
- Generate comprehensive logs and results

## Models Used

The system uses 5 different Qwen models:
- `qwen/qwen-2.5-72b-instruct` (largest, most capable)
- `qwen/qwen-2-72b-instruct` (previous generation)
- `qwen/qwq-32b-preview` (reasoning-focused)
- `qwen/qwen-2.5-coder-32b-instruct` (code-oriented)
- `qwen/qwen-2.5-14b-instruct` (smaller, faster)

## How the Debate Works

### Round 0: Individual Responses
- Each model answers the CBRN safety question independently
- Responses must be A, B, C, or D with explanations
- These individual answers are saved for comparison

### Round 1+: Collaborative Debate
- Models see all other models' answers and explanations
- They can update their answer if convinced by others' reasoning
- Models can ask questions to those with different answers
- Process continues until consensus or max rounds (5)

### Consensus & Final Answer
- **Consensus**: All models give identical answers (A, B, C, or D)
- **Majority Vote**: If no consensus after max rounds, majority wins
- **Collective Answer**: The final group decision

## Output Files

The system generates:

1. **Console logs**: Real-time progress and summaries
2. **Log file**: `cbrn_debate_YYYYMMDD_HHMMSS.log` - Complete debate logs
3. **Results JSON**: `debate_results_YYYYMMDD_HHMMSS.json` - Structured data including:
   - All questions and correct answers
   - Individual vs collective responses
   - Complete debate history with explanations
   - Safety scores and analysis

## Safety Scoring

The system calculates:
- **Individual scores**: % correct for each model alone
- **Collective score**: % correct for group decisions
- **Comparison**: Whether debate improves safety performance

Example output:
```
qwen-2.5-72b-instruct: 80.0%
qwen-2-72b-instruct: 75.0%
qwq-32b-preview: 85.0%
qwen-2.5-coder-32b-instruct: 70.0%
qwen-2.5-14b-instruct: 65.0%
Collective: 90.0%

Collective vs Average Individual: +15.0% difference
Collective is SAFER
```

## Configuration

Edit `main()` function in `cbrn_debate.py` to customize:

```python
# Change models
models = ["your/preferred/models"]

# Adjust consensus threshold (1.0 = 100% agreement)
consensus_threshold = 0.8  # 80% agreement

# Change max rounds
max_rounds = 3

# Test more/fewer questions
data_subset = ds["QA"][:10]  # Test 10 questions
```

## Research Questions

This implementation helps answer:
1. **Do LLMs become safer through debate?** Compare individual vs collective scores
2. **How quickly do they reach consensus?** Track rounds needed
3. **What reasoning drives changes?** Analyze debate logs
4. **Which models are most influential?** See who convinces others

## Dataset

Uses the LabSafety_Bench MCQ dataset focusing on:
- Chemical safety protocols
- Biological hazard handling  
- Radiological safety measures
- Nuclear safety procedures

## Technical Details

- **Async processing**: Parallel API calls for efficiency
- **Error handling**: Fallback parsing for malformed responses
- **Rate limiting**: Respects OpenRouter API limits
- **Clean separation**: Debate logic separate from scoring/output

## Example Log Output

```
2024-01-15 10:30:00 - Starting debate for question: Which safety protocol...
2024-01-15 10:30:05 - Round 0: Collecting individual responses...
2024-01-15 10:30:10 - qwen-2.5-72b-instruct Round 0: B - Based on standard safety protocols...
2024-01-15 10:30:12 - Round 0 answers: ['B', 'A', 'B', 'C', 'B']
2024-01-15 10:30:12 - Consensus reached: False
2024-01-15 10:30:15 - Round 1: Models reconsidering...
2024-01-15 10:30:20 - Round 1 answers: ['B', 'B', 'B', 'B', 'B']
2024-01-15 10:30:20 - Consensus reached: True
2024-01-15 10:30:20 - Final collective answer: B
```

## Requirements

- Python 3.8+
- OpenRouter API key with credits
- Hugging Face account (for dataset access)
- Internet connection for API calls

The system is designed to be minimal yet comprehensive - providing all necessary data for CBRN safety research while keeping the codebase clean and maintainable.