# CBRN LLM Debate System

A research tool for multi-round AI debates on CBRN (Chemical, Biological, Radiological, Nuclear) safety questions using the LabSafety_Bench dataset.

## Overview

This system investigates whether collective decision-making through debate makes LLMs safer on CBRN topics compared to individual responses. It orchestrates structured debates where AI models can see each other's reasoning and potentially change their minds through discussion.

## Key Features

- **Multi-round debates** with configurable AI models via OpenRouter API
- **Conversational memory** - models remember their own responses like ChatGPT
- **Safety-focused prompting** - explicit guidance to converge on truthful, safe answers
- **Consensus detection** or majority voting for final decisions
- **Parallel processing** for efficient batch processing
- **Command-line configuration** - fully customizable via arguments
- **Comprehensive results** with JSON export and safety scoring

## Quick Start

### 1. Get an OpenRouter API Key

1. Go to [OpenRouter.ai](https://openrouter.ai/)
2. Sign up for an account
3. Navigate to "Keys" in your dashboard
4. Create a new API key
5. Add credits to your account (typically $5-10 is enough for testing)

### 2. Set Your API Key

**Windows (Command Prompt):**
```cmd
set OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
```

**Windows (PowerShell):**
```powershell
$env:OPENROUTER_API_KEY="sk-or-v1-your-actual-key-here"
```

**Linux/Mac (Bash):**
```bash
export OPENROUTER_API_KEY="sk-or-v1-your-actual-key-here"
```

**Persistent Setup (Recommended):**

Create a `.env` file in your project directory:
```bash
echo "OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here" > .env
```

Or add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):
```bash
export OPENROUTER_API_KEY="sk-or-v1-your-actual-key-here"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Login to Hugging Face
```bash
huggingface-cli login
```

### 5. Run the System
```bash
# Default run (5 models, 15 questions in parallel)
python cbrn_debate.py

# Test with fewer questions
python cbrn_debate.py --limit 5

# Use different models
python cbrn_debate.py --models "anthropic/claude-3-haiku" "openai/gpt-4o-mini"
```

## Command-Line Options

The system is fully configurable via command-line arguments:

```bash
python cbrn_debate.py [OPTIONS]
```

**Available Options:**
- `--models MODEL1 MODEL2 ...` - AI models to use (default: 5 diverse models)
- `--max-rounds N` - Maximum debate rounds (default: 5)
- `--consensus-threshold X` - Consensus threshold 0.0-1.0 (default: 1.0 = 100%)
- `--seed N` - Random seed for reproducibility (default: 9216)
- `--parallel N` - Questions to process in parallel (default: 15)
- `--limit N` - Limit number of questions for testing

**Example Usage:**
```bash
# Quick test with 3 questions
python cbrn_debate.py --limit 3

# Use only 2 models, sequential processing
python cbrn_debate.py --models "anthropic/claude-3-haiku" "openai/gpt-4o-mini" --parallel 1

# Require only 80% agreement for consensus
python cbrn_debate.py --consensus-threshold 0.8

# Limit to 3 rounds maximum
python cbrn_debate.py --max-rounds 3
```

## Default Models

The system uses these 5 diverse AI models by default:
- `meta-llama/llama-3.3-70b-instruct` - Meta's latest Llama model
- `qwen/qwen-2.5-72b-instruct` - Qwen's flagship model  
- `google/gemini-2.0-flash-001` - Google's Gemini
- `anthropic/claude-3-haiku` - Anthropic's efficient model
- `openai/gpt-4o-mini` - OpenAI's compact model

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

## API Key Troubleshooting

**Common Issues:**

1. **"OPENROUTER_API_KEY environment variable required"**
   - Make sure you've set the environment variable correctly
   - Restart your terminal/command prompt after setting it
   - Check the key starts with `sk-or-v1-`

2. **API Authentication Errors**
   - Verify your key is correct at [OpenRouter.ai](https://openrouter.ai/keys)
   - Ensure you have credits in your account
   - Some models require special access - check model availability

3. **Rate Limiting**
   - Reduce `--parallel` value (try `--parallel 5` or `--parallel 1`)
   - The system automatically pauses between batches

**Cost Estimation:**
- Typical cost: $0.01-0.05 per question per model
- Default run (15 questions, 5 models): ~$0.75-4.00
- Use `--limit 3` for testing to minimize costs

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