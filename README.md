# Phish Forensics: LLM vs. Human Phishing Detection

This repository contains a research pipeline and web app for evaluating and comparing the ability of Large Language Models (LLMs) and humans to detect phishing emails. The experiment quantifies evasion rates for real-world (North Korean) vs. AI-generated phishing, using legitimate emails as a benchmark.

---

## Table of Contents
- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Experiment Pipeline (LLM Evaluator)](#experiment-pipeline-llm-evaluator)
  - [Setup](#setup)
  - [Running the Experiment](#running-the-experiment)
  - [Switching LLM Models](#switching-llm-models)
  - [Environment Variables](#environment-variables)
  - [Dataset Format](#dataset-format)
- [Results & Visualization](#results--visualization)
- [Phish Swipe Game (Web App)](#phish-swipe-game-web-app)
- [Reproducibility & Research Notes](#reproducibility--research-notes)
- [License](#license)

---

## Overview

- **Goal:** Quantify and compare the evasion rates of real-world (NK) vs. AI-generated phishing emails using LLMs, ensuring legitimate emails are not misclassified.
- **Components:**
  - Python pipeline for LLM-based phishing detection and analysis
  - Next.js web app for human evaluation (Phish Swipe Game)
  - Automated visualizations and export for further analysis

---

## Directory Structure

- `/llm_evaluator/` — Python experiment pipeline (main script, requirements)
- `/data/` — Email datasets (CSV format)
- `/results/` — Output: metrics, confusion matrices, CSVs, visualizations
- `/phish_swipe_game/` — Next.js web app for human evaluation
- `visualize_results.py` — Script for generating plots from experiment output

---

## Experiment Pipeline (LLM Evaluator)

### Setup

1. **Install Python dependencies:**
   ```bash
   cd llm_evaluator
   pip install -r requirements.txt
   ```
2. **Set your LLM API key:**
   - For OpenAI: `export LLM_API_KEY=sk-...`
   - For Google, Anthropic, DeepSeek: use the appropriate key for your provider.
   - You can also pass the key via `--llm_api_key`.

### Running the Experiment

Run the main script with your dataset:
```bash
python llm_evaluator.py path/to/Dataset_FinalRun.csv --llm_model=openai/gpt-4.1
```
- **Arguments:**
  - `Dataset_FinalRun.csv`: CSV containing all emails (real-world phishing, AI-generated phishing, and legitimate)
  - `--llm_model`: Model to use (see below)
  - `--output_dir`: (Optional) Where to save results (default: `./`)
  - `--prompt_template`: (Optional) Custom prompt (must include `{email_body}`)
  - `--log_file`: (Optional) Log file path

### Switching LLM Models

- Use the `--llm_model` flag to specify the provider/model:
  - OpenAI: `openai/gpt-4.1`, `openai/gpt-4-turbo`, etc.
  - Google: `google/gemini-pro`, etc.
  - Anthropic: `anthropic/claude-3-opus`, etc.
  - DeepSeek: `deepseek/deepseek-llm`
- Example:
  ```bash
  python llm_evaluator.py path/to/Dataset_FinalRun.csv --llm_model=anthropic/claude-3-opus
  ```

### Environment Variables

- `LLM_API_KEY`: API key for the selected LLM provider
- (Web app) `.env.local` for Supabase keys (see below)

### Dataset Format

- CSV columns required: `Email_ID`, `Email_Body`, `True_Label`
- `True_Label` values: `phishing`, `ai-generated phishing`, `legitimate`
- Example:
  ```csv
  Email_ID,Email_Body,True_Label
  1,"Your account has been compromised. Click here to reset.",phishing
  2,"Quarterly report attached.",legitimate
  3,"Please verify your credentials at this link.",ai-generated phishing
  ```
- See `/data/Dataset_example.csv` for a template.

---

## Results & Visualization

- Results are saved in `/results/`:
  - `human_results.csv`, `ai_results.csv`: Detailed predictions
  - `summary_report.json`: Metrics, confusion matrices, breakdowns
  - PNGs: Confusion matrices, per-class metrics, accuracy comparisons
  - CSVs: Misclassified emails, evasion breakdowns
- To generate all plots:
  ```bash
  python visualize_results.py
  ```
- For 3D/creative visualizations, use `visualization_data.json` (Three.js, etc.)

---

## Phish Swipe Game (Web App)

- Located in `/phish_swipe_game/` (Next.js + Supabase)
- **Setup:**
  1. `cd phish_swipe_game`
  2. `npm install`
  3. `cp .env.local.example .env.local` (fill in Supabase keys)
  4. Run: `npm run dev`
- **Data import:**
  - Use: `npm run import-data -- --human=path/to/human.csv --ai=path/to/ai.csv [--clear]`
  - Requires: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` in `.env.local`
- **Database:**
  - Set up tables using `supabase/schema.sql`

---

## Reproducibility & Research Notes

- All code, prompts, and metrics are versioned for repeatable experiments.
- Caching avoids redundant LLM calls (see `llm_cache.json`).
- Legitimate emails are included as a benchmark to ensure improvements do not come at the cost of misclassifying real communications.
- For methods, see the detailed prompt and rationale in `llm_evaluator.py`.

---

## License

For academic use only. 