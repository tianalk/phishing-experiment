# Phish Forensics - AI vs. Human Detection

This project aims to investigate the impact of AI (Large Language Models) on the effectiveness of spear phishing emails, comparing AI-generated vs. human-generated examples using both automated LLM analysis and human judgment via a web game.

Refer to `PRD.txt` for detailed project requirements.

## Project Components

*   `/data`: Contains the email datasets (CSV format).
*   `/llm_evaluator`: Python script for evaluating LLM classification performance on the datasets.
*   `/phish_swipe_game`: Next.js application for the human evaluation game.

## Setup & Usage

### LLM Evaluator

The LLM Evaluator is a Python script that analyzes email datasets using various LLM providers to determine how well AI can detect phishing attempts.

#### Setup

1. Navigate to the LLM evaluator directory:
   ```bash
   cd llm_evaluator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your LLM API key as an environment variable:
   ```bash
   export LLM_API_KEY=your-api-key
   ```
   
   Or provide it directly as a command line argument.

#### Usage

Run the evaluator with datasets:

```bash
python llm_evaluator.py path/to/human_emails.csv path/to/ai_emails.csv --llm_model=openai/gpt-4-turbo
```

Supported providers include:
- OpenAI (format: `openai/model-name`)
- Google (format: `google/model-name`)
- Anthropic (format: `anthropic/model-name`)
- DeepSeek (format: `deepseek/model-name`)

Additional options:
- `--prompt_template`: Custom prompt template with {email_body} placeholder
- `--output_dir`: Directory to save results
- `--log_file`: Path to log file

### Phish Swipe Game

An interactive game for testing and improving phishing detection skills. Users swipe right on suspicious emails (phishing attempts) and left on legitimate emails, learning to identify common signs of phishing.

## Features

- Swipe interface for email classification
- Real-time feedback on user decisions
- Streak counting to track successful classifications
- Automatic detection of common phishing indicators
- Tutorial mode for first-time users
- Complete scoring and performance summary

## Technical Details

This project uses Next.js with React and includes:
- Responsive design for all screen sizes
- Interactive UI with swipe gestures
- Optional backend integration with Supabase for storing results

## Research Project

This game is part of a research project comparing human and AI phishing detection capabilities.

## Getting Started

1. Clone the repository
2. Install dependencies: `npm install`
3. Run the development server: `npm run dev`
4. Open [http://localhost:3000](http://localhost:3000) in your browser

## License

© 2025 Phish Forensics Research 