import argparse
import pandas as pd
import logging
import sys
import time
import os
import json
from typing import Dict, List, Tuple, Optional, Any, Union

# For metrics calculation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# LLM provider libraries
import openai
import google.generativeai as genai
from anthropic import Anthropic
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default classification prompt template
DEFAULT_PROMPT_TEMPLATE = """
You are a cybersecurity expert analyzing emails for potential phishing attempts.

Please classify the following email as either 'phishing' or 'legitimate'.
Respond with ONLY the word 'phishing' or 'legitimate' and no other text.

Email:
{email_body}
"""

def load_data(file_path: str) -> pd.DataFrame | None:
    """Loads and validates the input CSV data."""
    required_columns = {'Email_ID', 'Email_Body', 'True_Label'}
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded {file_path}. Shape: {df.shape}")

        # Validate columns - handling potential extra spaces in headers
        df.columns = df.columns.str.strip()
        if not required_columns.issubset(df.columns):
            missing_cols = required_columns - set(df.columns)
            logging.error(f"Missing required columns in {file_path}: {missing_cols}")
            return None

        # Basic check for empty data
        if df.empty:
            logging.warning(f"File {file_path} is empty.")
            return None # Or handle as appropriate

        # Normalize labels to lowercase for consistency
        if 'True_Label' in df.columns:
             df['True_Label'] = df['True_Label'].str.lower().str.strip()

        return df

    except FileNotFoundError:
        logging.error(f"Error: File not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        logging.error(f"Error: File {file_path} is empty or has no columns.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading {file_path}: {e}")
        return None

def parse_llm_response(response: str) -> str:
    """
    Extract 'phishing' or 'legitimate' from the LLM response.
    This function handles variations in output format across different LLMs.
    """
    response = response.lower().strip()
    
    # If the response is a single word, just check it
    if response in ['phishing', 'legitimate']:
        return response
    
    # If the response contains phishing or legitimate keywords
    if 'phishing' in response and 'legitimate' not in response:
        return 'phishing'
    elif 'legitimate' in response and 'phishing' not in response:
        return 'legitimate'
    
    # If both keywords are present or neither, try to find a clear indicator
    if 'classification: phishing' in response or 'classify as: phishing' in response:
        return 'phishing'
    if 'classification: legitimate' in response or 'classify as: legitimate' in response:
        return 'legitimate'
    
    # Default to error indicator if we can't determine
    logging.warning(f"Could not parse LLM response clearly: '{response}'")
    return 'error'

def retry_with_backoff(func, max_retries=3, initial_delay=1, backoff_factor=2):
    """Helper function to retry API calls with exponential backoff."""
    def wrapper(*args, **kwargs):
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    logging.error(f"Failed after {max_retries} attempts: {e}")
                    raise
                
                logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= backoff_factor
    return wrapper

def classify_with_openai(email_body: str, api_key: str, model_name: str, prompt_template: str) -> str:
    """Classify email using OpenAI API."""
    client = openai.OpenAI(api_key=api_key)
    prompt = prompt_template.format(email_body=email_body)
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a cybersecurity expert. Respond with ONLY 'phishing' or 'legitimate'."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,  # Lower temperature for more deterministic results
        max_tokens=10,    # We only need a short response
    )
    
    return response.choices[0].message.content.strip()

def classify_with_google(email_body: str, api_key: str, model_name: str, prompt_template: str) -> str:
    """Classify email using Google Gemini API."""
    genai.configure(api_key=api_key)
    prompt = prompt_template.format(email_body=email_body)
    
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    
    return response.text

def classify_with_anthropic(email_body: str, api_key: str, model_name: str, prompt_template: str) -> str:
    """Classify email using Anthropic Claude API."""
    client = Anthropic(api_key=api_key)
    prompt = prompt_template.format(email_body=email_body)
    
    message = client.messages.create(
        model=model_name,
        max_tokens=10,
        temperature=0.0,
        system="You are a cybersecurity expert. Respond with ONLY 'phishing' or 'legitimate'.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return message.content[0].text

def classify_with_deepseek(email_body: str, api_key: str, model_name: str, prompt_template: str) -> str:
    """Classify email using DeepSeek API."""
    url = "https://api.deepseek.com/v1/chat/completions"
    prompt = prompt_template.format(email_body=email_body)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a cybersecurity expert. Respond with ONLY 'phishing' or 'legitimate'."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 10
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()  # Raises an exception for HTTP errors
    
    return response.json()["choices"][0]["message"]["content"].strip()

def classify_email_with_llm(email_body: str, api_key: str, model_identifier: str, prompt_template: str = DEFAULT_PROMPT_TEMPLATE) -> str:
    """
    Classify an email using an LLM of the specified provider.
    
    Args:
        email_body: The content of the email to classify
        api_key: API key for the chosen LLM provider
        model_identifier: String in the format 'provider/model-name' (e.g., 'openai/gpt-4o')
        prompt_template: Template string for the classification prompt, with {email_body} placeholder
        
    Returns:
        Classification string ('phishing', 'legitimate', or 'error')
    """
    if not email_body or not isinstance(email_body, str):
        logging.error("No valid email body provided for classification")
        return "error"
    
    try:
        # Parse the provider and model name
        if '/' not in model_identifier:
            logging.error(f"Invalid model identifier format: {model_identifier}. Expected 'provider/model-name'")
            return "error"
        
        provider, model_name = model_identifier.lower().split('/', 1)
        
        # Dispatch to the appropriate provider's API
        if provider == 'openai':
            classify_func = retry_with_backoff(classify_with_openai)
            response = classify_func(email_body, api_key, model_name, prompt_template)
        elif provider == 'google':
            classify_func = retry_with_backoff(classify_with_google)
            response = classify_func(email_body, api_key, model_name, prompt_template)
        elif provider == 'anthropic':
            classify_func = retry_with_backoff(classify_with_anthropic)
            response = classify_func(email_body, api_key, model_name, prompt_template)
        elif provider == 'deepseek':
            classify_func = retry_with_backoff(classify_with_deepseek)
            response = classify_func(email_body, api_key, model_name, prompt_template)
        else:
            logging.error(f"Unsupported LLM provider: {provider}")
            return "error"
        
        # Parse the response to extract the classification
        return parse_llm_response(response)
        
    except Exception as e:
        logging.error(f"LLM classification error: {str(e)}")
        return "error"

def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate classification metrics from true labels and predictions.
    
    Args:
        df: DataFrame with 'True_Label' and 'LLM_Prediction' columns
        
    Returns:
        Dictionary of metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
    """
    if 'True_Label' not in df.columns or 'LLM_Prediction' not in df.columns:
        logging.error("Required columns missing from DataFrame. Expecting 'True_Label' and 'LLM_Prediction'")
        return {}
    
    # Filter out rows with error predictions or missing values
    valid_indices = df[
        (df['LLM_Prediction'] != 'error') & 
        (df['LLM_Prediction'].notna()) & 
        (df['True_Label'].notna())
    ].index
    
    if len(valid_indices) == 0:
        logging.warning("No valid predictions found for metric calculation")
        return {
            'Accuracy': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'F1-Score': 0.0,
            'Valid_Samples': 0,
            'Error_Rate': 1.0
        }
    
    df_valid = df.loc[valid_indices]
    y_true = df_valid['True_Label']
    y_pred = df_valid['LLM_Prediction']
    
    # Calculate metrics (focusing on 'phishing' as the positive class)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, 
        pos_label='phishing', 
        average='binary',
        zero_division=0
    )
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, 
        labels=['legitimate', 'phishing']
    ).ravel()
    
    # Percentage of errors (non-valid predictions)
    error_rate = 1 - (len(valid_indices) / len(df))
    
    return {
        'Accuracy': float(accuracy),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1-Score': float(f1),
        'Confusion_Matrix': {
            'True_Negative': int(tn),
            'False_Positive': int(fp),
            'False_Negative': int(fn),
            'True_Positive': int(tp)
        },
        'Valid_Samples': len(valid_indices),
        'Total_Samples': len(df),
        'Error_Rate': float(error_rate)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM's ability to detect phishing emails.")
    parser.add_argument("human_emails_csv", help="Path to the CSV file containing human-generated emails.")
    parser.add_argument("ai_emails_csv", help="Path to the CSV file containing AI-generated emails.")
    parser.add_argument("--llm_api_key", help="API key for the LLM. Alternatively, set via environment variable LLM_API_KEY.")
    parser.add_argument("--llm_model", default="openai/gpt-4-turbo", 
                        help="Provider and model in format 'provider/model-name' (e.g., 'openai/gpt-4-turbo', 'anthropic/claude-3-opus').")
    parser.add_argument("--prompt_template", default=DEFAULT_PROMPT_TEMPLATE,
                        help="Custom prompt template. Must include {email_body} placeholder.")
    parser.add_argument("--output_dir", default=".", help="Directory to save detailed results.")
    parser.add_argument("--log_file", default="llm_evaluator.log", help="Path to the log file.")
    
    args = parser.parse_args()
    
    # Setup file logging
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    # Check for API key in environment variable if not provided as argument
    api_key = args.llm_api_key
    if not api_key:
        api_key = os.environ.get('LLM_API_KEY')
        if not api_key:
            logging.error("No API key provided. Use --llm_api_key argument or set LLM_API_KEY environment variable.")
            sys.exit(1)
        else:
            logging.info("Using API key from environment variable.")
    
    # Validate prompt template
    if "{email_body}" not in args.prompt_template:
        logging.error("Prompt template must include {email_body} placeholder.")
        sys.exit(1)

    logging.info("Starting LLM evaluation process.")
    logging.info(f"Using LLM Model: {args.llm_model}")

    # --- Process Human-Generated Emails ---
    logging.info("Processing human-generated emails...")
    df_human = load_data(args.human_emails_csv)

    if df_human is not None:
        logging.info(f"Processing {len(df_human)} human-generated emails with {args.llm_model}...")
        predictions_human = []
        
        # Process emails with progress updates
        total_emails = len(df_human)
        for index, row in df_human.iterrows():
            if index > 0 and index % 10 == 0:
                logging.info(f"Progress: {index}/{total_emails} human-generated emails processed.")
                
            prediction = classify_email_with_llm(
                row['Email_Body'], 
                api_key, 
                args.llm_model,
                args.prompt_template
            )
            predictions_human.append(prediction)
            
            # Add a small delay to avoid hitting rate limits
            time.sleep(0.1)
            
        df_human['LLM_Prediction'] = predictions_human
        
        # Log error rate
        error_count = predictions_human.count('error')
        logging.info(f"Human-generated emails processed. Errors: {error_count}/{total_emails} ({error_count/total_emails:.1%})")

        # Calculate and log metrics for human data
        metrics_human = calculate_metrics(df_human)
        logging.info(f"Metrics for Human-Generated Data: {json.dumps(metrics_human, indent=2)}")

        # Save detailed results
        human_output_path = f"{args.output_dir}/human_results.csv"
        try:
            df_human.to_csv(human_output_path, index=False)
            logging.info(f"Detailed results for human data saved to {human_output_path}")
        except Exception as e:
            logging.error(f"Failed to save human results to {human_output_path}: {e}")
    else:
        logging.error("Skipping processing for human-generated emails due to loading errors.")


    # --- Process AI-Generated Emails ---
    logging.info("\nProcessing AI-generated emails...")
    df_ai = load_data(args.ai_emails_csv)

    if df_ai is not None:
        logging.info(f"Processing {len(df_ai)} AI-generated emails with {args.llm_model}...")
        predictions_ai = []
        
        # Process emails with progress updates
        total_emails = len(df_ai)
        for index, row in df_ai.iterrows():
            if index > 0 and index % 10 == 0:
                logging.info(f"Progress: {index}/{total_emails} AI-generated emails processed.")
                
            prediction = classify_email_with_llm(
                row['Email_Body'], 
                api_key, 
                args.llm_model,
                args.prompt_template
            )
            predictions_ai.append(prediction)
            
            # Add a small delay to avoid hitting rate limits
            time.sleep(0.1)
            
        df_ai['LLM_Prediction'] = predictions_ai
        
        # Log error rate
        error_count = predictions_ai.count('error')
        logging.info(f"AI-generated emails processed. Errors: {error_count}/{total_emails} ({error_count/total_emails:.1%})")

        # Calculate and log metrics for AI data
        metrics_ai = calculate_metrics(df_ai)
        logging.info(f"Metrics for AI-Generated Data: {json.dumps(metrics_ai, indent=2)}")

        # Save detailed results
        ai_output_path = f"{args.output_dir}/ai_results.csv"
        try:
            df_ai.to_csv(ai_output_path, index=False)
            logging.info(f"Detailed results for AI data saved to {ai_output_path}")
        except Exception as e:
            logging.error(f"Failed to save AI results to {ai_output_path}: {e}")
    else:
         logging.error("Skipping processing for AI-generated emails due to loading errors.")

    # --- Save Summary Report ---
    if df_human is not None and df_ai is not None:
        try:
            summary = {
                "Human_Generated_Metrics": metrics_human,
                "AI_Generated_Metrics": metrics_ai,
                "LLM_Model": args.llm_model,
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Human_Dataset": args.human_emails_csv,
                "AI_Dataset": args.ai_emails_csv
            }
            
            summary_path = f"{args.output_dir}/summary_report.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=4)
            logging.info(f"Summary report saved to {summary_path}")
        except Exception as e:
            logging.error(f"Failed to save summary report: {e}")

    logging.info("\nLLM evaluation process finished.")

if __name__ == "__main__":
    main() 