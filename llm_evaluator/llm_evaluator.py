import argparse
import pandas as pd
import logging
import sys
import time
import os
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Union

# For metrics calculation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# LLM provider libraries
import openai
import google.generativeai as genai
from anthropic import Anthropic
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ADDED/MODIFIED: Adjust the prompt to be more neutral and caution the LLM not to over-flag legitimate emails
DEFAULT_PROMPT_TEMPLATE = """
You are analyzing an email to see if it is suspicious (phishing) or not. 
Legitimate emails may also reference words like 'account', 'password', or 'security' without being malicious. Some legitimate emails may inform recipients of security incidents, unauthorized access, or misconfigurations, yet still be benign. 
Only classify this email as 'phishing' if it strongly indicates malicious intent (e.g., suspicious links, unverified sender, or unnatural requests for sensitive data).

Please respond with ONLY 'phishing' or 'legitimate' and no other text.

Email:
{email_body}
"""

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """Loads and validates the input CSV data, supporting both old and new formats. Adds a binary label column."""
    column_map = {
        'Email ID': 'Email_ID',
        'Email_ID': 'Email_ID',
        'Email_Body': 'Email_Body',
        'Label': 'True_Label',
        'Label ': 'True_Label',
        'True_Label': 'True_Label',
    }
    required_columns = {'Email_ID', 'Email_Body', 'True_Label'}
    valid_labels = {"phishing", "ai-generated phishing", "legitimate"}
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded {file_path}. Shape: {df.shape}")
        df.columns = df.columns.str.strip()
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        if not required_columns.issubset(df.columns):
            missing_cols = required_columns - set(df.columns)
            logging.error(f"Missing required columns in {file_path}: {missing_cols}")
            return None
        if df.empty:
            logging.warning(f"File {file_path} is empty.")
            return None
        # Normalize labels and filter
        if 'True_Label' in df.columns:
            df['True_Label'] = (
                df['True_Label']
                .astype(str)
                .str.lower()
                .str.strip()
                .str.replace('"', '')
                .str.replace("'", '')
            )
            df = df[df['True_Label'].isin(valid_labels)]
            # Add binary label: both 'phishing' and 'ai-generated phishing' become 'phishing'
            df['Binary_Label'] = df['True_Label'].replace({'ai-generated phishing': 'phishing'})
        df = df[list(required_columns) + ['Binary_Label']]
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
    
    if response in ['phishing', 'legitimate']:
        return response
    
    if 'phishing' in response and 'legitimate' not in response:
        return 'phishing'
    elif 'legitimate' in response and 'phishing' not in response:
        return 'legitimate'
    
    if 'classification: phishing' in response or 'classify as: phishing' in response:
        return 'phishing'
    if 'classification: legitimate' in response or 'classify as: legitimate' in response:
        return 'legitimate'
    
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
            {
                "role": "system",
                # ADDED/MODIFIED: Reinforce that multiple references to 'account' or 'password' alone is not phishing
                "content": (
                    "You received the following email. Is it phishin or legitimate? Many legitimate emails reference 'account', 'password', or 'security' "
                    "but are not phishing unless they contain genuinely malicious indicators such as suspicious links, "
                    "unverified senders, or unnatural requests for personal data. Respond ONLY with 'phishing' or 'legitimate'."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=10
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
        # ADDED/MODIFIED: Clarify non-malicious security terms are common
        system=(
            "You are a cybersecurity analyst. While analyzing emails for potential phishing, remember many legitimate messages "
            "use terms like 'account', 'urgent', or 'password' innocently. Only label an email 'phishing' if there are clear "
            "malicious cues such as suspicious links, unknown domains, or unnatural data requests. Respond solely with "
            "'phishing' or 'legitimate'."
        ),
        messages=[{"role": "user", "content": prompt}]
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
            {
                "role": "system",
                # ADDED/MODIFIED: Clarify that not all references to security = phishing
                "content": (
                    "You are a cybersecurity analyst. Recognize that legitimate emails often discuss 'account' or 'password'. "
                    "Only classify this as 'phishing' if it contains strongly malicious indicators, such as suspicious URLs, "
                    "fake sender addresses, or unnatural data requests. Respond exclusively with 'phishing' or 'legitimate'."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 10
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

def classify_email_with_llm(email_body: str, api_key: str, model_identifier: str, prompt_template: str = DEFAULT_PROMPT_TEMPLATE, **kwargs) -> str:
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
        if '/' not in model_identifier:
            logging.error(f"Invalid model identifier format: {model_identifier}. Expected 'provider/model-name'")
            return "error"
        
        provider, model_name = model_identifier.lower().split('/', 1)
        
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
        
        return parse_llm_response(response)
        
    except Exception as e:
        logging.error(f"LLM classification error: {str(e)} | Email_ID: {kwargs.get('Email_ID', 'N/A')} | Email_Body: {email_body}")
        return "error"

def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate binary classification metrics from binary labels and predictions.
    """
    if 'Binary_Label' not in df.columns or 'LLM_Prediction' not in df.columns:
        logging.error("Required columns missing from DataFrame. Expecting 'Binary_Label' and 'LLM_Prediction'")
        return {}
    valid_indices = df[
        (df['LLM_Prediction'] != 'error') & 
        (df['LLM_Prediction'].notna()) & 
        (df['Binary_Label'].notna())
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
    y_true = df_valid['Binary_Label']
    y_pred = df_valid['LLM_Prediction']
    labels = ['legitimate', 'phishing']
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    error_rate = 1 - (len(valid_indices) / len(df))
    per_class = {
        label: {
            'Precision': float(precision[i]),
            'Recall': float(recall[i]),
            'F1-Score': float(f1[i]),
            'Support': int(support[i])
        }
        for i, label in enumerate(labels)
    }
    return {
        'Accuracy': float(accuracy),
        'Macro_Precision': float(macro_precision),
        'Macro_Recall': float(macro_recall),
        'Macro_F1-Score': float(macro_f1),
        'Per_Class': per_class,
        'Confusion_Matrix': cm.tolist(),
        'Labels': labels,
        'Valid_Samples': len(valid_indices),
        'Total_Samples': len(df),
        'Error_Rate': float(error_rate)
    }

def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Compute a hash of the relevant columns to detect dataset changes."""
    relevant = df[['Email_ID', 'Email_Body', 'True_Label']].astype(str)
    data_str = relevant.to_csv(index=False)
    return hashlib.sha256(data_str.encode('utf-8')).hexdigest()

def load_llm_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)
    return {}

def save_llm_cache(cache_path: str, cache: dict):
    with open(cache_path, 'w') as f:
        json.dump(cache, f)

def get_email_key(email_id: str, email_body: str) -> str:
    # Use a hash of the email body for uniqueness
    body_hash = hashlib.sha256(email_body.encode('utf-8')).hexdigest()
    return f"{email_id}:{body_hash}"

def phishing_breakdown(df: pd.DataFrame) -> dict:
    """Return LLM binary performance for AI-generated and NK phishing emails separately."""
    breakdown = {}
    for label in ["ai-generated phishing", "phishing"]:
        subset = df[df['True_Label'] == label]
        if len(subset) == 0:
            continue
        y_true = subset['Binary_Label']
        y_pred = subset['LLM_Prediction']
        labels = ['legitimate', 'phishing']
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        per_class = {
            l: {
                'Precision': float(precision[i]),
                'Recall': float(recall[i]),
                'F1-Score': float(f1[i]),
                'Support': int(support[i])
            }
            for i, l in enumerate(labels)
        }
        breakdown[label] = {
            'Accuracy': float((y_true == y_pred).mean()),
            'Macro_Precision': float(macro_precision),
            'Macro_Recall': float(macro_recall),
            'Macro_F1-Score': float(macro_f1),
            'Per_Class': per_class,
            'Confusion_Matrix': cm.tolist(),
            'Labels': labels,
            'Valid_Samples': len(subset),
        }
    return breakdown

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM's ability to detect phishing emails.")
    parser.add_argument("input_csv", help="Path to the CSV file containing all emails (real, AI-generated phishing, and legitimate).")
    parser.add_argument("--llm_api_key", help="API key for the LLM. Alternatively, set via environment variable LLM_API_KEY.")
    parser.add_argument("--llm_model", default="openai/gpt-4.1", 
                        help="Provider and model in format 'provider/model-name' (e.g., 'openai/gpt-4.1', 'anthropic/claude-3-opus').")
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

    # Ensure results directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    cache_path = os.path.join(args.output_dir, 'llm_cache.json')
    llm_cache = load_llm_cache(cache_path)

    # --- Process All Emails ---
    logging.info("Processing all emails from input file...")
    df = load_data(args.input_csv)

    if df is not None:
        # Randomize order for potential bias reduction
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        dataset_hash = compute_dataset_hash(df)
        logging.info(f"Dataset hash: {dataset_hash}")
        logging.info(f"Processing {len(df)} emails with {args.llm_model}...")
        predictions = []
        total_emails = len(df)
        for index, row in df.iterrows():
            if index > 0 and index % 10 == 0:
                logging.info(f"Progress: {index}/{total_emails} emails processed.")
            email_key = get_email_key(str(row['Email_ID']), str(row['Email_Body']))
            if email_key in llm_cache:
                prediction = llm_cache[email_key]
            else:
                prediction = classify_email_with_llm(
                    row['Email_Body'],
                    api_key,
                    args.llm_model,
                    args.prompt_template,
                    Email_ID=row.get('Email_ID', row.get('Email ID', index))
                )
                llm_cache[email_key] = prediction
                # Save cache every 10 emails for robustness
                if index % 10 == 0:
                    save_llm_cache(cache_path, llm_cache)
            predictions.append(prediction)
            time.sleep(0.1)
        save_llm_cache(cache_path, llm_cache)
        df['LLM_Prediction'] = predictions
        
        # Log error rate
        error_count = predictions.count('error')
        logging.info(f"Emails processed. Errors: {error_count}/{total_emails} ({error_count/total_emails:.1%})")

        # Calculate and log metrics for all data
        metrics = calculate_metrics(df)
        logging.info(f"Metrics for All Data: {json.dumps(metrics, indent=2)}")

        # Save detailed results
        output_path = f"{args.output_dir}/all_results.csv"
        try:
            df.to_csv(output_path, index=False)
            logging.info(f"Detailed results saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save results to {output_path}: {e}")

        # Add breakdown by phishing type
        breakdown = phishing_breakdown(df)
        metrics['Phishing_Type_Breakdown'] = breakdown
    else:
        logging.error("Skipping processing due to loading errors.")

    # --- Save Summary Report ---
    if df is not None:
        try:
            summary = {
                "All_Data_Metrics": metrics,
                "LLM_Model": args.llm_model,
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Dataset": args.input_csv
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
