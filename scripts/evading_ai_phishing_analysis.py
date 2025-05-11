import pandas as pd
from textblob import TextBlob
from collections import Counter
import re

# AI-generated phishing analysis
ai_df = pd.read_csv('results/ai_results.csv')
evade = ai_df[(ai_df['True_Label'] == 'ai-generated phishing') & (ai_df['LLM_Prediction'] == 'legitimate')]
detected = ai_df[(ai_df['True_Label'] == 'ai-generated phishing') & (ai_df['LLM_Prediction'] == 'phishing')]

# Human-generated phishing analysis
df = pd.read_csv('results/human_results.csv')
human_detected = df[(df['True_Label'] == 'phishing') & (df['LLM_Prediction'] == 'phishing')]

phishing_keywords = [
    'account', 'verify', 'login', 'password', 'urgent', 'click', 'update', 'security',
    'confirm', 'suspend', 'limited', 'access', 'bank', 'invoice', 'payment', 'alert',
    'unusual', 'activity', 'reset', 'credentials', 'immediately', 'important', 'notice',
    'ssn', 'social security', 'refund', 'transfer', 'wire', 'locked', 'unlock',
    'statement', 'review', 'authenticate', 'validate', 'secure', 'risk', 'threat', 'breach',
    'compromise', 'win', 'prize', 'free', 'gift', 'congratulations', 'selected', 'winner',
    'limited time', 'offer', 'deal', 'exclusive', 'act now', 'action required', 'required',
    'final notice', 'reminder', 'notification', 'update your', 'click here', 'log in', 'sign in'
]
def count_phishing_keywords(text):
    text = str(text).lower()
    return sum(kw in text for kw in phishing_keywords)

def analyze_group(group, label):
    sentiment = group['Email_Body'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    lengths = group['Email_Body'].apply(len)
    words = group['Email_Body'].apply(lambda x: len(str(x).split()))
    phishing_counts = group['Email_Body'].apply(count_phishing_keywords)
    def get_ngrams(text, n):
        tokens = re.findall(r'\w+', str(text).lower())
        return zip(*[tokens[i:] for i in range(n)])
    bigrams = Counter()
    trigrams = Counter()
    for body in group['Email_Body']:
        bigrams.update(get_ngrams(body, 2))
        trigrams.update(get_ngrams(body, 3))
    link_count = group['Email_Body'].str.lower().str.contains('link').sum()
    attach_count = group['Email_Body'].str.lower().str.contains('attachment').sum()
    print(f'--- {label} ---')
    print(f'Count: {len(group)}')
    print(f'Avg length: {lengths.mean():.1f} chars, {words.mean():.1f} words')
    print(f'Avg phishing-related keywords: {phishing_counts.mean():.2f}')
    print(f'Avg sentiment polarity: {sentiment.mean():.2f}')
    print(f'Emails mentioning "link": {link_count}/{len(group)}')
    print(f'Emails mentioning "attachment": {attach_count}/{len(group)}')
    print("Top bigrams:", [" ".join(bg) for bg, _ in bigrams.most_common(10)])
    print("Top trigrams:", [" ".join(tg) for tg, _ in trigrams.most_common(10)])
    print()

analyze_group(evade, 'AI-generated phishing that evaded detection')
analyze_group(human_detected, 'Human-generated phishing that was detected') 