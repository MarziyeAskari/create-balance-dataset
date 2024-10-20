import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load ParsBERT model and tokenizer
model_name = "HooshvareLab/bert-base-parsbert-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# Tokenize job titles
def get_embeddings(text_list):
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the [CLS] token embedding as a representation of the whole sentence
    return outputs.last_hidden_state[:, 0, :].numpy()


def balance_dataset(df):
    # Get embeddings for all job titles
    X = df['anchor'].apply(lambda x: get_embeddings(x))
    y = df['job_titles'].apply(lambda x: get_embeddings(x))
    z = df['labels']

    # Apply SMOTE for oversampling
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, z_resampled = smote.fit_resample(X, z)
    y_resampled, z_resampled = smote.fit_resample(y, z)

    # Check the new class distribution
    print("Original class distribution:", Counter(z))
    print("Resampled class distribution:", Counter(z_resampled))

    # Convert back to DataFrame if needed
    df_resampled = pd.DataFrame({'anchor': X_resampled, 'job_titles': y_resampled})
    df_resampled['labels'] = z_resampled
    return df_resampled