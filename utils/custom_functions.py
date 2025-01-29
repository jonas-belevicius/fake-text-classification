import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from tqdm import tqdm
import random
from collections import Counter
from time import time
from scipy.stats import linregress, shapiro, ttest_ind, mannwhitneyu, levene
from scipy.sparse import csr_matrix
from typing import List, Dict, Tuple, Union, Any, Optional
import shap
import json
from joblib import Parallel, delayed

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from autocorrect import Speller
from langdetect import detect, LangDetectException
import re
import string
import nlpaug.augmenter.word as naw

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    roc_auc_score,
)

import umap.umap_ as umap
from sklearn.cluster import DBSCAN
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import Pipeline

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.loggers import Logger
from torch.utils.data import DataLoader
from torchmetrics import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    AUROC,
)
from config import (
    DEVICE,
    RANDOM_SEED,
    NLP,
    GRADIENT_CLIP,
)

project_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
utils_dir = os.path.join(project_dir, "utils")
data_dir = os.path.join(project_dir, "data")
sys.path.append(utils_dir)

nrc_lex = pd.read_csv(
    os.path.join(project_dir, "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"),
    names=["word", "emotion", "association"],
    sep="\t",
)
nrc_lex = nrc_lex[nrc_lex["association"] == 1]


def compare_groups(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Union[Tuple[float, float, bool], Tuple[float, float], str]]:
    """
    Compare two groups for statistical differences.

    Parameters:
    - group1: array-like, first group of data.
    - group2: array-like, second group of data.
    - alpha: float, significance level for statistical tests (default=0.05).

    Returns:
    - result: dict with the test performed, test statistic, p-value, and
    conclusion.
    """
    result = {}

    stat1, p1 = shapiro(group1)
    stat2, p2 = shapiro(group2)
    is_normal1 = p1 > alpha
    is_normal2 = p2 > alpha

    result["normality_group1"] = (stat1, p1, is_normal1)
    result["normality_group2"] = (stat2, p2, is_normal2)

    n1, n2 = len(group1), len(group2)
    ranks = np.sum(np.array(group1)[:, None] > np.array(group2))
    delta = (2 * ranks - n1 * n2) / (n1 * n2)

    if is_normal1 and is_normal2:
        stat_levene, p_levene = levene(group1, group2)
        equal_var = p_levene > alpha
        result["levene"] = (stat_levene, p_levene, equal_var)

        stat_ttest, p_ttest = ttest_ind(group1, group2, equal_var=equal_var)
        result["test"] = "t-test"
        result["statistic"] = stat_ttest
        result["p_value"] = p_ttest
        result["conclusion"] = "Significant" if p_ttest < alpha else "Not significant"
    else:
        stat_mwu, p_mwu = mannwhitneyu(group1, group2, alternative="two-sided")
        result["test"] = "Mann-Whitney U"
        result["statistic"] = stat_mwu
        result["p_value"] = p_mwu
        result["conclusion"] = "Significant" if p_mwu < alpha else "Not significant"
        result["cliffs_delta"] = delta

    return result


def count_all_caps(text: str) -> int:
    """
    Count the number of all-capitalized words in the text.
    """
    caps_list = re.findall(r"\b[A-Z]+\b", text)
    return len(caps_list)


def return_all_caps(text: str) -> List[str]:
    """
    Return all-capitalized words in the text.
    """
    caps_list = re.findall(r"\b[A-Z]+\b", text)
    return caps_list


def remove_first_caps_word(
    text: str,
    words_to_remove: List[str],
):
    """
    Remove the first capitalized word from the text.
    """
    pattern = r"^(?:" + "|".join(words_to_remove) + r")\b"
    return re.sub(pattern, "", text).strip()


def remove_reuters(text: str) -> str:
    """
    Remove the word "Reuters" from the text.
    """
    return re.sub(r"\(Reuters\)", "", text).strip()


def count_all_longer_caps(text: str) -> int:
    """
    Count the number of all-capitalized words longer than 1 character in the
    text.
    """
    caps_list = re.findall(r"\b[A-Z]+\b", text)
    caps_list = [word for word in caps_list if len(word) > 1]
    return len(caps_list)


def count_html_tags(text: str) -> int:
    """Counts the number of HTML tags in the text."""
    return len(re.findall(r"<.*?>", text))


def count_url_links(text: str) -> int:
    """Counts the number of URL links in the text."""
    return len(re.findall(r"http\S+|www.\S+", text))


def count_special_chars(text: str) -> int:
    """Counts the number of special characters in the text."""
    return len(re.findall(r"[^\w\s]", text))


def count_numbers(text: str) -> int:
    """Counts the number of numbers in the text."""
    return len(re.findall(r"\d+", text))


def count_emotions(text: str) -> int:
    """Counts the number of emojis in the text."""
    return len(re.findall(r"[\U0001F600-\U0001F64F]", text))


def count_hash_mentions(text: str) -> int:
    """Counts the number of hashtags and mentions in the text."""
    return len(re.findall(r"@\w+|#\w+", text))


def get_top_ngram(
    corpus: List[str],
    n: Optional[int] = None,
) -> List[Tuple[str, int]]:
    """
    Get the top n-grams from a corpus of text.

    Parameters:
    - corpus: list, a list of text strings.
    - n: int, the n-gram range (default=None).
    """
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:10]


def calculate_ttr(text: str) -> float:
    """
    Calculate the type-token ratio of a text.
    """
    tokens = word_tokenize(text)
    total_tokens = len(tokens)
    unique_tokens = len(set(tokens))
    return unique_tokens / total_tokens if total_tokens > 0 else 0


def get_word_frequencies(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the word frequencies of a text.
    """
    tokens = word_tokenize(text)
    token_count = Counter(tokens)
    sorted_token_count = token_count.most_common()
    token_ranks = np.arange(1, len(sorted_token_count) + 1)
    frequencies = np.array([count for _, count in sorted_token_count])
    return token_ranks, frequencies


def get_regression_results(
    log_ranks: np.ndarray,
    log_frequencies: np.ndarray,
) -> Dict[str, float]:
    """
    Get the regression results of the log-rank and log-frequency data.
    """
    slope, intercept, r_value, p_value, std_err = linregress(
        log_ranks,
        log_frequencies,
    )

    return dict(
        slope=slope,
        intercept=intercept,
        r_value=r_value,
        p_value=p_value,
        std_err=std_err,
    )


def get_similarity_matrix(texts: pd.Series) -> np.ndarray:
    """
    Get the similarity matrix of a list of text strings.
    """
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(matrix)

    return similarity_matrix


def get_similar_texts(
    similarity_matrix: np.ndarray,
    texts: pd.Series,
    threshold: float = 0.8,
) -> Dict[str, Dict[str, float]]:
    """
    Find similar texts based on a similarity matrix.

    Args:
    similarity_matrix (array-like):
        A 2D matrix where element (i, j) represents the similarity score
        between text i and text j.
    texts (list):
        A list of text strings corresponding to the rows/columns of the
        similarity matrix.threshold (float): Minimum similarity score to
        consider texts as similar.

    Returns:
    dict:
        A dictionary where each text is mapped to another dictionary of similar
        texts and their scores.
    """
    similar_texts = {}
    text_idxs = texts.index

    for idx, similarity_row in zip(text_idxs, similarity_matrix):
        text = texts[idx]

        if text not in similar_texts:
            similar_texts[text] = {}

        for other_idx, similarity_score in zip(text_idxs, similarity_row):
            other_text = texts[other_idx]
            if similarity_score >= threshold and idx != other_idx:
                similar_texts[text][other_text] = similarity_score

    return similar_texts


def get_similar_text_idx(
    similarity_matrix: np.ndarray,
    texts: pd.Series,
    threshold: float = 0.8,
) -> Dict[str, Dict[str, float]]:
    """
    Find similar texts based on a similarity matrix.

    Args:
    similarity_matrix (array-like):
        A 2D matrix where element (i, j) represents the similarity score
        between text i and text j.
    texts (list):
        A list of text strings corresponding to the rows/columns of the
        similarity matrix.threshold (float): Minimum similarity score to
        consider texts as similar.

    Returns:
    dict:
        A dictionary where each text is mapped to another dictionary of similar
        texts and their scores.
    """
    similar_idx = {}
    text_idxs = texts.index

    for idx, similarity_row in zip(text_idxs, similarity_matrix):
        for other_idx, similarity_score in zip(text_idxs, similarity_row):

            if similarity_score >= threshold and idx != other_idx:
                similar_idx[idx] = {}
                similar_idx[idx][other_idx] = similarity_score

    return similar_idx


def detect_language(text: str) -> str:
    """
    Detect the language of a text string.
    """
    try:
        if text and text.strip():
            return detect(text)
        else:
            return "unknown"
    except LangDetectException:
        return "unknown"


def get_tag_counts(text: str) -> Counter:
    """
    Get the counts of POS tags in a text.
    """
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    tags = []
    for word, tag in pos_tags:
        tags.append(tag)

    return Counter(tags)


def get_text_sentiment(text: str) -> Tuple[float, float]:
    """
    Get the sentiment polarity and subjectivity of a text.
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    return polarity, subjectivity


def get_text_emotions(text: str):
    """
    Get the emotions of a text based on the NRC Emotion Lexicon.
    """
    words = text.lower().split()
    emotions = nrc_lex[nrc_lex["word"].isin(words)]["emotion"].value_counts()
    return emotions


def get_text_embeddings(
    text: str,
    tokenizer: Any,
    model: Any,
) -> torch.Tensor:
    """
    Get the text embeddings from a pre-trained transformer model.
    """
    input = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model(**input)

    return outputs.last_hidden_state.mean(dim=1)


def get_dbscan_outliers(
    similarity_matrix: np.ndarray, eps: float = 0.5, min_samples: int = 20
) -> List[int]:
    """
    Get the outliers from a similarity matrix using DBSCAN clustering.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = dbscan.fit_predict(1 - similarity_matrix)
    outliers = [i for i, label in enumerate(labels) if label == -1]
    return outliers


def get_vectorized_matrix(texts: pd.Series) -> np.ndarray:
    """
    Get the vectorized matrix of a list of text strings.
    """
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)

    return matrix


def get_pca_outliers(
    vectorized_matrix: np.ndarray,
    pca_components: int = 2,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the outliers from a vectorized matrix using PCA.
    """
    pca = KernelPCA(n_components=pca_components, random_state=RANDOM_SEED)
    reduced_data = pca.fit_transform(vectorized_matrix.toarray())
    distances = np.linalg.norm(reduced_data, axis=1)
    outliers = np.where(distances > threshold)[0]

    return outliers, reduced_data


def get_umap_outliers(
    vectorized_matrix: np.ndarray,
    n_neighbors: int = 3,
    min_dist: float = 0.1,
    threshold: float = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the outliers from a vectorized matrix using UMAP.
    """
    umap_reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine"
    )

    reduced_data = umap_reducer.fit_transform(vectorized_matrix.toarray())
    distances = np.linalg.norm(reduced_data, axis=1)
    outliers = np.where(distances > threshold)[0]

    return outliers, reduced_data


def delete_empty_col_rows(
    df: pd.DataFrame,
    empty_col_to_check: str = "text",
) -> pd.DataFrame:
    """
    Delete rows with empty values in a specific column of a DataFrame.
    """
    df = df.copy()
    for idx, row in df.iterrows():
        if len(row[empty_col_to_check]) == 0:
            df.drop(index=idx, inplace=True)

    return df


def check_for_empty_col_rows(
    df: pd.DataFrame,
    empty_col_to_check: str = "text",
) -> List[int]:
    """
    Check for rows with empty values in a specific column of a DataFrame.
    """
    df = df.copy()
    empty_rows = []
    for idx, row in df.iterrows():
        if len(row[empty_col_to_check]) == 0:
            empty_rows.append(idx)

    return empty_rows


def clean_punctuation(text: str) -> str:
    """
    Clean the punctuation from a text string.
    """
    cleaned_text = re.sub(r"[.!?]{2,}", ".", text)
    cleaned_text = re.sub(r",+", ",", cleaned_text)
    cleaned_text = re.sub(r"-{2,}", "-", cleaned_text)
    cleaned_text = re.sub(r"\s+[.,!?;:-]+\s+", " ", cleaned_text)
    cleaned_text = re.sub(r"^[.,!?;:-]+", "", cleaned_text)
    cleaned_text = re.sub(r"[.,!?;:-]+$", "", cleaned_text)
    cleaned_text = re.sub(
        r"(?<!\w)[.,!?;:-]+|\b[.,!?;:-]+\b",
        "",
        cleaned_text,
    )
    cleaned_text = cleaned_text.rstrip(string.punctuation)
    return cleaned_text


def remove_named_entities(text: str) -> str:
    """
    Remove named entities from a text string.
    """
    doc = NLP(text)
    tokens = [token.text for token in doc if not token.ent_type_]
    return " ".join(tokens)


def remove_rare_tokens(texts: pd.Series, min_freq: int = 5) -> pd.Series:
    """
    Remove rare tokens from a list of text strings.
    """
    texts = texts.copy()
    all_tokens = [token for text in texts for token in text.split()]
    token_freq = Counter(all_tokens)

    frequent_tokens = {token for token, freq in token_freq.items() if freq >= min_freq}

    def filter_text(text):
        return " ".join([word for word in text.split() if word in frequent_tokens])

    return [filter_text(text) for text in texts]


def replace_t_with_not(text: str) -> str:
    """
    Replace the contraction "t" with "not" in a text string.
    """
    text = re.sub(r"\b(\w+)'t\b", r"\1 not", text)
    return text


def remove_at(text: str) -> str:
    """
    Remove the "@" symbol from a text string.
    """
    text = re.sub(r"@\w+", "", text)
    return text


def remove_punctuation_except(text: str, punctuation_to_except: set) -> str:
    """
    Remove punctuation from a text string except for specific characters.
    """
    text = re.sub(
        rf"[{re.escape(string.punctuation.replace(punctuation_to_except, ''))}]",
        "",
        text,
    )
    return text


def text_augment(text: str, word_vec: naw.ContextualWordEmbsAug) -> str:
    """
    Augment a text string using a word vector.
    """
    text_augmented = word_vec.augment(text)
    return text_augmented


def augment_texts(
    texts: pd.Series,
    size_to_augment: float = 0.0,
    action: str = "insert",
) -> pd.Series:
    """
    Augment a list of text strings using a word vector.
    """
    texts = texts.copy()
    aug = naw.ContextualWordEmbsAug(
        model_path="bert-base-uncased",
        action=action,
    )
    if not (0 < size_to_augment < 1):
        raise ValueError("size_to_augment must be between 0 and 1.")

    sample_size = int(len(texts) * size_to_augment)
    sample_indices = texts.sample(sample_size).index

    tqdm.pandas(desc=f"Augmenting Texts, action={action}")
    texts.loc[sample_indices] = texts.loc[sample_indices].progress_apply(
        lambda text: "".join(aug.augment(text))
    )

    return texts


def batch_text_augment(
    texts: pd.Series, word_vec: naw.ContextualWordEmbsAug, n_jobs: int = -1
) -> List[str]:
    """
    Batch augment a list of text strings using a word vector.
    """
    texts = texts.copy()
    augmented_texts = Parallel(n_jobs=n_jobs)(
        delayed(text_augment)(text, word_vec) for text in texts
    )
    return augmented_texts


def random_deletion(text: str, p: float = 0.1) -> str:
    """
    Randomly delete words from a text string.
    """
    words = text.split()
    if len(words) == 1:
        return text
    return " ".join([word for word in words if random.uniform(0, 1) > p])


def random_swap(text: str, n: int = 2) -> str:
    """
    Randomly swap words in a text string.
    """
    words = text.split()
    if len(words) >= n:
        for _ in range(min(n, len(words))):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]

    return " ".join(words)


def remove_stopwords(text: str, stopwords: set) -> str:
    """
    Remove stopwords from a text string.
    """
    return " ".join([word for word in text.split() if word not in stopwords])


def remove_stopwords_except(
    text: str,
    stopwords: set,
    stopwords_to_except: set,
) -> str:
    """
    Remove stopwords from a text string except for specific words.
    """
    if not text:
        return ""
    stop_words = stopwords - stopwords_to_except
    return " ".join([word for word in text.split() if word not in stop_words])


def remove_top_overused_tokens(texts: pd.Series, top_tokens: int = 50) -> pd.Series:
    """
    Remove the top overused tokens from a list of text strings.
    """
    texts = texts.copy()
    all_tokens = [token for text in texts for token in text.split()]
    token_freq = Counter(all_tokens)

    token_freq_df = pd.DataFrame(
        token_freq.items(), columns=["token", "count"]
    ).sort_values("count", ascending=False)
    frequent_tokens = []
    for idx, row in token_freq_df.iterrows():
        frequent_tokens.append(row["token"])
        if len(frequent_tokens) == top_tokens:
            break

    def filter_text(text: str) -> str:
        """
        Filter out frequent tokens from a text string.
        """
        return " ".join([word for word in text.split() if word not in frequent_tokens])

    return [filter_text(text) for text in texts]


def remove_proper_nouns(text: str, nlp: spacy.Language) -> str:
    """
    Remove proper nouns from a text string.
    """
    doc = nlp(text)
    return " ".join([token.text for token in doc if token.pos_ != "PROPN"])


def after_augmentation_cleanup(text: str) -> str:
    """
    Clean up a text string after augmentation.
    """
    words = text.split()
    cleaned_words = [word for word in words if "_" not in word]
    return " ".join(cleaned_words)


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation from a text string.
    """
    return re.sub(r"[^\w\s]", "", text)


def clean_whitespace(text: str) -> str:
    """
    Clean the whitespace from a text string.
    """
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def remove_urls(text: str) -> str:
    """
    Remove URLs from a text string.
    """
    return re.sub(r"http\S+|www\S+|https\S+", "", text)


def remove_mentions_hashes(text: str) -> str:
    """
    Remove mentions and hashtags from a text string.
    """
    return re.sub(r"@\w+|#\w+", "", text)


def remove_html(text: str) -> str:
    """
    Remove HTML tags from a text string
    """
    return re.sub(r"<.*?>", "", text)


def fix_spelling(text: str) -> str:
    """
    Fix the spelling of a text string.
    """
    spell = Speller()
    return spell(text)


def get_wordnet_pos(treebank_tag: str) -> str:
    """
    Get the WordNet POS tag from a treebank POS tag.
    """
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def remove_chars_tags(text: str) -> str:
    """
    Remove characters and tags from a text string.
    """
    cleaned_text = re.sub(r"<.*?>", "", text)
    cleaned_text = re.sub(r"http\S+|www.\S+", "", cleaned_text)
    cleaned_text = re.sub(r"[^\w\s]", "", cleaned_text)
    cleaned_text = re.sub(r"\d+", "", cleaned_text)
    cleaned_text = re.sub(r"[\U0001F600-\U0001F64F]", "", cleaned_text)
    return re.sub(r"@\w+|#\w+", "", cleaned_text)


def lemmatize_text(text: str) -> str:
    """
    Lemmatize a text string.
    """
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    lemmatized_text = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags
    ]
    result = " ".join(lemmatized_text)
    return result


def change_tags_to_custom_tokens(
    text: str, url_tag: str = "<URL>", html_tag: str = "<HTML>"
) -> str:
    """
    Change tags to custom tokens in a text string.
    """
    text = re.sub(r"http[s]?://\S+", url_tag, text)
    text = re.sub(r"<[^>]+>", html_tag, text)
    return text


def train_model(
    model: LightningModule,
    data_module: LightningDataModule,
    callbacks: ModelCheckpoint,
    logger: Logger,
    epochs: int,
    lr: float,
) -> pl.Trainer:
    """
    Trains a PyTorch Lightning model using the provided data module, checkpoint
    callback, and logger.

    Args:
        model (LightningModule): The model to be trained.
        data_module (LightningDataModule): The data module that provides the
        training and validation data.
        checkpoint_callback (ModelCheckpoint): Callback to save model
        checkpoints during training.
        logger (Logger): Logger for tracking training progress and metrics.
        epochs (int): The number of training epochs.
        lr (float): The learning rate for the model.

    Returns:
        pl.Trainer: The PyTorch Lightning Trainer instance after training.
    """

    model.lr = lr
    model = model.to("cuda")
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=2,
        devices=torch.cuda.device_count(),
        accelerator="gpu",
        enable_progress_bar=True,
        accumulate_grad_batches=2,
        precision=32,
        gradient_clip_val=GRADIENT_CLIP,
    )

    trainer.fit(model, data_module)
    return trainer


def calculate_text_title_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate text and title features from a DataFrame of text data.

    Returns:
    pd.DataFrame: DataFrame with text and title features.
    - title_caps_proportion: Proportion of capitalized words in the title.
    - text_caps_proportion: Proportion of capitalized words in the text.
    - special_chars_combination_title: Combination of special characters in
    the title.
    - title_special_chars_proportion: Proportion of special characters in the
    title.
    - title_numbers_proportion: Proportion of numbers in the title.
    - ttr_text: Type-token ratio of the text.
    - ttr_title: Type-token ratio of the title.
    - text_subjectivity: Subjectivity of the text.
    - text_polarity: Polarity of the text.
    - text_stopword_proportion: Proportion of stopwords in the text.
    - negative: Value of negative emotions in the text.
    - fear: Value of fear emotions in the text.
    - sadness: Value of sadness emotions in the text.
    - anger: Value of anger emotions in the text.
    - positive: Value of positive emotions in the text.
    - disgust: Value of disgust emotions in the text.
    - trust: Value of trust emotions in the text.
    - anticipation: Value of anticipation emotions in the text.
    - surprise: Value of surprise emotions in the text.
    - joy: Value of joy emotions in the text.

    """

    def get_stopword_proportion(text: str) -> float:
        """Calculates proportion of stopwords in text string."""
        stop_words = set(stopwords.words("english"))
        words = text.split()
        stop_words_count = len([word for word in words if word in stop_words])
        return stop_words_count / len(words) if len(words) > 0 else 0

    nrc_lex = pd.read_csv(
        os.path.join(project_dir, "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"),
        names=["word", "emotion", "association"],
        sep="\t",
    )
    nrc_lex = nrc_lex[nrc_lex["association"] == 1]

    data = data.copy()

    print("Calculating title caps proportion")
    data["title_word_count"] = data["title"].apply(lambda x: len(x.split()))
    data["title_caps_words"] = data["title"].apply(return_all_caps)
    data["title_caps_count"] = data["title"].apply(count_all_caps)
    data["title_caps_proportion"] = np.where(
        data["title_word_count"] > 0,
        data["title_caps_count"] / data["title_word_count"],
        0,
    )

    print("Calculating text caps proportion")
    data["text_word_count"] = data["text"].apply(lambda x: len(x.split()))
    data["text_caps_words"] = data["text"].apply(return_all_caps)
    data["text_caps_count"] = data["text"].apply(count_all_caps)

    data["text_caps_words_longer"] = data["text_caps_words"].apply(
        lambda x: len([item for item in x if len(item) > 1])
    )

    data["text_caps_proportion"] = np.where(
        data["text_word_count"] > 0,
        data["text_caps_words_longer"] / data["text_word_count"],
        0,
    )

    print("Calculating special characters in title")
    data["title_special_chars_count"] = data["text"].apply(count_special_chars)
    data["title_numbers_count"] = data["text"].apply(count_numbers)
    data["title_html_tags_count"] = data["text"].apply(count_html_tags)
    data["title_url_links_count"] = data["text"].apply(count_url_links)
    data["title_emotions_count"] = data["text"].apply(count_emotions)
    data["title_hash_mentions_count"] = data["text"].apply(count_hash_mentions)

    print("Calculating special characters combinations in title")
    data["special_chars_combination_title"] = list(
        zip(
            (data["title_html_tags_count"] > 0).astype(int),
            (data["title_url_links_count"] > 0).astype(int),
            (data["title_special_chars_count"] > 0).astype(int),
            (data["title_numbers_count"] > 0).astype(int),
            (data["title_emotions_count"] > 0).astype(int),
            (data["title_hash_mentions_count"] > 0).astype(int),
        )
    )

    print("Calculating special characters proportion in title")
    data["title_special_chars_proportion"] = np.where(
        data["title_word_count"] > 0,
        data["title_special_chars_count"] / data["title_word_count"],
        0,
    )

    data["title_numbers_proportion"] = np.where(
        data["title_word_count"],
        data["title_numbers_count"] / data["title_word_count"],
        0,
    )

    print("Calculating type-token ratio")
    data["ttr_text"] = data["text"].apply(calculate_ttr)
    data["ttr_title"] = data["title"].apply(calculate_ttr)

    print("Calculating text sentiment")
    polarity_subjectivity = data["text"].apply(get_text_sentiment)
    data["text_subjectivity"] = [item[1] for item in polarity_subjectivity]
    data["text_polarity"] = [item[0] for item in polarity_subjectivity]

    data["text_stopword_proportion"] = data["text"].apply(
        get_stopword_proportion,
    )

    print("Calculating text emotions")
    emotions = data["text"].apply(get_text_emotions)
    emotions = emotions.fillna(0)

    data = data.merge(emotions, left_index=True, right_index=True)

    return data[
        [
            "title_caps_proportion",
            "text_caps_proportion",
            "special_chars_combination_title",
            "title_special_chars_proportion",
            "title_numbers_proportion",
            "ttr_text",
            "ttr_title",
            "text_subjectivity",
            "text_polarity",
            "text_stopword_proportion",
            "negative",
            "fear",
            "sadness",
            "anger",
            "positive",
            "disgust",
            "trust",
            "anticipation",
            "surprise",
            "joy",
        ]
    ]


def count_frequency(
    data_series: pd.Series,
    bins: Optional[int] = None,
) -> Counter:
    """
    Count the frequency of elements in a pandas Series, with optional binning
    for continuous values.

    Args:
    - data_series (pd.Series): The pandas Series containing the data.
    - bins (Optional[int], optional): The number of bins for continuous data.
    - Defaults to None.

    Returns:
    - Counter: Counter object containing the frequency of each element or bin.
    """
    if pd.api.types.is_float_dtype(data_series):
        cuts = pd.cut(data_series, bins=bins)
        counts = Counter(cuts)
    elif pd.api.types.is_integer_dtype(data_series):
        counts = Counter(data_series)
    else:
        counts = Counter(data_series)

    return counts


def get_column_true_fake(df: pd.DataFrame, col: str) -> Tuple[pd.Series, pd.Series]:
    """
    Get the values of a column for True and Fake articles.
    """
    true = df.loc[df["status"] == "True"][col]
    fake = df.loc[df["status"] == "Fake"][col]
    return true, fake


def count_fake_true_ratios(
    true_freq_dict: Dict[str, int], fake_freq_dict: Dict[str, int]
) -> pd.DataFrame:
    """
    Count the ratios of fake to true article frequencies for each bin or value.
    """
    df = pd.DataFrame(
        true_freq_dict.items(),
        columns=["bin/value", "true_count"],
    )
    df = df.merge(
        pd.DataFrame(
            fake_freq_dict.items(),
            columns=["bin/value", "fake_count"],
        )
    )
    df["ratio"] = df["fake_count"] / df["true_count"]
    df.sort_values("ratio", ascending=False, inplace=True)
    return df.loc[df["ratio"] >= 5]


def heuristic_model_classifier(article: pd.Series) -> str:
    """
    Heuristic model to classify an article as True or Fake based on features.

    Args:
    - article (pd.Series): A row containing the article features.

    Returns:
    - str: The predicted class label ("True" or "Fake").
    """

    if article["special_chars_combination_title"] == (0, 1, 1, 1, 0, 1):
        return "Fake"

    elif (
        article["text_caps_proportion"] > 0.123
        and article["text_caps_proportion"] <= 0.247
    ):
        return "Fake"

    elif (
        article["title_caps_proportion"] > 0.417
        and article["title_caps_proportion"] <= 0.556
    ):
        return "Fake"

    elif (
        article["text_caps_proportion"] > 0.247
        and article["text_caps_proportion"] <= 0.37
    ):
        return "Fake"

    elif (
        article["text_stopword_proportion"] > 0.455
        and article["text_stopword_proportion"] <= 0.53
    ):
        return "Fake"

    elif article["special_chars_combination_title"] == (0, 1, 1, 0, 0, 0):
        return "Fake"

    elif article["special_chars_combination_title"] == (0, 1, 1, 1, 0, 0):
        return "Fake"

    elif (
        article["text_stopword_proportion"] > 0.0758
        and article["text_stopword_proportion"] <= 0.152
    ):
        return "Fake"

    elif article["disgust"] > 13.111 and article["disgust"] <= 19.667:
        return "Fake"

    elif article["special_chars_combination_title"] == (0, 0, 1, 1, 0, 1):
        return "Fake"

    elif article["special_chars_combination_title"] == (1, 0, 1, 1, 0, 1):
        return "Fake"

    elif article["ttr_title"] > 0.929 and article["ttr_title"] <= 0.964:
        return "Fake"

    elif article["sadness"] > 23.0 and article["sadness"] <= 30.667:
        return "Fake"

    elif article["negative"] > 41.111 and article["sadness"] <= 61.667:
        return "Fake"

    elif article["text_polarity"] > 0.778 and article["text_polarity"] <= 1.0:
        return "Fake"

    elif (
        article["text_caps_proportion"] > 0.278
        and article["text_caps_proportion"] <= 0.417
    ):
        return "Fake"

    elif article["ttr_text"] > 0.905 and article["ttr_text"] <= 1.0:
        return "Fake"

    else:
        return "True"


def get_prediction_metrics(
    labels: List[str],
    predictions: List[str],
    pred_probs: Optional[Dict[str, float]] = None,
    processing_time: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """
    Calculate prediction metrics for a classification task.

    Args:
    - labels (List[str]): The true labels of the data.
    - predictions (List[str]): The predicted labels from the model.
    - pred_probs (Optional[Dict[str, float]], optional): The predicted
    probabilities from the model. Defaults to None.
    - processing_time (Optional[float], optional): The processing time for
    making predictions. Defaults to None.
    """
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    predictions = label_encoder.transform(predictions)

    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    conf_matrix = confusion_matrix(labels, predictions, normalize="pred")
    if pred_probs is not None:
        precisions, recalls, thresholds = precision_recall_curve(
            labels,
            pred_probs,
        )
        pr_auc = auc(recalls, precisions)
        roc_auc = roc_auc_score(labels, pred_probs)
    else:
        pr_auc = None
        roc_auc = None
    return dict(
        accuracy=acc,
        precision=precision,
        recall=recall,
        f1=f1,
        conf_matrix=conf_matrix,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        processing_time=processing_time,
    )


def review_prediction_metrics(
    prediction_metrics_dict: Dict[str, Dict[str, Optional[float]]]
) -> pd.DataFrame:
    """
    Review the prediction metrics for multiple classification tasks.
    """
    df = pd.DataFrame(prediction_metrics_dict).T
    df.drop(["conf_matrix"], axis=1, inplace=True)
    return df


def train_and_store_model(
    model: Any,
    x_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    x_val: Union[np.ndarray, pd.DataFrame],
    model_cache: Dict[str, Pipeline],
    model_predictions: Dict[str, np.ndarray],
    model_predicted_probabilities: Dict[str, np.ndarray],
    model_params: Dict[str, Dict[str, Any]],
    model_processing_time: Dict[float, Dict[float, Any]],
    model_param_values: Dict[str, Union[int, float]] = None,
) -> None:
    """
    Train a model with given parameters and store the model, predictions,
    and probabilities.

    Parameters:
    - model (Any): The model to train.
    - x_train (Union[np.ndarray, pd.DataFrame]): The training features.
    - y_train (Union[np.ndarray, pd.Series]): The training labels.
    - x_val (Union[np.ndarray, pd.DataFrame]): The validation features.
    - model_cache (Dict[str, Pipeline]): The cache to store the trained models.
    - model_predictions (Dict[str, np.ndarray]): The cache to store the model
    predictions.
    - model_predicted_probabilities (Dict[str, np.ndarray]): The cache to store
    the model predicted probabilities.
    - model_params (Dict[str, Dict[str, Any]]): The cache to store the model
    parameters.
    - model_processing_time (Dict[float, Dict[float, Any]]): The cache to store
    the model processing time.
    - model_param_values (Dict[str, Union[int, float]], optional): The
    parameter values for the model. Defaults to None.
    """

    time_start = time()
    model_name = model.__name__.lower()
    print(f"Fitting {model_name}")
    if model_param_values is not None:
        model_fit = model(**model_param_values)
    else:
        model_fit = model()

    model_fit.fit(x_train, y_train)
    time_end = time()

    model_predictions[model_name] = model_fit.predict(x_val)
    model_predicted_probabilities[model_name] = model_fit.predict_proba(x_val)[:, 1]
    model_cache[model_name] = model_fit
    model_params[model_name] = model_fit.get_params
    model_processing_time[model_name] = time_end - time_start


def review_prediction_confusion_matrix(
    prediction_metrics_dict: Dict[str, Dict[str, Optional[float]]],
    ax: Optional[plt.Axes] = None,
) -> None:
    """
    Review the confusion matrix for multiple classification tasks.
    """
    conf_matrix = prediction_metrics_dict["conf_matrix"]
    sns.heatmap(
        conf_matrix,
        annot=True,
        annot_kws={"size": 8},
        fmt=".2f",
        cmap="crest",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")


def rename_dict(
    dictionary: Dict[str, Any],
    name_addition: str,
) -> None:
    """
    Rename the keys in a dictionary with a name addition.
    Keys that can be chaged are:
    - catboostclassifier
    - logisticregression
    - decisiontreeclassifier
    - lgbmclassifier
    - xgbclassifier
    - heuristic_model
    """
    if "catboostclassifier" in dictionary:
        dictionary[f"catboostclassifier_{name_addition}"] = dictionary.pop(
            "catboostclassifier"
        )
    else:
        pass
    if "logisticregression" in dictionary:
        dictionary[f"logisticregression_{name_addition}"] = dictionary.pop(
            "logisticregression"
        )
    else:
        pass
    if "decisiontreeclassifier" in dictionary:
        dictionary[f"decisiontreeclassifier_{name_addition}"] = dictionary.pop(
            "decisiontreeclassifier"
        )
    else:
        pass
    if "lgbmclassifier" in dictionary:
        dictionary[f"lgbmclassifier_{name_addition}"] = dictionary.pop(
            "lgbmclassifier",
        )
    else:
        pass
    if "xgbclassifier" in dictionary:
        dictionary[f"xgbclassifier_{name_addition}"] = dictionary.pop(
            "xgbclassifier",
        )
    else:
        pass
    if "heuristic_model" in dictionary:
        dictionary[f"heuristic_model_{name_addition}"] = dictionary.pop(
            "heuristic_model"
        )


def explain_shap(
    X_input: csr_matrix,
    model: Any,
    feature_names: List[str],
    sample_size: int,
    ax: plt.Axes,
    plot_size: Tuple = (3, 3),
    title: Optional[str] = None,
) -> None:
    """
    Explain a model using SHAP values.
    """
    sample_idx = np.random.randint(0, len(X_input.toarray()), size=sample_size)

    x_sample = X_input[sample_idx, :]
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(x_sample)
    if not isinstance(shap_values, np.ndarray):
        shap_values = shap_values.toarray()
    plt.sca(ax)
    shap.summary_plot(
        shap_values,
        x_sample.toarray(),
        feature_names=feature_names,
        color_bar=False,
        plot_size=plot_size,
        max_display=10,
        alpha=0.7,
        show=False,
    )
    ax.set_title(title)


def eval_model(
    model: LightningModule,
    test_loader: DataLoader,
    output_path: str,
    model_name: str,
    device: torch.device,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Evaluates a model on a test dataloader. Outputs predictions, true labels
    and losses. Saves outputs to output_path.

    Args:
        model (LightningModule): A trained model to be evaluated.
        test_loader (DataLoader): DataLoader object with test data.
        output_path (str): Path where predictions, true labels, and losses
        will be saved as JSON files.
        model_name (str): The name of the model to use for naming output files.

    Returns:
        Tuple[List[float], List[float], List[float]]: A tuple containing true
        labels, predicted labels, and losses.
    """
    model.eval().to(device)

    all_logits = []
    all_labels = []
    all_losses = []

    os.makedirs(output_path, exist_ok=True)

    with torch.no_grad():
        progress_bar = tqdm(
            test_loader, desc="Evaluating", total=len(test_loader), leave=True
        )
        for i, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            output = model(input_ids, attention_mask)

            loss = model.criterion(output.logits, labels)

            all_logits += output.logits.tolist()
            all_labels += labels.tolist()
            all_losses.append(loss.item())

            progress_bar.set_postfix(loss=loss.item())

    with open(
        os.path.join(output_path, f"{model_name}_logits.json"),
        "w",
    ) as f:
        json.dump(all_logits, f)

    with open(
        os.path.join(output_path, f"{model_name}_labels.json"),
        "w",
    ) as f:
        json.dump(all_labels, f)

    with open(
        os.path.join(output_path, f"{model_name}_losses.json"),
        "w",
    ) as f:
        json.dump(all_losses, f)

    return dict(
        all_logits=all_logits,
        all_labels=all_labels,
        all_losses=all_losses,
    )


def get_model_characteristics(
    model_metrics_dict: Dict[str, List[float]],
    model_results_dict: Dict[str, List[float]],
    result_name: str,
):
    """
    Generates and plots various model characteristics including regression and
    classification metrics.

    Args:
        model_metrics_dict (Dict[str, List[float]]): A dictionary containing
        model metrics.
            - 'roc_curve': ROC curve values for classification.
            - 'auroc': AUROC for classification.
            - 'conf_matrix': Confusion matrix for classification.
        model_results_dict (Dict[str, List[float]]): A dictionary containing
        model predictions and labels.
            - 'all_logits': logits from the model.
            - 'all_labels': Ground truth labels.
        result_name (str): The name of the experiment or model result to be
        used in the plot title.

    Returns:
        None: The function generates plots showing the characteristics of the
        model.
    """

    fpr = model_metrics_dict["roc"][0]
    tpr = model_metrics_dict["roc"][1]

    auroc = model_metrics_dict["auroc"]
    conf_matrix = model_metrics_dict["conf_matrix"]

    metrics = {
        "auroc": float(model_metrics_dict["auroc"]),
        "precision": float(model_metrics_dict["precision"]),
        "recall": float(model_metrics_dict["recall"]),
        "accuracy": float(model_metrics_dict["acc"]),
        "f1": float(model_metrics_dict["f1"]),
    }
    metrics_df = pd.DataFrame.from_dict(metrics.items())
    metrics_df.columns = ["metrics", "metric_value"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes = axes.ravel()

    sns.heatmap(
        conf_matrix,
        annot=True,
        cmap="crest",
        ax=axes[0],
    )
    axes[0].set_title("Fake text prediction")

    barplot = sns.barplot(
        data=metrics_df,
        y="metrics",
        x="metric_value",
        ax=axes[1],
    )
    for bar, value in zip(barplot.patches, metrics_df["metric_value"]):
        axes[1].text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            color="black",
            va="center",
        )

    sns.lineplot(x=fpr, y=tpr, ax=axes[2], label="First Model")
    sns.lineplot(
        x=[0, 1],
        y=[0, 1],
        label="Random Classifier",
        color="red",
        linestyle="--",
        ax=axes[2],
    )
    axes[2].set_xlabel("False Positive Rate (FPR)")
    axes[2].set_ylabel("True Positive Rate (TPR)")
    axes[2].set_title("ROC")
    axes[2].text(0.6, 0.4, f"AUC={auroc:.2f}")

    plt.suptitle(f"{result_name} regression/classification results", y=1.0)


def get_model_metrics(
    model_resul_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Calculates various evaluation metrics for a model's predictions.

    This function computes the following metrics:
    - Classification metrics: Accuracy, precision, recall, F1 score,
    AUROC, confusion matrix, and ROC curve.

    Args:
    model_resul_dict (Dict[str, torch.Tensor]): Dictionary containing model
    outputs and labels.

    Returns:
    Dict[str, torch.Tensor]: A dictionary containing computed metrics:
    - Metrics: Accuracy, precision, recall, F1 score, AUROC, confusion matrix,
    and ROC curve.
    """
    acc = Accuracy(task="binary")
    precision = Precision(task="binary")
    recall = Recall(task="binary")
    f1 = F1Score(task="binary")
    auroc = AUROC(task="binary", num_classes=2, average="macro")

    logits = torch.tensor(model_resul_dict["all_logits"])
    prob = torch.softmax(torch.tensor(logits), dim=-1)
    predictions = torch.argmax(prob, dim=-1)
    labels = torch.tensor(model_resul_dict["all_labels"])

    acc = acc(predictions, labels)
    precision = precision(predictions, labels)
    recall = recall(predictions, labels)
    f1 = f1(predictions, labels)
    auroc = auroc(predictions, labels)

    roc = roc_curve(labels, prob[:, 1])

    conf_matrix = confusion_matrix(labels, predictions, normalize="true")

    return dict(
        acc=acc,
        precision=precision,
        recall=recall,
        f1=f1,
        conf_matrix=conf_matrix,
        auroc=auroc,
        roc=roc,
        predictions=predictions,
        prob=prob,
    )


def calculate_text_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate text features from a DataFrame of text data.

    Returns:
    pd.DataFrame: DataFrame with text features.

    - text_caps_proportion: Proportion of capitalized words in the text.
    - ttr_text: Type-token ratio of the text.
    - text_subjectivity: Subjectivity of the text.
    - text_polarity: Polarity of the text.
    - text_stopword_proportion: Proportion of stopwords in the text.
    - text_word_count: Number of words in the text.
    - negative: Value of negative emotions in the text.
    - fear: Value of fear emotions in the text.
    - sadness: Value of sadness emotions in the text.
    - anger: Value of anger emotions in the text.
    - positive: Value of positive emotions in the text.
    - disgust: Value of disgust emotions in the text.
    - trust: Value of trust emotions in the text.
    """

    def get_stopword_proportion(text: str) -> float:
        """Calculates proportion of stopwords in text string."""
        stop_words = set(stopwords.words("english"))
        words = text.split()
        stop_words_count = len([word for word in words if word in stop_words])
        return stop_words_count / len(words) if len(words) > 0 else 0

    nrc_lex = pd.read_csv(
        os.path.join(project_dir, "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"),
        names=["word", "emotion", "association"],
        sep="\t",
    )
    nrc_lex = nrc_lex[nrc_lex["association"] == 1]

    data = data.copy()

    print("Calculating text caps proportion")
    data["text_word_count"] = data["text"].apply(lambda x: len(x.split()))
    data["text_caps_words"] = data["text"].apply(lambda x: return_all_caps(x))
    data["text_caps_count"] = data["text"].apply(lambda x: count_all_caps(x))

    data["text_caps_words_longer"] = data["text_caps_words"].apply(
        lambda x: len([item for item in x if len(item) > 1])
    )

    data["text_caps_proportion"] = np.where(
        data["text_word_count"] > 0,
        data["text_caps_words_longer"] / data["text_word_count"],
        0,
    )

    print("Calculating type-token ratio")
    data["ttr_text"] = data["text"].apply(calculate_ttr)

    print("Calculating text sentiment")
    polarity_subjectivity = data["text"].apply(get_text_sentiment)
    data["text_subjectivity"] = [item[1] for item in polarity_subjectivity]
    data["text_polarity"] = [item[0] for item in polarity_subjectivity]

    data["text_stopword_proportion"] = data["text"].apply(
        get_stopword_proportion,
    )

    print("Calculating text emotions")
    emotions = data["text"].apply(get_text_emotions)
    emotions = emotions.fillna(0)

    data = data.merge(emotions, left_index=True, right_index=True)

    return data[
        [
            "text_caps_proportion",
            "ttr_text",
            "text_subjectivity",
            "text_polarity",
            "text_stopword_proportion",
            "text_word_count",
            "negative",
            "fear",
            "sadness",
            "anger",
            "positive",
            "disgust",
            "trust",
            "anticipation",
            "surprise",
            "joy",
        ]
    ]
