"""
This script is used to preprocess the data for the deep learning models.
It defines several preprocessing pipelines that can be used to clean the text
data.
"""

import pandas as pd
from tqdm import tqdm
import os
import re
from config import (
    RANDOM_SEED,
    NLP,
    stop_words,
)
import custom_functions as cf
from sklearn.model_selection import train_test_split
import spacy

PROJECT_DIR = os.path.abspath("../")
DATASET_NAME = "combined_dataset5.csv"
ADDITIONAL_DATASET_NAME = "fake_or_real_news.csv"
DATA_DIR = os.path.join(PROJECT_DIR, "data")


def clean_train_pipeline(
    texts: pd.Series,
    nlp: spacy.lang.en.English,
    min_freq: int = 50,
    top_tokens: int = 50,
    sizes_to_augment: float = 0.05,
):
    """
    Pipeline does the follwoing preprocessing steps with texts:
    * Removes named entities like 'Trump' and others.
    * Removes top overused tokens.
    * Removes proper nouns.
    * Removes special characters like HTML tags, digits, links, email address.
    * Removes rare tokens.
    * Applies text augmentation.
    * Removes all punctuation.
    * Cleans whitespaces.

    Pipeline is used for training data preprocessing.

    Args:
    texts: pd.Series, texts to preprocess.
    nlp: spacy.lang.en.English, spacy model for named entity recognition.
    min_freq: int, minimum frequency of a token to be kept in the text.
    top_tokens: int, number of top overused tokens to remove.
    sizes_to_augment: float, size of the text to augment.

    """
    tqdm.pandas()

    print("Removing named entities")
    cleaned_texts = texts.progress_apply(cf.remove_named_entities)
    print("Removing top tokens")
    cleaned_texts = pd.Series(
        cf.remove_top_overused_tokens(texts, top_tokens=top_tokens)
    )
    print("Removing proper nouns")
    cleaned_texts = cleaned_texts.progress_apply(
        lambda text: cf.remove_proper_nouns(text, nlp),
    )
    print("Removing chars and tags")
    cleaned_texts = cleaned_texts.progress_apply(cf.remove_chars_tags)
    print("Removing rare tokens")
    cleaned_texts = pd.Series(
        cf.remove_rare_tokens(
            cleaned_texts,
            min_freq=min_freq,
        )
    )
    print("Augmenting texts, insert")
    cleaned_texts = cf.augment_texts(
        cleaned_texts, size_to_augment=sizes_to_augment, action="insert"
    )
    print("Augmenting texts, substitution")
    cleaned_texts = cf.augment_texts(
        cleaned_texts, size_to_augment=sizes_to_augment, action="substitute"
    )
    print("Removing punctuation")
    cleaned_texts = cleaned_texts.progress_apply(cf.remove_punctuation)
    print("Cleaning whitespaces")
    cleaned_texts = cleaned_texts.progress_apply(cf.clean_whitespace)

    return cleaned_texts


def clean_val_test_pipeline(
    texts: pd.Series,
):
    """
    Pipeline does the following preprocessing steps with texts:
    * Removes special characters, tags etc.
    * Removes all punctuation.
    * Cleans whitespaces.

    Pipeline is used for validation and test data preprocessing.

    Args:
    texts: pd.Series, texts to preprocess.
    """
    tqdm.pandas()
    print("Removing chars and tags")
    cleaned_texts = texts.progress_apply(cf.remove_chars_tags)
    print("Removing punctuation")
    cleaned_texts = cleaned_texts.progress_apply(cf.remove_punctuation)
    print("Cleaning whitespaces")
    cleaned_texts = cleaned_texts.progress_apply(cf.clean_whitespace)

    return cleaned_texts


def pipeline_short(text: str) -> str:
    """
    Short text cleaning pipeline cleaning punctuation, whitespaces and
    changing tags to custom tokens.
    """
    cleaned_text = cf.clean_punctuation(text)
    cleaned_text = cf.clean_whitespace(cleaned_text)
    cleaned_text = cf.change_tags_to_custom_tokens(cleaned_text)
    return cleaned_text


def pipeline_long(text: str) -> str:
    """
    Long text cleaning pipeline cleaning punctuation, whitespaces, spelling,
    lemmatization, and changing tags to custom tokens.
    """
    cleaned_text = cf.fix_spelling(text)
    cleaned_text = cf.lemmatize_text(cleaned_text)
    cleaned_text = cf.clean_punctuation(cleaned_text)
    cleaned_text = cf.clean_whitespace(cleaned_text)
    cleaned_text = cf.change_tags_to_custom_tokens(cleaned_text)
    return cleaned_text


def clean_pipeline_one(
    texts: pd.Series,
    nlp: spacy.lang.en.English,
):
    """
    Pipeline does the following preprocessing steps with texts:
    * Removes named entities like 'Trump' and others.
    * Removes proper nouns.
    * Removes special characters like HTML tags, digits, links, email addresses.
    * Removes all punctuation.
    * Cleans whitespaces.

    Args:
    texts: pd.Series, texts to preprocess.
    """
    tqdm.pandas()

    print("Removing named entities")
    cleaned_texts = texts.progress_apply(cf.remove_named_entities)
    print("Removing proper nouns")
    cleaned_texts = cleaned_texts.progress_apply(
        lambda text: cf.remove_proper_nouns(text, nlp),
    )
    print("Removing chars and tags")
    cleaned_texts = cleaned_texts.progress_apply(cf.remove_chars_tags)
    print("Removing punctuation")
    cleaned_texts = cleaned_texts.progress_apply(cf.remove_punctuation)
    print("Cleaning whitespaces")
    cleaned_texts = cleaned_texts.progress_apply(cf.clean_whitespace)

    return cleaned_texts


def clean_pipeline_two(
    texts: pd.Series,
):
    """
    Pipeline does the following preprocessing steps with texts:
    * Lowercasing every word in the sentence.
    * Changing 't' to 'not'.
    * Removing '@name'
    * Isolating and removing punctuations except '?'.
    * Removing other special characters.
    * Removing stop words except "not" and "can".
    * Removing trailing whitespaces.

    Pipeline is based on recommendations described here:
    DOI:10.1016/j.ijcce.2022.03.003

    Args:
    texts: pd.Series, texts to preprocess.
    """
    tqdm.pandas()

    print("Lowercasing text")
    cleaned_texts = texts.progress_apply(lambda text: text.lower())
    print('Replacing "t" with "not"')
    cleaned_texts = cleaned_texts.progress_apply(cf.replace_t_with_not)
    print("Removing special characters, tags, numbers")
    cleaned_texts = cleaned_texts.progress_apply(cf.remove_chars_tags)
    print("Removing stopwords except 'not' and 'can'")
    cleaned_texts = cleaned_texts.progress_apply(
        lambda text: cf.remove_stopwords_except(
            text, stopwords=stop_words, stopwords_to_except={"not", "can"}
        )
    )
    print("Removing punctuation except for '?'")
    cleaned_texts = cleaned_texts.progress_apply(
        lambda text: cf.remove_punctuation_except(text, "?")
    )
    print("Cleaning whitespaces")
    cleaned_texts = cleaned_texts.progress_apply(cf.clean_whitespace)

    return cleaned_texts


def clean_pipeline_three(
    texts: pd.Series,
):
    """
    Pipeline does the following text processing steps:
    * Lowercases all text.
    * Removes URLs, HTML tags, hashtags, special characters.
    * Retains key punctuation like '?', '!' and '.'. Removes all other
    punctuation.
    * Removes all stopwords.
    * Cleans whitespaces.

    Args:
    texts: pd.Series, texts to preprocess.
    """
    tqdm.pandas()

    print("Lowercasing text")
    cleaned_texts = texts.progress_apply(lambda text: text.lower())
    print("Removing URLs, HTML tags, hashtags, special characters")
    cleaned_texts = cleaned_texts.progress_apply(cf.remove_mentions_hashes)
    cleaned_texts = cleaned_texts.progress_apply(cf.remove_urls)
    cleaned_texts = cleaned_texts.progress_apply(cf.remove_html)
    print("Removing stopwords")
    cleaned_texts = cleaned_texts.progress_apply(
        lambda text: cf.remove_stopwords(text, stopwords=stop_words)
    )
    print("Removing punctuation except for '?, !, .'")
    cleaned_texts = cleaned_texts.progress_apply(
        lambda text: re.sub(r"[^a-zA-Z0-9\s?!]", "", text)
    )
    print("Cleaning whitespaces")
    cleaned_texts = cleaned_texts.progress_apply(cf.clean_whitespace)

    return cleaned_texts


def clean_pipeline_four(
    texts: pd.Series,
):
    """
    Pipeline does the following preprocessing steps:
    * Lowercases all texts.
    * Removes URL, HTML tags, hashtags and other special characters.
    * Retains key punctuation like '?', '!' and '.'. Removes all other
    punctuation.
    * Removes repeated words in a text.
    * Cleans whitespaces.

    Args:
    texts: pd.Series, texts to preprocess.
    """
    tqdm.pandas()

    print("Lowercasing text")
    cleaned_texts = texts.progress_apply(lambda text: text.lower())
    print("Removing special characters, tags, numbers")
    cleaned_texts = cleaned_texts.progress_apply(cf.remove_chars_tags)
    print("Removing punctuation except for '?, !, .'")
    cleaned_texts = cleaned_texts.progress_apply(
        lambda text: re.sub(r"[^a-zA-Z0-9\s?!]", "", text)
    )
    print("Removing repeated words")
    cleaned_texts = cleaned_texts.progress_apply(
        lambda text: re.sub(r"\b(\w+)\s+\1\b", r"\1", text)
    )
    print("Cleaning whitespaces")
    cleaned_texts = cleaned_texts.progress_apply(cf.clean_whitespace)

    return cleaned_texts


def clean_pipeline_five(
    texts: pd.Series,
    nlp: spacy.lang.en.English,
):
    """
    Pipeline does the following preprocessing steps:
    * Removes named entities and proper nouns like 'Trump' and others.
    * Lowercases all texts.
    * Replaces 't' with 'not'.
    * Removes URL, HTML tags, hashtags and other special characters.
    * Retains key punctuation like '?', '!' and '.'. Removes all other
    punctuation.
    * Removes stopwords except 'not' and 'can'.
    * Cleans whitespaces.

    Args:
    texts: pd.Series, texts to preprocess.
    nlp: spacy.lang.en.English, spacy model for named entity recognition.
    """
    tqdm.pandas()

    print("Removing named entities")
    cleaned_texts = texts.progress_apply(cf.remove_named_entities)
    print("Removing proper nouns")
    cleaned_texts = cleaned_texts.progress_apply(
        lambda text: cf.remove_proper_nouns(text, nlp),
    )
    print("Lowercasing text")
    cleaned_texts = cleaned_texts.progress_apply(lambda text: text.lower())
    print('Replacing "t" with "not"')
    cleaned_texts = cleaned_texts.progress_apply(cf.replace_t_with_not)
    print("Removing special characters, tags, numbers")
    cleaned_texts = cleaned_texts.progress_apply(cf.remove_chars_tags)
    print("Removing punctuation except for '?, !, .'")
    cleaned_texts = cleaned_texts.progress_apply(
        lambda text: re.sub(r"[^a-zA-Z0-9\s?!]", "", text)
    )
    print("Removing stopwords")
    cleaned_texts = cleaned_texts.progress_apply(
        lambda text: cf.remove_stopwords(text, stopwords=stop_words)
    )
    # print("Removing stopwords except 'not' and 'can'")
    # cleaned_texts = cleaned_texts.progress_apply(
    #     lambda text: cf.remove_stopwords_except(
    #         text, stopwords=stop_words, stopwords_to_except={"not", "can"}
    #     )
    # )
    print("Cleaning whitespaces")
    cleaned_texts = cleaned_texts.progress_apply(cf.clean_whitespace)

    return cleaned_texts


def clean_pipeline_six(
    texts: pd.Series,
    nlp: spacy.lang.en.English,
):
    """
    Pipeline does the following preprocessing steps:
    * Removes named entities and proper nouns like 'Trump' and others.
    * Lowercases all texts.
    * Replaces 't' with 'not'.
    * Removes URL, HTML tags, hashtags and other special characters.
    * Retains key punctuation like '?'. Removes all other
    punctuation.
    * Removes repeated words in a text.
    * Cleans whitespaces.

    Args:
    texts: pd.Series, texts to preprocess.
    nlp: spacy.lang.en.English, spacy model for named entity recognition.
    """
    tqdm.pandas()

    print("Removing named entities")
    cleaned_texts = texts.progress_apply(cf.remove_named_entities)
    print("Removing proper nouns")
    cleaned_texts = cleaned_texts.progress_apply(
        lambda text: cf.remove_proper_nouns(text, nlp),
    )
    print("Lowercasing text")
    cleaned_texts = cleaned_texts.progress_apply(lambda text: text.lower())
    print('Replacing "t" with "not"')
    cleaned_texts = cleaned_texts.progress_apply(cf.replace_t_with_not)
    print("Removing special characters, tags, numbers")
    cleaned_texts = cleaned_texts.progress_apply(cf.remove_chars_tags)
    print("Removing punctuation except for '?, !, .'")
    cleaned_texts = cleaned_texts.progress_apply(
        lambda text: re.sub(r"[^a-zA-Z0-9\s?!]", "", text)
    )
    print("Removing repeated words")
    cleaned_texts = cleaned_texts.progress_apply(
        lambda text: re.sub(r"\b(\w+)\s+\1\b", r"\1", text)
    )
    print("Cleaning whitespaces")
    cleaned_texts = cleaned_texts.progress_apply(cf.clean_whitespace)

    return cleaned_texts


if __name__ == "__main__":

    data = pd.read_csv(os.path.join(DATA_DIR, DATASET_NAME))
    additional_data = pd.read_csv(
        os.path.join(
            DATA_DIR,
            ADDITIONAL_DATASET_NAME,
        )
    )

    train_data, temp_data = train_test_split(
        data,
        test_size=0.15,
        random_state=RANDOM_SEED,
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=RANDOM_SEED
    )

    pipelines_to_run = [
        clean_pipeline_one,
        clean_pipeline_two,
        clean_pipeline_three,
        clean_pipeline_four,
        clean_pipeline_five,
        clean_pipeline_six,
    ]

    pipeline_names = [
        "pipeline_one",
        "pipeline_two",
        "pipeline_three",
        "pipeline_four",
        "pipeline_five",
        "pipeline_six",
    ]

    for i, pipeline in enumerate(pipelines_to_run):
        print(f"Running pipeline {pipeline_names[i]}")
        print("Train set processing")
        changed_train_data = train_data.copy()
        changed_train_data["text"] = pipeline(changed_train_data["text"], NLP)
        changed_train_data.to_csv(
            os.path.join(
                DATA_DIR,
                f"train_processed_p{pipeline_names[i]}.csv",
            )
        )

        print("Validation set processing")
        changed_val_data = val_data.copy()
        changed_val_data["text"] = pipeline(changed_val_data["text"], NLP)
        changed_val_data.to_csv(
            os.path.join(
                DATA_DIR,
                f"val_processed_p{pipeline_names[i]}.csv",
            )
        )

        print("Test set processing")
        changed_test_data = test_data.copy()
        changed_test_data["text"] = pipeline(changed_test_data["text"], NLP)
        changed_test_data.to_csv(
            os.path.join(
                DATA_DIR,
                f"test_processed_p{pipeline_names[i]}.csv",
            )
        )

        print("Additional set processing")
        changed_additional_data = additional_data.copy()
        changed_additional_data["text"] = pipeline(
            changed_additional_data["text"],
            NLP,
        )
        changed_additional_data.to_csv(
            os.path.join(
                DATA_DIR,
                f"additional_set_processed_p{pipeline_names[i]}.csv",
            )
        )
