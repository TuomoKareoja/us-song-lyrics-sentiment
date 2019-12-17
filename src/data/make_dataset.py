import logging
import os
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from afinn import Afinn
from dotenv import find_dotenv, load_dotenv
from langdetect import DetectorFactory, detect
from nltk.stem import WordNetLemmatizer

# setting random seed
random_seed = 123


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    logger.info("Loading data")
    df = pd.read_csv(os.path.join("data", "raw", "billboard100_1960-2015.csv"), sep=";")

    logger.info("Changing song duration to minutes")
    df = change_song_duration(df)

    logger.info("Fixing weird spelling")
    df = fix_weird_spellings(df)

    logger.info("Recognizing language")
    df = recognize_language(df)

    logger.info("Removing or replacing special characters")
    df = remove_or_replace_special_characters(df)

    logger.info("Finding instrumental pieces")
    df = find_instrumentals(df)

    logger.info("Lemmatizing lyrics")
    df = lemmatize(df)

    logger.info("Calculating aggregate data about lyrics")
    df = add_lyrics_agg_info(df)

    logger.info("Calculating Afinn sentiment")
    df = calculate_afinn_sentiment(df)

    logger.info("Calculating Emolex sentiment")
    df = calculate_emolex_sentiment(df)

    logger.info("Marking lyrics columns null if something weird about them")
    df = mark_lyrics_values_na(df)

    logger.info("Saving to data/processed/")
    df.to_csv(
        os.path.join("data", "processed", "lyrics_processed.csv"), sep=";", index=False
    )


def change_song_duration(df):

    df["duration_min"] = np.round(df["duration_ms"] / (1000 * 60), 1)
    # no song should be less than 1 min
    df["duration_min"].mask(df["duration_min"] <= 1, np.nan, inplace=True)
    df.drop(columns=["duration_ms"], inplace=True)

    return df


def find_instrumentals(df):
    # finding instrumental songs (if no lyrics then mark None)
    # if between 1 and 150 characters in lyrics also mark as instrumental
    # if spotify instrumentalness is more than 0.7 then instrumental
    # if contains word instrumental in lyrics then instrumental
    df["instrumental"] = np.where(
        ((df["lyrics"].str.len() > 0) & (df["lyrics"].str.len() <= 150))
        | (df["instrumentalness"] > 0.7),
        1,
        np.where(df["lyrics"].isna(), np.nan, 0),
    )

    # remove lyrics from instrumental songs
    df["lyrics"].where(df["instrumental"] != 1, np.nan, inplace=True)

    # remove language from instrumental songs
    df["language"].where(df["instrumental"] != 1, np.nan, inplace=True)

    return df


def remove_or_replace_special_characters(df):

    # marking not found as truly missing
    df["lyrics"].replace("Not found", np.nan, inplace=True)

    # lowercase
    df["lyrics"] = df["lyrics"].str.lower()

    # Remove everything within [] and () and {} including themselves
    # these are usually background vocals and metainfomation (chorus, verse, etc.)
    within_brackets_and_such = r"\[[^]]*\]|\([^)]*\)|\{[^}]*\}"
    # words marking the song structure or repetitions
    # also some found misspellings included
    song_structure_info = r"chorus|bridge|verse|chorous|repeat|instrumental"

    # markdown parts
    markdown_parts = r"\&amp quot|\&amp|\&quot"

    # remove x times repetition markings
    repeat_x_first = r"\sx\s?[0-9]\s"
    repeat_number_first = r"\s[0-9]*\s?x\s"

    all_patterns = "|".join(
        [
            within_brackets_and_such,
            song_structure_info,
            markdown_parts,
            repeat_x_first,
            repeat_number_first,
        ]
    )

    df["lyrics"].replace(all_patterns, "", regex=True, inplace=True)

    # remove \x
    df["lyrics"].replace(r"\x", "", inplace=True)

    df["lyrics"].replace("[^a-z ]+", "", regex=True, inplace=True)

    # strip extra spaces
    df["lyrics"] = df["lyrics"].str.strip()
    df["lyrics"].replace(r"\s+", " ", regex=True, inplace=True)

    return df


def fix_weird_spellings(df):

    unbeeped_curse_words = {"sh~t": "shit", "b~itch": "bitch"}

    with open(os.path.join("data", "lexicons", "quote_words.txt"), "r") as text:
        quote_words = eval(text.read())

    df["lyrics"].replace(unbeeped_curse_words, inplace=True)
    df["lyrics"].replace(quote_words, inplace=True)

    return df


def recognize_language(df):

    # making sure there are no nones and just seeds
    df["lyrics"].fillna(np.nan, inplace=True)

    # language inference is non deterministic so
    # to keep things from changing around use seed
    DetectorFactory.seed = random_seed

    song_languages = []

    for lyric in df["lyrics"]:
        if lyric != lyric:
            song_languages.append(np.nan)
        else:
            language = detect(lyric)
            song_languages.append(language)

    # only real other languages are es, it, de, fr, others are mistakes
    song_languages = [
        language if language in ["en", "es", "it", "de", "fr", np.nan] else "en"
        for language in song_languages
    ]

    df["language"] = song_languages

    return df


def lemmatize(df):

    wordnet_lemmatizer = WordNetLemmatizer()

    lemmatized_lyrics = []

    for lyric, language in zip(df["lyrics"], df["language"]):

        if language == "en" and lyric is not np.nan:
            lyric_words = nltk.word_tokenize(lyric)

            lemmatized_words = []

            for word in lyric_words:
                lemmatized_words.append(wordnet_lemmatizer.lemmatize(word))
                lemmatized_words.append(" ")

            lemmatized_words = "".join(lemmatized_words)
            lemmatized_lyrics.append(lemmatized_words)

        else:
            lemmatized_lyrics.append("")

    df["lyrics_lemmatized"] = lemmatized_lyrics

    return df


def add_lyrics_agg_info(df):

    df["wordlist"] = df["lyrics_lemmatized"].str.split()
    df["wordcount"] = df["wordlist"].str.len()
    df["words_per_min"] = df["wordcount"] / df["duration_min"]
    df["unique_words"] = df["wordlist"].apply(lambda x: len(set(x)))
    df["unique_words_per_min"] = df["unique_words"] / df["duration_min"]
    df["perc_words_unique"] = df["unique_words"] * 100.0 / df["wordcount"]

    return df


def calculate_afinn_sentiment(df):

    af = Afinn(emoticons=False)

    # Positive sentiment. Negative sentiment words count as 0
    df["sentiment_positive"] = [
        np.sum([np.max(af.score(word), 0) for word in lyric.split()])
        for lyric in df["lyrics_lemmatized"]
    ]

    # Negative sentiment. Positive sentiment words count as 0
    df["sentiment_negative"] = [
        np.sum([np.absolute(np.min(af.score(word), 0)) for word in lyric.split()])
        for lyric in df["lyrics_lemmatized"]
    ]

    df["sentiment"] = df["sentiment_positive"] - df["sentiment_negative"]

    # Absolute value of the sentiment. Emotionally ladden lyrics
    # get high score regardles of valence
    df["sentiment_abs"] = df["sentiment_positive"] + df["sentiment_negative"]

    # sentiment per word
    df["sentiment_positive_per_word"] = df["sentiment_positive"].divide(df["wordcount"])
    df["sentiment_negative_per_word"] = df["sentiment_negative"].divide(df["wordcount"])
    df["sentiment_abs_per_word"] = df["sentiment_abs"].divide(df["wordcount"])

    # sentiment per minute
    df["sentiment_positive_per_min"] = df["sentiment_positive"].divide(
        df["duration_min"]
    )
    df["sentiment_negative_per_min"] = df["sentiment_negative"].divide(
        df["duration_min"]
    )
    df["sentiment_abs_per_min"] = df["sentiment_abs"].divide(df["duration_min"])

    return df


def calculate_emolex_sentiment(df):

    # read data in
    nrc_df = pd.read_csv(
        os.path.join("data", "lexicons", "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"),
        sep="\t",
        header=None,
        names=["word", "emotion", "score"],
    )

    # naming emotions so that it is clearly nrc
    # this is useful later as we use these values
    # as column names
    nrc_df["emotion"] = "nrc_" + nrc_df["emotion"]

    lyrics_df = df[["wordlist"]]
    lyrics_df = lyrics_df.explode("wordlist")
    lyrics_df.columns = ["word"]
    lyrics_df["song_index"] = lyrics_df.index
    lyrics_df = pd.merge(lyrics_df, nrc_df, how="left", on="word")
    lyrics_df = lyrics_df.groupby(["song_index", "emotion"], as_index=False)[
        "score"
    ].sum()
    lyrics_df = (
        lyrics_df.set_index(["song_index", "emotion"])["score"].unstack().reset_index()
    )

    df = pd.merge(df, lyrics_df, how="left", left_index=True, right_on="song_index")
    df.drop(columns="song_index", inplace=True)

    # nrc sentiment per word and minute
    nrc_columns = [column for column in df.columns if "nrc" in column]

    for emotion in nrc_columns:
        df[emotion + "_per_word"] = df[emotion].divide(df["wordcount"])
        df[emotion + "_per_min"] = df[emotion].divide(df["duration_min"])

    return df


def mark_lyrics_values_na(df):

    no_lyrics = df["lyrics_lemmatized"] == ""

    lyric_columns = [
        "nrc_anger_per_min",
        "nrc_anger_per_word",
        "nrc_anger",
        "nrc_anticipation_per_min",
        "nrc_anticipation_per_word",
        "nrc_anticipation",
        "nrc_disgust_per_min",
        "nrc_disgust_per_word",
        "nrc_disgust",
        "nrc_fear_per_min",
        "nrc_fear_per_word",
        "nrc_fear",
        "nrc_joy_per_min",
        "nrc_joy_per_word",
        "nrc_joy",
        "nrc_negative_per_min",
        "nrc_negative_per_word",
        "nrc_negative",
        "nrc_positive_per_min",
        "nrc_positive_per_word",
        "nrc_positive",
        "nrc_sadness_per_min",
        "nrc_sadness_per_word",
        "nrc_sadness",
        "nrc_surprise_per_min",
        "nrc_surprise_per_word",
        "nrc_surprise",
        "nrc_trust_per_min",
        "nrc_trust_per_word",
        "nrc_trust",
        "perc_words_unique",
        "sentiment_abs_per_min",
        "sentiment_abs_per_word",
        "sentiment_abs",
        "sentiment_negative_per_min",
        "sentiment_negative_per_word",
        "sentiment_negative",
        "sentiment_positive_per_min",
        "sentiment_positive_per_word",
        "sentiment_positive",
        "sentiment",
        "unique_words_per_min",
        "unique_words",
        "wordcount",
        "wordlist",
        "words_per_min",
    ]

    for column in lyric_columns:
        df[column].mask(no_lyrics, other=np.nan, inplace=True)

    return df


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
