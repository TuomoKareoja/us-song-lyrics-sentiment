# %%

import os
import re

import matplotlib.pyplot as plt
import collections
import pandas as pd
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell
from wordcloud import WordCloud
from nltk.corpus import stopwords

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True, rc={"figure.figsize": (10.7, 8.27)})

# setting random seed
random_seed = 123


# %% load_data

df = pd.read_csv(os.path.join("data", "processed", "lyrics_processed.csv"), sep=";")

# %% missing

df.isna().sum().plot.bar()

# %% missing timeseries

columns_of_interest = ["lyrics", "duration_min"]

df.groupby("year")[columns_of_interest].apply(lambda x: x.isna().sum()).plot.line()
plt.title("Number of Songs with Missing Information")
plt.show()

df[(df["instrumental"] == 0) & (df["language"] == "en")].groupby("year")[
    columns_of_interest
].apply(lambda x: x.isna().mean().multiply(100)).plot.line()
plt.title("Percent of Non-instrumental English Songs with Missing Information")
plt.show()

# %% Wordcloud

lyrics_text = " ".join(df["lyrics_lemmatized"].dropna())
wordcloud = WordCloud(max_font_size=40, background_color="white").generate(lyrics_text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Wordcloud of lyrics")
plt.show()

# %% Plot of word frequencies

word_counter = collections.Counter(lyrics_text.split())
word_freq_df = pd.DataFrame.from_dict(word_counter, orient="index").reset_index()
word_freq_df.columns = ["word", "count"]
# removing stopwords
stopwords_no_quotes = [
    re.sub("'", "", stopword) for stopword in stopwords.words("english")
]
word_freq_df = word_freq_df[~word_freq_df["word"].isin(stopwords_no_quotes)]
word_freq_df.sort_values(by="count", ascending=False, inplace=True)
sns.barplot(x="word", y="count", data=word_freq_df.iloc[:20, :])
plt.title("Top 10 Words by frequency")
plt.xlabel("Year")
plt.ylabel("Word Frequency")
plt.show()

# %% wordcount and unique words by time

sns.lineplot(x="year", y="wordcount", data=df)
plt.title("Average Wordcount by Year")
plt.ylim(bottom=0)
plt.xlabel("Year")
plt.ylabel("Wordcount")
plt.show()

sns.lineplot(x="year", y="perc_words_unique", data=df)
plt.title("Percent of Words Unique by Year")
plt.ylim(bottom=0)
plt.xlabel("Year")
plt.ylabel("% of Words Unique")

plt.show()

sns.lineplot(x="year", y="words_per_min", data=df)
plt.title("Average Words per Minute by Year")
plt.ylim(bottom=0)
plt.xlabel("Year")
plt.ylabel("Wordcount")
plt.savefig(os.path.join("reports", "figures", "average_words_per_minute_by_year.png"))
plt.show()

sns.lineplot(x="year", y="unique_words_per_min", data=df)
plt.title("Average of Unique Words per Minute by Year")
plt.ylim(bottom=0)
plt.xlabel("Year")
plt.ylabel("Unique Words per Minute")
plt.show()


# %% Spotify audio features over time

sns.lineplot(x="year", y="duration_min", data=df)
plt.ylim(bottom=0)
plt.title("Song Duration by Year")
plt.xlabel("Year")
plt.ylabel("Average Duration in Seconds")
plt.savefig(os.path.join("reports", "figures", "average_duration_by_year.png"))
plt.show()

sns.lineplot(x="year", y="loudness", data=df)
plt.title("Song Decibels by Year")
plt.xlabel("Year")
plt.ylabel("Average Decibels by Year")
plt.savefig(os.path.join("reports", "figures", "average_decibels_by_year.png"))
plt.show()

sns.lineplot(x="year", y="tempo", data=df)
plt.ylim(bottom=0)
plt.title("Song Tempo by Year")
plt.xlabel("Year")
plt.ylabel("Average Beats per Minute")
plt.savefig(os.path.join("reports", "figures", "average_tempo_by_year.png"))
plt.show()

sns.lineplot(x="year", y="danceability", data=df, ci=None, label="Danceability")
sns.lineplot(x="year", y="energy", data=df, ci=None, label="Energy")
sns.lineplot(x="year", y="speechiness", data=df, ci=None, label="Speechiness")
sns.lineplot(x="year", y="acousticness", data=df, ci=None, label="Acousticness")
sns.lineplot(x="year", y="instrumentalness", data=df, ci=None, label="Instrumentalness")
sns.lineplot(x="year", y="valence", data=df, ci=None, label="Valence")
plt.ylim(bottom=0)
plt.title("Average Value of Spotify Audio Features by Year")
plt.xlabel("Year")
plt.ylabel("Feature Value")
plt.show()

sns.lineplot(x="year", y="danceability", data=df, ci=None, label="Danceability")
sns.lineplot(x="year", y="energy", data=df, ci=None, label="Energy")
sns.lineplot(x="year", y="acousticness", data=df, ci=None, label="Acousticness")
plt.ylim(bottom=0)
plt.title("Average Value of Spotify Audio Features by Year")
plt.xlabel("Year")
plt.ylabel("Feature Value")
plt.savefig(os.path.join("reports", "figures", "the_victory_of_electric_music.png"))
plt.show()

sns.lineplot(x="year", y="valence", data=df)
plt.ylim(bottom=0)
plt.title("Average Valence by Year")
plt.xlabel("Year")
plt.ylabel("Valence")
plt.savefig(os.path.join("reports", "figures", "spotify_valence_by_year.png"))
plt.show()


# %% Afinn sentiment with time

sns.lineplot(x="year", y="sentiment_abs", data=df, label="Absolute Sentiment")
sns.lineplot(x="year", y="sentiment_positive", data=df, label="Positive Sentiment")
sns.lineplot(x="year", y="sentiment_negative", data=df, label="Negative Sentiment")
plt.ylim(bottom=0)
plt.title("Average Afinn Sentiment by Year and Type")
plt.xlabel("Year")
plt.ylabel("Sentiment")
plt.show()

# sns.lineplot(x="year", y="sentiment_abs_per_word", data=df, label="Absolute Sentiment")
sns.lineplot(
    x="year", y="sentiment_positive_per_word", data=df, label="Positive Sentiment"
)
sns.lineplot(
    x="year", y="sentiment_negative_per_word", data=df, label="Negative Sentiment"
)
plt.ylim(bottom=0)
plt.title("Average Afinn Sentiment per Word by Year and Type")
plt.xlabel("Year")
plt.ylabel("Sentiment per Word")
plt.savefig(os.path.join("reports", "figures", "average_afinn_sentiment_by_year.png"))
plt.show()

sns.lineplot(x="year", y="sentiment_abs_per_min", data=df, label="Absolute Sentiment")
sns.lineplot(
    x="year", y="sentiment_positive_per_min", data=df, label="Positive Sentiment"
)
sns.lineplot(
    x="year", y="sentiment_negative_per_min", data=df, label="Negative Sentiment"
)
plt.ylim(bottom=0)
plt.title("Average Afinn Sentiment per Minute by Year and Type")
plt.xlabel("Year")
plt.ylabel("Sentiment per Minute")
plt.show()


# %% NRC sentiment with time

sns.lineplot(x="year", y="nrc_anticipation", ci=None, data=df, label="Anticipation")
sns.lineplot(x="year", y="nrc_disgust", ci=None, data=df, label="Disgust")
sns.lineplot(x="year", y="nrc_surprise", ci=None, data=df, label="Surprise")
sns.lineplot(x="year", y="nrc_trust", ci=None, data=df, label="Trust")
sns.lineplot(x="year", y="nrc_sadness", ci=None, data=df, label="Sadness")
sns.lineplot(x="year", y="nrc_anger", ci=None, data=df, label="Anger")
sns.lineplot(x="year", y="nrc_joy", ci=None, data=df, label="Joy")
sns.lineplot(x="year", y="nrc_fear", ci=None, data=df, label="Fear")
plt.ylim(bottom=0)
plt.title("Average NRC Sentiment by Year and Type")
plt.xlabel("Year")
plt.ylabel("Sentiment")
plt.show()

sns.lineplot(
    x="year", y="nrc_anticipation_per_word", ci=None, data=df, label="Anticipation"
)
sns.lineplot(x="year", y="nrc_disgust_per_word", ci=None, data=df, label="Disgust")
sns.lineplot(x="year", y="nrc_surprise_per_word", ci=None, data=df, label="Surprise")
sns.lineplot(x="year", y="nrc_trust_per_word", ci=None, data=df, label="Trust")
sns.lineplot(x="year", y="nrc_sadness_per_word", ci=None, data=df, label="Sadness")
sns.lineplot(x="year", y="nrc_anger_per_word", ci=None, data=df, label="Anger")
sns.lineplot(x="year", y="nrc_joy_per_word", ci=None, data=df, label="Joy")
sns.lineplot(x="year", y="nrc_fear_per_word", ci=None, data=df, label="Fear")
plt.ylim(bottom=0)
plt.title("Average NRC Sentiment per Word by Year and Type")
plt.xlabel("Year")
plt.ylabel("Sentiment")
plt.savefig(os.path.join("reports", "figures", "average_nrc_sentiment_by_year.png"))
plt.show()

sns.lineplot(
    x="year", y="nrc_anticipation_per_min", ci=None, data=df, label="Anticipation"
)
sns.lineplot(x="year", y="nrc_disgust_per_min", ci=None, data=df, label="Disgust")
sns.lineplot(x="year", y="nrc_surprise_per_min", ci=None, data=df, label="Surprise")
sns.lineplot(x="year", y="nrc_trust_per_min", ci=None, data=df, label="Trust")
sns.lineplot(x="year", y="nrc_sadness_per_min", ci=None, data=df, label="Sadness")
sns.lineplot(x="year", y="nrc_anger_per_min", ci=None, data=df, label="Anger")
sns.lineplot(x="year", y="nrc_joy_per_min", ci=None, data=df, label="Joy")
sns.lineplot(x="year", y="nrc_fear_per_min", ci=None, data=df, label="Fear")
plt.ylim(bottom=0)
plt.title("Average NRC Sentiment per Minute by Year and Type")
plt.xlabel("Year")
plt.ylabel("Sentiment per Minute")
plt.show()

sns.lineplot(x="year", y="nrc_positive", data=df, label="Positive")
sns.lineplot(x="year", y="nrc_negative", data=df, label="Negative")
plt.ylim(bottom=0)
plt.title("Average Overall NRC Sentiment by Year")
plt.xlabel("Year")
plt.ylabel("Sentiment")
plt.show()

sns.lineplot(x="year", y="nrc_positive_per_word", data=df, label="Positive")
sns.lineplot(x="year", y="nrc_negative_per_word", data=df, label="Negative")
plt.ylim(bottom=0)
plt.title("Average Overall NRC Sentiment per Word by Year")
plt.xlabel("Year")
plt.ylabel("Sentiment per Word")
plt.savefig(
    os.path.join("reports", "figures", "average_nrc_emotions_overall_by_year.png")
)
plt.show()

sns.lineplot(x="year", y="nrc_positive_per_min", data=df, label="Positive")
sns.lineplot(x="year", y="nrc_negative_per_min", data=df, label="Negative")
plt.ylim(bottom=0)
plt.title("Average Overall NRC Sentiment per Minute by Year")
plt.xlabel("Year")
plt.ylabel("Sentiment per Minute")
plt.show()

# %% Creating index of positivity and negativity

index_columns = [
    "nrc_positive_per_word",
    "sentiment_positive_per_word",
    "valence",
    "nrc_negative_per_word",
    "sentiment_negative_per_word",
    "valence",
]

for column in index_columns:
    df[column + "_scaled"] = df[column] - df[column].mean() / (
        df[column].max() - df[column].min()
    )

positivity_index = (
    df["nrc_positive_per_word_scaled"]
    + df["sentiment_positive_per_word_scaled"]
    + df["valence_scaled"]
)

negativity_index = (
    df["nrc_negative_per_word_scaled"]
    + df["sentiment_negative_per_word_scaled"]
    - df["valence_scaled"]
)

# scaling the indices
df["positivity_index"] = (
    positivity_index
    - positivity_index.mean() / positivity_index.max()
    - positivity_index.min()
)

df["negativity_index"] = (
    negativity_index
    - negativity_index.mean() / negativity_index.max()
    - negativity_index.min()
)

df["emotion_index"] = df["positivity_index"] - df["negativity_index"]

# %%

sns.lineplot(x="year", y="positivity_index", data=df, label="Positivity")
sns.lineplot(x="year", y="negativity_index", data=df, label="Negativity")
sns.lineplot(x="year", y="emotion_index", data=df, label="Overall")
plt.ylim(bottom=0)
plt.title("Average Emotion Index by Year")
plt.xlabel("Year")
plt.ylabel("Emotion Index")
plt.savefig(os.path.join("reports", "figures", "emotion_index_by_year.png"))
plt.show()


# %% Most negative songs

df[df['emotion_index'] == df['emotion_index'].min()]

# %% Most positive songs

df[df['emotion_index'] == df['emotion_index'].max()]

# %% most lyrics

df[df['wordcount'] == df['wordcount'].max()]

# %% most unique lyrics

df[df['unique_words'] == df['unique_words'].max()]

# %% most lyrics per minute

df[df['words_per_min'] == df['words_per_min'].max()]

# %%
