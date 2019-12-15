# %%

import os

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from afinn import Afinn
from IPython.core.interactiveshell import InteractiveShell
from langdetect import DetectorFactory, detect
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True, rc={"figure.figsize": (12.7, 9.27)})

# setting random seed
random_seed = 123


# %% load_data

df = pd.read_csv(os.path.join("data", "raw", "billboard100_1960-2015.csv"), sep=";")

# %% # marking not found as truly missing

df["lyrics"].replace("Not found", np.nan, inplace=True)

# %% lyrics missing

print(round(df["lyrics"].isna().sum() * 100 / len(df["lyrics"])), "% of lyrics missing")


# %% lowercase

df["lyrics"] = df["lyrics"].str.lower()

# %% some instrumental songs have these marking

df["lyrics"].replace("---", "", inplace=True)

# %% strip extra spaces

df["lyrics"] = df["lyrics"].str.strip()
df["lyrics"].replace(r"\s+", " ", regex=True, inplace=True)

# %% find instrumental songs

# finding instrumental songs (if no lyrics then mark None)
# if less than 150 characters in lyrics also mark as instrumental
# if spotify instrumentalness is more than 0.7 then instrumental
df["instrumental"] = np.where(
    (df["lyrics"].str.len() < 150) | (df["instrumentalness"] > 0.7),
    1,
    np.where(df["lyrics"].isna(), np.nan, 0),
)

# remove lyrics from instrumental songs
df["lyrics"].where(df["instrumental"] != 1, np.nan, inplace=True)

# %% unbeeping curse words

unbeeped_curse_words = {"sh~t": "shit", "b~itch": "bitch"}

df["lyrics"].replace(unbeeped_curse_words, inplace=True)

# %%

# Remove everything within [] and () and {} including themselves
# these are usually background vocals and metainfomation (chorus, verse, etc.)
within_brackets_and_such = r"\[[^]]*\]|\([^)]*\)|\{[^}]*\}"

# words marking the song structure or repetitions
# also some found misspellings included
song_structure_info = r"chorus|bridge|verse|chorous|repeat"

# markdown parts
markdown_parts = r"\&amp quot|\&amp|\&quot"

all_patterns = "|".join([within_brackets_and_such, song_structure_info, markdown_parts])
df["lyrics"].replace(all_patterns, "", regex=True, inplace=True)

# %% remove special characters

# TODO:
# remove \x
df["lyrics"].replace(r"\x", "", inplace=True)

# remove
# still leaves single quotes as these are used in actual words
df["lyrics"].replace(r'[+=#*.;:!?,"(){}\[\]]', "", regex=True, inplace=True)

# replacing with space
# - is still kept as it is used in actual words
df["lyrics"].replace(r"(<br\s*/><br\s*/>)|(\/)|[_]", " ", regex=True, inplace=True)

# %% remove x times repetition markings

repeat_x_first = r"\sx\s?[0-9]\s"
repeat_number_first = r"\s[0-9]*\s?x\s"

all_patterns = "|".join([repeat_x_first, repeat_number_first])

# replacing with space so we don't combine words by accident
df["lyrics"].replace(r"(<br\s*/><br\s*/>)|(\/)", " ", regex=True, inplace=True)

# %% remove numbers

df["lyrics"].replace("[0-9]+", "", regex=True, inplace=True)

# %% making the single quote marks proper

df["lyrics"].replace(r"[“”’`]", "'", regex=True, inplace=True)

# %% Finding which are the most common words with ' in them

words = []
df[~df["lyrics"].isna()]["lyrics"].str.split().apply(words.extend)
words_df = pd.DataFrame(words)
words_df.columns = ["words"]
quote_words_df = words_df[words_df["words"].str.contains(r"^\'|\'$", regex=True)]
quote_words_counted = quote_words_df["words"].value_counts()
quote_words_counted[quote_words_counted > 5].index.to_list()

# %%

replace_quote_words = {
    "'cause": "because",
    "'em": "them",
    "lovin'": "loving",
    "'til": "until",
    "'bout": "about",
    "goin'": "going",
    "tryin'": "trying",
    "nothin'": "nothing",
    "lookin'": "looking",
    "feelin'": "feeling",
    "gon'": "going",
    "gettin'": "getting",
    "makin'": "making",
    "comin'": "coming",
    "'round": "around",
    "livin'": "living",
    "darlin'": "darling",
    "doin'": "doing",
    "runnin'": "running",
    "thinkin'": "thinking",
    "somethin'": "something",
    "'cos": "because",
    "talkin'": "talking",
    "dancin'": "danging",
    "'till": "until",
    "movin'": "moving",
    "turnin'": "turning",
    "rockin'": "rocking",
    "cryin'": "crying",
    "rollin'": "rolling",
    "fallin'": "falling",
    "sayin'": "saying",
    "walkin'": "walking",
    "an'": "and",
    "takin'": "taking",
    "singin'": "singing",
    "twistin'": "twisting",
    "burnin'": "burning",
    "playin'": "playing",
    "waitin'": "waiting",
    "mornin'": "morning",
    "givin'": "giving",
    "sittin'": "sitting",
    "dreamin'": "dreaming",
    "tellin'": "telling",
    "holdin'": "holding",
    "lil'": "little",
    "hangin'": "hanging",
    "slippin'": "slipping",
    "ol'": "old",
    "callin'": "calling",
    "'cuz": "because",
    "leavin'": "leaving",
    "fuckin'": "fucking",
    "drivin'": "driving",
    "shakin'": "shaking",
    "shinin'": "shining",
    "groovin'": "grooving",
    "smilin'": "smiling",
    "lyin'": "lying",
    "havin'": "having",
    "ridin'": "riding",
    "surfin'": "surfing",
    "workin'": "working",
    "diggin'": "digging",
    "searchin'": "searching",
    "wonderin'": "wondering",
    "laughin'": "laughing",
    "swingin'": "swinging",
    "dyin'": "dying",
    "standin'": "standing",
    "breakin'": "breaking",
    "losin'": "losing",
    "watchin'": "watching",
    "startin'": "starting",
    "missin'": "missing",
    "hatin'": "hating",
    "hurtin'": "hurting",
    "wishin'": "wishing",
    "kissin'": "kissing",
    "keepin'": "keeping",
    "jumpin'": "jumping",
    "hopin'": "hoping",
    "bein'": "being",
    "tossin'": "tossing",
    "wit'": "with",
    "kickin'": "kicking",
    "truckin'": "trucking",
    "bouncin'": "bouncing",
    "changin'": "changing",
    "beggin'": "begging",
    "ramblin'": "rambling",
    "showin'": "showing",
    "messin'": "messing",
    "knowin'": "knowing",
    "evenin'": "evening",
    "rainin'": "raining",
    "askin'": "asking",
    "screamin'": "screaming",
    "'cross": "across",
    "blowin'": "blowing",
    "wastin'": "wasting",
    "sleepin'": "sleeping",
    "pressurin'": "pressuring",
    "poppin'": "popping",
    "drinkin'": "drinking",
    "checkin'": "checking",
    "steppin'": "stepping",
    "barefootin'": "barefooting",
    "smokin'": "smoking",
    "trippin'": "tripping",
    "wantin'": "wanting",
    "pushin'": "pushing",
    "puttin'": "putting",
    "bumpin'": "bumping",
    "wearin'": "wearing",
    "stoppin'": "stopping",
    "'fore": "before",
    "knockin'": "knocking",
    "seein'": "seeing",
    "nuttin'": "nothing",
    "shootin'": "shooting",
    "stayin'": "staying",
    "findin'": "finding",
    "bringin'": "bringing",
    "listenin'": "listening",
    "hitchin'": "hitching",
    "reelin'": "reeling",
    "blastin'": "blasting",
    "believin'": "believing",
    "chasin'": "chasing",
    "spendin'": "spending",
    "chillin'": "chilling",
    "spinnin'": "spinning",
    "growin'": "growing",
    "stealin'": "stealing",
    "actin'": "acting",
    "lettin'": "letting",
    "flyin'": "flying",
    "cookin'": "cooking",
    "hollerin'": "hollering",
    "touchin'": "touching",
    "ringin'": "ringing",
    "foolin'": "fooling",
    "draggin'": "dragging",
    "killin'": "killing",
    "throwin'": "throwing",
    "pumpin'": "pumping",
    "»lil'": "little",
    "'nuff": "enough",
    "pickin'": "picking",
    "creepin'": "creeping",
    "cruisin'": "cruising",
    "morn'": "morning",
    "sellin'": "selling",
    "'stead": "instead",
    "complanin'": "complaining",
    "payin'": "paying",
    "sailin'": "sailing",
    "layin'": "laying",
    "travelin'": "traveling",
    "rubbin'": "rubbing",
    "frontin'": "fronting",
    "'head": "ahead",
    "befo'": "before",
    "bustin'": "busting",
    "soakin'": "soaking",
    "jammin'": "jamming",
    "beatin'": "beating",
    "mothafuckin'": "motherfucking",
    "flowin'": "flowing",
    "hittin'": "hitting",
    "sippin'": "sipping",
    "needin'": "needing",
    "headin'": "heading",
    "floatin'": "floating",
    "countin'": "counting",
    "eatin'": "eating",
    "prayin'": "praying",
    "survivin'": "surviving",
    "til'": "until",
    "cheatin'": "cheating",
    "tho'": "though",
    "risin'": "rising",
    "happenin'": "happening",
    "fumblin'": "fumbling",
    "breathin'": "breathing",
    "fightin'": "fighting",
    "forgettin'": "forgetting",
    "'magination": "imagination",
    "slammin'": "slamming",
    "'neath": "underneath",
    "pimpin'": "pimping",
    "perculatin'": "perculating",
    "travellin'": "traveling",
    "settin'": "setting",
    "treatin'": "treating",
    "hustlin'": "hustling",
    "noddin'": "nodding",
    "gonna'": "going to",
    "followin'": "following",
    "reachin'": "reaching",
    "'cept": "except",
    "trickin'": "tricking",
    "rhymin'": "rhyming",
    "tumblin'": "tumbling",
    "hoppin'": "hopping",
    "dealin'": "dealing",
    "bout'": "about",
    "hidin'": "hiding",
    "stallin'": "stalling",
    "motherfuckin'": "motherfucking",
    "wanna'": "want to",
    "shoutin'": "shouting",
    "sinkin'": "sinking",
    "learnin'": "learning",
    "thumpin'": "thumping",
    "pullin'": "pulling",
    "tearin'": "tearing",
    "buyin'": "buying",
    "swayin'": "swaying",
    "datin'": "dating",
    "huggin'": "hugging",
    "straightenin'": "straightening",
    "charmin'": "charming",
    "gatherin'": "gathering",
    "liftin'": "lifting",
    "humpin'": "humping",
    "stowin'": "stowing",
    "whippin'": "whipping",
    "stumblin'": "stumbling",
    "questionin'": "questioning",
    "flippin'": "flipping",
    "yellin'": "yelling",
    "shufflin'": "shuffling",
    "passin'": "passing",
    "grindin'": "grinding",
    "cuttin'": "cutting",
    "chokin'": "choking",
    "pleasin'": "pleasing",
    "droppin'": "dropping",
    "parkin'": "parking",
}

df["lyrics"].replace(replace_quote_words, inplace=True)

# removing single quote marks finally

df["lyrics"].replace("'", "", inplace=True)

# %% strip extra spaces again

df["lyrics"] = df["lyrics"].str.strip()
df["lyrics"].replace(r"\s+", " ", regex=True, inplace=True)

# %% recognise language

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


# %% Lemmatize

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


# %% Information about the lyrics

df["wordlist"] = df["lyrics_lemmatized"].str.split()
df["wordcount"] = df["wordlist"].str.len()
df["unique_words"] = df["wordlist"].apply(lambda x: len(set(x)))
df["perc_words_unique"] = df["unique_words"] / df["wordcount"]


# %% Calculating the Afinn sentiment

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


# %% Calculating EmoLex sentiment

# read data in
nrc_df = pd.read_csv(
    os.path.join("data", "lexicons", "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"),
    sep="\t",
    header=None,
    names=["word", "emotion", "score"],
)

# %%

# naming emotions so that it is clearly nrc
# this is useful later as we use these values
# as column names
nrc_df["emotion"] = "nrc_" + nrc_df["emotion"]

# %% combining to data

lyrics_df = df[["wordlist"]]
lyrics_df = lyrics_df.explode("wordlist")
lyrics_df.columns = ["word"]
lyrics_df["song_index"] = lyrics_df.index
lyrics_df = pd.merge(lyrics_df, nrc_df, how="left", on="word")
lyrics_df = lyrics_df.groupby(["song_index", "emotion"], as_index=False)["score"].sum()
lyrics_df = (
    lyrics_df.set_index(["song_index", "emotion"])["score"].unstack().reset_index()
)

df = pd.merge(df, lyrics_df, how="left", left_index=True, right_on="song_index")
df.drop(columns="song_index", inplace=True)


# %% nrc sentiment per word

nrc_columns = [column for column in df.columns if "nrc" in column]

for emotion in nrc_columns:
    df[emotion + "_per_word"] = df[emotion].divide(df["wordcount"])


# %% replacing lyrics related columns with missing if there are no lyrics

no_lyrics = df["lyrics"].str.len() == 0
instrumental = df["instrumental"] == 1
language_not_english = df["language"] != "en"

lyric_columns = [
    "lyrics",
    "lyrics_lemmatized",
    "wordlist",
    "wordcount",
    "unique_words",
    "perc_words_unique",
    "sentiment",
    "sentiment_positive",
    "sentiment_negative",
    "sentiment_abs",
    "sentiment_positive_per_word",
    "sentiment_negative_per_word",
    "sentiment_abs_per_word",
    "nrc_anger",
    "nrc_anticipation",
    "nrc_disgust",
    "nrc_fear",
    "nrc_joy",
    "nrc_negative",
    "nrc_positive",
    "nrc_sadness",
    "nrc_surprise",
    "nrc_trust",
    "nrc_anger_per_word",
    "nrc_anticipation_per_word",
    "nrc_disgust_per_word",
    "nrc_fear_per_word",
    "nrc_joy_per_word",
    "nrc_negative_per_word",
    "nrc_positive_per_word",
    "nrc_sadness_per_word",
    "nrc_surprise_per_word",
    "nrc_trust_per_word",
]

df[lyric_columns].mask(no_lyrics | instrumental | language_not_english, inplace=True)

# %% Wordcloud

lyrics_text = " ".join(df["lyrics_lemmatized"])
wordcloud = WordCloud(max_font_size=40).generate(lyrics_text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Wordcloud of lyrics")
plt.show()

# %% Plot of word frequencies

# sns.countplot(lyrics_text.str.split())
# plt.title("Words by frequency")
# plt.xlabel("Year")
# plt.ylabel("% of Words Unique")
# plt.show()

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


# %% Spotify audio features over time

sns.lineplot(x="year", y="duration_ms", data=df)
plt.ylim(bottom=0)
plt.title("Song Duration by Year")
plt.xlabel("Year")
plt.ylabel("Average Duration in Millisecods")
plt.show()

sns.lineplot(x="year", y="loudness", data=df)
plt.title("Song Decibels by Year")
plt.xlabel("Year")
plt.ylabel("Average Decibels by Year")
plt.show()

sns.lineplot(x="year", y="tempo", data=df)
plt.ylim(bottom=0)
plt.title("Song Tempo by Year")
plt.xlabel("Year")
plt.ylabel("Average Beats per Minute")
plt.show()

sns.lineplot(x="year", y="danceability", data=df, label="Danceability")
sns.lineplot(x="year", y="energy", data=df, label="Energy")
sns.lineplot(x="year", y="speechiness", data=df, label="Speechiness")
sns.lineplot(x="year", y="acousticness", data=df, label="Acousticness")
sns.lineplot(x="year", y="instrumentalness", data=df, label="Instrumentalness")
sns.lineplot(x="year", y="valence", data=df, label="Valence")
plt.ylim(bottom=0)
plt.title("Average Value of Spotify Audio Features by Year")
plt.xlabel("Year")
plt.ylabel("Sentiment")
plt.show()


# %% Afinn sentiment with time

sns.lineplot(x="year", y="sentiment", data=df)
plt.ylim(bottom=0)
plt.title("Average Sentiment by Year")
plt.xlabel("Year")
plt.ylabel("% of Words Unique")
plt.show()

sns.lineplot(x="year", y="sentiment_abs", data=df, label="Absolute Sentiment")
sns.lineplot(x="year", y="sentiment_positive", data=df, label="Positive Sentiment")
sns.lineplot(x="year", y="sentiment_negative", data=df, label="Negative Sentiment")
plt.ylim(bottom=0)
plt.title("Average Afinn Sentiment by Year and Type")
plt.xlabel("Year")
plt.ylabel("Sentiment")
plt.show()

sns.lineplot(x="year", y="sentiment_abs_per_word", data=df, label="Absolute Sentiment")
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
plt.title("Average NRC Sentiment per Word by Year and Type")
plt.xlabel("Year")
plt.ylabel("Sentiment")
plt.show()

sns.lineplot(x="year", y="nrc_positive", data=df, label="Positive")
sns.lineplot(x="year", y="nrc_negative", data=df, label="Negative")
plt.title("Average Overall NRC Sentiment by Year")
plt.xlabel("Year")
plt.ylabel("Sentiment")
plt.show()

sns.lineplot(x="year", y="nrc_positive_per_word", data=df, label="Positive")
sns.lineplot(x="year", y="nrc_negative_per_word", data=df, label="Negative")
plt.title("Average Overall NRC Sentiment per Word by Year")
plt.xlabel("Year")
plt.ylabel("Sentiment")
plt.show()

# %%
