# -*- coding: utf-8 -*-
import logging
import os
import re
import time
from contextlib import closing
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import find_dotenv, load_dotenv
from requests import get
from requests.exceptions import RequestException


def main():
    """ Fetches songs and artist be webscraping and then
        uses this data to fetch the song lyrics from an API
    """

    apikey = os.getenv("RAPIDAPIKEY")

    logger = logging.getLogger(__name__)

    logger.info("Fetching songs and artist")

    years = [year for year in range(1960, 2016)]
    urls = create_urls(years)
    songs_df = fetch_and_parse(urls, years)

    logger.info("Adding 2013 info from data/raw/ (nasty format in the website)")

    # have to use ; for separator as song names contain commas
    songs_df_2013 = pd.read_csv(
        os.path.join("data", "raw", "2013_top_100.csv"), sep=";"
    )
    songs_df = pd.concat([songs_df, songs_df_2013], ignore_index=True)

    logger.info("Saving song and artist data to disk")

    songs_df.to_csv(
        os.path.join("data", "raw", "billboard100_1960-2015.csv"), index=False, sep=";"
    )

    logger.info("Fetching song lyrics")

    # songs_df = pd.read_csv(
    #     os.path.join("data", "raw", "billboard100_1960-2015.csv"), sep=";"
    # )

    lyrics = []

    songs_amount = len(songs_df)

    for row_index, row in songs_df.iterrows():
        logger.info(f"Song {row_index + 1} / {songs_amount}")
        lyric = get_lyrics(row["artist"], row["song"], apikey, use_spotify_api=False)
        # making sure that we don't go over the API limit
        time.sleep(0.51)
        if not lyric:
            # spotify api is slower, but finds results easier
            lyric = get_lyrics(row["artist"], row["song"], apikey, use_spotify_api=True)
            time.sleep(0.51)

        lyrics.append(lyric)

    songs_df["lyrics"] = lyrics

    logger.info("Saving to disk")

    songs_df.to_csv(
        os.path.join("data", "raw", "billboard100_1960-2015_with_lyrics.csv"),
        index=False,
    )


def get_lyrics(artist, song_title, apikey, use_spotify_api):
    """Fetches song lyrics for provided artist and song from Mouritz Lyrics API

    :param artist: Name of the artist
    :type artist: str
    :param song_title: Name of the song
    :type song_title: str
    :param apikey: API key for Mouritz Lyrics API
    :type apikey: str
    :param use_spotify_api: Should Spotify API be used. Slower, but more reliable
    :type use_spotify_api: boolean
    :return: Lyrics of the song
    :rtype: str
    """

    # "featuring" makes the string messy and Spotify API can find the song
    # without this info
    artist = artist.lower().split("feat", 1)[0].strip()
    song_title = song_title.lower().strip()

    url = "https://mourits-lyrics.p.rapidapi.com"

    headers = {
        "x-rapidapi-host": "mourits-lyrics.p.rapidapi.com",
        "x-rapidapi-key": apikey,
    }

    if use_spotify_api:
        payload = {"q": artist + " " + song_title}
    else:
        payload = {"a": artist, "s": song_title}

    try:
        r = requests.get(url, params=payload, headers=headers)
        lyric = r.json()["result"]["lyrics"]
        # removing line changes with spaces
        lyric = lyric.replace("\n", " ")
        print(lyric)
        return lyric

    except Exception as e:
        return "Exception occurred \n" + str(e)


def simple_get(url):

    """Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.

    :return: Content of the url
    :rtype: str
    """

    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error("Error during requests to {0} : {1}".format(url, str(e)))
        return None


def is_good_response(resp):
    """Returns True if the response seems to be HTML, False otherwise.

    :param resp: Response code
    :type resp: int
    :return: is the response status OK
    :rtype: boolean
    """
    content_type = resp.headers["Content-Type"].lower()
    return (
        resp.status_code == 200
        and content_type is not None
        and content_type.find("html") > -1
    )


def log_error(e):
    """It is always a good idea to log errors.
    This function just prints them, but you can
    make it do anything.

    :param e: Error message
    :type e: str
    """
    print(e)


def create_urls(years):
    """Generates urls that lead to billboard top 100 chart for provided year.

    :param years: List of years
    :type years: list
    :return: List of urls for provided years
    :rtype: list
    """
    urls = []
    for year in years:
        url = f"http://billboardtop100of.com/{year}-2/"
        urls.append(url)
    return urls


def fetch_and_parse(urls, years):
    """Fetches raw HTML from provided urls and parses
    it for artists, songs and positions. Years need to
    be provided to add this information.

    :param urls: list of urls to parse
    :type urls: list
    :param years: list of years which were used to create the urls
    :type years: list
    :return: dataframe with song information
    :rtype: dataframe
    """
    artists = []
    songs = []
    chart_years = []
    positions = []

    linebreak_and_lyrics_cleaner = re.compile(r"[\n]|LYRICS")

    for url, year in zip(urls, years):
        raw_html = simple_get(url)
        soup = BeautifulSoup(raw_html, "html.parser")
        table = soup.find("table")
        try:
            for tr in table.find_all("tr"):
                tds = tr.find_all("td")
                chart_years.append(year)
                positions.append(linebreak_and_lyrics_cleaner.sub("", tds[0].text))
                artists.append(linebreak_and_lyrics_cleaner.sub("", tds[1].text))
                songs.append(linebreak_and_lyrics_cleaner.sub("", tds[2].text))
        except AttributeError:
            continue

    df = pd.DataFrame(
        {"position": positions, "artist": artists, "song": songs, "year": chart_years}
    )

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
