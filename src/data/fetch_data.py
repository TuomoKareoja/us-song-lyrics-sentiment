# -*- coding: utf-8 -*-
import logging
import os
import re
import time
from contextlib import closing
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
import requests
import spotipy
from bs4 import BeautifulSoup
from dotenv import find_dotenv, load_dotenv
from requests import get
from requests.exceptions import RequestException
from spotipy.oauth2 import SpotifyClientCredentials


def main():
    """ Fetches songs and artist be webscraping and then
        uses this data to fetch the song lyrics from an API
    """

    rapidapi_key = os.getenv("RAPIDAPIKEY")
    geniuslyrics_key = os.getenv("GENIUSLYRICSKEY")
    spotify_client_id = os.getenv("SPOTIFYCLIENTID")
    spotify_secret_key = os.getenv("SPOTIFYSECRETKEY")

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

    songs_df["lyrics"] = "Not searched"
    songs_df["lyrics_source"] = None

    logger.info("Saving song and artist data to disk")

    songs_df.to_csv(
        os.path.join("data", "raw", "billboard100_1960-2015.csv"), index=False, sep=";"
    )

    logger.info("Fetching song lyrics")

    songs_df = pd.read_csv(
        os.path.join("data", "raw", "billboard100_1960-2015.csv"), sep=";"
    )

    songs_amount = len(songs_df)
    fetched_songs = 0

    for row_index, row in songs_df.iterrows():
        logger.info(f"Song {row_index + 1} / {songs_amount}")

        if row["lyrics"] == "Not searched" or row["lyrics"] == "Not found":

            # slowing down requests so that we cause no trouble
            time.sleep(0.5)

            lyric, source = get_lyric_from_apis(
                artist=row["artist"],
                song_title=row["song"],
                rapidapi_key=rapidapi_key,
                geniuslyrics_key=geniuslyrics_key,
            )
            songs_df.iloc[row_index, songs_df.columns.get_loc("lyrics")] = lyric
            songs_df.iloc[row_index, songs_df.columns.get_loc("lyrics_source")] = source

            fetched_songs += 1
            print(lyric)

            # saving every after every 100 fetched lyrics
            if fetched_songs > 0 and fetched_songs % 100 == 0:
                print("Saving progress")
                songs_df.to_csv(
                    os.path.join("data", "raw", "billboard100_1960-2015.csv"),
                    sep=";",
                    index=False,
                )

    songs_df.to_csv(
        os.path.join("data", "raw", "billboard100_1960-2015.csv"), sep=";", index=False
    )

    songs_df = pd.read_csv(
        os.path.join("data", "raw", "billboard100_1960-2015.csv"), sep=";"
    )

    logger.info("Fetching audio features from Spotify API")

    audio_features_df = get_spotify_audiofeatures(
        artists=songs_df["artist"],
        song_titles=songs_df["song"],
        spotify_client_id=spotify_client_id,
        spotify_secret_key=spotify_secret_key,
    )
    songs_df = pd.concat([songs_df, audio_features_df], axis="columns")

    logger.info("Saving final dataset to disk")

    songs_df.to_csv(
        os.path.join("data", "raw", "billboard100_1960-2015.csv"), sep=";", index=False
    )


def get_lyrics_mouritz(artist, song_title, use_spotify_api, rapidapi_key):
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
        "x-rapidapi-key": rapidapi_key,
    }

    if use_spotify_api:
        payload = {"q": artist + " " + song_title}
    else:
        payload = {"a": artist, "s": song_title}
    try:
        r = requests.get(url, params=payload, headers=headers)
        lyric = r.json()["result"]["lyrics"]

        return lyric, "mourits"

    except Exception:
        return None, None


def get_lyrics_chartlyrics(artist, song_title):

    url = f"http://api.chartlyrics.com/apiv1.asmx/SearchLyricDirect"

    payload = {"artist": artist, "song": song_title}

    try:
        r = requests.get(url, params=payload)
        tree = ElementTree.fromstring(r.text)
        lyric = tree.find("{http://api.chartlyrics.com/}Lyric").text

        return lyric, "chartlyrics"

    except Exception:
        return None, None


def get_lyrics_geniuslyrics(artist, song_title, geniuslyrics_key):

    base_url = "http://api.genius.com"
    search_url = base_url + "/search"

    headers = {f"Authorization": "Bearer {geniuslyrics_key}"}
    data = {"q": song_title}

    lyrics = None
    song_info = None

    try:

        response = requests.get(search_url, data=data, headers=headers)
        json = response.json()

        # find the right artist from all the results
        for hit in json["response"]["hits"]:
            if hit["result"]["primary_artist"]["name"] == artist:
                song_info = hit
                break

        # if song for the right artist found scrape the lyrics
        if song_info:
            song_api_path = song_info["result"]["api_path"]
            song_url = base_url + song_api_path
            response = requests.get(song_url, headers=headers)
            json = response.json()
            path = json["response"]["song"]["path"]
            page_url = "http://genius.com" + path
            page = requests.get(page_url)
            html = BeautifulSoup(page.text, "html.parser")
            # remove script tags in the lyrics
            [h.extract() for h in html("script")]
            lyrics = html.find("div", class_="lyrics").get_text()

            return lyrics, "genius"

        else:
            return None, None

    except Exception:
        return None, None


def get_lyric_from_apis(artist, song_title, rapidapi_key, geniuslyrics_key):

    lyric = None
    source = None

    lyric, source = get_lyrics_mouritz(
        artist, song_title, use_spotify_api=False, rapidapi_key=rapidapi_key
    )
    # if lyric is None or (1 > len(lyric.strip()) > 100000):
    # lyric, source = get_lyrics_chartlyrics(artist, song_title)
    # if lyric is None or (1 > len(lyric.strip()) > 100000):
    #     lyric, source = get_lyrics_geniuslyrics(
    #         artist, song_title, geniuslyrics_key=geniuslyrics_key
    #     )
    if lyric is None or (1 > len(lyric.strip()) > 100000):

        # wait because second time addressing mourits
        # and we don't want to go over the limit
        time.sleep(0.3)
        lyric, source = get_lyrics_mouritz(
            artist, song_title, use_spotify_api=True, rapidapi_key=rapidapi_key
        )

    if lyric is None or (1 > len(lyric.strip()) > 100000):
        lyric = "Not found"
        source = None

    else:
        # semicolons would mess up the separators in csv
        lyric = lyric.replace(";", " ")
        # line changes also mess up the csv
        lyric = lyric.replace("\n", " ")

    return lyric, source


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


def get_spotify_audiofeatures(
    artists, song_titles, spotify_client_id, spotify_secret_key
):

    audio_features_to_keep = [
        "duration_ms",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "valence",
        "tempo",
        "time_signature",
    ]

    # Authenticating to spotify client
    client_credentials_manager = SpotifyClientCredentials(
        client_id=spotify_client_id, client_secret=spotify_secret_key
    )
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    audio_features = []

    songs_amount = len(artists)

    for i, (artist, song_title) in enumerate(zip(artists, song_titles)):

        search_str = artist + " " + song_title

        # search for track information
        result = sp.search(q=search_str, limit=1, type="track")

        # if no results found create and empty dictionary
        if result["tracks"]["total"] == 0:
            song_audio_features = {
                audio_feat_key: None for audio_feat_key in audio_features_to_keep
            }

            print(f"Song {i + 1} / {songs_amount}: not found")

        # if results found search audio features with the ID
        else:
            # extracting ID
            song_id = result["tracks"]["items"][0]["id"]
            # fetching audio features
            result = sp.audio_features(tracks=[song_id])
            # keeping just the dict
            result = result[0]

            # trying to get audiofeatures from result
            # if fail then return an empty dict
            try:

                # dropping unnecessary features
                song_audio_features = {
                    audio_feat_key: result[audio_feat_key]
                    for audio_feat_key in audio_features_to_keep
                    if audio_feat_key in result
                }

                print(f"Song {i + 1} / {songs_amount}: {song_audio_features}")

            except Exception:

                song_audio_features = {
                    audio_feat_key: None for audio_feat_key in audio_features_to_keep
                }

                print(f"Song {i + 1} / {songs_amount}: could not be extracted")

        audio_features.append(song_audio_features)

    # create a dataframe with the list of dictionaries
    df = pd.DataFrame(data=audio_features)

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
