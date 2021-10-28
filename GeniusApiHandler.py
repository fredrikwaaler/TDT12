import os
import lyricsgenius as lg
from settings import *
import re
import unicodedata
from utils import slugify


class GeniusApiHandler:

    def __init__(self, access_token, skip_non_songs=True, excluded_terms=("(Remix)", "(Live)", "(Demo)")):
        """
        Initialize a GeniusApiHandler that is used to scour Genius.com for song-lyrics.
        :param access_token: The Genius-API access token.
        :param skip_non_songs: True if you want to skip "non-songs" -
        this includes for instance track-lists and other non-song elements by the artists.
        :param excluded_terms: Exclude songs with these terms when searching
        """
        self.genius = lg.Genius(access_token, skip_non_songs=skip_non_songs, excluded_terms=excluded_terms)

    def _get_songs_from_artist(self, artist, max_songs=None, include_features=False):
        """
        Returns the songs found at genius, than was found when searching for the given artist (Only for top hit).
        :param artist: The artist to search for
        :param max_songs: The max number of songs to include. None if you want to include all.
        :param include_features: Whether or not to include songs where the artist is only featured.
        :return: List of songs
        """
        artist = self.genius.search_artist(artist, max_songs=max_songs, sort='popularity',
                                           include_features=include_features)
        return artist.songs

    @staticmethod
    def _get_lyric_blocks(lyrics):
        """
        Returns the lyrical blocks (verses, refrain, bridge, etc.) from the lyrics.
        :param lyrics: The lyrics to return blocks from
        :return: The lyrical blocks from the lyrics
        """
        blocks = []
        block = []
        lines = lyrics.splitlines()
        for line in lines:
            if line == "":
                blocks.append(block)
                block = []
            # To combat a bug where newlines sometimes is not added by genius between verses
            elif "[" in line and len(block) > 1:
                blocks.append(block)
                block = [line]
            else:
                block.append(line)
        return blocks

    @staticmethod
    def _remove_section_headers(lyrics):
        """
        Removes the section headers (i.e. [Verse] or [Bridge]) from the lyrics)
        :param lyrics: The lyrics
        :return: The lyrics without section headers
        """
        new_lyrics = ""
        for line in lyrics.splitlines():
            if "[" in line and "]" in line:
                pass
            else:
                new_lyrics += f"{line}\n"
        return new_lyrics

    @staticmethod
    def _remove_adlibs(lyrics):
        """
        Removes the adlibs (everything between parenthesis from the lyrics.
        :param lyrics: The lyrics
        :return: The lyrics without adlibs
        """
        new_lyrics = ""
        for line in lyrics.splitlines():
            line_without_adlibs = re.sub("[(].*?[)]", "", line)
            new_lyrics += f"{line_without_adlibs}\n"
        return new_lyrics

    def _remove_verses_not_by_artist(self, artist, lyrics):
        """
        Removes all the verses, bridges, chorus etc. from the lyrics where the given artist is not included.
        This also automatically removes the section headers from the lyrics.
        :param artist: The given artist
        :param lyrics: The lyrics
        :return: The new lyrics that only has lyrical blocks where the given artist participates
        """
        new_lyrics = ""
        lyric_blocks = self._get_lyric_blocks(lyrics)
        for block in lyric_blocks:
            if len(block) > 0:
                # If block header has a ":" it enlists the artists active in the verse
                if ":" in block[0] and artist.lower() not in block[0].lower():
                    pass
                else:
                    for line in block[1:]:
                        new_lyrics += f"{line}\n"
                    new_lyrics += "\n"
        return new_lyrics

    def write_lyrics_to_folder(self, artist, max_songs=None, include_features=False, artist_exclusively=True,
                               remove_adlibs=False, folder=None):
        """
        Writes all the lyrics from the given artists songs (capped at max_songs) to separate files in a folder.
        :param artist: The artist name
        :param max_songs: The max number of songs
        :param include_features: Whether or not to include songs where the artist is
        :param artist_exclusively: Whether or not the written lyrics should only contain the lyrics where the
        given artist appears in the lyrical block (i.e. verse or bridge)
        :param remove_adlibs: Whether or not to remove adlibs (anything between a parenthesis is considered an adlib)
        :param folder: The folder (relative to cwd) to store the lyrics. If none, the name of this is same as artist.
        """
        songs = self._get_songs_from_artist(artist, max_songs, include_features)

        if not folder:
            folder = artist.replace(" ", "")
        folder = os.path.join(os.getcwd(), folder)
        if not os.path.isdir(folder):
            os.mkdir(folder)

        for song in songs:
            lyrics = song.lyrics
            if artist_exclusively:
                lyrics = self._remove_verses_not_by_artist(artist, lyrics)
            else:
                lyrics = self._remove_section_headers(lyrics)
            if remove_adlibs:
                lyrics = self._remove_adlibs(lyrics)
            song_slug = slugify(song.title) + ".txt"
            song_file = os.path.join(folder, song_slug)
            with open(song_file, "w+", encoding='utf-8') as file:
                file.write(lyrics)


if __name__ == '__main__':
    handler = GeniusApiHandler(CLIENT_ACCESS_TOKEN)
    handler.write_lyrics_to_folder("Travis Scott", max_songs=150)


