from typing import List

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet


class LemmaTokenizer:
    """
    Given a sentence, returns a list of lemmatized tokens.
    """

    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, title: str) -> List[str]:
        """
        When called, takes in a string and returns a list of lemmatized tokens.

        Parameter(s)
        ------------
        title: str
            The title of an article passed as a string.
        
        Returns
        -------
        lemmatized_tokens: List[str]
            A list of lemmatized tokens.
        """

        lemmatized_tokens = [self.lemmatizer.lemmatize(t, self.get_wordnet_pos(t)) for t in word_tokenize(title)]
        return lemmatized_tokens

    def get_wordnet_pos(self, word: str) -> wordnet:
        """
        Map part of speech tags to work with lemmatize().

        Parameter(s)
        ------------
        word: str
            A word to tag with its part of speech

        Returns
        -------
        mapped_pos: wordnet
            The mapped wordnet pos object
        """

        ## pos_tag expects a list so need to pass [word]
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        ## defaults to wordnet.NOUN if no tag is found
        mapped_pos = tag_dict.get(tag, wordnet.NOUN)
        return mapped_pos