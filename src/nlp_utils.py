"""
Natural Language Processing Utilities for Chatbot.
Provides tokenization, Porter stemming, and Bag-of-Words vectorization
with zero external corpus download dependencies.
"""

import re
import numpy as np


class SimplePorterStemmer:
    """
    Lightweight, self-contained implementation of the Porter Stemmer algorithm.
    Ensures zero external dependency downloads (e.g., no nltk corpora required)
    for reliable deployment in restricted or containerized environments.
    """
    def __init__(self):
        self.b = ""
        self.k = 0
        self.k0 = 0
        self.j = 0

    def _cons(self, i):
        if self.b[i] in 'aeiou':
            return False
        if self.b[i] == 'y':
            if i == self.k0:
                return True
            else:
                return not self._cons(i - 1)
        return True

    def _m(self):
        n = 0
        i = self.k0
        while True:
            if i > self.j:
                return n
            if not self._cons(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > self.j:
                    return n
                if self._cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > self.j:
                    return n
                if not self._cons(i):
                    break
                i += 1
            i += 1

    def _vowelinstem(self):
        for i in range(self.k0, self.j + 1):
            if not self._cons(i):
                return True
        return False

    def _doublec(self, i):
        if i < self.k0 + 1:
            return False
        if self.b[i] != self.b[i - 1]:
            return False
        return self._cons(i)

    def _cvc(self, i):
        if i < self.k0 + 2 or not self._cons(i) or self._cons(i - 1) or not self._cons(i - 2):
            return False
        ch = self.b[i]
        if ch in ('w', 'x', 'y'):
            return False
        return True

    def _ends(self, s):
        length = len(s)
        if s[length - 1] != self.b[self.k]:
            return False
        if length > (self.k - self.k0 + 1):
            return False
        if self.b[self.k - length + 1:self.k + 1] != s:
            return False
        self.j = self.k - length
        return True

    def _setto(self, s):
        length = len(s)
        self.b = self.b[:self.j + 1] + s + self.b[self.j + length + 1:]
        self.k = self.j + length

    def _r(self, s):
        if self._m() > 0:
            self._setto(s)

    def _step1ab(self):
        if self.b[self.k] == 's':
            if self._ends("sses"):
                self.k -= 2
            elif self._ends("ies"):
                self._setto("i")
            elif self.b[self.k - 1] != 's':
                self.k -= 1
        if self._ends("eed"):
            if self._m() > 0:
                self.k -= 1
        elif (self._ends("ed") or self._ends("ing")) and self._vowelinstem():
            self.k = self.j
            if self._ends("at"):
                self._setto("ate")
            elif self._ends("bl"):
                self._setto("ble")
            elif self._ends("iz"):
                self._setto("ize")
            elif self._doublec(self.k):
                self.k -= 1
                ch = self.b[self.k]
                if ch in ('l', 's', 'z'):
                    self.k += 1
            elif self._m() == 1 and self._cvc(self.k):
                self._setto("e")

    def _step1c(self):
        if self._ends("y") and self._vowelinstem():
            self.b = self.b[:self.k] + 'i' + self.b[self.k + 1:]

    def _step2(self):
        mapping = {
            "ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance",
            "izer": "ize", "bli": "ble", "alli": "al", "entli": "ent", "eli": "e",
            "ousli": "ous", "ization": "ize", "ation": "ate", "ator": "ate",
            "alism": "al", "iveness": "ive", "fulness": "ful", "ousness": "ous",
            "aliti": "al", "iviti": "ive", "biliti": "ble"
        }
        for suffix, replacement in mapping.items():
            if self._ends(suffix):
                self._r(replacement)
                break

    def _step3(self):
        mapping = {
            "icate": "ic", "ative": "", "alize": "al",
            "iciti": "ic", "ical": "ic", "ful": "", "ness": ""
        }
        for suffix, replacement in mapping.items():
            if self._ends(suffix):
                self._r(replacement)
                break

    def _step4(self):
        suffixes = (
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
            "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize"
        )
        for suffix in suffixes:
            if self._ends(suffix):
                if self._m() > 1:
                    self.k = self.j
                break
        if self._ends("ion") and self.j >= 0 and self.b[self.j] in ('s', 't'):
            if self._m() > 1:
                self.k = self.j

    def _step5(self):
        self.j = self.k
        if self.b[self.k] == 'e':
            a = self._m()
            if a > 1 or (a == 1 and not self._cvc(self.k - 1)):
                self.k -= 1
        if self.b[self.k] == 'l' and self._doublec(self.k) and self._m() > 1:
            self.k -= 1

    def stem(self, word):
        """Stem a single word to its root form."""
        word = word.lower()
        if len(word) <= 2:
            return word
        self.b = word
        self.k = len(word) - 1
        self.k0 = 0
        self._step1ab()
        self._step1c()
        self._step2()
        self._step3()
        self._step4()
        self._step5()
        return self.b[:self.k + 1]


# Singleton stemmer instance
_stemmer = SimplePorterStemmer()


def tokenize(text):
    """
    Splits text into words/tokens, expanding common English contractions.
    """
    if not text:
        return []
    text = text.lower()
    # Expand contractions
    contractions = {
        "what's": "what is",
        "how's": "how is",
        "who's": "who is",
        "it's": "it is",
        "i'm": "i am",
        "can't": "cannot",
        "don't": "do not",
        "you're": "you are",
        "there's": "there is",
        "let's": "let us",
        "that's": "that is"
    }
    for contr, expanded in contractions.items():
        text = text.replace(contr, expanded)

    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text)
    return tokens


def stem(word):
    """Returns the stemmed version of the input word."""
    return _stemmer.stem(word)


def bag_of_words(tokenized_sentence, words):
    """
    Returns a binary Bag-of-Words numpy array indicating presence of words.
    """
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag
