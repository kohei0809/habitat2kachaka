#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from collections import Counter
from typing import List

from habitat.core.logging import logger
from habitat.core.simulator import ShortestPathPoint
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.geometry_utils import quaternion_to_list

SENTENCE_SPLIT_REGEX = re.compile(r"([^\w-]+)")

"""
 Tokenize and vocabulary utils originally authored by @apsdehal and are
 taken from Pythia.
"""


def tokenize(
    sentence, regex=SENTENCE_SPLIT_REGEX, keep=["'s"], remove=[",", "?"]
):
    sentence = sentence.lower()

    for token in keep:
        sentence = sentence.replace(token, " " + token)

    for token in remove:
        sentence = sentence.replace(token, "")

    tokens = regex.split(sentence)
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens


def load_str_list(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines


class VocabDict:
    UNK_TOKEN = "<unk>"
    PAD_TOKEN = "<pad>"
    START_TOKEN = "<s>"
    END_TOKEN = "</s>"

    def __init__(self, word_list=None, filepath=None):
        if word_list is not None:
            self.word_list = word_list
            self._build()

        elif filepath:
            self.word_list = load_str_list(filepath)
            self._build()

    def _build(self):
        if self.UNK_TOKEN not in self.word_list:
            self.word_list = [self.UNK_TOKEN] + self.word_list

        self.word2idx_dict = {w: n_w for n_w, w in enumerate(self.word_list)}

        # String (word) to integer (index) dict mapping
        self.stoi = self.word2idx_dict
        # Integer to string (word) reverse mapping
        self.itos = self.word_list
        self.num_vocab = len(self.word_list)

        self.UNK_INDEX = (
            self.word2idx_dict[self.UNK_TOKEN]
            if self.UNK_TOKEN in self.word2idx_dict
            else None
        )

        self.PAD_INDEX = (
            self.word2idx_dict[self.PAD_TOKEN]
            if self.PAD_TOKEN in self.word2idx_dict
            else None
        )

    def idx2word(self, n_w):
        return self.word_list[n_w]

    def __len__(self):
        return len(self.word_list)

    def get_size(self):
        return len(self.word_list)

    def get_unk_index(self):
        return self.UNK_INDEX

    def get_unk_token(self):
        return self.UNK_TOKEN

    def word2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.UNK_INDEX is not None:
            return self.UNK_INDEX
        else:
            raise ValueError(
                "word %s not in dictionary \
                             (while dictionary does not contain <unk>)"
                % w
            )

    def tokenize_and_index(
        self,
        sentence,
        regex=SENTENCE_SPLIT_REGEX,
        keep=["'s"],
        remove=[",", "?"],
    ) -> List[int]:
        inds = [
            self.word2idx(w)
            for w in tokenize(sentence, regex=regex, keep=keep, remove=remove)
        ]
        return inds


class VocabFromText(VocabDict):
    DEFAULT_TOKENS = [
        VocabDict.PAD_TOKEN,
        VocabDict.UNK_TOKEN,
        VocabDict.START_TOKEN,
        VocabDict.END_TOKEN,
    ]

    def __init__(
        self,
        sentences,
        min_count=1,
        regex=SENTENCE_SPLIT_REGEX,
        keep=[],
        remove=[],
        only_unk_extra=False,
    ):
        token_counter = Counter()

        for sentence in sentences:
            tokens = tokenize(sentence, regex=regex, keep=keep, remove=remove)
            token_counter.update(tokens)

        token_list = []
        for token in token_counter:
            if token_counter[token] >= min_count:
                token_list.append(token)

        extras = self.DEFAULT_TOKENS

        if only_unk_extra:
            extras = [self.UNK_TOKEN]

        self.word_list = extras + token_list
        self._build()
