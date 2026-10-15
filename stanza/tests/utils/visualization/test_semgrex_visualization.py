"""
Tests for the semgrex visualizer which don't need CoreNLP

The semgrex results are built by hand, so only spacy is needed for
the displacy HTML
"""

import re

import pytest

pytest.importorskip("spacy")
pytest.importorskip("IPython")

from stanza.models.common.doc import Document
from stanza.protobuf import SemgrexResponse
from stanza.utils.conll import CoNLL
from stanza.utils.visualization import semgrex_visualizer
from stanza.utils.visualization.semgrex_visualizer import get_sentences_html, semgrexify_html

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

WORDS = "The quick brown fox jumps over the lazy dog near the old barn".split()
EMPTY_WORD_SENTENCE = """
# text = Sue likes coffee and Bill tea.
1\tSue\tSue\tPROPN\tNNP\tNumber=Sing\t2\tnsubj\t2:nsubj\t_
2\tlikes\tlike\tVERB\tVBZ\tNumber=Sing|Person=3|Tense=Pres\t0\troot\t0:root\t_
3\tcoffee\tcoffee\tNOUN\tNN\tNumber=Sing\t2\tobj\t2:obj\t_
4\tand\tand\tCCONJ\tCC\t_\t5\tcc\t5.1:cc\t_
5\tBill\tBill\tPROPN\tNNP\tNumber=Sing\t2\tconj\t5.1:nsubj\t_
5.1\tlikes\tlike\tVERB\tVBZ\tNumber=Sing|Person=3|Tense=Pres\t_\t_\t2:conj:and\tCopyOf=2
6\ttea\ttea\tNOUN\tNN\tNumber=Sing\t5\torphan\t5.1:obj\tSpaceAfter=No
7\t.\t.\tPUNCT\t.\t_\t2\tpunct\t2:punct\t_
""".lstrip()

def build_doc():
    sentence = []
    for idx, word in enumerate(WORDS):
        sentence.append({
            "id": idx + 1,
            "text": word,
            "lemma": word.lower(),
            "upos": "VERB" if word == "jumps" else "NOUN",
            "head": 0 if word == "jumps" else 5,
            "deprel": "root" if word == "jumps" else "dep",
        })
    return Document([sentence], " ".join(WORDS))

def build_sentence_result(num_matches):
    """
    One pattern which matched each of the first num_matches words, as {}=word would
    """
    sentence_result = SemgrexResponse.SentenceResult()
    pattern_result = sentence_result.pattern.add()
    for idx in range(num_matches):
        match = pattern_result.match.add()
        match.matchIndex = idx + 1
        node = match.node.add()
        node.name = "word"
        node.matchIndex = idx + 1
    return sentence_result

def build_empty_word_doc():
    return CoNLL.conll2doc(input_str=EMPTY_WORD_SENTENCE, ignore_gapping=False)

def build_empty_word_semgrex_result():
    result = SemgrexResponse()
    sentence_result = result.sentence.add()
    pattern_result = sentence_result.pattern.add()
    match = pattern_result.match.add()
    match.sentenceIndex = 0
    match.matchIndex = 5
    match.matchEmptyIndex = 1
    node = match.node.add()
    node.name = "empty"
    node.matchIndex = 5
    node.emptyIndex = 1
    return result

def test_more_matches_than_colors():
    """
    There are seven colors; more matches than that reuse them in order
    """
    html = get_sentences_html(build_doc(), "en")[0]
    edited = semgrexify_html(html, build_sentence_result(len(WORDS)))

    # each match gets its word bolded and a label
    assert edited.count('class="bolded"') == len(WORDS)
    assert edited.count(">Wor.</tspan>") == len(WORDS)
    # the eighth match is colored the same as the first
    assert edited.count('fill="#4477AA"') == 2 * 2
    assert edited.count('fill="#66CCEE"') == 2 * 2

@pytest.mark.parametrize(
    "language,expected_words,expected_heads,empty_position,ordinary_word_position",
    [
        (
            "en",
            ["Sue", "likes", "coffee", "and", "Bill", "likes", "tea", "."],
            [1, 1, 1, 4, 1, 1, 4, 1],
            5,
            4,
        ),
        (
            "ar",
            [".", "tea", "likes", "Bill", "and", "coffee", "likes", "Sue"],
            [6, 3, 6, 6, 3, 6, 6, 6],
            2,
            3,
        ),
    ],
)
def test_empty_word_is_rendered_and_highlighted(language, expected_words, expected_heads,
                                                empty_position, ordinary_word_position, monkeypatch):
    """Empty-node IDs must map to their SVG token, including in RTL rendering."""
    captured = {}
    original_render = semgrex_visualizer.displacy.render

    def capture_render(spacy_doc, *args, **kwargs):
        captured["words"] = [token.text for token in spacy_doc]
        captured["heads"] = [token.head.i for token in spacy_doc]
        return original_render(spacy_doc, *args, **kwargs)

    class StubSemgrex:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def process(self, doc, *patterns):
            return build_empty_word_semgrex_result()

    monkeypatch.setattr(semgrex_visualizer.displacy, "render", capture_render)
    monkeypatch.setattr(semgrex_visualizer, "Semgrex", StubSemgrex)

    html = semgrex_visualizer.visualize_search_doc(
        build_empty_word_doc(), ["{}=empty"], language, render=False
    )[0]

    assert captured["words"] == expected_words
    assert captured["heads"] == expected_heads

    text_elements = re.findall(r'<text class="displacy-token"[^>]*>.*?</text>', html, flags=re.DOTALL)
    assert len(text_elements) > empty_position
    assert "likes" in text_elements[empty_position]
    assert 'class="bolded"' in text_elements[empty_position]
    assert 'class="bolded"' not in text_elements[ordinary_word_position]
