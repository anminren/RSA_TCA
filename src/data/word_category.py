"""基于 spaCy 的词汇类别识别。

将每个词分类为：
- "ner": 命名实体词（如人名、地名、组织名等）
- "noun": 名词
- "verb": 动词
- "other": 其他词类

用于为不同词汇类别选择不同的 EEG 时间窗口。
"""

import logging
from typing import Dict, List

logger = logging.getLogger("eeg_bart")

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except ImportError:
            raise ImportError(
                "spacy is required for word category tagging. "
                "Install with: pip install spacy"
            )
        except OSError:
            raise OSError(
                "spaCy model 'en_core_web_sm' not found. "
                "Install with: python -m spacy download en_core_web_sm"
            )
    return _nlp


def categorize_words(words: List[str]) -> List[str]:
    """对词列表进行 NER 和 POS 分类。

    Args:
        words: 按阅读顺序排列的词列表。

    Returns:
        与 words 等长的类别列表，每个元素为 "ner" / "noun" / "verb" / "other"。
    """
    nlp = _get_nlp()
    text = " ".join(words)
    doc = nlp(text)

    ner_char_spans = set()
    for ent in doc.ents:
        for i in range(ent.start_char, ent.end_char):
            ner_char_spans.add(i)

    categories = []
    char_offset = 0

    for word in words:
        word_start = text.find(word, char_offset)
        if word_start == -1:
            categories.append("other")
            char_offset += len(word) + 1
            continue

        word_end = word_start + len(word)

        is_ner = any(i in ner_char_spans for i in range(word_start, word_end))
        if is_ner:
            categories.append("ner")
        else:
            token_found = False
            for token in doc:
                if token.idx >= word_start and token.idx < word_end:
                    if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                        categories.append("noun")
                        token_found = True
                        break
                    elif token.pos_ == "VERB" or token.pos_ == "AUX":
                        categories.append("verb")
                        token_found = True
                        break
            if not token_found:
                categories.append("other")

        char_offset = word_end

    return categories


class WordCategoryCache:
    """缓存每篇文章的词汇类别结果。"""

    def __init__(self):
        self._cache: Dict[str, List[str]] = {}

    def get_categories(self, article_key: str, words: List[str]) -> List[str]:
        if article_key not in self._cache:
            self._cache[article_key] = categorize_words(words)
        return self._cache[article_key]

    def get_word_category(
        self, article_key: str, words: List[str], word_position: int
    ) -> str:
        categories = self.get_categories(article_key, words)
        if word_position < len(categories):
            return categories[word_position]
        return "other"
