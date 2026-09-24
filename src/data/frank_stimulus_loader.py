"""Frank 数据集刺激文本加载器。

从 test.source 文件加载 8 篇文章文本，使用正则分词匹配 EEG epoch。
Frank 数据集将标点符号作为独立 token，因此用 \\w+|[^\\w\\s] 分词。
"""

import re
from pathlib import Path
from typing import Dict, List


class FrankStimulusLoader:
    """加载 Frank 数据集的刺激文本。

    数据来源: test.source 文件，每行一篇文章。
    分词方式: 正则 \\w+|[^\\w\\s]，将标点视为独立 token。
    """

    def __init__(
        self,
        data_dir: str,
        article_ids: List[int] = None,
    ):
        self.data_dir = Path(data_dir)
        if article_ids is None:
            article_ids = list(range(1, 9))
        self.article_ids = article_ids

        source_path = self.data_dir / "test.source"
        with open(source_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        self.articles: Dict[int, List[str]] = {}
        for aid in article_ids:
            line_idx = aid - 1
            if line_idx < len(lines):
                tokens = re.findall(r"\w+|[^\w\s]", lines[line_idx].strip())
                self.articles[aid] = tokens

    def get_context_for_word(self, article_id: int, word_position: int) -> str:
        words = self.articles[article_id]
        if word_position < 0 or word_position >= len(words):
            raise IndexError(
                f"word_position={word_position} out of range [0, {len(words) - 1}], "
                f"article_{article_id}"
            )
        return " ".join(words[: word_position + 1])

    def get_target_word(self, article_id: int, word_position: int) -> str:
        return self.articles[article_id][word_position]

    def get_article_length(self, article_id: int) -> int:
        return len(self.articles[article_id])

    def get_article_words(self, article_id: int) -> List[str]:
        return self.articles[article_id]

    def __repr__(self) -> str:
        lengths = {aid: len(words) for aid, words in self.articles.items()}
        return f"FrankStimulusLoader(articles={lengths})"
