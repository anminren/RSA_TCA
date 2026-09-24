"""阅读任务刺激文本加载器（Derco 数据集）。

加载 article_X.pkl 文件，为每个目标词构建因果上下文。
"""

import pickle
from pathlib import Path
from typing import Dict, List


class _DataOnlyUnpickler(pickle.Unpickler):
    """Load primitive pickle data without permitting code execution.

    DERCo's published ``article_*.pkl`` files are plain lists of strings.  A
    normal ``pickle.load`` can import and execute arbitrary globals, which is
    unnecessary for this data format and unsafe for files from unknown sources.
    """

    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(
            f"Refusing non-data pickle global: {module}.{name}"
        )


def _load_word_list(path: Path) -> List[str]:
    with path.open("rb") as handle:
        words = _DataOnlyUnpickler(handle).load()
    if not isinstance(words, list) or not all(isinstance(word, str) for word in words):
        raise ValueError(f"Expected a list of strings in {path}")
    return words


class StimulusLoader:
    """加载阅读任务的刺激文本，提供因果上下文查询。

    数据结构：每篇文章一个 .pkl 文件，包含词列表。
    因果上下文 = 从文章第一个词到目标词（含）的拼接。
    """

    def __init__(
        self,
        data_dir: str,
        article_ids: List[int] = None,
    ):
        """初始化 StimulusLoader。

        Args:
            data_dir: EEG-Data 根目录，包含 article_X.pkl 文件。
            article_ids: 要加载的文章编号列表，默认 [0,1,2,3,4]。
        """
        self.data_dir = Path(data_dir)
        if article_ids is None:
            article_ids = [0, 1, 2, 3, 4]
        self.article_ids = article_ids

        self.articles: Dict[int, List[str]] = {}
        for aid in article_ids:
            pkl_path = self.data_dir / f"article_{aid}.pkl"
            self.articles[aid] = _load_word_list(pkl_path)

    def get_context_for_word(self, article_id: int, word_position: int) -> str:
        """获取目标词的因果上下文。

        Args:
            article_id: 文章编号。
            word_position: 词在文章中的位置（0-indexed）。

        Returns:
            从第 0 个词到第 word_position 个词（含）的空格拼接文本。
        """
        words = self.articles[article_id]
        if word_position < 0 or word_position >= len(words):
            raise IndexError(
                f"word_position={word_position} 超出范围 [0, {len(words) - 1}], "
                f"article_{article_id}"
            )
        return " ".join(words[: word_position + 1])

    def get_target_word(self, article_id: int, word_position: int) -> str:
        """获取目标词文本。"""
        words = self.articles[article_id]
        return words[word_position]

    def get_article_length(self, article_id: int) -> int:
        return len(self.articles[article_id])

    def get_article_words(self, article_id: int) -> List[str]:
        return self.articles[article_id]

    def __repr__(self) -> str:
        lengths = {aid: len(words) for aid, words in self.articles.items()}
        return f"StimulusLoader(articles={lengths})"
