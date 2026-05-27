import re

from app.schemas import ResumeChunk

SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


class TextChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, candidate: str, text: str) -> list[ResumeChunk]:
        segments = self._segments(text)
        if not segments:
            return []
        chunks: list[ResumeChunk] = []
        current: list[str] = []
        index = 0

        for segment in segments:
            candidate_text = self._join(current + [segment])
            if current and len(candidate_text) > self.chunk_size:
                chunks.append(self._chunk(candidate, index, self._join(current)))
                index += 1
                current = self._overlap_segments(current)
                if len(self._join(current + [segment])) > self.chunk_size:
                    current = []
            current.append(segment)

        if current:
            chunks.append(self._chunk(candidate, index, self._join(current)))
        return chunks

    def _segments(self, text: str) -> list[str]:
        blocks = [
            " ".join(block.split())
            for block in re.split(r"(?:\r?\n\s*){2,}", text.replace("\u200b", " "))
            if block.strip()
        ]
        segments: list[str] = []
        for block in blocks:
            if len(block) <= self.chunk_size:
                segments.append(block)
                continue
            for sentence in SENTENCE_BOUNDARY_RE.split(block):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(sentence) <= self.chunk_size:
                    segments.append(sentence)
                else:
                    segments.extend(self._word_segments(sentence))
        return segments

    def _word_segments(self, text: str) -> list[str]:
        segments: list[str] = []
        current: list[str] = []
        for word in text.split():
            candidate = self._join(current + [word])
            if current and len(candidate) > self.chunk_size:
                segments.append(self._join(current))
                current = []
            current.append(word)
        if current:
            segments.append(self._join(current))
        return segments

    def _overlap_segments(self, segments: list[str]) -> list[str]:
        if self.overlap <= 0:
            return []
        tail: list[str] = []
        length = 0
        for segment in reversed(segments):
            length += len(segment) + (1 if tail else 0)
            if length > self.overlap and tail:
                break
            if length > self.chunk_size // 2:
                break
            tail.insert(0, segment)
        return tail

    @staticmethod
    def _join(values: list[str]) -> str:
        return " ".join(value.strip() for value in values if value.strip()).strip()

    @staticmethod
    def _chunk(candidate: str, index: int, text: str) -> ResumeChunk:
        return ResumeChunk(
            chunk_id=f"{candidate}-{index}",
            candidate=candidate,
            text=text,
        )
