from app.schemas import ResumeChunk


class TextChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, candidate: str, text: str) -> list[ResumeChunk]:
        normalized = " ".join(text.split())
        if not normalized:
            return []
        chunks: list[ResumeChunk] = []
        start = 0
        index = 0
        while start < len(normalized):
            end = min(start + self.chunk_size, len(normalized))
            chunk_text = normalized[start:end].strip()
            if chunk_text:
                chunks.append(
                    ResumeChunk(
                        chunk_id=f"{candidate}-{index}",
                        candidate=candidate,
                        text=chunk_text,
                    )
                )
                index += 1
            if end == len(normalized):
                break
            start = max(0, end - self.overlap)
        return chunks
