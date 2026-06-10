from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.schemas import ResumeChunk


class TextChunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator="end",
        )

    def split(self, candidate: str, text: str) -> list[ResumeChunk]:
        return [
            ResumeChunk(chunk_id=f"{candidate}-{index}", candidate=candidate, text=chunk)
            for index, chunk in enumerate(self.splitter.split_text(text))
        ]
