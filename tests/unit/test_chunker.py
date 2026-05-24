from app.rag.chunker import TextChunker


def test_chunker_respects_overlap() -> None:
    chunker = TextChunker(chunk_size=20, overlap=5)
    chunks = chunker.split("Maria", "abcdefghijklmnopqrstuvwxyz")
    assert len(chunks) == 2
    assert chunks[0].text == "abcdefghijklmnopqrst"
    assert chunks[1].text.startswith("pqrst")
