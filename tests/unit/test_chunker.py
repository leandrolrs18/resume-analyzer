from app.rag.chunker import TextChunker


# Garante que o chunker respeita parágrafos e frases antes de quebrar o texto.
def test_chunker_prefers_paragraph_and_sentence_boundaries() -> None:
    text = (
        "Primeiro parágrafo com experiência em backend e APIs RESTful.\n\n"
        "Segundo parágrafo com pipelines de dados e dashboards analíticos. "
        "Também liderou integrações em sistemas distribuídos."
    )

    chunks = TextChunker(chunk_size=85, overlap=0).split("Ana", text)

    assert [chunk.text for chunk in chunks] == [
        "Primeiro parágrafo com experiência em backend e APIs RESTful.",
        "Segundo parágrafo com pipelines de dados e dashboards analíticos.",
        "Também liderou integrações em sistemas distribuídos.",
    ]


# Garante que o chunker não corta palavras longas ao dividir um texto maior.
def test_chunker_does_not_split_inside_words_when_sentence_is_long() -> None:
    text = " ".join(f"palavra{i}" for i in range(20))

    chunks = TextChunker(chunk_size=55, overlap=0).split("Ana", text)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 55 for chunk in chunks)
    assert all(" " in chunk.text for chunk in chunks)
    assert "avra" not in chunks[1].text[:5]
