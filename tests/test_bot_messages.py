from tg_rag_bot.bot import indexed_message


def test_fragments_plural() -> None:
    assert indexed_message(1, "a.md", "x").startswith("Добавил 1 фрагмент из")
    assert indexed_message(2, "a.md", "x").startswith("Добавил 2 фрагмента из")
    assert indexed_message(5, "a.md", "x").startswith("Добавил 5 фрагментов из")
    assert indexed_message(11, "a.md", "x").startswith("Добавил 11 фрагментов из")
    assert indexed_message(22, "a.md", "x").startswith("Добавил 22 фрагмента из")


def test_file_name_escaped_for_html() -> None:
    # Сообщение уходит с parse_mode=HTML: голый «&» или «<» Telegram отвергает.
    text = indexed_message(3, "R&D <черновик>.pdf", "abc")
    assert "R&amp;D &lt;черновик&gt;.pdf" in text
    assert text.endswith("<code>abc</code>")
