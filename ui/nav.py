"""Список страниц интерфейса — их порядок в верхнем меню."""
from __future__ import annotations

import streamlit as st

from . import client, expert, glossary, home, results

PAGES = {
    "home": st.Page(home.render, title="Главная", icon=":material/home:", url_path="home", default=True),
    "results": st.Page(results.render, title="Результаты", icon=":material/insights:", url_path="results"),
    "client": st.Page(client.render, title="Оценить клиента", icon=":material/person_search:", url_path="client"),
    "glossary": st.Page(glossary.render, title="Словарь", icon=":material/menu_book:", url_path="glossary"),
    "expert": st.Page(expert.render, title="Для специалистов", icon=":material/science:", url_path="expert"),
}
