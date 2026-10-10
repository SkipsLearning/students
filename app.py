"""Веб-интерфейс проекта: результаты простыми словами, оценка клиента, словарь, раздел для специалистов.

Запуск:  streamlit run app.py   (или двойной щелчок по start.command / start.bat)
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui.nav import PAGES  # noqa: E402

st.set_page_config(page_title="Скоринг: единая модель", page_icon=":material/insights:", layout="wide")
st.navigation(list(PAGES.values()), position="top").run()
