#!/bin/bash
# Запуск интерфейса проекта на macOS: двойной щелчок по этому файлу.
# Первый запуск ставит библиотеки (1–3 минуты, нужен интернет), следующие — сразу.
cd "$(dirname "$0")" || exit 1

fail() {
  rm -rf .venv
  echo "$1"
  read -r -p "Нажмите Enter, чтобы закрыть окно"; exit 1
}

if [ ! -x .venv/bin/python ]; then
  PY=""
  for c in python3.12 python3.11 python3.13 "$HOME/.pyenv/shims/python3.12" python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
  [ -n "$PY" ] || fail "Python не найден. Установите Python 3.12 с https://www.python.org/downloads/ и запустите снова."

  echo "Создаю окружение ($PY) и ставлю библиотеки…"
  "$PY" -m venv .venv || fail "Не удалось создать окружение."
  # uv качает пакеты параллельно и не компилирует байткод — в разы быстрее pip.
  # Если uv поставить не вышло, ставим обычным pip.
  if ! { .venv/bin/python -m pip install -q --disable-pip-version-check uv \
         && .venv/bin/python -m uv pip install --python .venv/bin/python -r requirements.txt; }; then
    echo "Быстрая установка не удалась, ставлю обычным способом…"
    .venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt \
      || fail "Не удалось установить библиотеки. Проверьте интернет и запустите снова."
  fi
fi

echo "Открываю интерфейс в браузере. Чтобы остановить — закройте это окно."
.venv/bin/python -m streamlit run app.py
