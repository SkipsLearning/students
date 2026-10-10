@echo off
rem Запуск интерфейса проекта на Windows: двойной щелчок по этому файлу.
rem Первый запуск ставит библиотеки (1-3 минуты, нужен интернет), следующие - сразу.
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Создаю окружение и ставлю библиотеки...
  py -3.12 -m venv .venv 2>nul || python -m venv .venv
  rem uv качает пакеты параллельно и не компилирует байткод - в разы быстрее pip.
  rem Если uv поставить не вышло, ставим обычным pip.
  .venv\Scripts\python -m pip install -q --disable-pip-version-check uv && .venv\Scripts\python -m uv pip install --python .venv\Scripts\python.exe -r requirements.txt || (
    echo Быстрая установка не удалась, ставлю обычным способом...
    .venv\Scripts\python -m pip install --disable-pip-version-check -r requirements.txt
  ) || (
    rmdir /s /q .venv
    echo Не удалось установить библиотеки. Проверьте интернет и Python 3.12.
    pause
    exit /b 1
  )
)
echo Открываю интерфейс в браузере. Чтобы остановить - закройте это окно.
.venv\Scripts\python -m streamlit run app.py
