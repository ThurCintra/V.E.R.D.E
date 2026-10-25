@echo off
rem V.E.R.D.E. - inicia o sistema no Windows. Uso (dentro da pasta do projeto, ou com duplo clique):
rem   iniciar          abre o dashboard
rem   iniciar api      sobe a API em http://localhost:8000/docs
rem   iniciar tudo     dashboard + API (a API abre em outra janela)
rem   iniciar testes   roda os testes automaticos
rem Na primeira vez cria o ambiente e instala as dependencias (alguns minutos).
setlocal EnableExtensions
cd /d "%~dp0"
set "VENV_PY=.venv\Scripts\python.exe"
if exist "%VENV_PY%" goto deps

echo [V.E.R.D.E.] Procurando o Python...
set "PY="
for %%P in ("%LOCALAPPDATA%\Programs\Python\Python312\python.exe" "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" "%ProgramFiles%\Python312\python.exe" "%ProgramFiles%\Python313\python.exe" "%ProgramFiles%\Python311\python.exe") do if not defined PY if exist "%%~P" set "PY=%%~P"
if not defined PY for /f "delims=" %%E in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do set "PY=%%E"
if not defined PY for /f "delims=" %%E in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "PY=%%E"
if defined PY if not exist "%PY%" set "PY="
if defined PY goto venv

echo [V.E.R.D.E.] Python nao encontrado. Instalando o Python 3.12 pelo winget...
winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined PY goto venv
echo [V.E.R.D.E.] Nao consegui instalar o Python. Instale o Python 3.12 em python.org e rode de novo.
goto falhou

:venv
echo [V.E.R.D.E.] Criando o ambiente com %PY%
"%PY%" -m venv .venv
if errorlevel 1 goto falhou

:deps
fc /b requirements.txt .venv\requirements.instalado >nul 2>&1
if not errorlevel 1 goto modelo
echo [V.E.R.D.E.] Instalando as dependencias. Na primeira vez leva alguns minutos: nao feche esta janela.
"%VENV_PY%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto falhou
copy /y requirements.txt .venv\requirements.instalado >nul

:modelo
if not exist models\yolov8n.pt "%VENV_PY%" scripts\download_model.py
if not exist models\yolov8n.pt goto falhou
if not exist data\esteira.mp4 "%VENV_PY%" scripts\prepare_data.py

if /i "%~1"=="api" goto api
if /i "%~1"=="testes" goto testes
if /i "%~1"=="tudo" start "V.E.R.D.E. - API" "%VENV_PY%" -m uvicorn src.api:app --port 8000

echo.
echo [V.E.R.D.E.] Dashboard em http://localhost:8501  -  para parar, feche esta janela ou aperte Ctrl+C
start "" /min cmd /c "timeout /t 6 /nobreak >nul && start http://localhost:8501"
"%VENV_PY%" -m streamlit run app\dashboard.py --server.headless true --browser.gatherUsageStats false
goto fim

:api
echo.
echo [V.E.R.D.E.] API em http://localhost:8000/docs  -  para parar, feche esta janela ou aperte Ctrl+C
"%VENV_PY%" -m uvicorn src.api:app --port 8000
goto fim

:testes
"%VENV_PY%" -m pytest
goto fim

:falhou
echo.
echo [V.E.R.D.E.] Algo deu errado. Copie as mensagens acima e envie para quem te ajuda.
pause
exit /b 1

:fim
endlocal
