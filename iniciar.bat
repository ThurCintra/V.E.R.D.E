@echo off
rem V.E.R.D.E. - inicia o sistema no Windows. Uso (dentro da pasta do projeto, ou com duplo clique):
rem   iniciar          abre o dashboard
rem   iniciar api      sobe a API em http://localhost:8000/docs
rem   iniciar tudo     dashboard + API (a API abre em outra janela)
rem   iniciar testes   roda os testes automaticos
rem   iniciar extras   instala os opcionais (Hugging Face, Gemini, YOLO-World)
rem   iniciar verificar  confere (e baixa) os modelos do motor de precisao
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
call :instalar requirements.txt
if errorlevel 1 goto falhou
copy /y requirements.txt .venv\requirements.instalado >nul

:modelo
if not exist models\yolov8s.pt "%VENV_PY%" scripts\download_model.py
if not exist models\yolov8s.pt goto falhou
if not exist data\esteira.mp4 "%VENV_PY%" scripts\prepare_data.py

if /i "%~1"=="api" goto api
if /i "%~1"=="testes" goto testes
if /i "%~1"=="extras" goto extras
if /i "%~1"=="verificar" goto verificar
if /i "%~1"=="tudo" call :instalar
rem Instala %1 com o uv (bem mais rapido que o pip); se o uv nao der certo, usa o pip.
if not exist ".venv\Scripts\uv.exe" "%VENV_PY%" -m pip install --disable-pip-version-check -q uv
if exist ".venv\Scripts\uv.exe" ".venv\Scripts\uv.exe" pip install --python "%VENV_PY%" -r %1
if exist ".venv\Scripts\uv.exe" if not errorlevel 1 exit /b 0
"%VENV_PY%" -m pip install --disable-pip-version-check -r %1
exit /b %errorlevel%

:liberar_porta 8000 uvicorn
if /i "%~1"=="tudo" start "V.E.R.D.E. - API" "%VENV_PY%" -m uvicorn src.api:app --port 8000
call :instalar
rem Instala %1 com o uv (bem mais rapido que o pip); se o uv nao der certo, usa o pip.
if not exist ".venv\Scripts\uv.exe" "%VENV_PY%" -m pip install --disable-pip-version-check -q uv
if exist ".venv\Scripts\uv.exe" ".venv\Scripts\uv.exe" pip install --python "%VENV_PY%" -r %1
if exist ".venv\Scripts\uv.exe" if not errorlevel 1 exit /b 0
"%VENV_PY%" -m pip install --disable-pip-version-check -r %1
exit /b %errorlevel%

:liberar_porta 8501 streamlit

echo.
echo [V.E.R.D.E.] Dashboard em http://localhost:8501  -  para parar, feche esta janela ou aperte Ctrl+C
start "" /min "%VENV_PY%" scripts\abrir_navegador.py http://localhost:8501
"%VENV_PY%" -m streamlit run app\dashboard.py --server.port 8501 --server.headless true --browser.gatherUsageStats false --server.fileWatcherType none
goto fim

:api
echo.
call :instalar
rem Instala %1 com o uv (bem mais rapido que o pip); se o uv nao der certo, usa o pip.
if not exist ".venv\Scripts\uv.exe" "%VENV_PY%" -m pip install --disable-pip-version-check -q uv
if exist ".venv\Scripts\uv.exe" ".venv\Scripts\uv.exe" pip install --python "%VENV_PY%" -r %1
if exist ".venv\Scripts\uv.exe" if not errorlevel 1 exit /b 0
"%VENV_PY%" -m pip install --disable-pip-version-check -r %1
exit /b %errorlevel%

:liberar_porta 8000 uvicorn
echo [V.E.R.D.E.] API em http://localhost:8000/docs  -  para parar, feche esta janela ou aperte Ctrl+C
"%VENV_PY%" -m uvicorn src.api:app --port 8000
goto fim

:verificar
"%VENV_PY%" scripts\verificar_modelos.py
pause
goto fim

:testes
fc /b requirements-dev.txt .venv\dev.instalado >nul 2>&1
if errorlevel 1 call :instalar requirements-dev.txt
if errorlevel 1 goto falhou
copy /y requirements-dev.txt .venv\dev.instalado >nul
"%VENV_PY%" -m pytest
goto fim

:extras
echo [V.E.R.D.E.] Instalando os extras (Hugging Face, Gemini, YOLO-World)...
call :instalar requirements-extras.txt
if errorlevel 1 goto falhou
echo.
echo [V.E.R.D.E.] Extras instalados. Configure o arquivo .env (veja o .env.example) e rode: iniciar verificar
pause
goto fim

:falhou
echo.
echo [V.E.R.D.E.] Algo deu errado. Copie as mensagens acima e envie para quem te ajuda.
pause
exit /b 1

:fim
endlocal
exit /b 0

:instalar
rem Instala %1 com o uv (bem mais rapido que o pip); se o uv nao der certo, usa o pip.
if not exist ".venv\Scripts\uv.exe" "%VENV_PY%" -m pip install --disable-pip-version-check -q uv
if exist ".venv\Scripts\uv.exe" ".venv\Scripts\uv.exe" pip install --python "%VENV_PY%" -r %1
if exist ".venv\Scripts\uv.exe" if not errorlevel 1 exit /b 0
"%VENV_PY%" -m pip install --disable-pip-version-check -r %1
exit /b %errorlevel%

:liberar_porta
rem Encerra uma copia antiga do V.E.R.D.E. (streamlit ou uvicorn) que ainda esteja usando a porta %1.
rem Sem isso, o novo dashboard sobe em outra porta e o navegador abre o antigo, com o codigo velho.
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort %1 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { $p = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $_.OwningProcess); if ($p -and $p.CommandLine -like '*%2*') { Write-Host '[V.E.R.D.E.] Encerrando a copia antiga na porta %1'; Stop-Process -Id $p.ProcessId -Force } }"
exit /b 0
