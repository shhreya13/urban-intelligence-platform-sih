@echo off
REM ---------------------------------------------------------------------
REM One-command demo (Windows).
REM
REM   demo.bat                             first clip in data\videos
REM   demo.bat data\videos\my_clip.mp4     a specific clip
REM   demo.bat --multi-bus                 3 buses -> cross-bus dedup visible
REM   demo.bat --dashboard                 NO model: load the shipped output
REM                                        and start the web + mobile apps
REM
REM Demo accounts: admin@ traffic@ roads@ safety@ transport@ field@
REM urbanintel.local.  The password below is for local demos only; set
REM UIP_AUTH__DEMO_PASSWORD yourself to change it, and set
REM auth.seed_demo_users: false in config\platform.yaml for real use.
REM ---------------------------------------------------------------------
setlocal
cd /d "%~dp0"

if "%PYTHON%"=="" set PYTHON=python
if "%UIP_AUTH__DEMO_PASSWORD%"=="" set "UIP_AUTH__DEMO_PASSWORD=UrbanIntel@2026"

if /i "%~1"=="--dashboard" (
  echo --- loading the shipped pipeline output, no model is run ---
  "%PYTHON%" scripts\import_events.py --sync
  if errorlevel 1 exit /b 1
  echo.
  echo   Demo accounts : admin@ traffic@ roads@ safety@ transport@ field@urbanintel.local
  echo   Password      : %UIP_AUTH__DEMO_PASSWORD%
  "%PYTHON%" run_backend.py %2 %3 %4 %5
  goto :eof
)

echo --- checking the environment ---
"%PYTHON%" scripts\check_setup.py
if errorlevel 1 (
  echo.
  echo Setup check failed.  Try:  %PYTHON% -m pip install -r requirements.txt
  exit /b 1
)

echo.
echo --- running the demo ---
if exist "%~1" (
  "%PYTHON%" scripts\run_demo.py --source "%~1" %2 %3 %4 %5 %6
) else (
  "%PYTHON%" scripts\run_demo.py %*
)
endlocal
