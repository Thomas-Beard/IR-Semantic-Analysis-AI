@echo off
setlocal

REM === CONFIG ===
set BASE_DIR=%~dp0
set JAVA_HOME=%BASE_DIR%jdk-17
set PATH=%JAVA_HOME%\bin;%PATH%
set ZK_DIR=%BASE_DIR%zookeeper
set SOLR_DIR=%BASE_DIR%solr-9.5.0
set SOLR_PORT=8990
set ZK_PORT=2181

REM === LAUNCH ZK ===
echo Starting ZooKeeper...
start /b "" "%ZK_DIR%\bin\zkServer.cmd"

echo Waiting 6 seconds...
timeout /t 6 >nul

REM === LAUNCH SOLR ===
echo Starting SolrCloud...
cd /d "%SOLR_DIR%"
call bin\solr.cmd start -c -z localhost:%ZK_PORT% -p %SOLR_PORT%

echo Waiting 20 seconds for Solr to initialize...
timeout /t 20 >nul

start http://localhost:%SOLR_PORT%/solr

endlocal
