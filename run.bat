 @echo off
echo ========================================
echo Starting River Management System...
echo ========================================
echo.

echo Checking Neo4j connection...
python -c "from py2neo import Graph; g = Graph('bolt://localhost:7687', auth=('neo4j', '12345678')); print('✓ Neo4j connection successful')" 2>nul
if %errorlevel% neq 0 (
    echo ✗ Neo4j connection failed
    echo Please ensure Neo4j is running with:
    echo - Host: localhost
    echo - Port: 7687
    echo - Username: neo4j
    echo - Password: 12345678
    echo.
    echo You can install Neo4j Desktop from: https://neo4j.com/download
    pause
    exit /b 1
)

echo.
echo Starting Flask application...
echo Access the application at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python run.py

pause