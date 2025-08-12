 @echo off
echo ========================================
echo River Water Quality Management System
echo Setup Script for Windows
echo ========================================
echo.

echo Step 1: Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo.
echo Step 2: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Step 3: Installing Python dependencies...
pip install Flask==2.3.3
pip install py2neo==2021.2.4
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install joblib==1.3.2
pip install openpyxl==3.1.2
pip install plotly==5.17.0
pip install requests==2.31.0
pip install python-dotenv==1.0.0

echo.
echo Step 4: Checking for required data files...
if exist "河流拓扑结构.xlsx" (
    echo ✓ Found: 河流拓扑结构.xlsx
) else (
    echo ✗ Missing: 河流拓扑结构.xlsx
    echo Please ensure this file is in the project directory
)

if exist "河道氨氮统计数据--环境容量.xlsx" (
    echo ✓ Found: 河道氨氮统计数据--环境容量.xlsx
) else (
    echo ✗ Missing: 河道氨氮统计数据--环境容量.xlsx
    echo Please ensure this file is in the project directory
)

if exist "train_tradedata.csv" (
    echo ✓ Found: train_tradedata.csv
) else (
    echo ✗ Missing: train_tradedata.csv
    echo Please ensure this file is in the project directory
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Install Neo4j Desktop from https://neo4j.com/download
echo 2. Create a database with password: 12345678
echo 3. Start the database
echo 4. Run: python run.py
echo 5. Open: http://localhost:5000
echo.
pause