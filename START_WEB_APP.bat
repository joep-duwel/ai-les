@echo off
:: 🌐 KNN Image Classifier - Web App Starter (Windows)
:: ===================================================
:: 
:: Batch bestand om de web-gebaseerde KNN Image Classifier te starten
:: Dit bestand start de Streamlit web applicatie vanuit Windows
::
:: Auteur: AI Student
:: Datum: 2024
:: Vak: AI Lessen - "Beter een K-nearest neighbour dan een verre vriend"

title KNN Image Classifier - Web App

echo.
echo ================================================================
echo            🌐 KNN IMAGE CLASSIFIER - WEB APP 🌐
echo ================================================================
echo 🎓 AI Lessen - "Beter een K-nearest neighbour dan een verre vriend"
echo 📸 Web-gebaseerde interface voor KNN beeldclassificatie
echo ================================================================
echo.

:: Controleer of Python beschikbaar is
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is niet gevonden op je systeem!
    echo 📥 Download Python van: https://python.org
    echo.
    pause
    exit /b 1
)

echo ✅ Python gevonden!
echo.

:: Probeer de web app te starten
echo 🚀 Starting KNN Web App...
echo 📱 De app wordt geopend in je browser op: http://localhost:8501
echo ⏹️  Sluit dit venster om de app te stoppen
echo.

:: Start de Python web app starter
python start_web_app.py

echo.
echo 👋 Web app gestopt. Tot ziens!
pause 