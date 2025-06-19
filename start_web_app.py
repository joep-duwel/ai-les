#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌐 KNN Image Classifier - Web App Starter
==========================================

Script om de web-gebaseerde KNN Image Classifier te starten.
Dit script controleert of alle dependencies geïnstalleerd zijn en start
vervolgens de Streamlit web applicatie.

Auteur: AI Student
Datum: 2024
Vak: AI Lessen
"""

import sys
import subprocess
import os
from pathlib import Path

def print_banner():
    """Print welkom banner."""
    print("\n" + "="*60)
    print("🌐 KNN IMAGE CLASSIFIER - WEB APP 🌐")
    print("="*60)
    print("🎓 AI Lessen - 'Beter een K-nearest neighbour dan een verre vriend'")
    print("📸 Web-gebaseerde interface voor KNN beeldclassificatie")
    print("="*60 + "\n")

def check_dependencies():
    """Controleer of alle benodigde packages geïnstalleerd zijn."""
    print("🔍 Controleren van dependencies...")
    
    required_packages = [
        'streamlit', 'numpy', 'scikit-learn', 'matplotlib', 
        'pandas', 'seaborn', 'plotly', 'Pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NIET GEVONDEN")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  De volgende packages ontbreken: {', '.join(missing_packages)}")
        print("🔧 Installeer ze met: pip install -r requirements.txt")
        return False
    
    print("✅ Alle dependencies zijn geïnstalleerd!\n")
    return True

def check_core_files():
    """Controleer of de core bestanden aanwezig zijn."""
    print("📁 Controleren van core bestanden...")
    
    required_files = [
        'knn_image_classifier.py',
        'knn_web_app.py'
    ]
    
    missing_files = []
    
    for file in required_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - NIET GEVONDEN")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  De volgende bestanden ontbreken: {', '.join(missing_files)}")
        return False
    
    print("✅ Alle core bestanden zijn aanwezig!\n")
    return True

def start_streamlit_app():
    """Start de Streamlit web applicatie."""
    print("🚀 Starting Streamlit web applicatie...")
    print("📱 De app wordt geopend in je browser op: http://localhost:8501")
    print("⏹️  Stop de app met Ctrl+C in dit terminal venster\n")
    
    try:
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "knn_web_app.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Fout bij starten van Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Web app gestopt. Tot ziens!")
        return True
    
    return True

def main():
    """Hoofdfunctie."""
    print_banner()
    
    # Controleer dependencies
    if not check_dependencies():
        print("🔄 Probeer eerst: pip install -r requirements.txt")
        input("\n⏎ Druk Enter om af te sluiten...")
        return
    
    # Controleer bestanden
    if not check_core_files():
        print("📁 Zorg dat alle bestanden in de juiste map staan.")
        input("\n⏎ Druk Enter om af te sluiten...")
        return
    
    print("🎉 Alles klaar! Starting web applicatie...")
    print("💡 Tip: Bookmark http://localhost:8501 voor snelle toegang\n")
    
    # Start de web app
    if start_streamlit_app():
        print("✅ Web app succesvol gestopt.")
    else:
        print("❌ Er ging iets mis bij het starten van de web app.")
    
    input("\n⏎ Druk Enter om af te sluiten...")

if __name__ == "__main__":
    main() 