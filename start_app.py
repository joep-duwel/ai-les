#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 KNN Image Classifier - Easy Starter
=======================================

Een eenvoudige starter voor de KNN Image Classifier applicatie.
Dit script zorgt voor een mooie welkomstboodschap en start automatisch de main app.

Gebruik: python start_app.py
"""

import sys
import os

def print_welcome():
    """Print een vriendelijke welkomstboodschap."""
    print("🌟" + "=" * 78 + "🌟")
    print("                     🎉 WELKOM BIJ DE KNN IMAGE CLASSIFIER! 🎉")
    print("")
    print("                    📸 'Beter een K-nearest neighbour dan een verre vriend' 📸")
    print("")
    print("                              🎓 AI Lessen Opdracht 🎓")
    print("🌟" + "=" * 78 + "🌟")
    print("")
    print("🤖 Deze app demonstreert het K-Nearest Neighbors algoritme voor beeldherkenning!")
    print("")
    print("📚 Wat ga je leren:")
    print("   • Hoe KNN werkt (K-Nearest Neighbors)")
    print("   • Afbeelding classificatie met machine learning")
    print("   • MNIST cijfers en Fashion-MNIST kledingstukken herkennen")
    print("   • Model evaluatie en optimalisatie")
    print("")
    print("🎮 Wat kun je doen:")
    print("   • Datasets laden (cijfers of kledingstukken)")
    print("   • KNN modellen trainen")
    print("   • Voorspellingen maken op nieuwe afbeeldingen")
    print("   • Prestaties evalueren en visualiseren")
    print("   • De optimale K-waarde vinden")
    print("")
    print("💡 Eerste keer? Probeer dit:")
    print("   1️⃣ Laad MNIST Cijfers Dataset")
    print("   2️⃣ Train KNN Model (K=3 is een goede start)")  
    print("   3️⃣ Voorspel Testafbeeldingen")
    print("   4️⃣ Evalueer Model Prestaties")
    print("")
    print("🔸" + "─" * 78 + "🔸")

def check_dependencies():
    """Check of alle vereiste packages geïnstalleerd zijn."""
    try:
        import numpy
        import sklearn
        import matplotlib
        import seaborn
        from PIL import Image
        return True
    except ImportError as e:
        print("❌ Ontbrekende dependencies gedetecteerd!")
        print(f"   Fout: {e}")
        print("")
        print("🔧 Oplossing:")
        print("   pip install -r requirements.txt")
        print("")
        print("   Of installeer handmatig:")
        print("   pip install numpy scikit-learn matplotlib seaborn pillow")
        print("")
        return False

def main():
    """Main functie om de app te starten."""
    # Clear screen voor nette presentatie
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Toon welkomstboodschap
    print_welcome()
    
    # Check dependencies
    print("🔍 Controleren van Python dependencies...")
    if not check_dependencies():
        input("⏎ Druk Enter om af te sluiten...")
        return
    
    print("✅ Alle dependencies zijn geïnstalleerd!")
    print("")
    
    # Ask user if they want to continue
    try:
        user_input = input("🚀 Klaar om te beginnen? Druk Enter om de app te starten (of 'q' om af te sluiten): ").strip().lower()
        
        if user_input == 'q':
            print("👋 Tot ziens!")
            return
            
        print("")
        print("🔄 KNN Image Classifier wordt geladen...")
        print("📂 Importeren van modules...")
        
        # Import en start de main app
        from knn_app_ui import main as start_main_app
        
        print("✅ Modules geladen!")
        print("🎯 Starting main application...")
        print("")
        
        # Start the main application
        start_main_app()
        
    except KeyboardInterrupt:
        print("\n\n👋 Applicatie wordt afgesloten door gebruiker...")
    except ImportError as e:
        print(f"❌ Fout bij importeren van de main app: {e}")
        print("🔧 Zorg ervoor dat knn_app_ui.py in dezelfde directory staat!")
    except Exception as e:
        print(f"❌ Onverwachte fout: {e}")
        print("🤔 Probeer de app opnieuw te starten of check de documentatie.")

if __name__ == "__main__":
    main() 