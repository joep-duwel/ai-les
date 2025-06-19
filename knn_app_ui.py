#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN Image Classifier UI - AI Assignment
=======================================

Een eenvoudige gebruikersinterface voor de KNN afbeeldings classifier.
Deze applicatie biedt een menu-driven interface waarmee gebruikers
verschillende opties kunnen kiezen voor afbeeldings classificatie.

Auteur: AI Student
Datum: 2024
Vak: AI Lessen
Opdracht: "Beter een K-nearest neighbour dan een verre vriend"
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from knn_image_classifier import KNNImageClassifier, DatasetLoader, ModelEvaluator, visualize_samples


class KNNImageClassifierApp:
    """
    Hoofdapplicatie klasse die de gebruikersinterface beheert.
    
    Deze klasse biedt een menu-systeem waarmee gebruikers verschillende
    opties kunnen kiezen voor het werken met de KNN classifier.
    """
    
    def __init__(self):
        """
        Initialiseer de applicatie.
        """
        self.knn_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.dataset_type = None
        self.class_names = None
        
        # Fashion-MNIST klasse namen
        self.fashion_class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
        
        # MNIST cijfer namen
        self.digit_class_names = [str(i) for i in range(10)]
    
    def clear_screen(self):
        """
        Maak het scherm leeg voor een nettere gebruikerservaring.
        """
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """
        Toon de applicatie header.
        """
        print("🌟" + "=" * 58 + "🌟")
        print("      📸🤖 KNN AFBEELDINGS CLASSIFIER 🤖📸")
        print("         🎓 AI Lessen - Opdracht 🎓")
        print("    'Beter een K-nearest neighbour dan een verre vriend'")
        print("🌟" + "=" * 58 + "🌟")
        print()
    
    def print_main_menu(self):
        """
        Toon het hoofdmenu met alle beschikbare opties.
        """
        print("🎮 HOOFDMENU - Wat wil je doen?")
        print("🔸" + "─" * 42 + "🔸")
        print("📊 1. Laad MNIST Cijfers Dataset (0-9)")
        print("👕 2. Laad Fashion-MNIST Dataset (kleding)")
        print("🧠 3. Train KNN Model")
        print("🔮 4. Voorspel Testafbeeldingen")
        print("📈 5. Evalueer Model Prestaties")
        print("⚙️ 6. Zoek Optimale K-waarde")
        print("🎨 7. Visualiseer Dataset Voorbeelden")
        print("ℹ️ 8. Model Informatie")
        print("🎓 9. Informatie over KNN")
        print("❌ 0. Afsluiten")
        print("🔸" + "─" * 42 + "🔸")
    
    def get_user_choice(self):
        """
        Krijg gebruiker keuze en valideer invoer.
        
        Returns:
        --------
        str
            Geldige gebruiker keuze
        """
        while True:
            try:
                choice = input("\n👉 Kies een optie (0-9): ").strip()
                if choice in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    return choice
                else:
                    print("❌ Ongeldige keuze! Kies een nummer tussen 0 en 9.")
            except KeyboardInterrupt:
                print("\n\n👋 Applicatie wordt afgesloten...")
                sys.exit(0)
    
    def load_mnist_digits(self):
        """
        Laad de MNIST cijfers dataset en bereid deze voor.
        """
        print("📊 MNIST Cijfers Dataset Laden")
        print("🔸" + "─" * 32 + "🔸")
        
        try:
            # Laad dataset
            X, y = DatasetLoader.load_mnist_digits()
            
            # Gebruik subset voor snellere verwerking
            print("Voor snellere verwerking, selecteren we een subset van de data...")
            subset_size = min(5000, len(X))
            indices = np.random.choice(len(X), subset_size, replace=False)
            X = X[indices]
            y = y[indices]
            
            # Split in train en test
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            self.dataset_type = 'digits'
            self.class_names = self.digit_class_names
            
            print(f"✅ Dataset succesvol geladen!")
            print(f"🎯 Training samples: {len(self.X_train)}")
            print(f"🧪 Test samples: {len(self.X_test)}")
            print(f"🔢 Aantal klassen: {len(np.unique(y))}")
            
        except Exception as e:
            print(f"❌ Fout bij laden dataset: {e}")
        
        input("\n⏎ Druk Enter om door te gaan...")
    
    def load_fashion_mnist(self):
        """
        Laad de Fashion-MNIST dataset en bereid deze voor.
        """
        print("Fashion-MNIST Dataset Laden")
        print("-" * 30)
        
        try:
            # Laad dataset
            X, y = DatasetLoader.load_fashion_mnist()
            
            # Gebruik subset voor snellere verwerking
            print("Voor snellere verwerking, selecteren we een subset van de data...")
            subset_size = min(5000, len(X))
            indices = np.random.choice(len(X), subset_size, replace=False)
            X = X[indices]
            y = y[indices]
            
            # Split in train en test
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            self.dataset_type = 'fashion'
            self.class_names = self.fashion_class_names
            
            print(f"Dataset succesvol geladen!")
            print(f"Training samples: {len(self.X_train)}")
            print(f"Test samples: {len(self.X_test)}")
            print(f"Aantal klassen: {len(np.unique(y))}")
            print(f"Klassen: {', '.join(self.class_names)}")
            
        except Exception as e:
            print(f"Fout bij laden dataset: {e}")
        
        input("\nDruk Enter om door te gaan...")
    
    def train_model(self):
        """
        Train het KNN model met de geladen data.
        """
        print("KNN Model Trainen")
        print("-" * 30)
        
        if self.X_train is None:
            print("Geen dataset geladen! Laad eerst een dataset (optie 1 of 2).")
            input("\nDruk Enter om door te gaan...")
            return
        
        try:
            # Vraag gebruiker om k waarde
            while True:
                try:
                    k = int(input("Voer k waarde in (aanbevolen: 3-7): "))
                    if k > 0:
                        break
                    else:
                        print("k moet groter zijn dan 0!")
                except ValueError:
                    print("Voer een geldig getal in!")
            
            # Vraag om afstandsmetriek
            print("\nBeschikbare afstandsmeteiken:")
            print("1. Euclidean (L2)")
            print("2. Manhattan (L1)")
            
            while True:
                metric_choice = input("Kies afstandsmetriek (1 of 2): ").strip()
                if metric_choice == '1':
                    distance_metric = 'euclidean'
                    break
                elif metric_choice == '2':
                    distance_metric = 'manhattan'
                    break
                else:
                    print("Ongeldige keuze! Kies 1 of 2.")
            
            # Maak en train model
            self.knn_model = KNNImageClassifier(k=k, distance_metric=distance_metric)
            self.knn_model.fit(self.X_train, self.y_train)
            
            print(f"\nModel succesvol getraind!")
            print(f"K waarde: {k}")
            print(f"Afstandsmetriek: {distance_metric}")
            
        except Exception as e:
            print(f"Fout bij trainen model: {e}")
        
        input("\nDruk Enter om door te gaan...")
    
    def predict_images(self):
        """
        Maak voorspellingen op test afbeeldingen.
        """
        print("Afbeeldingen Voorspellen")
        print("-" * 30)
        
        if self.knn_model is None:
            print("Geen model getraind! Train eerst een model (optie 3).")
            input("\nDruk Enter om door te gaan...")
            return
        
        try:
            # Vraag aantal test samples
            while True:
                try:
                    n_samples = int(input(f"Aantal test samples om te voorspellen (max {len(self.X_test)}): "))
                    if 0 < n_samples <= len(self.X_test):
                        break
                    else:
                        print(f"Voer een getal tussen 1 en {len(self.X_test)} in!")
                except ValueError:
                    print("Voer een geldig getal in!")
            
            # Maak voorspellingen
            print(f"\nVoorspellingen maken voor {n_samples} samples...")
            predictions = self.knn_model.predict(self.X_test[:n_samples])
            actual = self.y_test[:n_samples]
            
            # Toon resultaten
            print(f"\nResultaten:")
            print("-" * 20)
            correct = 0
            for i in range(min(10, n_samples)):  # Toon maximaal 10 resultaten
                pred_class = self.class_names[predictions[i]] if self.class_names else str(predictions[i])
                actual_class = self.class_names[actual[i]] if self.class_names else str(actual[i])
                status = "✓" if predictions[i] == actual[i] else "✗"
                print(f"Sample {i+1}: Voorspeld={pred_class}, Werkelijk={actual_class} {status}")
                if predictions[i] == actual[i]:
                    correct += 1
            
            if n_samples > 10:
                print(f"... (en {n_samples-10} meer)")
            
            # Bereken en toon accuracy
            accuracy = np.mean(predictions == actual)
            print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
        except Exception as e:
            print(f"Fout bij voorspellen: {e}")
        
        input("\nDruk Enter om door te gaan...")
    
    def evaluate_model(self):
        """
        Evalueer model prestaties met uitgebreide metrics.
        """
        print("Model Prestaties Evalueren")
        print("-" * 30)
        
        if self.knn_model is None:
            print("Geen model getraind! Train eerst een model (optie 3).")
            input("\nDruk Enter om door te gaan...")
            return
        
        try:
            print("Model evaluatie uitvoeren...")
            
            # Gebruik subset voor snellere evaluatie
            n_eval = min(200, len(self.X_test))
            predictions = self.knn_model.predict(self.X_test[:n_eval])
            actual = self.y_test[:n_eval]
            
            # Gebruik ModelEvaluator voor uitgebreide evaluatie
            ModelEvaluator.evaluate_model(actual, predictions, self.class_names)
            
        except Exception as e:
            print(f"Fout bij evaluatie: {e}")
        
        input("\nDruk Enter om door te gaan...")
    
    def find_optimal_k(self):
        """
        Zoek de optimale K waarde door verschillende waardes te testen.
        """
        print("Optimale K-waarde Zoeken")
        print("-" * 30)
        
        if self.X_train is None:
            print("Geen dataset geladen! Laad eerst een dataset (optie 1 of 2).")
            input("\nDruk Enter om door te gaan...")
            return
        
        try:
            # Maak validatie set
            X_train_sub, X_val, y_train_sub, y_val = train_test_split(
                self.X_train, self.y_train, test_size=0.2, random_state=42
            )
            
            # Gebruik kleinere subsets voor snelheid
            X_train_sub = X_train_sub[:500]
            y_train_sub = y_train_sub[:500]
            X_val = X_val[:100]
            y_val = y_val[:100]
            
            print("Zoeken naar optimale k waarde...")
            print("Dit kan enkele minuten duren...")
            
            # Zoek optimale k
            optimal_k = ModelEvaluator.find_optimal_k(
                X_train_sub, y_train_sub, X_val, y_val, 
                k_range=range(1, 11)  # Test k van 1 tot 10
            )
            
            print(f"\nAanbevolen k waarde: {optimal_k}")
            
        except Exception as e:
            print(f"Fout bij zoeken optimale k: {e}")
        
        input("\nDruk Enter om door te gaan...")
    
    def visualize_dataset(self):
        """
        Visualiseer voorbeelden uit de dataset.
        """
        print("Dataset Voorbeelden Visualiseren")
        print("-" * 30)
        
        if self.X_train is None:
            print("Geen dataset geladen! Laad eerst een dataset (optie 1 of 2).")
            input("\nDruk Enter om door te gaan...")
            return
        
        try:
            print("Tonen van dataset voorbeelden...")
            visualize_samples(self.X_train, self.y_train, self.class_names)
            
        except Exception as e:
            print(f"Fout bij visualisatie: {e}")
        
        input("\nDruk Enter om door te gaan...")
    
    def show_model_info(self):
        """
        Toon informatie over het huidige model.
        """
        print("Model Informatie")
        print("-" * 30)
        
        if self.knn_model is None:
            print("Geen model getraind!")
        else:
            print(f"Model Type: K-Nearest Neighbors")
            print(f"K waarde: {self.knn_model.k}")
            print(f"Afstandsmetriek: {self.knn_model.distance_metric}")
            print(f"Training samples: {len(self.knn_model.X_train) if self.knn_model.X_train is not None else 'Geen'}")
        
        if self.dataset_type:
            print(f"\nDataset Type: {'MNIST Cijfers' if self.dataset_type == 'digits' else 'Fashion-MNIST'}")
            if self.X_train is not None:
                print(f"Training samples: {len(self.X_train)}")
                print(f"Test samples: {len(self.X_test)}")
                print(f"Aantal features: {self.X_train.shape[1]}")
                print(f"Aantal klassen: {len(np.unique(self.y_train))}")
        else:
            print("\nGeen dataset geladen.")
        
        input("\nDruk Enter om door te gaan...")
    
    def show_knn_info(self):
        """
        Toon informatie over het KNN algoritme.
        """
        print("Informatie over K-Nearest Neighbors")
        print("=" * 40)
        
        info_text = """
        KNN (K-Nearest Neighbors) is een eenvoudig maar krachtig machine learning algoritme.
        
        Hoe werkt het?
        --------------
        1. Voor elke nieuwe afbeelding die geclassificeerd moet worden:
        2. Bereken de afstand tot alle trainingsafbeeldingen
        3. Selecteer de K dichtstbijzijnde buren
        4. Stem af: de klasse die het meest voorkomt bij de K buren wint
        
        Afstandsmeteiken:
        -----------------
        • Euclidean (L2): sqrt(sum((x1-x2)²)) - meest gebruikt
        • Manhattan (L1): sum(|x1-x2|) - minder gevoelig voor uitschieters
        
        Voordelen:
        ----------
        • Eenvoudig te begrijpen en implementeren
        • Geen training fase nodig ('lazy learning')
        • Goed voor niet-lineaire data
        • Kan worden gebruikt voor classificatie en regressie
        
        Nadelen:
        --------
        • Langzaam bij voorspelling (moet alle afstanden berekenen)
        • Gevoelig voor 'vloek van dimensionaliteit'
        • Gevoelig voor irrelevante features
        • Heeft veel geheugen nodig
        
        Keuze van K:
        ------------
        • K te klein (bijv. K=1): Overfitting, gevoelig voor ruis
        • K te groot: Underfitting, verliest lokale patronen
        • Aanbeveling: Probeer oneven getallen (3, 5, 7) om ties te voorkomen
        """
        
        print(info_text)
        input("\nDruk Enter om door te gaan...")
    
    def run(self):
        """
        Start de hoofdapplicatie loop.
        """
        print("🚀 KNN Image Classifier wordt gestart...")
        print("💡 Tip: Begin met optie 1 (MNIST) → optie 3 (Train) → optie 4 (Voorspel)!")
        input("\n⏎ Druk Enter om te beginnen...")
        
        while True:
            self.clear_screen()
            self.print_header()
            self.print_main_menu()
            
            choice = self.get_user_choice()
            
            self.clear_screen()
            self.print_header()
            
            if choice == '0':
                print("👋 Applicatie wordt afgesloten...")
                print("🎉 Bedankt voor het gebruiken van de KNN Image Classifier!")
                print("💝 Veel succes met je AI Lessen opdracht!")
                break
            elif choice == '1':
                self.load_mnist_digits()
            elif choice == '2':
                self.load_fashion_mnist()
            elif choice == '3':
                self.train_model()
            elif choice == '4':
                self.predict_images()
            elif choice == '5':
                self.evaluate_model()
            elif choice == '6':
                self.find_optimal_k()
            elif choice == '7':
                self.visualize_dataset()
            elif choice == '8':
                self.show_model_info()
            elif choice == '9':
                self.show_knn_info()


def main():
    """
    Hoofdfunctie om de applicatie te starten.
    """
    try:
        app = KNNImageClassifierApp()
        app.run()
    except KeyboardInterrupt:
        print("\n\nApplicatie wordt afgesloten door gebruiker...")
    except Exception as e:
        print(f"\nEr is een onverwachte fout opgetreden: {e}")
        print("Applicatie wordt afgesloten...")


if __name__ == "__main__":
    main() 