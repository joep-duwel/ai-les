#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Image Tester - Verdiepende Opdracht
==========================================

Een uitbreiding voor de KNN Image Classifier die een map met afbeeldingen
kan inladen, classificeren en de nauwkeurigheid berekenen op basis van
de bestandsnamen.

Deze module implementeert de verdiepende opdracht uit de assignment:
"Maak een map met verschillende afbeeldingen. Geef de afbeeldingen de 
correcte naam (bijv. "T-Shirt.jpg") en schrijf een functie die alle 
afbeeldingen inlaadt, door je algoritme laat classificeren en de 
nauwkeurigheid uitrekent aan de hand van de bestandsnaam."

Auteur: AI Student
Datum: 2024
Vak: AI Lessen
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from knn_image_classifier import KNNImageClassifier, DatasetLoader


class BatchImageTester:
    """
    Klasse voor het batch testen van afbeeldingen uit een directory.
    
    Deze klasse kan een map met gelabelde afbeeldingen inladen,
    classificeren met een getraind KNN model, en de accuracy berekenen
    op basis van de bestandsnamen.
    """
    
    def __init__(self, knn_model, class_names=None):
        """
        Initialiseer de batch tester.
        
        Parameters:
        -----------
        knn_model : KNNImageClassifier
            Een getraind KNN model
        class_names : list, optional
            Namen van de klassen voor MNIST/Fashion-MNIST
        """
        self.knn_model = knn_model
        self.class_names = class_names or []
        
        # Fashion-MNIST klasse mapping (voor bestandsnaam naar klasse index)
        self.fashion_name_to_index = {
            't-shirt': 0, 'tshirt': 0, 'top': 0,
            'trouser': 1, 'trousers': 1, 'pants': 1,
            'pullover': 2,
            'dress': 3,
            'coat': 4,
            'sandal': 5, 'sandals': 5,
            'shirt': 6,
            'sneaker': 7, 'sneakers': 7,
            'bag': 8, 'bags': 8,
            'ankle boot': 9, 'boot': 9, 'boots': 9
        }
        
        # MNIST cijfer mapping
        self.digit_name_to_index = {str(i): i for i in range(10)}
    
    def load_image_from_file(self, image_path):
        """
        Laad en preprocess een afbeelding van bestand.
        
        Parameters:
        -----------
        image_path : str
            Pad naar de afbeelding
            
        Returns:
        --------
        numpy.ndarray
            Gepreprocesste afbeelding als 784-dimensionale vector
        """
        try:
            # Laad afbeelding
            img = Image.open(image_path)
            
            # Converteer naar grayscale als nodig
            if img.mode != 'L':
                img = img.convert('L')
            
            # Resize naar 28x28 (MNIST/Fashion-MNIST formaat)
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
            
            # Converteer naar numpy array
            img_array = np.array(img, dtype=np.float32)
            
            # Normaliseer pixel waarden naar 0-1 bereik
            img_array = img_array / 255.0
            
            # Flatten naar 784-dimensionale vector
            img_vector = img_array.flatten()
            
            return img_vector
            
        except Exception as e:
            print(f"Fout bij laden afbeelding {image_path}: {e}")
            return None
    
    def extract_label_from_filename(self, filename, dataset_type='fashion'):
        """
        Extraheer het label uit de bestandsnaam.
        
        Parameters:
        -----------
        filename : str
            Bestandsnaam (zonder pad)
        dataset_type : str
            Type dataset ('fashion' of 'digits')
            
        Returns:
        --------
        int or None
            Klasse index, of None als niet herkend
        """
        # Haal de bestandsnaam zonder extensie
        name_without_ext = os.path.splitext(filename)[0].lower()
        
        if dataset_type == 'fashion':
            # Zoek in Fashion-MNIST klassen
            for class_name, class_index in self.fashion_name_to_index.items():
                if class_name in name_without_ext:
                    return class_index
            
            # Als geen match, probeer numerieke waarde
            try:
                return int(name_without_ext)
            except ValueError:
                return None
                
        elif dataset_type == 'digits':
            # Zoek cijfer in bestandsnaam
            for digit_str in name_without_ext:
                if digit_str.isdigit():
                    return int(digit_str)
            return None
        
        return None
    
    def test_directory(self, directory_path, dataset_type='fashion', show_results=True):
        """
        Test alle afbeeldingen in een directory.
        
        Parameters:
        -----------
        directory_path : str
            Pad naar de directory met test afbeeldingen
        dataset_type : str
            Type dataset ('fashion' of 'digits')
        show_results : bool
            Of resultaten getoond moeten worden
            
        Returns:
        --------
        dict
            Resultaten dictionary met accuracy en details
        """
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory bestaat niet: {directory_path}")
        
        # Zoek alle afbeeldingsbestanden
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for extension in image_extensions:
            image_files.extend(glob.glob(os.path.join(directory_path, extension)))
            image_files.extend(glob.glob(os.path.join(directory_path, extension.upper())))
        
        if not image_files:
            raise ValueError(f"Geen afbeeldingsbestanden gevonden in {directory_path}")
        
        print(f"Gevonden {len(image_files)} afbeeldingen in {directory_path}")
        print("=" * 50)
        
        # Verwerk elke afbeelding
        results = {
            'total': 0,
            'correct': 0,
            'accuracy': 0.0,
            'predictions': [],
            'actual_labels': [],
            'filenames': [],
            'errors': []
        }
        
        for image_path in image_files:
            filename = os.path.basename(image_path)
            
            # Laad afbeelding
            img_vector = self.load_image_from_file(image_path)
            if img_vector is None:
                results['errors'].append(f"Kon {filename} niet laden")
                continue
            
            # Extraheer verwacht label uit bestandsnaam
            expected_label = self.extract_label_from_filename(filename, dataset_type)
            if expected_label is None:
                results['errors'].append(f"Kon label niet extraheren uit {filename}")
                continue
            
            # Maak voorspelling
            try:
                predicted_label = self.knn_model.predict_single(img_vector)
                
                # Sla resultaten op
                results['total'] += 1
                results['predictions'].append(predicted_label)
                results['actual_labels'].append(expected_label)
                results['filenames'].append(filename)
                
                if predicted_label == expected_label:
                    results['correct'] += 1
                    status = "✓ CORRECT"
                else:
                    status = "✗ FOUT"
                
                if show_results:
                    pred_name = self.class_names[predicted_label] if self.class_names else str(predicted_label)
                    actual_name = self.class_names[expected_label] if self.class_names else str(expected_label)
                    print(f"{filename:30} | Verwacht: {actual_name:12} | Voorspeld: {pred_name:12} | {status}")
                
            except Exception as e:
                results['errors'].append(f"Fout bij voorspelling voor {filename}: {e}")
        
        # Bereken accuracy
        if results['total'] > 0:
            results['accuracy'] = results['correct'] / results['total']
        
        return results
    
    def create_sample_images_directory(self, output_dir, dataset_type='fashion', n_samples=20):
        """
        Maak een voorbeelddirectory met gelabelde afbeeldingen voor testen.
        
        Deze functie haalt samples uit de geladen dataset en slaat ze op
        met de juiste bestandsnamen voor batch testing.
        
        Parameters:
        -----------
        output_dir : str
            Directory waar voorbeeldafbeeldingen opgeslagen worden
        dataset_type : str
            Type dataset ('fashion' of 'digits')
        n_samples : int
            Aantal samples om op te slaan
        """
        # Maak directory als het niet bestaat
        os.makedirs(output_dir, exist_ok=True)
        
        # Laad dataset voor samples
        if dataset_type == 'fashion':
            X, y = DatasetLoader.load_fashion_mnist()
            class_names = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                          'sandal', 'shirt', 'sneaker', 'bag', 'boot']
        else:
            X, y = DatasetLoader.load_mnist_digits()
            class_names = [str(i) for i in range(10)]
        
        # Selecteer random samples
        indices = np.random.choice(len(X), n_samples, replace=False)
        
        print(f"Maken van {n_samples} voorbeeldafbeeldingen in {output_dir}")
        
        for i, idx in enumerate(indices):
            # Haal sample en label
            sample = X[idx]
            label = y[idx]
            class_name = class_names[label]
            
            # Reshape naar 28x28
            img_array = sample.reshape(28, 28)
            
            # Converteer naar 0-255 range
            img_array = (img_array * 255).astype(np.uint8)
            
            # Maak afbeelding
            img = Image.fromarray(img_array, mode='L')
            
            # Sla op met gelabelde naam
            filename = f"{class_name}_{i+1:03d}.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
        
        print(f"Voorbeeldafbeeldingen opgeslagen in {output_dir}")
        return output_dir
    
    def visualize_results(self, results, max_display=12):
        """
        Visualiseer de resultaten van batch testing.
        
        Parameters:
        -----------
        results : dict
            Resultaten van test_directory()
        max_display : int
            Maximum aantal afbeeldingen om te tonen
        """
        if results['total'] == 0:
            print("Geen resultaten om te visualiseren")
            return
        
        # Selecteer subset voor visualisatie
        n_display = min(max_display, results['total'])
        indices = np.random.choice(results['total'], n_display, replace=False)
        
        # Bepaal grid grootte
        cols = 4
        rows = (n_display + cols - 1) // cols
        
        plt.figure(figsize=(15, 4 * rows))
        
        for i, idx in enumerate(indices):
            plt.subplot(rows, cols, i + 1)
            
            filename = results['filenames'][idx]
            predicted = results['predictions'][idx]
            actual = results['actual_labels'][idx]
            
            # Probeer afbeelding te laden voor visualisatie
            # (Dit is een vereenvoudigde versie - in praktijk zou je de originele afbeelding opnieuw laden)
            
            pred_name = self.class_names[predicted] if self.class_names else str(predicted)
            actual_name = self.class_names[actual] if self.class_names else str(actual)
            
            color = 'green' if predicted == actual else 'red'
            status = "✓" if predicted == actual else "✗"
            
            plt.text(0.5, 0.7, f"Bestand: {filename}", ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=8)
            plt.text(0.5, 0.5, f"Verwacht: {actual_name}", ha='center', va='center',
                    transform=plt.gca().transAxes, fontsize=10)
            plt.text(0.5, 0.3, f"Voorspeld: {pred_name}", ha='center', va='center',
                    transform=plt.gca().transAxes, fontsize=10, color=color)
            plt.text(0.5, 0.1, status, ha='center', va='center',
                    transform=plt.gca().transAxes, fontsize=14, color=color)
            
            plt.axis('off')
        
        plt.tight_layout()
        plt.suptitle(f'Batch Test Resultaten - Accuracy: {results["accuracy"]:.2%}', 
                    fontsize=16, y=1.02)
        plt.show()


def demo_batch_testing():
    """
    Demonstratie van de batch testing functionaliteit.
    """
    print("Batch Image Tester Demo")
    print("=" * 30)
    
    # Laad dataset en train een klein model voor demo
    print("Laden van dataset en trainen van model voor demo...")
    X, y = DatasetLoader.load_fashion_mnist()
    
    # Gebruik kleine subset voor snelle demo
    X_small = X[:1000]
    y_small = y[:1000]
    
    # Train KNN model
    knn = KNNImageClassifier(k=3)
    knn.fit(X_small, y_small)
    
    # Fashion-MNIST klasse namen
    fashion_class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    
    # Maak batch tester
    batch_tester = BatchImageTester(knn, fashion_class_names)
    
    # Maak voorbeelddirectory
    sample_dir = "sample_images"
    batch_tester.create_sample_images_directory(sample_dir, 'fashion', 10)
    
    # Test de directory
    print(f"\nTesten van afbeeldingen in {sample_dir}:")
    results = batch_tester.test_directory(sample_dir, 'fashion')
    
    # Toon resultaten
    print(f"\n{'='*50}")
    print(f"BATCH TEST RESULTATEN")
    print(f"{'='*50}")
    print(f"Totaal geteste afbeeldingen: {results['total']}")
    print(f"Correct geclassificeerd: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    
    if results['errors']:
        print(f"\nFouten ({len(results['errors'])}):")
        for error in results['errors']:
            print(f"  - {error}")
    
    print(f"\nVoorbeeldafbeeldingen zijn opgeslagen in '{sample_dir}' directory.")
    print("Je kunt je eigen afbeeldingen toevoegen met de juiste naamgeving voor verdere tests!")


if __name__ == "__main__":
    demo_batch_testing() 