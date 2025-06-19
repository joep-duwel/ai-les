#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN Image Classifier - AI Assignment
=====================================

Een K-nearest neighbour implementatie voor afbeeldings classificatie.
Dit script implementeert een KNN classifier die kan worden gebruikt voor
het classificeren van handgeschreven cijfers (MNIST) of kledingstukken (Fashion-MNIST).

Auteur: AI Student
Datum: 2024
Vak: AI Lessen
Opdracht: "Beter een K-nearest neighbour dan een verre vriend"
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class KNNImageClassifier:
    """
    K-Nearest Neighbours Classifier voor beeldclassificatie.
    
    Deze klasse implementeert een KNN classifier die kan worden gebruikt
    voor het classificeren van afbeeldingen op basis van pixel waarden.
    """
    
    def __init__(self, k=3, distance_metric='euclidean'):
        """
        Initialiseer de KNN classifier.
        
        Parameters:
        -----------
        k : int, default=3
            Het aantal naaste buren om te bekijken voor classificatie
        distance_metric : str, default='euclidean'
            De afstandsmetriek om te gebruiken ('euclidean' of 'manhattan')
        """
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        
    def fit(self, X_train, y_train):
        """
        Train de KNN classifier door de trainingsdata op te slaan.
        
        KNN is een 'lazy learning' algoritme, wat betekent dat het geen
        expliciete training fase heeft. Het slaat simpelweg alle trainingsdata op.
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Trainingsafbeeldingen (n_samples, n_features)
        y_train : numpy.ndarray
            Trainingslabels (n_samples,)
        """
        print(f"Training KNN classifier met {len(X_train)} samples...")
        self.X_train = X_train
        self.y_train = y_train
        print("Training voltooid!")
        
    def _calculate_distance(self, x1, x2):
        """
        Bereken de afstand tussen twee datapunten.
        
        Parameters:
        -----------
        x1, x2 : numpy.ndarray
            Twee afbeeldingsvectoren
            
        Returns:
        --------
        float
            De afstand tussen x1 en x2
        """
        if self.distance_metric == 'euclidean':
            # L2 afstand: sqrt(sum((x1 - x2)^2))
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            # L1 afstand: sum(|x1 - x2|)
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError("Ongeldige distance_metric. Gebruik 'euclidean' of 'manhattan'.")
    
    def _get_k_nearest_neighbors(self, test_sample):
        """
        Vind de k naaste buren voor een test sample.
        
        Parameters:
        -----------
        test_sample : numpy.ndarray
            Een enkele test afbeelding
            
        Returns:
        --------
        list
            Labels van de k naaste buren
        """
        # Bereken afstanden tot alle trainingssamples
        distances = []
        for i, train_sample in enumerate(self.X_train):
            distance = self._calculate_distance(test_sample, train_sample)
            distances.append((distance, self.y_train[i]))
        
        # Sorteer op afstand en neem de eerste k
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        
        # Return alleen de labels
        return [label for _, label in k_nearest]
    
    def predict_single(self, test_sample):
        """
        Voorspel de klasse van een enkele afbeelding.
        
        Parameters:
        -----------
        test_sample : numpy.ndarray
            Een enkele test afbeelding
            
        Returns:
        --------
        int
            Voorspelde klasse
        """
        if self.X_train is None:
            raise ValueError("Model is niet getraind! Roep eerst fit() aan.")
        
        # Krijg k naaste buren
        neighbors = self._get_k_nearest_neighbors(test_sample)
        
        # Stem af op de meest voorkomende klasse (majority voting)
        vote_counts = Counter(neighbors)
        predicted_class = vote_counts.most_common(1)[0][0]
        
        return predicted_class
    
    def predict(self, X_test):
        """
        Voorspel klassen voor meerdere test afbeeldingen.
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            Test afbeeldingen (n_samples, n_features)
            
        Returns:
        --------
        numpy.ndarray
            Voorspelde klassen
        """
        print(f"Voorspelling maken voor {len(X_test)} test samples...")
        predictions = []
        
        for i, test_sample in enumerate(X_test):
            if i % 50 == 0:  # Progress indicator
                print(f"Verwerkt: {i}/{len(X_test)} samples")
            
            prediction = self.predict_single(test_sample)
            predictions.append(prediction)
        
        print("Voorspellingen voltooid!")
        return np.array(predictions)


class DatasetLoader:
    """
    Klasse voor het laden en voorbereiden van datasets.
    """
    
    @staticmethod
    def load_mnist_digits():
        """
        Laad de MNIST cijfers dataset.
        
        Returns:
        --------
        tuple
            (X, y) waar X de afbeeldingen zijn en y de labels
        """
        print("MNIST cijfers dataset laden...")
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X = mnist.data.astype(np.float32)
        y = mnist.target.astype(np.int32)
        
        # Normaliseer pixel waarden naar 0-1 bereik
        X = X / 255.0
        
        print(f"Dataset geladen: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    @staticmethod
    def load_fashion_mnist():
        """
        Laad de Fashion-MNIST dataset.
        
        Returns:
        --------
        tuple
            (X, y) waar X de afbeeldingen zijn en y de labels
        """
        print("Fashion-MNIST dataset laden...")
        fashion_mnist = fetch_openml('Fashion-MNIST', version=1, parser='auto')
        X = fashion_mnist.data.astype(np.float32)
        y = fashion_mnist.target.astype(np.int32)
        
        # Normaliseer pixel waarden naar 0-1 bereik
        X = X / 255.0
        
        print(f"Dataset geladen: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y


class ModelEvaluator:
    """
    Klasse voor het evalueren van model prestaties.
    """
    
    @staticmethod
    def evaluate_model(y_true, y_pred, class_names=None):
        """
        Evalueer model prestaties en toon resultaten.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            Werkelijke labels
        y_pred : numpy.ndarray
            Voorspelde labels
        class_names : list, optional
            Namen van de klassen voor betere visualisatie
        """
        # Bereken accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Toon classificatie rapport
        print("\nClassificatie Rapport:")
        print(classification_report(y_true, y_pred))
        
        # Toon confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Voorspelde Labels')
        plt.ylabel('Werkelijke Labels')
        
        if class_names:
            plt.xticks(range(len(class_names)), class_names, rotation=45)
            plt.yticks(range(len(class_names)), class_names, rotation=0)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def find_optimal_k(X_train, y_train, X_val, y_val, k_range=range(1, 21)):
        """
        Vind de optimale k waarde door verschillende k waardes te testen.
        
        Parameters:
        -----------
        X_train, y_train : numpy.ndarray
            Trainingsdata
        X_val, y_val : numpy.ndarray
            Validatiedata
        k_range : range
            Bereik van k waardes om te testen
            
        Returns:
        --------
        int
            Optimale k waarde
        """
        print("Zoeken naar optimale k waarde...")
        accuracies = []
        
        for k in k_range:
            print(f"Testen k={k}...")
            knn = KNNImageClassifier(k=k)
            knn.fit(X_train, y_train)
            
            # Test op een kleinere subset voor snelheid
            subset_size = min(100, len(X_val))
            predictions = knn.predict(X_val[:subset_size])
            accuracy = accuracy_score(y_val[:subset_size], predictions)
            accuracies.append(accuracy)
            print(f"k={k}, Accuracy: {accuracy:.4f}")
        
        # Plot resultaten
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, accuracies, 'bo-')
        plt.xlabel('k waarde')
        plt.ylabel('Validatie Accuracy')
        plt.title('K waarde vs Accuracy')
        plt.grid(True)
        plt.show()
        
        # Vind optimale k
        optimal_k = k_range[np.argmax(accuracies)]
        print(f"Optimale k waarde: {optimal_k} (Accuracy: {max(accuracies):.4f})")
        
        return optimal_k


def visualize_samples(X, y, class_names=None, n_samples=10):
    """
    Visualiseer enkele voorbeelden uit de dataset.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Afbeeldingsdata
    y : numpy.ndarray
        Labels
    class_names : list, optional
        Namen van de klassen
    n_samples : int
        Aantal voorbeelden om te tonen
    """
    plt.figure(figsize=(15, 6))
    
    for i in range(n_samples):
        plt.subplot(2, 5, i + 1)
        # Reshape van 784 naar 28x28 voor MNIST
        image = X[i].reshape(28, 28)
        plt.imshow(image, cmap='gray')
        
        if class_names:
            title = f"Klasse: {class_names[y[i]]}"
        else:
            title = f"Label: {y[i]}"
        
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Deze code wordt uitgevoerd als het script direct wordt aangeroepen
    print("KNN Image Classifier - Test Run")
    print("=" * 40)
    
    # Laad een kleine subset van data voor snelle test
    print("Laden van test data...")
    X, y = DatasetLoader.load_mnist_digits()
    
    # Gebruik een kleine subset voor snelle test
    X_small = X[:1000]
    y_small = y[:1000]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_small, y_small, test_size=0.2, random_state=42
    )
    
    # Maak en train model
    knn = KNNImageClassifier(k=3)
    knn.fit(X_train, y_train)
    
    # Test voorspelling
    print("Test voorspelling...")
    predictions = knn.predict(X_test[:10])  # Test alleen eerste 10
    print(f"Voorspellingen: {predictions}")
    print(f"Werkelijke labels: {y_test[:10]}")
    
    print("\nTest voltooid!") 