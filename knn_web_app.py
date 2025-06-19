#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌐 KNN Image Classifier - Web App
==================================

Een moderne web-gebaseerde interface voor de KNN afbeeldings classifier.
Deze applicatie biedt een gebruiksvriendelijke webinterface voor het werken
met K-Nearest Neighbors voor beeldclassificatie.

Auteur: AI Student
Datum: 2024
Vak: AI Lessen
Opdracht: "Beter een K-nearest neighbour dan een verre vriend"

Start met: streamlit run knn_web_app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from PIL import Image
import io
import base64

# Import our KNN classes
from knn_image_classifier import KNNImageClassifier, DatasetLoader, ModelEvaluator

# Configure Streamlit page
st.set_page_config(
    page_title="🤖 KNN Image Classifier",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert > div {
        padding-top: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialiseer de Streamlit session state variabelen."""
    if 'knn_model' not in st.session_state:
        st.session_state.knn_model = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'dataset_type' not in st.session_state:
        st.session_state.dataset_type = None
    if 'class_names' not in st.session_state:
        st.session_state.class_names = None

def show_header():
    """Toon de applicatie header."""
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1>🤖📸 KNN Image Classifier 📸🤖</h1>
        <h3>🎓 AI Lessen - Opdracht</h3>
        <p style='font-style: italic; color: #666;'>"Beter een K-nearest neighbour dan een verre vriend"</p>
    </div>
    """, unsafe_allow_html=True)

def show_sidebar():
    """Toon de sidebar met navigatie en status."""
    st.sidebar.title("🎮 Navigatie")
    
    # Status informatie
    st.sidebar.markdown("### 📊 Status")
    
    if st.session_state.dataset_type:
        if st.session_state.dataset_type == 'digits':
            st.sidebar.success("📊 MNIST Cijfers geladen")
        else:
            st.sidebar.success("👕 Fashion-MNIST geladen")
        
        if st.session_state.X_train is not None:
            st.sidebar.info(f"🎯 Training: {len(st.session_state.X_train)} samples")
            st.sidebar.info(f"🧪 Test: {len(st.session_state.X_test)} samples")
    else:
        st.sidebar.warning("📊 Geen dataset geladen")
    
    if st.session_state.knn_model:
        st.sidebar.success("🧠 Model getraind")
        st.sidebar.info(f"⚙️ K = {st.session_state.knn_model.k}")
        st.sidebar.info(f"📏 Metric: {st.session_state.knn_model.distance_metric}")
    else:
        st.sidebar.warning("🧠 Model niet getraind")
    
    st.sidebar.markdown("---")
    
    # KNN uitleg
    with st.sidebar.expander("🎓 Wat is KNN?"):
        st.markdown("""
        **K-Nearest Neighbors** is een eenvoudig maar krachtig algoritme:
        
        **Hoe het werkt:**
        1. 🔍 Bereken afstand tot alle training afbeeldingen
        2. 👥 Selecteer de K dichtstbijzijnde buren
        3. 🗳️ Stem af: meest voorkomende klasse wint!
        
        **Voordelen:**
        - ✅ Eenvoudig te begrijpen
        - ✅ Geen training nodig
        - ✅ Werkt goed voor complexe patronen
        
        **Nadelen:**
        - ❌ Langzaam bij voorspelling
        - ❌ Heeft veel geheugen nodig
        """)

def load_dataset_page():
    """Dataset laden pagina."""
    st.header("📊 Dataset Laden")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔢 MNIST Cijfers")
        st.markdown("""
        - 70.000 handgeschreven cijfers (0-9)
        - 28x28 pixels, grayscale
        - Klassieke benchmark dataset
        """)
        
        if st.button("📊 Laad MNIST Cijfers", type="primary"):
            with st.spinner("📥 MNIST dataset laden..."):
                try:
                    X, y = DatasetLoader.load_mnist_digits()
                    
                    # Gebruik subset voor snellere verwerking
                    subset_size = min(5000, len(X))
                    indices = np.random.choice(len(X), subset_size, replace=False)
                    X = X[indices]
                    y = y[indices]
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    # Sla op in session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.dataset_type = 'digits'
                    st.session_state.class_names = [str(i) for i in range(10)]
                    
                    st.success("✅ MNIST dataset succesvol geladen!")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Fout bij laden dataset: {e}")
    
    with col2:
        st.subheader("👕 Fashion-MNIST")
        st.markdown("""
        - 70.000 kledingstukken (10 categorieën)
        - 28x28 pixels, grayscale
        - T-shirts, broeken, schoenen, etc.
        """)
        
        if st.button("👕 Laad Fashion-MNIST", type="primary"):
            with st.spinner("📥 Fashion-MNIST dataset laden..."):
                try:
                    X, y = DatasetLoader.load_fashion_mnist()
                    
                    # Gebruik subset voor snellere verwerking
                    subset_size = min(5000, len(X))
                    indices = np.random.choice(len(X), subset_size, replace=False)
                    X = X[indices]
                    y = y[indices]
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    # Fashion klasse namen
                    fashion_names = [
                        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
                    ]
                    
                    # Sla op in session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.dataset_type = 'fashion'
                    st.session_state.class_names = fashion_names
                    
                    st.success("✅ Fashion-MNIST dataset succesvol geladen!")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Fout bij laden dataset: {e}")
    
    # Toon dataset info als geladen
    if st.session_state.X_train is not None:
        st.markdown("---")
        st.subheader("📈 Dataset Informatie")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Training Samples", len(st.session_state.X_train))
        with col2:
            st.metric("🧪 Test Samples", len(st.session_state.X_test))
        with col3:
            st.metric("🔢 Features", st.session_state.X_train.shape[1])
        with col4:
            st.metric("🏷️ Klassen", len(np.unique(st.session_state.y_train)))

def visualize_dataset_page():
    """Dataset visualisatie pagina."""
    st.header("🎨 Dataset Visualisatie")
    
    if st.session_state.X_train is None:
        st.warning("⚠️ Laad eerst een dataset!")
        return
    
    # Sample afbeeldingen tonen
    st.subheader("🖼️ Voorbeeldafbeeldingen")
    
    n_samples = st.slider("Aantal voorbeelden", 5, 20, 10)
    
    if st.button("🎲 Toon Random Voorbeelden"):
        # Selecteer random samples
        indices = np.random.choice(len(st.session_state.X_train), n_samples, replace=False)
        
        # Maak grid
        cols = 5
        rows = (n_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else []
        
        for i, idx in enumerate(indices):
            if i < len(axes):
                # Reshape afbeelding
                img = st.session_state.X_train[idx].reshape(28, 28)
                label = st.session_state.y_train[idx]
                
                axes[i].imshow(img, cmap='gray')
                if st.session_state.class_names:
                    title = f"{st.session_state.class_names[label]}"
                else:
                    title = f"Label: {label}"
                axes[i].set_title(title, fontsize=10)
                axes[i].axis('off')
        
        # Verberg lege subplots
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Klasse distributie
    st.subheader("📊 Klasse Distributie")
    
    # Bereken distributie
    unique, counts = np.unique(st.session_state.y_train, return_counts=True)
    
    if st.session_state.class_names:
        labels = [st.session_state.class_names[i] for i in unique]
    else:
        labels = [str(i) for i in unique]
    
    # Plotly bar chart
    fig = px.bar(
        x=labels, 
        y=counts,
        title="Distributie van Klassen in Training Set",
        labels={'x': 'Klasse', 'y': 'Aantal Samples'},
        color=counts,
        color_continuous_scale='viridis'
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def train_model_page():
    """Model training pagina."""
    st.header("🧠 KNN Model Trainen")
    
    if st.session_state.X_train is None:
        st.warning("⚠️ Laad eerst een dataset!")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("⚙️ Model Parameters")
        
        # K waarde selectie
        k_value = st.slider(
            "🔢 K waarde (aantal buren)", 
            min_value=1, 
            max_value=20, 
            value=3, 
            step=2,
            help="Oneven getallen worden aanbevolen om ties te voorkomen"
        )
        
        # Afstandsmetriek
        distance_metric = st.selectbox(
            "📏 Afstandsmetriek",
            ["euclidean", "manhattan"],
            help="Euclidean (L2) is meest gebruikt, Manhattan (L1) is minder gevoelig voor uitschieters"
        )
        
        # Train knop
        if st.button("🚀 Train KNN Model", type="primary"):
            with st.spinner("🧠 Model trainen..."):
                try:
                    # Maak en train model
                    knn = KNNImageClassifier(k=k_value, distance_metric=distance_metric)
                    knn.fit(st.session_state.X_train, st.session_state.y_train)
                    
                    # Sla model op
                    st.session_state.knn_model = knn
                    
                    st.success("✅ Model succesvol getraind!")
                    st.balloons()
                    
                    # Toon model info
                    st.markdown(f"""
                    <div class='success-box'>
                    <h4>🎉 Model Getraind!</h4>
                    <p><strong>K waarde:</strong> {k_value}</p>
                    <p><strong>Afstandsmetriek:</strong> {distance_metric}</p>
                    <p><strong>Training samples:</strong> {len(st.session_state.X_train)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Fout bij trainen: {e}")
    
    with col2:
        st.subheader("💡 Tips")
        st.markdown("""
        **K waarde kiezen:**
        - 🔸 K te klein (1,3): Gevoelig voor ruis
        - 🔸 K te groot (15+): Verliest lokale patronen
        - 🔸 Aanbeveling: Start met 3, probeer 5, 7
        
        **Afstandsmetriek:**
        - 📐 **Euclidean**: Meest gebruikt, goed voor meeste toepassingen
        - 📏 **Manhattan**: Minder gevoelig voor extreme waarden
        """)

def predict_page():
    """Voorspellingen pagina."""
    st.header("🔮 Voorspellingen Maken")
    
    if st.session_state.knn_model is None:
        st.warning("⚠️ Train eerst een KNN model!")
        return
    
    if st.session_state.X_test is None:
        st.warning("⚠️ Geen test data beschikbaar!")
        return
    
    st.subheader("🧪 Test Sample Voorspellingen")
    
    # Aantal samples selecteren
    n_samples = st.slider("Aantal test samples", 1, min(50, len(st.session_state.X_test)), 10)
    
    if st.button("🎯 Maak Voorspellingen", type="primary"):
        with st.spinner("🔮 Voorspellingen maken..."):
            try:
                # Maak voorspellingen
                predictions = st.session_state.knn_model.predict(st.session_state.X_test[:n_samples])
                actual = st.session_state.y_test[:n_samples]
                
                # Bereken accuracy
                accuracy = accuracy_score(actual, predictions)
                
                # Toon accuracy
                st.metric("🎯 Accuracy", f"{accuracy:.2%}")
                
                # Toon resultaten tabel
                results_df = pd.DataFrame({
                    'Sample': range(1, n_samples + 1),
                    'Werkelijk': [st.session_state.class_names[a] if st.session_state.class_names else str(a) for a in actual],
                    'Voorspeld': [st.session_state.class_names[p] if st.session_state.class_names else str(p) for p in predictions],
                    'Correct': ['✅' if p == a else '❌' for p, a in zip(predictions, actual)]
                })
                
                st.dataframe(results_df, use_container_width=True)
                
                # Visualiseer enkele voorbeelden
                if st.checkbox("🖼️ Toon afbeeldingen"):
                    show_samples = min(8, n_samples)
                    cols = 4
                    rows = (show_samples + cols - 1) // cols
                    
                    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
                    axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else []
                    
                    for i in range(show_samples):
                        img = st.session_state.X_test[i].reshape(28, 28)
                        pred_name = st.session_state.class_names[predictions[i]] if st.session_state.class_names else str(predictions[i])
                        actual_name = st.session_state.class_names[actual[i]] if st.session_state.class_names else str(actual[i])
                        
                        axes[i].imshow(img, cmap='gray')
                        color = 'green' if predictions[i] == actual[i] else 'red'
                        axes[i].set_title(f"Pred: {pred_name}\nActual: {actual_name}", color=color, fontsize=9)
                        axes[i].axis('off')
                    
                    # Verberg lege subplots
                    for i in range(show_samples, len(axes)):
                        axes[i].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
            except Exception as e:
                st.error(f"❌ Fout bij voorspellen: {e}")

def evaluate_model_page():
    """Model evaluatie pagina."""
    st.header("📈 Model Evaluatie")
    
    if st.session_state.knn_model is None:
        st.warning("⚠️ Train eerst een KNN model!")
        return
    
    # Evaluatie uitvoeren
    if st.button("📊 Voer Evaluatie Uit", type="primary"):
        with st.spinner("📈 Model evalueren..."):
            try:
                # Gebruik subset voor snellere evaluatie
                n_eval = min(200, len(st.session_state.X_test))
                predictions = st.session_state.knn_model.predict(st.session_state.X_test[:n_eval])
                actual = st.session_state.y_test[:n_eval]
                
                # Bereken metrics
                accuracy = accuracy_score(actual, predictions)
                
                # Toon accuracy
                st.metric("🎯 Overall Accuracy", f"{accuracy:.2%}")
                
                # Classification report
                st.subheader("📋 Classification Report")
                report = classification_report(actual, predictions, output_dict=True)
                
                # Converteer naar DataFrame
                report_df = pd.DataFrame(report).transpose()
                
                # Filter relevante rijen
                metrics_df = report_df.iloc[:-3]  # Exclude accuracy, macro avg, weighted avg
                
                if st.session_state.class_names:
                    metrics_df.index = [st.session_state.class_names[int(i)] if i.isdigit() else i for i in metrics_df.index]
                
                st.dataframe(metrics_df, use_container_width=True)
                
                # Confusion Matrix
                st.subheader("🔥 Confusion Matrix")
                cm = confusion_matrix(actual, predictions)
                
                # Plotly heatmap
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    title="Confusion Matrix",
                    labels=dict(x="Voorspelde Labels", y="Werkelijke Labels"),
                    color_continuous_scale='Blues'
                )
                
                if st.session_state.class_names:
                    unique_labels = sorted(list(set(actual) | set(predictions)))
                    class_labels = [st.session_state.class_names[i] for i in unique_labels]
                    fig.update_layout(
                        xaxis=dict(tickmode='array', tickvals=list(range(len(class_labels))), ticktext=class_labels),
                        yaxis=dict(tickmode='array', tickvals=list(range(len(class_labels))), ticktext=class_labels)
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Per-class accuracy
                st.subheader("📊 Per-Class Performance")
                
                class_accuracies = []
                for i in range(len(np.unique(actual))):
                    mask = actual == i
                    if np.sum(mask) > 0:
                        class_acc = accuracy_score(actual[mask], predictions[mask])
                        class_name = st.session_state.class_names[i] if st.session_state.class_names else str(i)
                        class_accuracies.append({'Klasse': class_name, 'Accuracy': class_acc})
                
                acc_df = pd.DataFrame(class_accuracies)
                
                fig = px.bar(
                    acc_df, 
                    x='Klasse', 
                    y='Accuracy',
                    title="Accuracy per Klasse",
                    color='Accuracy',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Fout bij evaluatie: {e}")

def optimize_k_page():
    """K waarde optimalisatie pagina."""
    st.header("⚙️ K-waarde Optimalisatie")
    
    if st.session_state.X_train is None:
        st.warning("⚠️ Laad eerst een dataset!")
        return
    
    st.markdown("""
    Deze functie test verschillende K-waardes om de optimale waarde te vinden.
    Het gebruikt cross-validation om de beste K te bepalen.
    """)
    
    # Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        k_range = st.slider("K bereik testen", 1, 15, (1, 11))
        k_values = list(range(k_range[0], k_range[1], 2))  # Alleen oneven getallen
    
    with col2:
        sample_size = st.slider("Sample grootte (voor snelheid)", 100, 1000, 500)
    
    if st.button("🔍 Zoek Optimale K", type="primary"):
        with st.spinner("🔍 K-waarde optimalisatie..."):
            try:
                # Maak validatie set
                X_train_sub, X_val, y_train_sub, y_val = train_test_split(
                    st.session_state.X_train, st.session_state.y_train, 
                    test_size=0.2, random_state=42
                )
                
                # Gebruik kleinere subsets voor snelheid
                X_train_sub = X_train_sub[:sample_size]
                y_train_sub = y_train_sub[:sample_size]
                X_val = X_val[:min(200, len(X_val))]
                y_val = y_val[:min(200, len(y_val))]
                
                accuracies = []
                progress_bar = st.progress(0)
                
                for i, k in enumerate(k_values):
                    # Update progress
                    progress_bar.progress((i + 1) / len(k_values))
                    
                    # Train model
                    knn = KNNImageClassifier(k=k)
                    knn.fit(X_train_sub, y_train_sub)
                    
                    # Test op validatie set
                    predictions = knn.predict(X_val)
                    accuracy = accuracy_score(y_val, predictions)
                    accuracies.append(accuracy)
                
                # Vind optimale K
                optimal_idx = np.argmax(accuracies)
                optimal_k = k_values[optimal_idx]
                optimal_accuracy = accuracies[optimal_idx]
                
                # Resultaten tonen
                st.success(f"🎯 Optimale K waarde: {optimal_k} (Accuracy: {optimal_accuracy:.2%})")
                
                # Plot resultaten
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=k_values,
                    y=accuracies,
                    mode='lines+markers',
                    name='Validation Accuracy',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8)
                ))
                
                # Markeer optimale K
                fig.add_trace(go.Scatter(
                    x=[optimal_k],
                    y=[optimal_accuracy],
                    mode='markers',
                    name=f'Optimal K={optimal_k}',
                    marker=dict(size=15, color='red', symbol='star')
                ))
                
                fig.update_layout(
                    title="K-waarde vs Validation Accuracy",
                    xaxis_title="K waarde",
                    yaxis_title="Accuracy",
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Resultaten tabel
                results_df = pd.DataFrame({
                    'K waarde': k_values,
                    'Accuracy': [f"{acc:.2%}" for acc in accuracies]
                })
                
                st.dataframe(results_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Fout bij optimalisatie: {e}")

def main():
    """Hoofdfunctie van de web app."""
    # Initialiseer session state
    init_session_state()
    
    # Toon header
    show_header()
    
    # Sidebar
    show_sidebar()
    
    # Main content
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dataset", "🎨 Visualisatie", "🧠 Training", 
        "🔮 Voorspelling", "📈 Evaluatie", "⚙️ Optimalisatie"
    ])
    
    with tab1:
        load_dataset_page()
    
    with tab2:
        visualize_dataset_page()
    
    with tab3:
        train_model_page()
    
    with tab4:
        predict_page()
    
    with tab5:
        evaluate_model_page()
    
    with tab6:
        optimize_k_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p>🎓 Gemaakt voor AI Lessen - "Beter een K-nearest neighbour dan een verre vriend"</p>
        <p>🤖 Powered by Streamlit & KNN Algorithm</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 