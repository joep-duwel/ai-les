# KNN Afbeeldings Classifier 📸🤖

**"Beter een K-nearest neighbour dan een verre vriend"**

Een complete implementatie van een K-Nearest Neighbors classifier voor afbeeldings classificatie, gebouwd voor de AI Lessen opdracht.

## 🚀 Hoe de App Starten

### 🌐 Web App - Moderne Interface (AANBEVOLEN!)

#### Stap 1: Installeer Dependencies
```bash
pip install -r requirements.txt
```

#### Stap 2: Start de Web App
- **Windows**: Dubbelklik `START_WEB_APP.bat`
- **Handmatig**: `python start_web_app.py`
- **Direct**: `streamlit run knn_web_app.py`

**De app opent automatisch in je browser op http://localhost:8501 🌐**

### 💻 Terminal App - Klassieke Interface

```bash
python knn_app_ui.py
```

**Beide versies hebben dezelfde functies - kies wat je prefereert! 🎉**

## 📁 Project Structuur

```
📦 KNN Image Classifier
├── 🌐 knn_web_app.py             # WEB APP - Modern & Interactief!
├── 💻 knn_app_ui.py              # Terminal app - Klassiek
├── 🧠 knn_image_classifier.py    # KNN implementatie
├── 📊 batch_image_tester.py      # Verdiepende opdracht
├── 🚀 start_web_app.py           # Web app starter
├── 📝 START_WEB_APP.bat          # Windows web starter
├── 📝 START_KNN_APP.bat          # Windows terminal starter  
├── 📋 requirements.txt           # Python dependencies
└── 📖 README.md                  # Deze handleiding
```

## 🎯 Wat Kan de App?

### ✅ Hoofdfuncties (Vereist voor Opdracht)
1. **📊 Dataset Laden**: MNIST Cijfers of Fashion-MNIST
2. **🧠 Model Trainen**: KNN classifier met instelbare K-waarde
3. **🔮 Voorspellingen**: Classificeer nieuwe afbeeldingen
4. **📈 Evaluatie**: Uitgebreide prestatie analyses
5. **🎨 Visualisatie**: Bekijk dataset voorbeelden
6. **⚙️ K-waarde Optimalisatie**: Vind de beste K automatisch
7. **ℹ️ Informatie**: Leer alles over KNN algoritme

### 🌟 Extra Features (Verdiepende Opdracht)
- **📁 Batch Testing**: Test een hele map met afbeeldingen
- **🏷️ Automatische Naamherkenning**: Labels uit bestandsnamen
- **📊 Accuracy Berekening**: Gebaseerd op bestandsnamen

## 🎮 Hoe Te Gebruiken

### 🌐 Web App - Voor Beginners (AANBEVOLEN):
1. **Start**: Dubbelklik `START_WEB_APP.bat` 
2. **Dataset**: Ga naar "📊 Dataset" tab → klik "📊 Laad MNIST Cijfers"
3. **Training**: Ga naar "🧠 Training" tab → klik "🚀 Train KNN Model" (K=3)
4. **Voorspelling**: Ga naar "🔮 Voorspelling" tab → maak voorspellingen
5. **Evaluatie**: Bekijk "📈 Evaluatie" tab voor prestaties

### 💻 Terminal App - Klassieke Interface:
1. Start: `python knn_app_ui.py`
2. Kies **optie 1** om MNIST cijfers te laden
3. Kies **optie 3** om het model te trainen (gebruik K=3)
4. Kies **optie 4** om voorspellingen te maken
5. Kies **optie 5** om resultaten te bekijken

### Voor Gevorderden:
1. Probeer **Fashion-MNIST** voor kledingstukken
2. Gebruik **"⚙️ Optimalisatie"** om de optimale K-waarde te vinden
3. Experiment met verschillende afstandsmeteiken (euclidean/manhattan)
4. Run `python batch_image_tester.py` voor batch testing

## 🧠 Wat is K-Nearest Neighbors?

KNN is een simpel maar krachtig machine learning algoritme:

### 🔍 Hoe Het Werkt:
1. **Afstand Berekenen**: Voor elke nieuwe afbeelding, bereken afstand tot alle trainingsafbeeldingen
2. **Buren Selecteren**: Kies de K dichtstbijzijnde afbeeldingen
3. **Stemmen**: De klasse die het meest voorkomt bij de K buren wint!

### 📏 Afstandsmeteiken:
- **Euclidean (L2)**: `√(sum((x1-x2)²))` - Meest gebruikt
- **Manhattan (L1)**: `sum(|x1-x2|)` - Minder gevoelig voor uitschieters

### ⚡ Voor- en Nadelen:

**✅ Voordelen:**
- Eenvoudig te begrijpen
- Geen training nodig ('lazy learning')
- Werkt goed voor complexe data patronen

**❌ Nadelen:**
- Langzaam bij voorspelling
- Heeft veel geheugen nodig
- Gevoelig voor irrelevante features

## 📊 Datasets

### 🔢 MNIST Cijfers
- 70.000 handgeschreven cijfers (0-9)
- 28x28 pixels, grayscale
- Klassieke benchmark dataset

### 👕 Fashion-MNIST
- 70.000 kledingstukken (10 categorieën)
- 28x28 pixels, grayscale
- Moderne alternative voor MNIST

**Categorieën:**
1. T-shirt/top
2. Trouser (broek)
3. Pullover
4. Dress (jurk)
5. Coat (jas)
6. Sandal
7. Shirt
8. Sneaker
9. Bag (tas)
10. Ankle boot (enkellaars)

## 🎯 Opdracht Checklist

### ✅ Verplichte Onderdelen:
- [x] Python app gemaakt
- [x] KNN algoritme uitgelegd en geïmplementeerd
- [x] Afbeelding classificatie werkend
- [x] Uitgebreide code documentatie
- [x] Eenvoudige UI met menu systeem
- [x] App kan afgesloten worden

### 🌟 Extra Onderdelen:
- [x] Verdiepende opdracht: Batch image testing
- [x] Accuracy berekening op bestandsnamen
- [x] Visualisaties en grafieken
- [x] K-waarde optimalisatie
- [x] Beide datasets ondersteund

## 🚨 Troubleshooting

### Probleem: ModuleNotFoundError
**Oplossing:** Installeer dependencies:
```bash
pip install -r requirements.txt
```

### Probleem: App start niet
**Oplossing:** Check Python versie (3.7+ vereist):
```bash
python --version
python knn_app_ui.py
```

### Probleem: Dataset download lukt niet
**Oplossing:** Check internet verbinding, scikit-learn downloadt automatisch

### Probleem: App is langzaam
**Oplossing:** Dit is normaal! KNN moet alle afstanden berekenen. Gebruik kleinere K-waarden voor snelheid.

## 🎓 Leermateriaal

### KNN Concepten:
- **K-waarde**: Aantal buren om te bekijken (oneven getallen aanbevolen)
- **Overfitting**: K te klein → te gevoelig voor ruis
- **Underfitting**: K te groot → verliest lokale patronen
- **Cross-validation**: Techniek om beste K te vinden

### Machine Learning Begrippen:
- **Training set**: Data om van te leren
- **Test set**: Data om prestaties te meten
- **Features**: Eigenschappen van data (pixels in ons geval)
- **Labels**: Correcte antwoorden (klassen)
- **Accuracy**: Percentage correct geclassificeerd

## 🏆 Tips voor Beste Resultaten

1. **K-waarde**: Begin met 3, probeer 5 en 7
2. **Dataset grootte**: Meer trainingsdata = betere resultaten
3. **Preprocessing**: Normalisatie is al ingebouwd
4. **Afstandsmetriek**: Euclidean werkt meestal het beste
5. **Geduld**: KNN kan langzaam zijn, maar resultaten zijn het waard!

## 🤝 Support

Bij vragen of problemen:
1. Check deze README eerst
2. Bekijk de code commentaren
3. Gebruik de app's ingebouwde help (optie 9)
4. Test met kleine datasets eerst

## 🎉 Veel Plezier!

Deze app demonstreert een van de fundamentele machine learning algoritmes. KNN is misschien simpel, maar het vormt de basis voor veel complexere AI systemen!

**Happy Learning! 🚀📚** 