# Blood Donation Campaign Dashboard (Detailed)

## Introduction  
Ce tableau de bord Streamlit est un outil complet pour suivre, analyser et visualiser les données relatives à une campagne de don de sang. Il présente différentes sections qui permettent de comprendre l’engagement des volontaires, d’explorer leur profil démographique et leur éligibilité, et de visualiser l’évolution de la campagne. Le code exploite plusieurs bibliothèques Python pour la manipulation des données, la cartographie, les graphiques et l’interface utilisateur.

---

## Table des matières  
1. Aperçu des sections principales  
2. Structure du projet  
3. Fonctionnalités principales  
4. Installation et exécution  
5. Dépendances  
6. Organisation du code  
7. Méthologie 

---

## Aperçu des sections principales  

Le tableau de bord est organisé en plusieurs sections accessibles via la barre latérale (sidebar) :

1. **Home** :  
   - Présente des informations clés (KPI) sur le nombre total de volontaires, de donneurs éligibles ou non, ainsi que les raisons majeures d’inéligibilité.  
   - Inclut des jauges montrant les pourcentages de différents statuts (ex. taux d’éligibilité, taux d’inéligibilité).  
   - Affiche des cartes et tableaux permettant de voir la distribution géographique (arrondissement, département, région) et un aperçu des données démographiques (répartition par genre, statut matrimonial, etc.).

2. **Donations** :  
   - Regroupe la partie “Registration”, où l’on peut renseigner de nouveaux donneurs via un formulaire. Les champs du formulaire incluent :  
     - Numéro d’enregistrement (génération d’un ID unique)  
     - Informations démographiques : date de naissance, âge calculé, niveau d’étude, profession, etc.  
     - Informations médicales : taux d’hémoglobine, éventuelle date du dernier don, etc.  
     - Vérification automatique de l’éligibilité via la fonction `check_eligibility`.  
   - Propose un bouton pour afficher le nouveau dataset qui inclut les enregistrements ajoutés.

3. **Campaign Insights** :  
   - Montre l’évolution temporelle (mensuelle, journalière) du nombre de donneurs.  
   - Visualise les raisons d’inéligibilité (temporaire ou définitive) via des nuages de mots et des diagrammes.  
   - Propose des graphiques radar, des line charts et heatmaps pour analyser la saisonnalité et la progression de la campagne.

4. **Eligibility and Profile** :  
   - Analyse les différents profils de donneurs et leur éligibilité.  
   - Met en avant l’“Ideal Donor” avec des critères clés (taux d’hémoglobine, âge, etc.).  
   - Fournit des indicateurs sur les catégories “Eligible”, “Temporairement Non-Eligible” et “Définitivement Non-Eligible”.

5. **Dataset Insights** :  
   - Propose des analyses plus détaillées sur la base de données :  
     - Répartition par niveau d’étude, profession, secteur, genre.  
     - Distribution de l’âge (histogramme, pyramide des âges).  
     - Visualisation de l’éligibilité au don selon différents filtres (arrondissement, genre, etc.).  
   - Permet de filtrer selon l’arrondissement de résidence ou l’état d’éligibilité.

6. **Cartography** :  
   - Affiche une carte interactive, centrée sur le Cameroun, et utilise Folium pour :  
     - Les “marqueurs” représentant les quartiers où résident les donneurs.  
     - Un mode “Choropleth” permettant d’observer la densité de volontaires par région, département ou arrondissement.  
   - Propose des tooltips indiquant le nom de la zone et le nombre de volontaires.

7. **Options** :  
   - Permet de personnaliser le thème (couleurs principales, arrière-plan, police) via un système de configuration Streamlit.  
   - Gère le choix de la langue (variable `st.session_state.language`) avec un fichier JSON `langage.json` qui contient les traductions.

8. **About** :  
   - Section prévue pour présenter des informations supplémentaires sur la campagne ou l’équipe.

---

## Structure du projet  
Voici un aperçu de la structure standard :  

```
.
├── es.py                        # Script principal du tableau de bord
├── functions.py                # Fonctions utilitaires (ex. check_eligibility)
├── langage.json                # Fichier JSON pour les traductions multilingues
├── donnees.xlsx                # Fichier Excel principal de la base de données
├── last.xlsx                   # Fichier Excel supplémentaire (historique)
├── gadm41_CMR_*.shp            # Shapefiles pour la cartographie
├── logo.jpeg                   # Logo utilisé dans la sidebar
├── requirements.txt            # Liste des dépendances Python
└── README.md                   # Documentation (présent fichier)
```

---

## Fonctionnalités principales

1. **Gestion de la base de données** :  
   - Lecture et fusion de deux fichiers Excel (`donnees.xlsx` et `last.xlsx`).  
   - Possibilité d’ajouter de nouveaux enregistrements via un formulaire Streamlit.  
   - Mise à jour automatique de la base de données dès l’ajout d’un nouveau donneur.

2. **Visualisations avancées** :  
   - Usage de Plotly (heatmaps, line charts, radar charts, etc.).  
   - Usage de Folium/Streamlit-Folium pour la cartographie.  
   - Usage de wordcloud pour la visualisation textuelle des raisons d’inéligibilité.

3. **Filtres dynamiques** :  
   - Filtrage par arrondissement, par date, par éligibilité, etc.  
   - Affichage dynamique des informations selon les filtres choisis.

4. **Multilingue et thèmes** :  
   - Choix entre plusieurs langues (via `langage.json`).  
   - Personnalisation de l’interface avec des thèmes prédéfinis (Clair, Sombre, Bleu).

5. **Analyse de l’éligibilité** :  
   - Vérification automatique basée sur plusieurs critères médicaux et biologiques (allaitement, date du dernier don, infection, etc.).  
   - Séparation des résultats en “Eligible”, “Temporairement Non-eligible” et “Définitivement Non-eligible”.

---

## Installation et exécution

1. **Cloner le dépôt ou télécharger les fichiers**.  
2. **Installer les dépendances** :  
   ```bash
   pip install -r requirements.txt
   ```  
3. **Lancer le tableau de bord** :  
   ```bash
   streamlit run es.py
   ```

---

## Dépendances  

Le fichier `requirements.txt` contient les dépendances exactes utilisées dans le projet :  

- **Streamlit** pour l’interface utilisateur.  
- **Pandas**, **GeoPandas**, **Numpy** pour la manipulation de données.  
- **Plotly** pour les graphiques interactifs.  
- **Folium** et **streamlit-folium** pour la cartographie.  
- **Openpyxl** pour la lecture/écriture de fichiers Excel.  
- **Geopy** pour la géolocalisation.  
- **Pillow** pour la gestion des images.  
- **streamlit-extras** et **echarts-streamlit** pour certaines fonctionnalités additionnelles.

---

## Organisation du code  

- `es.py` :  
  - Initialise la configuration de la page (titre, icône, etc.).  
  - Gère la navigation via la barre latérale (Home, Donations, Cartography, etc.).  
  - Charge les datasets et les fusionne.  
  - Crée et affiche les différents composants du tableau de bord (cartes, graphiques, formulaires).

- `functions.py` (éventuel) :  
  - Contient les fonctions utilitaires (ex. `check_eligibility()`) pour calculer l’éligibilité en se basant sur les champs médicaux.

- `langage.json` :  
  - Fichier de traduction pour gérer l’affichage multilingue.

---

## Méthodologie 

Le programme développé permet de déterminer l’éligibilité d’un donneur de sang en fonction de critères médicaux et des conditions de santé spécifiées. Le processus repose sur l'extraction et l'analyse de données fournies par le donneur, telles que l'âge, le poids, les antécédents médicaux et les informations relatives aux dons précédents. L'approche suit plusieurs étapes :

**Extraction des données pertinentes** :  
   Le programme commence par extraire les informations essentielles du jeu de données. Il s'agit des données relatives à l'âge, au poids, au genre, à la présence d'éléments médicaux spécifiques (par exemple, taux d'hémoglobine) et aux antécédents de don de sang. Ces informations permettent d’évaluer si le donneur répond aux critères de santé requis.

**Vérification des critères de base** :  
   - L'âge et le poids sont les premiers critères vérifiés. Le donneur doit être âgé de 18 à 70 ans et peser au minimum 50 kg pour être éligible au don. Si l'un de ces critères est non respecté, le donneur est immédiatement considéré comme *non éligible* de façon permanente.
   
**Vérification du taux d'hémoglobine** :  
   Le taux d'hémoglobine est une donnée importante pour évaluer la capacité d’un donneur à supporter un prélèvement de sang. Les seuils varient en fonction du genre (13 g/dL pour les hommes et 12 g/dL pour les femmes). Si le taux d’hémoglobine est trop bas, le donneur est temporairement non éligible.

**Vérification de la date du dernier don** :  
   Le programme prend en compte le délai depuis le dernier don effectué par le donneur. Si le donneur a effectué un don moins de 56 jours auparavant (soit 8 semaines), il sera temporairement non éligible. Ce critère est fondé sur les recommandations des autorités sanitaires.

**Vérification des contre-indications permanentes** :  
   Certaines conditions médicales sont des contre-indications permanentes au don de sang. Ces conditions sont spécifiées sous forme de colonnes dans les données (par exemple, drépanocytose, VIH, hépatites, diabète, maladie cardiaque). Si le donneur présente l'une de ces conditions, il est définitivement *non éligible*.

**Vérification des contre-indications temporaires** :  
   Le programme identifie également des contre-indications temporaires, telles que la grossesse, l’allaitement, la présence d’une infection récente, ou la prise de certains médicaments (comme les antibiotiques). Si l’une de ces contre-indications est présente, le donneur sera temporairement *non éligible*.

**Détermination de l'éligibilité finale** :  
   Une fois toutes les conditions vérifiées, l'éligibilité finale est déterminée. Si des raisons permanentes d’inéligibilité sont trouvées, le donneur est classé comme *définitivement non éligible*. Si des raisons temporaires sont détectées, il sera classé comme *temporairement non éligible*. Si aucune des conditions défavorables n'est remplie, le donneur est considéré comme *éligible*.

**Retour du résultat** :  
   Le programme retourne un dictionnaire contenant l'éligibilité du donneur ainsi que les raisons détaillées pour lesquelles il est éligible ou non éligible. Cela permet de fournir une transparence sur les critères utilisés pour évaluer chaque cas.

Ainsi, ce programme est un outil pratique et automatisé pour faciliter l’évaluation de l’éligibilité des donneurs de sang en fonction de critères médicaux, tout en assurant la conformité avec les exigences sanitaires.

Il est important de noter que des méthodes telles que le web scraping a été utilisée notamment pour collecter les donnée notamments sur la localisation des quartiers et des arrondissements de résidence
---

## Conclusion  
Ce tableau de bord centralise toutes les informations nécessaires pour suivre la progression d’une campagne de don de sang. Il permet d’examiner la démographie des donneurs, leur éligibilité, leur localisation, ainsi que l’efficacité et les tendances de la campagne. Grâce à ses filtres et visualisations interactives, il constitue un outil puissant d’aide à la décision pour les responsables de campagne et les parties prenantes.
