import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
from collections import Counter
import plotly.graph_objects as go
from streamlit_echarts import st_echarts
from PIL import Image
from wordcloud import WordCloud
from collections import Counter
from pyecharts import options as opts
from pyecharts.charts import Bar
from streamlit_echarts import st_pyecharts
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
from io import BytesIO
import random
import math
import warnings
from wordcloud import WordCloud
import nltk
from wordcloud import STOPWORDS
from nltk.corpus import stopwords
from wordcloud import STOPWORDS
from nltk.corpus import stopwords
import string,re
from PIL import Image, ImageDraw

@st.cache_resource
def check_eligibility(donor_data):
    """
    Détermine l'éligibilité d'un donneur de sang selon les critères médicaux.
    
    Retourne une des valeurs suivantes:
    - 'Eligible': Le donneur remplit tous les critères
    - 'Temporarily Not Eligible': Le donneur est temporairement non éligible
    - 'Not Eligible': Le donneur est définitivement non éligible
    """
    
    # Extraction des données pertinentes pour l'évaluation
    age = donor_data['age'].iloc[0]
    poids = donor_data['poids'].iloc[0]
    date_jour = datetime.now().date()
    
    # Initialisation des variables pour les raisons d'inéligibilité
    permanent_reasons = []
    temporary_reasons = []
    
    # Vérification des critères de base (âge et poids)
    if age < 18 or age > 70:
        permanent_reasons.append(f"Âge non conforme ({age} ans)")
    
    if poids < 50:
        permanent_reasons.append(f"Poids insuffisant ({poids} kg)")
    
    # Vérification du taux d'hémoglobine
    if 'taux_dhemoglobine' in donor_data.columns and not pd.isna(donor_data['taux_dhemoglobine'].iloc[0]):
        taux_hb = donor_data['taux_dhemoglobine'].iloc[0]
        genre = donor_data['genre'].iloc[0]
        
        # Seuils généralement admis (peuvent varier selon les pays)
        if (genre.lower() == 'homme' or genre.lower() == 'masculin') and taux_hb < 13:
            temporary_reasons.append(f"Taux d'hémoglobine bas ({taux_hb} g/dL)")
        elif (genre.lower() == 'femme' or genre.lower() == 'féminin') and taux_hb < 12:
            temporary_reasons.append(f"Taux d'hémoglobine bas ({taux_hb} g/dL)")
    
    # Vérification du dernier don
    if donor_data['deja_donne_sang'].iloc[0] == 'Yes' and donor_data['date_dernier_don'].iloc[0]:
        try:
            date_dernier_don = donor_data['date_dernier_don'].iloc[0]  # Assuming this is already a datetime.date object

            delai_depuis_dernier_don = (date_jour - date_dernier_don).days
            
            if delai_depuis_dernier_don < 56:  # 8 weeks = 56 days
                temporary_reasons.append(f"Dernier don trop récent ({delai_depuis_dernier_don} jours)")
        except ValueError:
            # Si la date n'est pas dans le bon format, on ne peut pas vérifier
            pass
    
    # Vérification des contre-indications permanentes
    permanent_conditions = [
        ('drepanocytaire', "Drépanocytose"),
        ('porteur_hiv_hbs_hcv', "Porteur VIH, hépatite B ou C"),
        ('diabetique', "Diabète"),
        ('cardiaque', "Maladie cardiaque"), 
        ('opere', "Opération récente"),
        ('tatoue', "Tatouage récent"),
        ('scarifie', "Scarification récente"),
    ]
    
    for col, reason in permanent_conditions:
        if col in donor_data.columns and donor_data[col].iloc[0] == 'Yes':
            permanent_reasons.append(reason)
    
    # Vérification des contre-indications temporaires
    temporary_conditions = [
        ('est_sous_anti_biotherapie', "Sous antibiothérapie"),
        ('taux_dhemoglobine_bas', "Taux d'hémoglobine bas"),
        ('date_dernier_don_3_mois', "Don récent (moins de 3 mois)"),
        ('ist_recente', "IST récente"),
        ('ddr_incorrecte', "DDR incorrecte"),
        ('allaitement', "Allaitement"),
        ('accouchement_6mois', "Accouchement récent (moins de 6 mois)"),
        ('interruption_grossesse', "Interruption de grossesse récente"),
        ('enceinte', "Grossesse en cours"),
        ('antecedent_transfusion', "Transfusion récente"),
        ('hypertendus', "Hypertension non contrôlée"),
        ('asthmatiques', "Crise d'asthme récente")
    ]
    
    for col, reason in temporary_conditions:
        if col in donor_data.columns and donor_data[col].iloc[0] == 'Yes':
            temporary_reasons.append(reason)
    
    # Détermination de l'éligibilité finale
    if permanent_reasons:
        eligibility = 'Définitivement non-eligible'
        reasons = permanent_reasons
    elif temporary_reasons:
        eligibility = 'Temporairement Non-eligible'
        reasons = temporary_reasons
    else:
        eligibility = 'Eligible'
        reasons = []
    
    return {
        'eligibility': eligibility,
        'reasons': reasons
    }


@st.cache_resource
def message(text):
    st.markdown(
        f"""
        <div class="celebration-container">
            <h1 class="celebration-text">{text}</h1>
        </div>
        
        <style>
            .celebration-container {{
                text-align: center;
                padding: 10px;
                position: relative;
                overflow: hidden;
                height: 200px;
            }}

            .celebration-text {{
                font-size: 4em;
                background: linear-gradient(45deg, #ff0000, #ff7700, #ffff00, #00ff00, #0000ff, #8b00ff);
                background-size: 600% 600%;
                animation: gradient 6s ease infinite, bounce 2s ease infinite;
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            }}

            .fireworks {{
                position: absolute;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background-color: #ffcc00;
                animation: fireworks 2s ease infinite;
                opacity: 0;
                left: 50%;
                bottom: 0;
            }}

            @keyframes gradient {{
                0% {{ background-position: 0% 50%; }}
                50% {{ background-position: 100% 50%; }}
                100% {{ background-position: 0% 50%; }}
            }}

            @keyframes bounce {{
                0%, 100% {{ transform: translateY(0); }}
                50% {{ transform: translateY(-20px); }}
            }}

            @keyframes fireworks {{
                0% {{
                    transform: translate(50%, 100%) scale(1);
                    opacity: 1;
                }}
                25% {{
                    opacity: 1;
                }}
                100% {{ 
                    transform: translate(var(--x), var(--y)) scale(3);
                    opacity: 0;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts

@st.cache_resource(show_spinner=False)
def compute_age_distribution(age_data):
    """
    Computes age distribution for an ECharts bar chart.
    
    Parameters:
    age_data (pd.Series or pd.DataFrame): Series containing age values
    
    Returns:
    dict: ECharts option dictionary for plotting
    """
    # Convert input to hashable format for proper caching
    if isinstance(age_data, pd.DataFrame):
        if len(age_data.columns) == 1:
            age_data = age_data.iloc[:, 0].values.tolist()
        else:
            age_data = str(age_data)  # Convert to string representation for caching
    elif isinstance(age_data, pd.Series):
        age_data = age_data.values.tolist()

    age_data_processed = pd.Series(age_data) if not isinstance(age_data, str) else pd.Series([])

    # Define age categories
    bins = [0, 17, 25, 35, 50, float('inf')]
    labels = ["Moins de 18 ans", "18-25 ans", "26-35 ans", "36-50 ans", "Plus de 50 ans"]

    # Categorize ages
    age_categories = pd.cut(age_data_processed, bins=bins, labels=labels, right=True)

    # Count occurrences in each category
    age_frequencies = age_categories.value_counts().reindex(labels, fill_value=0).tolist()

    # Define the ECharts option
    option = {
        "tooltip": {"trigger": "axis", "formatter": "{b}: {c}"},
        "xAxis": {
            "type": "category",
            "data": labels,
            "name": "Age Group",
            "nameLocation": "middle",
            "nameGap": 35,
            "axisLabel": {"rotate": 0}
        },
        "yAxis": {"name": "Number of People"},
        "series": [
            {
                "name": "Count",
                "type": "bar",
                "data": age_frequencies,
                "barWidth": "90%",
                "itemStyle": {
                    "color": "rgba(99, 110, 250, 0.8)",
                    "borderColor": "rgba(50, 50, 150, 1)",
                    "borderWidth": 1,
                    "borderRadius": [3, 3, 0, 0]
                },
                "label": {"show": True, "position": "top", "formatter": "{c}"}
            }
        ],
        "grid": {"top": "10%", "bottom": "15%", "left": "8%", "right": "5%", "containLabel": True},
        "toolbox": {"feature": {"saveAsImage": {}, "dataZoom": {}, "restore": {}}, "right": "10%"}
    }
    
    return option




@st.cache_resource
def prepare_radar_chart_options(data, categories, title=""):
    """
    Prépare les options pour le graphique radar et les retourne sans les afficher.
    Cette fonction peut être mise en cache en toute sécurité.
    """
    # Define radar chart indicators with auto-adjusted max values
    max_value = max([max(values) for values in data.values()]) * 1.2
    indicators = [{"name": cat, "max": max_value} for cat in categories]

    # Prepare data for ECharts
    series_data = [
        {
            "name": key,
            "value": value,
            "areaStyle": {"opacity": 0.5},  # Add filled color
            "itemStyle": {"color": "red"} 
        }
        for key, value in data.items()
    ]

    # Define ECharts options
    option = {
       "title": {"text": title, "textStyle": {"fontSize": 18, "color": "#333"}},
        "tooltip": {"trigger": "item"},
        "legend": {"data": list(data.keys()), "left": "14%"},
        "radar": {
            "indicator": indicators,
            "shape": "circle",  # Make radar circular
            "splitNumber": 5,  # Reduced number of grid levels for more spacing
            "axisName": {"color": "#333"},  # Category label color
            "splitLine": {
                "lineStyle": {
                    "type": "solid",  # Changed to dashed for better visibility
                    "width": 2,
                    "dashOffset": 0,  # Add dash offset
                    "cap": "round"
                }
            },  # Grid lines
            "splitArea": {
                "show": False,
                "areaStyle": {
                    "color": ["red", "white"]
                }  # Alternating background colors
            },
        },
        "series": [
            {
                "name": title,
                "type": "radar",
                "data": series_data,
                "tooltip": {
                    "show": True
                }
            }
        ]
    }
    
    return option

def plot_radar_chart(data, categories, title="", height="400px"):
    """
    Fonction de rendu qui affiche le graphique en utilisant les options préparées
    par la fonction mise en cache.
    """
    # Obtenir les options depuis la fonction mise en cache
    options = prepare_radar_chart_options(data, categories, title)
    
    # Rendre le graphique (cette partie ne peut pas être mise en cache)
    st_echarts(options=options, height=height, width="100%")
df = pd.read_excel('last.xlsx')
#df['Date de remplissage de la fiche'] = pd.to_datetime(df['Date de remplissage de la fiche'])
# m= df['Date de remplissage de la fiche'].dt.month
# month_counts = m.value_counts().sort_index()
# categories = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
# data = {"Nombre d'occurences": month_counts.reindex(range(1, 13), fill_value=0).tolist()}
# plot_radar_chart(data, categories)


from collections import Counter
import streamlit as st
from streamlit_echarts import st_echarts

# Fonctions de préparation (mises en cache)
@st.cache_resource
def prepare_frequency_pie_options_wl(data_list, lege):
    """
    Prépare les options pour le graphique pie et les retourne sans les afficher.
    """
    # Count the frequency of each value
    frequency = Counter(data_list)
    
    # Convert to the format needed for ECharts and sort by value in descending order
    pie_data = [{"name": str(key), "value": value} for key, value in sorted(frequency.items(), key=lambda x: x[1], reverse=True)]
    
    # Define pie chart options
    pie_options = {
        "tooltip": {
            "trigger": "item",
            "formatter": "{b} : <br/>{c} occurences ({d}%)"
        },
        "legend": {
            "show": True,
            "orient": "vertical",
            "left": "70%",
            "top": "60%"
        },
        "series": [
            {
                "type": "pie",
                "radius": "95%",
                "center": ["52%", "40%"],  # Position the pie [x, y]
                "data": pie_data,
                "roseType": "radius",
                "label": {
                    "show": False  # Remove text labels on pie segments
                },
                "labelLine": {
                    "show": False  # Hide label lines
                },
                "itemStyle": {
                    "shadowBlur": 30,
                    "shadowColor": "rgba(0, 0, 0, 0.5)",
                },
                "animationType": "scale",
                "animationEasing": "elasticOut",
            }
        ],
        # Set a colorful palette for the pie chart
        "color": ["#FA668F", "#B22222", "#DC143C", "#EC8282", "#FFB6B6", "#FFD1D1"]
    }
    
    return pie_options

@st.cache_resource
def prepare_frequency_pie_options(data_list, legend_left="70%", legend_top="50%"):
    """
    Prépare les options pour le graphique pie standard et les retourne sans les afficher.
    """
    # Count the frequency of each value
    frequency = Counter(data_list)
    
    # Convert to the format needed for ECharts and sort by value in descending order
    pie_data = [{"name": str(key), "value": value} for key, value in sorted(frequency.items(), key=lambda x: x[1], reverse=True)]
    
    # Define pie chart options
    pie_options = {
        "tooltip": {
            "trigger": "item",
            "formatter": "<span style='font-size:11px; font-weight:bold;'>{b} :</span> <br/>"
                     "<span style='font-size:12px;'>{c} occurrences ({d}%)</span>"
        },
        "legend": {
            "show": True,
            "orient": "vertical",
            "left": legend_left,
            "top": legend_top,
            "textStyle": {
                "fontSize": 11, # Optional: Makes the text bold
                "color": "#333"  # Optional: Sets the text color
            }
        },
        "series": [
            {
                "type": "pie",
                "radius": "105%",
                "center": ["60%", "55%"],  # Position the pie [x, y]
                "data": pie_data,
                "roseType": "radius",
                "label": {
                    "show": False  # Remove text labels on pie segments
                },
                "labelLine": {
                    "show": False # Hide label lines
                },
                "itemStyle": {
                    "shadowBlur": 30,
                    "shadowColor": "rgba(0, 0, 0, 0.5)",
                },
                "animationType": "scale",
                "animationEasing": "elasticOut",
            }
        ],
        # Set a colorful palette for the pie chart
        "color": ["#FA668F",   "SlateBlue", "SandyBrown", "pink", "cyan","green"]
    }
    
    return pie_options

@st.cache_resource
def prepare_frequency_pieh_options(data_list, legend_left="70%", legend_top="85%"):
    """
    Prépare les options pour le graphique pie avec trou (donut) et les retourne sans les afficher.
    """
    # Count the frequency of each value
    frequency = Counter(data_list)

    # Convert to the format needed for ECharts and sort by value in descending order
    pie_data = [{"name": str(key), "value": value} for key, value in sorted(frequency.items(), key=lambda x: x[1], reverse=True)]

    # Define pie chart options
    pie_options = {
        "tooltip": {
            "trigger": "item",
            "formatter": "{b} : <br/>{c} occurrences ({d}%)"
        },
        "legend": {
            "show": True,
            "orient": "vertical",
            "left": legend_left,
            "top": legend_top
        },
        "series": [
            {
                "type": "pie",
                "radius": ["25%", "70%"],  # Creates a donut effect
                "center": ["50%", "48%"],  # Position of the pie chart
                "data": pie_data,
                "roseType": "radius",
                "label": {
                    "show": True  # Show labels on pie segments
                },
                "labelLine": {
                    "show": True  # Show label lines
                },
                "itemStyle": {
                    "shadowBlur": 30,
                    "shadowColor": "rgba(0, 0, 0, 0.5)",
                },
                "animationType": "scale",
                "animationEasing": "elasticOut",
            }
        ],
        # Set a colorful palette for the pie chart
        "color": ["#FA668F",  "SlateBlue", "tomato","SandyBrown", "pink", "#FF3434","cyan","skyblue"]
    }

    return pie_options

# Fonctions de rendu (non mises en cache)
def render_frequency_pie_wl(data_list, lege):
    """
    Fonction de rendu pour graphique pie type waterloo
    """
    options = prepare_frequency_pie_options_wl(data_list, lege)
    st_echarts(options=options, height="400px")

def render_frequency_pie(data_list, legend_left="70%", legend_top="50%"):
    """
    Fonction de rendu pour graphique pie standard
    """
    options = prepare_frequency_pie_options(data_list, legend_left, legend_top)
    st_echarts(options=options, height="400px")

def render_frequency_pieh(data_list, legend_left="70%", legend_top="65%", key=None):
    """
    Fonction de rendu pour graphique pie avec trou (donut)
    """
    options = prepare_frequency_pieh_options(data_list, legend_left, legend_top)
    st_echarts(options=options, height="350px", key=key)
#render_frequency_pie2(pd.read_excel("last.xlsx")["Niveau_d'etude"])


@st.cache_data
def count_frequencies(data_list):
    """
    Count the frequency of each value in the data list.

    Parameters:
    - data_list: List of values to count frequencies.

    Returns:
    - List of dictionaries with names and values for ECharts.
    """
    frequency = Counter(data_list)
    return [{"name": str(key), "value": value} for key, value in sorted(frequency.items(), key=lambda x: x[1], reverse=True)]

def render_frequency_pie2(pie_data, legend_left="23%", legend_top="80%"):
    """
    Render a pie chart with a hole (donut chart) using ECharts in Streamlit.

    Parameters:
    - pie_data: List of dictionaries with names and values for ECharts.
    - legend_left: Horizontal position of the legend (default: "75%").
    - legend_top: Vertical position of the legend (default: "50%").
    """
    # Define pie chart options
    pie_options = {
        "tooltip": {
            "trigger": "item",
            "formatter": "{b} : <br/>{c} Values<br/>({d}%)"
        },
        "legend": {
            "show": True,
            "orient": "vertical",
            "left": legend_left,
            "top": legend_top
        },
        "series": [
            {
                "type": "pie",
                "radius": ["25%", "70%"],  # Creates a donut effect
                "center": ["50%", "45%"],  # Position of the pie chart
                "data": pie_data,
                "roseType": "radius",
                "label": {
                    "show": True  # Show labels on pie segments
                },
                "labelLine": {
                    "show": True  # Show label lines
                },
                "itemStyle": {
                    "shadowBlur": 30,
                    "shadowColor": "rgba(0, 0, 0, 0.5)",
                },
                "animationType": "scale",
                "animationEasing": "elasticOut",
            }
        ],
        # Set a colorful palette for the pie chart
        "color": ["#FA668F",  "SlateBlue", "tomato","SandyBrown", "pink", "#FF3434","cyan","skyblue"]
    }

    # Render the chart in Streamlit
    st_echarts(options=pie_options, height="300px")

@st.cache_resource
def plot_age_pyramid(df, age_col='Age', gender_col='Genre_', bin_size=4, height=600):
    """
    Plots an age pyramid using Plotly in Streamlit.

    Parameters:
    df (DataFrame): The dataset containing age and gender columns.
    age_col (str): Column name for age data.
    gender_col (str): Column name for gender data.
    bin_size (int): Size of age bins.
    height (int): Height of the figure in pixels.

    Returns:
    None (Displays the plot in Streamlit).
    """
    # Define age bins
    bin_edges = np.arange(df[age_col].min(), df[age_col].max() + bin_size, bin_size)
    bin_labels = [f"{int(age)}-{int(age + bin_size - 1)}" for age in bin_edges[:-1]]

    # Assign age groups
    df['AgeGroup'] = pd.cut(df[age_col], bins=bin_edges, labels=bin_labels, right=False)

    # Group by age group and gender
    hommes = df[df[gender_col] == 'Homme'].groupby('AgeGroup').size().reset_index(name='Count')
    femmes = df[df[gender_col] == 'Femme'].groupby('AgeGroup').size().reset_index(name='Count')

    # Assign gender labels
    hommes['Genre'] = 'Hommes'
    femmes['Genre'] = 'Femmes'

    # Concatenate data
    data = pd.concat([hommes, femmes])

    # Invert female values for visualization
    data.loc[data['Genre'] == 'Femmes', 'Count'] *= -1

    # Create the age pyramid plot
    fig = px.bar(data, 
                 y='AgeGroup', 
                 x='Count', 
                 color='Genre', 
                 orientation='h',
                 labels={'Count': 'Number of people ', 'AgeGroup': 'Âge'},
                 color_discrete_map={'Hommes': '#DC143C', 'Femmes':  'SlateBlue'})

    # Customize layout
    fig.update_layout(
        height=height,  # Set the figure height
        yaxis=dict(showgrid=True, categoryorder="category ascending"),
        xaxis=dict(title='Nombre de personnes', showgrid=True),
        plot_bgcolor='rgba(0,0,0,0)',
        barmode='overlay',
        bargap=0.1,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    st.plotly_chart(fig, use_container_width=True)

@st.cache_resource
def number_line(df):

    nombre_total_donneurs = df.shape[0]

    a = df['Date de remplissage de la fiche'].dropna()

    # Convertir en datetime avec gestion des erreurs
    dates = pd.to_datetime(a, format="%m/%d/%Y %H:%M", errors='coerce').dropna()

    a = dates.value_counts()
    a.sort_index(inplace=True)
    #a = a[(a.index >= "2000-01-01") & (a.index <= "2022-12-31")]

    # Make the line thinner (reduced from 3 to 1.5)
    fig = px.line(a, x=a.index, y=a.values, labels={'x': 'Années', 'y': 'Nombre d\'occurrences'}, line_shape='spline')
    fig.update_traces(line_color='#f83e8c', line_width=1.5, 
                      fill='tozeroy', fillcolor='rgba(232, 62, 140, 0.1)') 
    
    # Enhance the layout
    fig.update_layout(
        height=400, 
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                thickness=0.05,
                bgcolor='rgba(0,0,0,0.05)',
            ),
            rangeslider_bordercolor='#aaaaaa', 
            rangeslider_borderwidth=1,
            showgrid=True,
            gridcolor='rgba(211,211,211,0.3)',
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(211,211,211,0.3)',
        ),
    )

    st.plotly_chart(fig, use_container_width=True)
@st.cache_data
def generate_wordcloud_and_barchart(df):
    # Extract words related to donation reasons
    words = " ".join(df["Raison_indisponibilité_fusionnée"].dropna().astype(str).tolist())

    # Compute word frequencies
    word_list = words.split()
    word_freq = {}
    for word in word_list:
        word_freq[word] = word_freq.get(word, 0) + np.random.randint(10, 50)  # Simulated frequencies

    # Boost specific keywords
    key_words = ["sang", "donneur", "vie", "santé", "sauver", "solidarité", "don"]
    for word in key_words:
        if word in word_freq:
            word_freq[word] *= 3  # Increase importance

    # Define a good color palette (blood and health theme)
    colors = ["#8B0000", "#B22222", "#DC143C", "#E9967A", "#FF6347", "#FFA07A"]

    # Define a color function for the WordCloud
    
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return np.random.choice(colors)

    # Create a circular mask
    x, y = np.ogrid[:300, :300]
    mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
    mask = 255 * mask.astype(int)

    # Generate the word cloud
    wc = WordCloud(
        background_color="white",
        max_words=150,
        mask=mask,
        color_func=color_func,  # Use custom colors
        max_font_size=80,
        random_state=42,
        width=800,
        height=800,
        contour_width=1.5,
        contour_color='#8B0000'  # Dark red contour
    ).generate_from_frequencies(word_freq)

    # Convert word cloud image to base64
    img = wc.to_image()
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Create a Plotly figure with the word cloud image
    fig = go.Figure()

    # Add image as a background
    fig.add_layout_image(
        dict(
            source='data:image/png;base64,' + img_str,
            x=0,
            y=1,
            xref="paper",
           
            sizex=1,
            sizey=1,
        )
    )

    # Configure layout for better visibility
    fig.update_layout(
        width=400,
        height=300,
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis={'showgrid': False, 'showticklabels': False, 'zeroline': False},
        yaxis={'showgrid': False, 'showticklabels': False, 'zeroline': False},
        margin=dict(l=140, r=0, t=0, b=0, pad=0)  # Remove all margins and padding

       
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Example usage
# df = pd.DataFrame({'Raison_indisponibilité_fusionnée': ['fatigue; maladie; voyage', 'travail; fatigue; grippe']})
# generate_interactive_wordcloud(df)


@st.cache_resource
def circle(df):
    # Définir les colonnes
    cols = ["Classe_Age", "Niveau_d'etude", 'Genre_', 'Religion_Catégorie', 
        "Situation_Matrimoniale_(SM)",  "categories"]

    # Extraire les modalités et leurs fréquences
    data = df[cols]
    groupes = []
    for col in cols:
        groupes.append(data[col].value_counts().index.tolist())
    groupes = sum(groupes, [])  # Aplatir la liste

    # Ajouter des sauts de ligne dans les étiquettes longues
    groupes = [label.replace(" ", "<br>") for label in groupes]  # Exemple : remplacer les espaces par <br>

    # Calculer les totaux par modalité
    modal = []
    for col in cols: 
        modal.append(data[col].value_counts().tolist())
    total = sum(modal, [])  # Aplatir la liste

    # Calculer les pourcentages par éligibilité pour chaque modalité
    pourcentages = []

    for col in cols:
        modalites = df[col].unique()
        for mod in modalites:
            pourcentages_modalite = []
            df_modalite = df[df[col] == mod]
            total_modalite = len(df_modalite)
            
            # Ne pas continuer si aucune donnée
            if total_modalite == 0:
                continue
                
            for eligibilite in ['Eligible', 'Temporairement Non-eligible', 'Définitivement non-eligible']:
                # S'assurer que la colonne existe
                if 'ÉLIGIBILITÉ_AU_DON.' in df.columns:
                    count = len(df_modalite[df_modalite['ÉLIGIBILITÉ_AU_DON.'] == eligibilite])
                    percentage = round((count / total_modalite) * 100)
                    pourcentages_modalite.append(percentage)
                else:
                    pourcentages_modalite.append(0)  # Valeur par défaut si la colonne n'existe pas
                    
            pourcentages.append(pourcentages_modalite)

    # Vérifier que les listes ont la même longueur
    min_length = min(len(groupes), len(total), len(pourcentages))
    groupes = groupes[:min_length]
    total = total[:min_length]
    pourcentages = pourcentages[:min_length]

    # Calculer les valeurs pour chaque catégorie d'éligibilité
    def_non_eligibles = [total[i] * pourcentages[i][0] / 100 for i in range(len(total))]
    temp_non_eligibles = [total[i] * pourcentages[i][1] / 100 for i in range(len(total))]
    eligibles = [total[i] * pourcentages[i][2] / 100 for i in range(len(total))]

    # Créer des angles pour chaque groupe
    theta = np.linspace(0, 2*np.pi, len(groupes), endpoint=False)
    # Ajuster l'ordre pour que le graphique commence en haut
    theta = np.roll(theta, len(theta)//4)
    groupes_roll = np.roll(groupes, len(groupes)//4).tolist()
    total_roll = np.roll(total, len(total)//4).tolist()
    def_non_eligibles_roll = np.roll(def_non_eligibles, len(eligibles)//4).tolist()
    temp_non_eligibles_roll = np.roll(temp_non_eligibles, len(temp_non_eligibles)//4).tolist()
    eligibles_roll = np.roll(eligibles, len(def_non_eligibles)//4).tolist()

    # Créer le graphique
    fig = go.Figure()

    # Ajouter "Définitivement non-éligibles" (gris clair)
    fig.add_trace(go.Barpolar(
        r=[def_non_eligibles_roll[i] + temp_non_eligibles_roll[i] + eligibles_roll[i] for i in range(len(groupes_roll))],
        theta=groupes_roll,
        name="Éligibles",
        marker_color="lightgreen", 
        width=1
    ))

    # Ajouter "Temporairement non-éligibles" (bleu)
    fig.add_trace(go.Barpolar(
        r=[temp_non_eligibles_roll[i] + eligibles_roll[i] for i in range(len(groupes_roll))],
        theta=groupes_roll,
        name="Temporairement non-éligibles",
        marker_color="#B22222", 
        width=1
    ))

    # Ajouter "Éligibles" (vert)
    fig.add_trace(go.Barpolar(
        r=eligibles_roll,
        theta=groupes_roll,
        name="Définitivement non-éligibles",
        marker_color="blue", 
        width=1
    ))

    # Ajouter du texte pour les totaux uniquement (sans les étiquettes)
    for i in range(len(groupes_roll)):
        fig.add_trace(go.Scatterpolar(
            r=[1.1*max(total_roll)],
            theta=[groupes_roll[i]],
            text=[f"{total_roll[i]}"],  # Suppression de {groupes_roll[i]}<br>
            mode="text",
            showlegend=False,
            textfont=dict(size=8)
        ))

    # Configurer le layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(total_roll)*1.2],
                tickvals=[0, max(total_roll)*0.2, max(total_roll)*0.4, max(total_roll)*0.6, max(total_roll)*0.8, max(total_roll)],
                ticktext=["0%", "20%", "40%", "60%", "80%", "100%"],
                tickfont=dict(size=11),
            gridcolor="rgba(20, 0, 0, 0.25)",  # ✅ Correct RGBA format
            griddash="dot"
            ),
            angularaxis=dict(
                direction="clockwise",
                tickfont=dict(size=12),
                gridcolor="rgba(20, 0, 0, 0.25)",  # ✅ Correct RGBA format
            griddash="dot"  # Ajuster la taille de la police des étiquettes
            )
        ),
        legend=dict(
            title="Type de donneurs:",
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        width=900,
        height=900
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)

@st.cache_data
def process_donation_dates(df):
    """
    Process donation dates from the dataframe.
    Returns a DataFrame with cleaned dates and weekday information.
    
    Args:
        df: Input DataFrame containing donation data
    
    Returns:
        DataFrame with processed dates
    """
    # Extract and clean dates
    dates_raw = df['Date de remplissage de la fiche'].dropna()

    # Convert to datetime with error handling
    dates = pd.to_datetime(dates_raw, format="%m/%d/%Y %H:%M", errors='coerce').dropna()

    # Create a DataFrame with dates
    dates_df = pd.DataFrame({'Date': dates})
    dates_df['Year'] = dates_df['Date'].dt.year
    dates_df['Weekday'] = dates_df['Date'].dt.day_name()  # Get day name (Monday, Tuesday, etc.)
    
    return dates_df

@st.cache_data
def analyze_weekday_donations(dates_df, year_filter=(2019, 2020)):
    """
    Analyze donations by weekday for specific years.
    
    Args:
        dates_df: DataFrame with processed dates
        year_filter: Tuple of years to filter by
    
    Returns:
        Tuple containing:
        - Weekday counts
        - Weekday order
        - Max donation day and count
    """
    # Define the order of days for sorting
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Filter for specified years
    filtered_data = dates_df[dates_df['Year'].isin(year_filter)]
    
    # Count donations by weekday
    weekday_counts = filtered_data['Weekday'].value_counts().reindex(weekday_order, fill_value=0)
    
    # Find the day with maximum donations
    max_count = weekday_counts.max()
    max_day = weekday_counts.idxmax() if not weekday_counts.empty else "No data"
    
    return weekday_counts, weekday_order, (max_day, max_count)

@st.cache_data
def prepare_chart_data(weekday_counts, weekday_order):
    """
    Prepare data for the scatter chart.
    
    Args:
        weekday_counts: Series with counts by weekday
        weekday_order: List with weekday order
    
    Returns:
        List of formatted data points with sizing
    """
    # Calculate the position for each day on x-axis (0 to 6)
    weekday_positions = {day: i for i, day in enumerate(weekday_order)}
    
    # Format the data for the chart
    formatted_data = []
    for day, count in weekday_counts.items():
        formatted_data.append([
            weekday_positions[day],  # Position on x-axis (0 to 6)
            count,                   # Number of donations (y-axis)
            count,                   # For symbol size (same as count)
            day,                     # Day name as label
            2019                     # Year (hardcoded as 2019 for now)
        ])
    
    # Calculate the symbol size for each point
    data_with_size = []
    for point in formatted_data:
        size = math.sqrt(point[2]) * 5  # Scaling factor for bubble size
        data_with_size.append({
            "value": point,
            "symbolSize": size,
            "name": point[3],
            "label": {
                "position": "top",
                "formatter": "{c}",
                "fontSize": 12,
                "color": "rgb(204, 46, 72)"
            }
        })
    
    return data_with_size

@st.cache_data
def create_chart_options(data_with_size, weekday_order):
    """
    Create the ECharts options for the scatter chart.
    
    Args:
        data_with_size: List of formatted data points
        weekday_order: List with weekday order
    
    Returns:
        Dict with chart options
    """
    option = {
        "backgroundColor": {
            "type": "radialGradient",
            "x": 0.3,
            "y": 0.3,
            "r": 0.8,
            "colorStops": [
                {
                    "offset": 0,
                    "color": "#f7f8fa"
                },
                {
                    "offset": 1,
                    "color": "#cdd0d5"
                }
            ]
        },
        "grid": {
            "left": "8%",
            "top": "15%",
            "right": "8%",
            "bottom": "12%"
        },
        "tooltip": {},
        "xAxis": {
            "type": "category",
            "data": weekday_order,
            "name": "Jour de la semaine",
            "nameLocation": "middle",
            "nameGap": 30,
            "axisLine": {
                "lineStyle": {
                    "color": "#999"
                }
            },
            "axisLabel": {
                "rotate": 0,
                "fontSize": 10
            }
        },
        "yAxis": {
            "type": "value",
            "name": "Nombre de dons",
            "nameLocation": "middle",
            "nameGap": 20,
            "axisLine": {
                "lineStyle": {
                    "color": "#999"
                }
            },
            "splitLine": {
                "lineStyle": {
                    "type": "dashed"
                }
            }
        },
        "series": [
            {
                "data": data_with_size,
                "type": "scatter",
                "emphasis": {
                    "focus": "series",
                    "itemStyle": {
                        "shadowBlur": 20,
                        "shadowColor": "rgba(120, 36, 50, 0.7)"
                    }
                },
                "itemStyle": {
                    "shadowBlur": 10,
                    "shadowColor": "rgba(120, 36, 50, 0.5)",
                    "shadowOffsetY": 5,
                    "color": {
                        "type": "radialGradient",
                        "x": 0.4,
                        "y": 0.3,
                        "r": 1,
                        "colorStops": [
                            {
                                "offset": 0,
                                "color": "rgb(251, 118, 123)"
                            },
                            {
                                "offset": 1,
                                "color": "rgb(204, 46, 72)"
                            }
                        ]
                    }
                }
            }
        ]
    }
    
    return option

def jour(df, height="200px"):
    """
    Main function to display the weekly donation pattern chart.
    
    Args:
        df: Input DataFrame containing donation data
        height: Chart height
    """
    # Process the data (cached)
    dates_df = process_donation_dates(df)
    
    # Analyze weekday donations (cached)
    weekday_counts, weekday_order, (max_day, max_count) = analyze_weekday_donations(dates_df)
    
    # Prepare chart data (cached)
    data_with_size = prepare_chart_data(weekday_counts, weekday_order)
    
    # Create chart options (cached)
    chart_options = create_chart_options(data_with_size, weekday_order)
    
      
    nombre_total_donneurs = df.shape[0]

    # Extract and clean dates
    dates_raw = df['Date de remplissage de la fiche'].dropna()

    # Convert to datetime with error handling
    dates = pd.to_datetime(dates_raw, format="%m/%d/%Y %H:%M", errors='coerce').dropna()

    # Create a DataFrame with dates
    dates_df = pd.DataFrame({'Date': dates})
    dates_df['Year'] = dates_df['Date'].dt.year
    dates_df['Weekday'] = dates_df['Date'].dt.day_name()  # Get day name (Monday, Tuesday, etc.)

    # Filter for 2019 and 2020
    data_2019 = dates_df[(dates_df['Year'] == 2019) | (dates_df['Year'] == 2020)]

    # Define the order of days for sorting
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Count donations by weekday for each year
    counts_2019 = data_2019['Weekday'].value_counts().reindex(weekday_order, fill_value=0)


    # Prepare data in the requested format
    # Using [index, count, total_for_sizing, 'Label', year]
    data_2019_formatted = []


    # Calculate the position for each day on x-axis (0 to 6)
    weekday_positions = {day: i for i, day in enumerate(weekday_order)}

    # For 2019
    for day, count in counts_2019.items():
        data_2019_formatted.append([
            weekday_positions[day],  # Position on x-axis (0 to 6)
            count,                   # Number of donations (y-axis)
            count,                   # For symbol size (same as count)
            day,                     # Day name as label
            2019                     # Year
        ])


    # Set up the data in the required structure
    data = [
        data_2019_formatted,  # Data for 2019
    ]

    # Pre-calculate symbol sizes for each data point
    data_2019_with_size = []


    # Find max values to determine which point should have a label
    max_count_2019 = 0
    max_day_2019 = ""
    ""

    # Calculate the size for each point and find max values
    for point in data[0]:
        size = math.sqrt(point[2]) * 5  # Scaling factor for bubble size
        if point[1] > max_count_2019:
            max_count_2019 = point[1]
            max_day_2019 = point[3]
        
        data_2019_with_size.append({
            "value": point,
            "symbolSize": size,
            "name": point[3]
        })



    # Add labels to all points
    for i, point_data in enumerate(data_2019_with_size):
        point_data["label"] = {
            #"show": True,
            "position": "top",
            "formatter": "{c}",  # Show both day name and count
            "fontSize": 12,
            "color": "rgb(204, 46, 72)"
        }


    # Define the option dictionary
    option = {
        "backgroundColor": {
            "type": "radialGradient",
            "x": 0.3,
            "y": 0.3,
            "r": 0.8,
            "colorStops": [
                {
                    "offset": 0,
                    "color": "#f7f8fa"
                },
                {
                    "offset": 1,
                    "color": "#cdd0d5"
                }
            ]
        },

        "grid": {
            "left": "8%",
            "top": "15%",
            "right": "8%",
            "bottom": "12%"
        },
        "tooltip": {
            #"formatter": "{c}: {b} donations"  # Corrigé: utiliser un string au lieu d'une fonction JavaScript
        },
        "xAxis": {
            "type": "category",
            "data": weekday_order,
            "name": "Jour de la semaine",
            "nameLocation": "middle",
            "nameGap": 30,
            "axisLine": {
                "lineStyle": {
                    "color": "#999"
                }
            },
            "axisLabel": {
                "rotate": 0,
                "fontSize": 10
            }
        },
        "yAxis": {
            "type": "value",
            "name": "Nombre de dons",
            "nameLocation": "middle",
            "nameGap": 20,
            "axisLine": {
                "lineStyle": {
                    "color": "#999"
                }
            },
            "splitLine": {
                "lineStyle": {
                    "type": "dashed"
                }
            }
        },
        "series": [
            {
                #"name": "2019",
                "data": data_2019_with_size,
                "type": "scatter",
                "emphasis": {
                    "focus": "series",
                    "itemStyle": {
                        "shadowBlur": 20,
                        "shadowColor": "rgba(120, 36, 50, 0.7)"
                    }
                },
                "itemStyle": {
                    "shadowBlur": 10,
                    "shadowColor": "rgba(120, 36, 50, 0.5)",
                    "shadowOffsetY": 5,
                    "color": {
                        "type": "radialGradient",
                        "x": 0.4,
                        "y": 0.3,
                        "r": 1,
                        "colorStops": [
                            {
                                "offset": 0,
                                "color": "rgb(251, 118, 123)"
                            },
                            {
                                "offset": 1,
                                "color": "rgb(204, 46, 72)"
                            }
                        ]
                    }
                }
            }
        ]
    }


    # Display the chart with adjustable height
    st_echarts(options=option)

@st.cache_data
def ideal_f():
    """
    Creates and displays an ideal donor profile tree chart.
    Using cache_data as the function returns deterministic output and doesn't involve widgets.
    """
    
    data = {
            "name": "Donneur idéal",
            "children": [
                {"name": "Statut professionnel", "children": [{"name": "Employé/ouvrier qualifié"}]},
                {"name": "Statut matrimonial", "children": [{"name": "Célibataire"}]},
                {"name": "Religion", "children": [{"name": "Christianisme"}]},
                {"name": "Âge", "children": [{"name": "26-35 ans"}]},
                {"name": "Niveau d'éducation", "children": [{"name": "Secondaire- Universitaire"}]},
                {"name": "Genre", "children": [{"name": "Homme"}]}
            ]
        }

    # Configuration des options pour le graphique (with dark pink color)
    option = {
        "tooltip": {
            "trigger": 'item',
            "triggerOn": 'mousemove'
        },
        "series": [
            {
                "type": 'tree',
                "data": [data],
                "top": '5%',
                "left": '20%',
                "bottom": '5%',
                "right": '30%',
                "symbolSize": 10,
                "label": {
                    "position": 'left',
                    "verticalAlign": 'middle',
                    "align": 'right',
                    "fontSize": 12,
                    "distance": 1
                },
                "leaves": {
                    "label": {
                        "position": 'right',
                        "verticalAlign": 'middle',
                        "align": 'left',
                        "fontSize": 12,
                        "distance": 0.7  # Reduced distance for leaves
                    }
                },
                "itemStyle": {  # Style for nodes
                    "color": "#C71585"  # Dark pink for nodes
                },
                "lineStyle": {  # Style for lines (edges)
                    "color": "#C71585",  # Dark pink for lines
                    "width": 1  # Optional: adjust line thickness
                },
                "emphasis": {
                    "focus": 'descendant',
                    "itemStyle": {
                        "color": "#FF69B4"  # Lighter pink on hover for contrast
                    },
                    "lineStyle": {
                        "color": "#FF69B4"
                    }
                },
                "expandAndCollapse": True,
                "animationDuration": 200,
                "animationDurationUpdate": 650
            }
        ]
    }

    return option

def display_ideal_chart_f():
    """
    Displays the ideal donor chart using the cached data.
    """
    option = ideal_f()
    st_echarts(options=option, height="200px")

@st.cache_data
def ideal_e():
    """
    Creates and displays an ideal donor profile tree chart.
    Using cache_data as the function returns deterministic output and doesn't involve widgets.
    """
    
    
    data = {
        "name": "Ideal Donor",
        "children": [
            {"name": "Professional Status", "children": [{"name": "Employee/Skilled Worker"}]},
            {"name": "Marital Status", "children": [{"name": "Single"}]},
            {"name": "Religion", "children": [{"name": "Christianity"}]},
            {"name": "Age", "children": [{"name": "26-35 years"}]},
            {"name": "Education Level", "children": [{"name": "Secondary - University"}]},
            {"name": "Gender", "children": [{"name": "Male"}]}
        ]
    }


    # Configuration des options pour le graphique (with dark pink color)
    option = {
        "tooltip": {
            "trigger": 'item',
            "triggerOn": 'mousemove'
        },
        "series": [
            {
                "type": 'tree',
                "data": [data],
                "top": '5%',
                "left": '20%',
                "bottom": '5%',
                "right": '30%',
                "symbolSize": 10,
                "label": {
                    "position": 'left',
                    "verticalAlign": 'middle',
                    "align": 'right',
                    "fontSize": 12,
                    "distance": 1
                },
                "leaves": {
                    "label": {
                        "position": 'right',
                        "verticalAlign": 'middle',
                        "align": 'left',
                        "fontSize": 12,
                        "distance": 0.7  # Reduced distance for leaves
                    }
                },
                "itemStyle": {  # Style for nodes
                    "color": "#C71585"  # Dark pink for nodes
                },
                "lineStyle": {  # Style for lines (edges)
                    "color": "#C71585",  # Dark pink for lines
                    "width": 1  # Optional: adjust line thickness
                },
                "emphasis": {
                    "focus": 'descendant',
                    "itemStyle": {
                        "color": "#FF69B4"  # Lighter pink on hover for contrast
                    },
                    "lineStyle": {
                        "color": "#FF69B4"
                    }
                },
                "expandAndCollapse": True,
                "animationDuration": 200,
                "animationDurationUpdate": 650
            }
        ]
    }

    return option

def display_ideal_chart_e():
    """
    Displays the ideal donor chart using the cached data.
    """
    option = ideal_e()
    st_echarts(options=option, height="200px")

def random_color():
    """Helper function to generate random hex color"""
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


@st.cache_data
def prepare_douala_data(data):
    """
    Prepares and filters data for Douala.
    This function is cached as it performs data transformations that are deterministic for the same input.
    """
    # Remplacer les NaN par une chaîne vide pour éviter l'erreur
    data = data.copy()  # Create a copy to avoid modifying the original
    data['Arrondissement_de_résidence_'] = data['Arrondissement_de_résidence_'].fillna('')
    # Filtrer les données pour Douala uniquement
    douala_data = data[data['Arrondissement_de_résidence_'].str.contains('Douala')]
    return douala_data


@st.cache_data
def filter_by_demographic(_data, demographic_var, selected_value):
    """
    Filters data based on selected demographic variable and value.
    Cached as it's a data transformation that's deterministic for inputs.
    """
    if demographic_var != 'Aucune':
        return _data[_data[demographic_var] == selected_value]
    return _data


@st.cache_data
def calculate_eligibility_stats(filtered_data):
    """
    Calculates eligibility statistics by commune.
    Cached as it performs deterministic data processing.
    """
    # Obtenir les catégories uniques de ÉLIGIBILITÉ_AU_DON.
    categories = filtered_data['ÉLIGIBILITÉ_AU_DON.'].unique()

    # Regrouper par arrondissement et éligibilité pour calculer les proportions
    eligibility_by_commune = filtered_data.groupby(
        ['Arrondissement_de_résidence_', 'ÉLIGIBILITÉ_AU_DON.']
    ).size().unstack(fill_value=0)

    # Calculer le pourcentage pour chaque catégorie
    eligibility_by_commune['Total'] = eligibility_by_commune.sum(axis=1)
    for category in categories:
        eligibility_by_commune[f'{category}_%'] = (
            eligibility_by_commune.get(category, 0) / eligibility_by_commune['Total'] * 100
        ).round(2)

    return eligibility_by_commune, categories


@st.cache_data
def create_radar_options(eligibility_by_commune, categories):
    """
    Creates the radar chart options based on eligibility statistics.
    Cached as it generates chart options deterministically based on inputs.
    """
    communes = eligibility_by_commune.index.tolist()
    indicators = [{"name": commune, "max": 100} for commune in communes]

    # Créer les options pour chaque graphique radar
    radar_options = {}
    for category in categories:
        values = eligibility_by_commune[f'{category}_%'].tolist()
        # Définir la couleur : rouge pour "Eligible", aléatoire pour les autres
        line_color = "#FF0000" if category == "Eligible" else random_color()
        
        radar_options[category] = {
            "title": {
                "text": f"{category} (%)",
                "textStyle": {"fontSize": 12}
            },
            "tooltip": {},
            "radar": {
                "indicator": indicators,
                "shape": "circle",
                "splitNumber": 5,
                "splitLine": {  # Ajout pour définir la couleur des cercles concentriques
                    "lineStyle": {
                        "color": "gray",  # Noir pour les lignes des cercles
                        "width": 0.5
                    }
                },
                "axisName": {
                    "fontSize": 6,
                    "color": "#fff",
                    "backgroundColor": "#666",
                    "borderRadius": 2,
                    "padding": [2, 2.5]
                }
            },
            "series": [
                {
                    "name": category,
                    "type": "radar",
                    "data": [
                        {
                            "value": values,
                            "name": f"{category} (%)"
                        }
                    ],
                    "areaStyle": {"opacity": 0.2},
                    "lineStyle": {
                        "width": 1.5,
                        "color": line_color
                    }
                }
            ],
            "legend": {
                "data": [f"{category} (%)"],
                "top": "bottom",
                "textStyle": {"fontSize": 10}
            }
        }
    
    return radar_options


def three(data):
    """
    Main function to display radar charts for eligibility analysis.
    Not cached because it contains widgets and interactive elements.
    """
    # Prepare data
    douala_data = prepare_douala_data(data)
    
    # UI elements for filtering
    demographic_var = st.selectbox(
        "Choose a variable",
        options=['Aucune'] + [col for col in douala_data.columns if col not in ['Arrondissement_de_résidence_', 'ÉLIGIBILITÉ_AU_DON.']],
        index=0
    )
    
    # Filter based on selected demographic variable
    if demographic_var != 'Aucune':
        unique_values = douala_data[demographic_var].unique()
        selected_value = st.selectbox(f"Choisissez une valeur pour {demographic_var} :", options=unique_values)
        filtered_data = filter_by_demographic(douala_data, demographic_var, selected_value)
    else:
        filtered_data = douala_data
    
    # Calculate statistics
    eligibility_by_commune, categories = calculate_eligibility_stats(filtered_data)
    
    # Create radar chart options
    radar_options = create_radar_options(eligibility_by_commune, categories)
    
    # Display radar charts
    cols = st.columns([1, 1, 1])
    for i, category in enumerate(categories):
        with cols[i]:
            st_echarts(options=radar_options[category], height="300px", width="90%", key=f"radar_{category}")



@st.cache_resource
def heatmap(df) : 
    df['Date de remplissage de la fiche'] = pd.to_datetime(df['Date de remplissage de la fiche'], errors='coerce')

    # Filtrer les lignes où la date est valide
    df = df.dropna(subset=['Date de remplissage de la fiche'])

    # Compter le nombre de dons par jour
    dons_par_jour = df.groupby(df['Date de remplissage de la fiche'].dt.date).size().reset_index(name='Nombre de dons')

    # Générer une plage complète de dates pour inclure les jours sans dons
    date_min = dons_par_jour['Date de remplissage de la fiche'].min()
    date_max = dons_par_jour['Date de remplissage de la fiche'].max()
    all_dates = pd.date_range(start=date_min, end=date_max, freq='D')
    all_dates_df = pd.DataFrame({'Date': all_dates})
    all_dates_df['Date'] = all_dates_df['Date'].dt.date

    # Fusionner avec les données de dons pour inclure les jours à 0 dons
    dons_par_jour = all_dates_df.merge(dons_par_jour, left_on='Date', right_on='Date de remplissage de la fiche', how='left')
    dons_par_jour['Nombre de dons'] = dons_par_jour['Nombre de dons'].fillna(0)

    # Ajouter les colonnes nécessaires pour le heatmap
    dons_par_jour['Jour de la semaine'] = pd.to_datetime(dons_par_jour['Date']).dt.dayofweek
    dons_par_jour['Semaine'] = pd.to_datetime(dons_par_jour['Date']).dt.isocalendar().week
    dons_par_jour['Année'] = pd.to_datetime(dons_par_jour['Date']).dt.year

    # Définir les jours de la semaine (0 = Lundi, 6 = Dimanche)
    jours_semaine = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']

    # Lister les années uniques
    annees_uniques = sorted(dons_par_jour['Année'].unique())

    # Palette de couleurs élégante : gris clair -> orange -> rouge profond
    custom_colorscale = [
        [0.0, 'rgb(240, 240, 240)'],  # Gris très clair
        [0.3, 'rgb(255, 204, 153)'],  # Orange doux
        [0.6, 'rgb(255, 102, 102)'],  # Rouge corail
        [1.0, 'rgb(153, 0, 0)']       # Rouge profond
    ]

    # Préparer les données pour chaque année
    traces = []
    for annee in annees_uniques:
        df_annee = dons_par_jour[dons_par_jour['Année'] == annee]
        semaines_uniques = sorted(df_annee['Semaine'].unique())
        
        # Créer une matrice pour le heatmap
        z_values = []
        for jour in range(7):
            row = []
            for semaine in semaines_uniques:
                data = df_annee[(df_annee['Jour de la semaine'] == jour) & (df_annee['Semaine'] == semaine)]
                row.append(data['Nombre de dons'].values[0] if not data.empty else 0)
            z_values.append(row)
        
        # Ajouter une trace pour chaque année avec visibilité initiale
        traces.append(
            go.Heatmap(
                z=z_values,
                x=semaines_uniques,
                y=jours_semaine,
                colorscale=custom_colorscale,
                showscale=True,
                colorbar=dict(
                    title='Nombre de dons',
                    tickfont=dict(size=12, color='black'),
                    thickness=20,
                    outlinecolor='black',
                    outlinewidth=1
                ),
                hoverinfo='x+y+z',
                zmin=0,
                hovertemplate='Semaine: %{x}<br>Jour: %{y}<br>Dons: %{z}<extra></extra>',
                visible=(annee == annees_uniques[0])  # Seule la première année visible au départ
            )
        )

    # Créer la figure avec toutes les traces
    fig = go.Figure(data=traces)

    # Ajouter le menu déroulant pour les années
    updatemenus = [
        dict(
            buttons=[
                dict(
                    args=[{
                        'visible': [annee == a for a in annees_uniques],
                        'title': f'Heatmap des dons - Année {annee}'
                    }],
                    label=str(annee),
                    method='update'
                ) for annee in annees_uniques
            ],
            direction='down',
            showactive=True,
            x=0.1,
            xanchor='left',
            y=1.15,
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='gray',
            font=dict(size=12, color='black')
        )
    ]

    # Personnaliser le layout pour un look moderne
    fig.update_layout(

        xaxis_title='Semaine de l\'année',
        yaxis_title='Jour de la semaine',
        
        
        updatemenus=updatemenus,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=80, r=40, t=100, b=40),
        showlegend=False,
        annotations=[
            dict(
                text="Source: vos données",
                xref="paper", yref="paper",
                x=1, y=-0.1,
                showarrow=False,
                font=dict(size=10, color='gray')
            )
        ]
    )

    # Ajouter des bordures légères aux cellules du heatmap
    fig.update_traces(
        zmin=0,
        connectgaps=False,
        xgap=1,  # Espacement horizontal entre les cellules
        ygap=1   # Espacement vertical entre les cellules
    )

    st.plotly_chart(fig,use_container_width=True)

def fiel_1(df) : 

    # Filtrer pour ne garder que les donneurs ayant répondu "Oui" à "A-t-il (elle) déjà donné le sang"
    df = df[df['A-t-il (elle) déjà donné le sang'] == 'Oui']

    # Étape 2 : Créer les flux entre chaque paire de catégories consécutives
    # On va créer des paires : (A-t-il (elle) déjà donné le sang → ÉLIGIBILITÉ_AU_DON.), (ÉLIGIBILITÉ_AU_DON. → Classe_Age), etc.

    # Flux 1 : A-t-il (elle) déjà donné le sang → ÉLIGIBILITÉ_AU_DON.
    flux1 = df.groupby(['A-t-il (elle) déjà donné le sang', 'ÉLIGIBILITÉ_AU_DON.']).size().reset_index(name='Nombre')

    # Flux 2 : ÉLIGIBILITÉ_AU_DON. → Classe_Age
    flux2 = df.groupby(['ÉLIGIBILITÉ_AU_DON.', 'Classe_Age']).size().reset_index(name='Nombre')

    # Flux 3 : Classe_Age → Genre_
    flux3 = df.groupby(['Classe_Age', 'Genre_']).size().reset_index(name='Nombre')

    # Flux 4 : Genre_ → Niveau_d_etude
    flux4 = df.groupby(['Genre_', "Niveau_d'etude"]).size().reset_index(name='Nombre')

    # Flux 5 : Niveau_d_etude → Religion_Catégorie
    flux5 = df.groupby(["Niveau_d'etude", 'Religion_Catégorie']).size().reset_index(name='Nombre')

    # Flux 6 : Religion_Catégorie → Situation_Matrimoniale_(SM)
    flux6 = df.groupby(['Religion_Catégorie', 'Situation_Matrimoniale_(SM)']).size().reset_index(name='Nombre')

    # Flux 7 : Situation_Matrimoniale_(SM) → categories
    flux7 = df.groupby(['Situation_Matrimoniale_(SM)', 'categories']).size().reset_index(name='Nombre')

    # Étape 3 : Créer la liste des nœuds (toutes les catégories uniques)
    donne_sang = df['A-t-il (elle) déjà donné le sang'].unique().tolist()  # Contient uniquement "Oui" après le filtre
    eligibilites = df['ÉLIGIBILITÉ_AU_DON.'].unique().tolist()
    classes_age = df['Classe_Age'].unique().tolist()
    genres = df['Genre_'].unique().tolist()
    niveaux_etude = df["Niveau_d'etude"].unique().tolist()
    religions = df['Religion_Catégorie'].unique().tolist()
    situations_matrimoniales = df['Situation_Matrimoniale_(SM)'].unique().tolist()
    categories = df['categories'].unique().tolist()

    # Liste complète des nœuds
    nodes = (donne_sang + eligibilites + classes_age + genres + niveaux_etude +
        religions + situations_matrimoniales + categories)

    # Étape 4 : Créer un dictionnaire pour mapper les nœuds à des indices
    node_dict = {node: idx for idx, node in enumerate(nodes)}

    # Étape 5 : Créer les liens (source, target, value) pour chaque flux
    # Liens pour Flux 1 : A-t-il (elle) déjà donné le sang → ÉLIGIBILITÉ_AU_DON.
    source1 = flux1['A-t-il (elle) déjà donné le sang'].map(node_dict).tolist()
    target1 = flux1['ÉLIGIBILITÉ_AU_DON.'].map(node_dict).tolist()
    value1 = flux1['Nombre'].tolist()

    # Liens pour Flux 2 : ÉLIGIBILITÉ_AU_DON. → Classe_Age
    source2 = flux2['ÉLIGIBILITÉ_AU_DON.'].map(node_dict).tolist()
    target2 = flux2['Classe_Age'].map(node_dict).tolist()
    value2 = flux2['Nombre'].tolist()

    # Liens pour Flux 3 : Classe_Age → Genre_
    source3 = flux3['Classe_Age'].map(node_dict).tolist()
    target3 = flux3['Genre_'].map(node_dict).tolist()
    value3 = flux3['Nombre'].tolist()

    # Liens pour Flux 4 : Genre_ → Niveau_d_etude
    source4 = flux4['Genre_'].map(node_dict).tolist()
    target4 = flux4["Niveau_d'etude"].map(node_dict).tolist()
    value4 = flux4['Nombre'].tolist()

    # Liens pour Flux 5 : Niveau_d_etude → Religion_Catégorie
    source5 = flux5["Niveau_d'etude"].map(node_dict).tolist()
    target5 = flux5['Religion_Catégorie'].map(node_dict).tolist()
    value5 = flux5['Nombre'].tolist()

    # Liens pour Flux 6 : Religion_Catégorie → Situation_Matrimoniale_(SM)
    source6 = flux6['Religion_Catégorie'].map(node_dict).tolist()
    target6 = flux6['Situation_Matrimoniale_(SM)'].map(node_dict).tolist()
    value6 = flux6['Nombre'].tolist()

    # Liens pour Flux 7 : Situation_Matrimoniale_(SM) → categories
    source7 = flux7['Situation_Matrimoniale_(SM)'].map(node_dict).tolist()
    target7 = flux7['categories'].map(node_dict).tolist()
    value7 = flux7['Nombre'].tolist()

    # Combiner tous les liens
    source = source1 + source2 + source3 + source4 + source5 + source6 + source7
    target = target1 + target2 + target3 + target4 + target5 + target6 + target7
    value = value1 + value2 + value3 + value4 + value5 + value6 + value7

    # Étape 6 : Définir les couleurs pour les nœuds
    # On attribue des couleurs différentes pour chaque groupe de nœuds
    num_donne_sang = len(donne_sang)
    num_eligibilites = len(eligibilites)
    num_classes_age = len(classes_age)
    num_genres = len(genres)
    num_niveaux_etude = len(niveaux_etude)
    num_religions = len(religions)
    num_situations_matrimoniales = len(situations_matrimoniales)
    num_categories = len(categories)

    # Couleurs pour chaque groupe
    colors_donne_sang = ['#ff00ff'] * num_donne_sang  # Magenta pour A-t-il (elle) déjà donné le sang
    colors_eligibilites = ['#00cc96'] * num_eligibilites  # Vert pour ÉLIGIBILITÉ_AU_DON.
    colors_classes_age = ['#f83e8c'] * num_classes_age  # Rose pour Classe_Age
    colors_genres = ['#8b008b'] * num_genres  # Violet foncé pour Genre_
    colors_niveaux_etude = ['#1e90ff'] * num_niveaux_etude  # Bleu pour Niveau_d_etude
    colors_religions = ['#ffd700'] * num_religions  # Jaune pour Religion_Catégorie
    colors_situations_matrimoniales = ['#ff4500'] * num_situations_matrimoniales  # Orange pour Situation_Matrimoniale_(SM)
    colors_categories = ['#00b7eb'] * num_categories  # Cyan pour categories

    # Combiner les couleurs
    node_colors = (colors_donne_sang + colors_eligibilites + colors_classes_age + colors_genres +
            colors_niveaux_etude + colors_religions + colors_situations_matrimoniales + colors_categories)

    # Étape 7 : Créer le diagramme de Sankey
    fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=nodes,
        color=node_colors
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color='rgba(200, 200, 200, 0.5)'  # Couleur des liens
    )
    )])

    # Étape 8 : Personnaliser le layout
    fig.update_layout(
    title_text="Flux des donneurs (A-t-il donné = Oui) : A-t-il donné → Éligibilité → Classe d'âge → Genre → Niveau d'étude → Religion → Situation matrimoniale → Catégorie",
    font=dict(size=10, color='black'),
    width=1200,  # Ajuster la largeur pour une meilleure lisibilité
    height=800  # Ajuster la hauteur
    )
    st.plotly_chart(fig,use_container_width=True)

def plot_top4_demographic(data_recurrent, data_non_recurrent, column, title_prefix, comparison=False, orientation='v'):
    """
    Génère un graphique Plotly avec les top 4 catégories pour une variable démographique.
    
    Parameters:
    - data_recurrent: DataFrame des donneurs récurrents
    - data_non_recurrent: DataFrame des donneurs non récurrents (pour comparaison)
    - column: Colonne démographique à analyser
    - title_prefix: Préfixe du titre du graphique
    - comparison: Booléen pour indiquer si on compare récurrents et non récurrents
    - orientation: 'v' pour vertical (par défaut), 'h' pour horizontal
    """
    # Compter les occurrences pour les donneurs récurrents et non récurrents
    if comparison:
        # Pour les graphiques de comparaison (récurrents vs non récurrents)
        count_recurrent = data_recurrent[column].value_counts()
        count_non_recurrent = data_non_recurrent[column].value_counts()
        
        # Fusionner les deux séries pour obtenir toutes les catégories
        all_categories = pd.concat([count_recurrent, count_non_recurrent], axis=1, sort=False)
        all_categories.columns = ['Récurrents', 'Non Récurrents']
        all_categories.fillna(0, inplace=True)
        
        # Calculer le total pour trier
        all_categories['Total'] = all_categories['Récurrents'] + all_categories['Non Récurrents']
        top4_categories = all_categories.sort_values('Total', ascending=False).head(4).index
        
        # Filtrer les données pour ne garder que les top 4 catégories
        count_recurrent = count_recurrent[count_recurrent.index.isin(top4_categories)]
        count_non_recurrent = count_non_recurrent[count_non_recurrent.index.isin(top4_categories)]
        
        # Créer le graphique de comparaison
        fig = go.Figure()
        
        if orientation == 'v':
            # Orientation verticale
            fig.add_trace(go.Bar(
                x=count_recurrent.index,
                y=count_recurrent.values,
                name='Récurrents (Oui)',
                marker_color='#EC8282',  # Vert
                text=count_recurrent.values,
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Nombre: %{y}<br>Catégorie: Récurrents<extra></extra>'
            ))
            
            fig.add_trace(go.Bar(
                x=count_non_recurrent.index,
                y=count_non_recurrent.values,
                name='Non Récurrents (Non)',
                marker_color='#ff5733',  # Orange
                text=count_non_recurrent.values,
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Nombre: %{y}<br>Catégorie: Non Récurrents<extra></extra>'
            ))
            
            xaxis_title = column
            yaxis_title = 'Nombre de donneurs'
            xaxis_config = dict(tickangle=45, title_standoff=25)
            yaxis_config = dict(gridcolor='rgba(0,0,0,0.1)', title_standoff=25)
            
        else:
            # Orientation horizontale
            fig.add_trace(go.Bar(
                y=count_recurrent.index,
                x=count_recurrent.values,
                name='Récurrents (Oui)',
                marker_color='#EC8282',  # Vert
                text=count_recurrent.values,
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Nombre: %{x}<br>Catégorie: Récurrents<extra></extra>',
                orientation='h'
            ))
            
            fig.add_trace(go.Bar(
                y=count_non_recurrent.index,
                x=count_non_recurrent.values,
                name='Non Récurrents (Non)',
                marker_color='#ff5733',  # Orange
                text=count_non_recurrent.values,
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Nombre: %{x}<br>Catégorie: Non Récurrents<extra></extra>',
                orientation='h'
            ))
            
            xaxis_title = 'Nombre de donneurs'
            yaxis_title = column
            xaxis_config = dict(gridcolor='rgba(0,0,0,0.1)', title_standoff=25)
            yaxis_config = dict(title_standoff=25)
            
    else:
        # Pour les graphiques de distribution (donneurs récurrents uniquement)
        count_recurrent = data_recurrent[column].value_counts()
        top4_categories = count_recurrent.head(4).index
        count_recurrent = count_recurrent[count_recurrent.index.isin(top4_categories)]
        
        # Créer le graphique de distribution
        fig = go.Figure()
        
        if orientation == 'v':
            # Orientation verticale
            fig.add_trace(go.Bar(
                x=count_recurrent.index,
                y=count_recurrent.values,
                name='Récurrents',
                marker_color='#EC8282',  # Vert
                text=count_recurrent.values,
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Nombre: %{y}<br>Catégorie: Récurrents<extra></extra>'
            ))
            
            xaxis_title = column
            yaxis_title = 'Nombre de donneurs'
            xaxis_config = dict(tickangle=45, title_standoff=25)
            yaxis_config = dict(gridcolor='rgba(0,0,0,0.1)', title_standoff=25)
            
        else:
            # Orientation horizontale
            fig.add_trace(go.Bar(
                y=count_recurrent.index,
                x=count_recurrent.values,
                name='Récurrents',
                marker_color='#EC8282',  # Vert
                text=count_recurrent.values,
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Nombre: %{x}<br>Catégorie: Récurrents<extra></extra>',
                orientation='h'
            ))
            
            xaxis_title = 'Nombre de donneurs'
            yaxis_title = column
            xaxis_config = dict(gridcolor='rgba(0,0,0,0.1)', title_standoff=25)
            yaxis_config = dict(title_standoff=25)
    
    # Personnaliser le layout
    fig.update_layout(
        title=f"{title_prefix} par {column} (Top 4)",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis=xaxis_config,
        yaxis=yaxis_config,
        legend=dict(
            title='Statut de récurrence',
            orientation='h',
            yanchor='bottom',
            y=-0.3,
            xanchor='center',
            x=0.5
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12, color='black'),
        margin=dict(l=50, r=50, t=80, b=150),
        width=800,
        height=600,
        bargap=0.2,
        barmode='group' if comparison else 'stack'
    )
    
    # Afficher le graphique
    st.plotly_chart(fig)
    orientation_dict = {
'Classe_Age': 'v',  # Vertical pour les tranches d'âge
'Genre_': 'v',      # Vertical pour le genre
'Niveau_d_etude': 'v',  # Vertical pour le niveau d'étude
'Religion_Catégorie': 'h',  # Horizontal pour les religions (étiquettes longues)
'Situation_Matrimoniale_(SM)': 'v',  # Vertical pour la situation matrimoniale
'categories': 'h',  # Horizontal pour les catégories professionnelles (étiquettes longues)
'Arrondissement_de_résidence_': 'h'  # Horizontal pour les arrondissements (étiquettes longues)
}
@st.cache_resource
# This function can be cached - it only processes data and returns the chart configuration
@st.cache_resource  # Keep the cache decorator here if it was present before
def prepare_months_status_data(df):
    data_1 = df[df['ÉLIGIBILITÉ_AU_DON.'] == 'Eligible']
    data_2 = df[df['ÉLIGIBILITÉ_AU_DON.'] == 'Temporairement Non-eligible']
    data_3 = df[df['ÉLIGIBILITÉ_AU_DON.'] == 'Définitivement non-eligible']

    S1 = []

    for df_subset in [data_1, data_2, data_3]:
        # Count total number of donors
        nombre_total_donneurs = df_subset.shape[0]

        # Extract and clean dates
        dates_raw = df_subset['Date de remplissage de la fiche'].dropna()

        # Convert to datetime with error handling
        dates = pd.to_datetime(dates_raw, format="%m/%d/%Y %H:%M", errors='coerce').dropna()

        # Create a DataFrame with dates
        dates_df = pd.DataFrame({'Date': dates})
        dates_df['Year'] = dates_df['Date'].dt.year
        dates_df['Month'] = dates_df['Date'].dt.month_name()  # Get month name

        # Filter for all data
        data_ = dates_df
        
        # Define the order of months for sorting
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                    'July', 'August', 'September', 'October', 'November', 'December']

        # Count donations by month for each year
        counts_ = data_['Month'].value_counts().reindex(month_order, fill_value=0)

        S1.append(counts_)

    option = {
    "color": ["#00DDFF", "#80FFA5","#FFBF00",  "#37A2FF", ],
    "tooltip": {
        "trigger": "axis",
        "axisPointer": {"type": "cross", "label": {"backgroundColor": "#6a7985"}},
    },
    "legend": {"data": ['Définitivement non-eligible', 'Temporairement Non-eligible', "Eligible",]},
    "toolbox": {"feature": {"saveAsImage": {}}},
    "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
    "xAxis": [
        {
            "type": "category",
            "boundaryGap": False,
            "data": ['January', 'February', 'March', 'April', 'May', 'June', 
                'July', 'August', 'September', 'October', 'November', 'December']
        }
    ],
    "yAxis": [{"type": "value"}],
    "series": [
        {
            "name": 'Définitivement non-eligible',
            "type": "line",
            "stack": "Total",
            "smooth": True,
            "lineStyle": {"width": 0},
            "showSymbol": False,
            "areaStyle": {
                "opacity": 0.8,
                "color": {
                    "type": "linear",
                    "x": 0,
                    "y": 0,
                    "x2": 0,
                    "y2": 1,
                    
                    "colorStops": [
                            {"offset": 0, "color": "#00DDFF"},
                            {"offset": 1, "color": "#00DDFF"},
                    ],
                    
                },
            },
            "data": S1[2].values.tolist(),
        },
        {
            "name": 'Temporairement Non-eligible',
            "type": "line",
            "stack": "Total",
            "smooth": True,
            "lineStyle": {"width": 0},
            "showSymbol": False,
            "areaStyle": {
                "opacity": 0.8,
                "color": {
                    "type": "linear",
                    "x": 0,
                    "y": 0,
                    "x2": 0,
                    "y2": 1,
                    "colorStops": [
                        {"offset": 0, "color":  "#37A2FF"},
                        {"offset": 1, "color": "rgb(77, 119, 255)"},
                    ],
                },
            },
            "data": S1[1].values.tolist(),
        },
        {
            "name": "Eligible",
            "type": "line",
            "stack": "Total",
            "smooth": True,
            "lineStyle": {"width": 0},
            "showSymbol": False,
            "areaStyle": {
                "opacity": 0.8,
                "color": {
                    "type": "linear",
                    "x": 0,
                    "y": 0,
                    "x2": 0,
                    "y2": 1,
                        "colorStops": [
                        {"offset": 0, "color": "#FFBF00"},
                        {"offset": 1, "color": "#FF0087"},
                    ],
                },
            },
            "data": S1[0].values.tolist(),
        },
    ],
    }
    
    return option

# This function should NOT be cached - it displays the widget

def by_months_status(df):
    # Get chart configuration from the cached function
    option = prepare_months_status_data(df)
    st_echarts(option, height="400px")
#@st.cache_resource
def options_women_reasons(df):
    data = df[['Genre_',
       'ÉLIGIBILITÉ_AU_DON.',
       'Raison de l’indisponibilité de la femme [La DDR est mauvais si <14 jour avant le don]',
       'Raison de l’indisponibilité de la femme [Allaitement ]',
       'Raison de l’indisponibilité de la femme [A accoucher ces 6 derniers mois  ]',
       'Raison de l’indisponibilité de la femme [Interruption de grossesse  ces 06 derniers mois]',
       'Raison de l’indisponibilité de la femme [est enceinte ]',
]]

# Récupération des données pour les femmes


    data = data[(data['Genre_'] == 'Femme') & 
            ((data['ÉLIGIBILITÉ_AU_DON.'] == 'Temporairement Non-eligible') | 
                (data['ÉLIGIBILITÉ_AU_DON.'] == 'Définitivement non-eligible'))]

    data.columns = ['Genre_',
        'ÉLIGIBILITÉ_AU_DON.','La DDR est mauvaise',
        'Allaitement','A accouchée ces 6 derniers mois',
        'Interruption de grossesse ces 06 derniers mois', 
        'Est enceinte'
        ]

    df2 = df[['Genre_',
        'ÉLIGIBILITÉ_AU_DON.',
        'Raison de non-eligibilité totale  [Antécédent de transfusion]',
        'Raison de non-eligibilité totale  [Porteur(HIV,hbs,hcv)]',
        'Raison de non-eligibilité totale  [Opéré]',
        'Raison de non-eligibilité totale  [Tatoué]',
        'Raison de non-eligibilité totale  [Diabétique]'
    ]]

    # Récupération des données pour les femmes


    dat = df2[(df2['Genre_'] == 'Femme') & (df2['ÉLIGIBILITÉ_AU_DON.'] == 'Définitivement non-eligible')]

    dat.columns = ['Genre_',
        'ÉLIGIBILITÉ_AU_DON.','Antécédent de transfusion',
        'Porteur(HIV,hbs,hcv)','Opéré',
        'Tatoué', 'Diabétique'
        ]

    symbols = [
    'path://M36.7,102.84c-1.17,2.54-2.99,4.98-3.39,7.63c-1.51,9.89-3.31,19.58-1.93,29.95 c0.95,7.15-2.91,14.82-3.57,22.35c-0.64,7.36-0.2,14.86,0.35,22.25c0.12,1.68,2.66,3.17,4.67,5.4c-0.6,0.82-1.5,2.22-2.58,3.48 c-0.96,1.12-1.96,2.35-3.21,3.04c-1.71,0.95-3.71,2.03-5.51,1.9c-1.18-0.08-3.04-2.13-3.16-3.43c-0.44-4.72,0-9.52-0.41-14.25 c-0.94-10.88-2.32-21.72-3.24-32.61c-0.49-5.84-1.63-12.01-0.35-17.54c3.39-14.56,2.8-28.84,0.36-43.4 c-2.71-16.16-1.06-32.4,0.54-48.59c0.91-9.22,4.62-17.36,8.53-25.57c1.32-2.77,1.88-6.84,0.87-9.62C21.89-3.77,18.09-11,14.7-18.38 c-0.56,0.1-1.13,0.21-1.69,0.31C10.17-11.52,6.29-5.2,4.71,1.65C2.05,13.21-4.42,22.3-11.43,31.28c-1.32,1.69-2.51,3.5-3.98,5.04 c-4.85,5.08-3.25,10.98-2.32,16.82c0.25,1.53,0.52,3.06,0.77,4.59c-0.53,0.22-1.07,0.43-1.6,0.65c-1.07-2.09-2.14-4.19-3.28-6.44 c-6.39,2.91-2.67,9.6-5.23,15.16c-1.61-3.31-2.77-5.68-3.93-8.06c0-0.33,0-0.67,0-1c6.96-16.08,14.63-31.9,20.68-48.31 C-5.24-4.07-2.03-18.55,2-32.73c0.36-1.27,0.75-2.53,0.98-3.82c1.36-7.75,4.19-10.23,11.88-10.38c1.76-0.04,3.52-0.21,5.76-0.35 c-0.55-3.95-1.21-7.3-1.45-10.68c-0.61-8.67,0.77-16.69,7.39-23.19c2.18-2.14,4.27-4.82,5.25-7.65c2.39-6.88,11.66-9,16.94-8.12 c5.92,0.99,12.15,7.93,12.16,14.12c0.01,9.89-5.19,17.26-12.24,23.68c-2.17,1.97-5.35,4.77-5.17,6.94c0.31,3.78,4.15,5.66,8.08,6.04 c1.82,0.18,3.7,0.37,5.49,0.1c5.62-0.85,8.8,2.17,10.85,6.73C73.38-27.19,78.46-14.9,84.2-2.91c1.52,3.17,4.52,5.91,7.41,8.09 c7.64,5.77,15.57,11.16,23.45,16.61c2.28,1.58,4.64,3.23,7.21,4.14c5.18,1.84,8.09,5.63,9.82,10.46c0.45,1.24,0.19,3.71-0.6,4.18 c-1.06,0.63-3.15,0.27-4.44-0.38c-7.05-3.54-12.84-8.88-19.14-13.5c-3.5-2.57-7.9-4-12.03-5.6c-9.44-3.66-17.73-8.42-22.5-18.09 c-2.43-4.94-6.09-9.27-9.69-14.61c-1.2,10.98-4.46,20.65,1.14,31.19c6.62,12.47,5.89,26.25,1.21,39.49 c-2.52,7.11-6.5,13.74-8.67,20.94c-1.91,6.33-2.2,13.15-3.23,19.75c-0.72,4.63-0.84,9.48-2.36,13.84 c-2.49,7.16-6.67,13.83-5.84,21.82c0.42,4.02,1.29,7.99,2.1,12.8c-3.74-0.49-7.47-0.4-10.67-1.66c-1.33-0.53-2.43-4.11-2.07-6.01 c1.86-9.94,3.89-19.69,0.07-29.74C34.55,108.63,36.19,105.52,36.7,102.84c1.25-8.45,2.51-16.89,3.71-24.9 c-0.83-0.58-0.85-0.59-0.87-0.61c-0.03,0.16-0.07,0.32-0.09,0.48C38.53,86.15,37.62,94.5,36.7,102.84z',
    'path://M40.02-99c2.07,1.21,4.26,2.25,6.19,3.66c5.94,4.34,8.23,12.57,4.95,19.79 c-3.21,7.08-6.82,14.03-10.86,20.67c-2.17,3.56-1.25,5.38,1.99,6.36c2.94,0.89,6.36,1.91,9.15,1.21c5.51-1.4,8.33,1.23,10.66,5.29 c4.71,8.22,9.72,16.29,13.84,24.8C81.06-6.65,89,0.4,99.56,5.17C109.82,9.8,120,14.7,129.85,20.15c4.72,2.61,9.09,6.37,10.24,12.97 c-2.89-1.93-5.2-3.75-7.78-5.04c-0.99-0.5-2.6,0.22-4.83,0.5c-5.36-9.35-16.8-9.4-26.74-12.62C91.68,13.04,81.82,11.37,75.66,3 c-5.98-8.13-11.61-16.52-17.4-24.79c-0.46-0.66-0.98-1.27-1.66-2.16c-3.21,7.75-6.78,15-9.12,22.63c-1.15,3.76-0.64,8.37,0.26,12.33 c0.81,3.59,3.01,6.92,4.87,10.22c6.73,11.95,2.41,22.89-2.91,33.75c-0.35,0.72-0.86,1.43-1.46,1.97 c-7.11,6.38-14.48,12.5-21.24,19.22c-2.08,2.07-3.1,5.7-3.62,8.77c-1.92,11.44-3.81,22.92-4.93,34.46 c-0.5,5.16,1.06,10.49,1.28,15.75c0.23,5.7,0.39,11.47-0.15,17.13c-1.15,12.11-2.83,24.17-4.11,36.27c-0.18,1.72,0.8,3.53,1.13,5.33 c0.88,4.76-0.22,6.23-4.71,5.17c-4.53-1.06-8.86-2.94-14.27-4.8c1.98-1.62,2.84-2.83,3.94-3.12c5.42-1.44,7-5.2,6.39-10.23 c-1.39-11.39-3.15-22.73-4.24-34.14c-0.53-5.56,0.16-11.23,0.24-16.85c0.06-4.49,0.01-8.97,0.01-14.72 c-2.79,1.53-5.2,2.27-6.79,3.83c-4.26,4.19-8.39,8.56-12.11,13.22c-1.55,1.95-2.19,4.76-2.79,7.29c-0.47,1.99,0.6,5.02-0.48,6.05 c-2.17,2.08-5.2,3.79-8.13,4.38c-3.61,0.73-7.49,0.18-12.26,0.18c6.34-8.69,11.91-16.11,17.22-23.71c3.29-4.71,6.23-9.67,9.24-14.58 c2.15-3.5,3.76-7.4,6.3-10.57c5.38-6.73,6.74-14.28,6.72-22.64C0.88,68.3,1.36,57.91,2.26,47.58c0.69-7.85,2.15-15.67,3.7-23.41 c0.77-3.83,2.89-7.39,3.72-11.22c1.83-8.4-1.9-16-4.38-23.95C2.96-5.34-0.31,0.12-1.5,6c-1.96,9.72-7.34,17.44-12.26,25.57 c-4.39,7.25-8.79,14.52-12.75,22.01c-2.64,5-4.5,10.41-6.83,15.92c-4.82-5.28-4.65-10.59-0.94-16.97 C-21.4,30.4-12.08,6.78-6.17-18.12c1.4-5.88,1.24-12.11,2.23-18.12c1.2-7.27,4.15-9.56,11.39-9.69c8.65-0.14,13.86-4.77,14.48-13.51 c0.35-5.01,0.16-10.11-0.28-15.12c-0.82-9.3,2.49-16.57,10.17-21.69c2.08-1.39,4.78-1.87,7.2-2.76C39.35-99,39.69-99,40.02-99z',
    'path://M-39,33.03c3.72-9.74,12.97-12.87,20.96-17.43c9.51-5.43,19.2-10.54,28.69-16 c1.77-1.02,3.35-2.85,4.33-4.67C21.44-17,27.82-28.95,33.95-41.04c2.13-4.2,4.95-6.01,9.7-6.09c3.68-0.06,7.52-0.92,10.97-2.25 c5.09-1.95,4.85-5.2,1.1-9.01c-5.12-5.21-10.89-10.1-13.23-17.54c-1.71-5.44,0.78-15.62,4.87-18.74 c4.12-3.15,12.55-3.84,16.69-0.12c3.39,3.04,6.44,7.27,7.8,11.56c1.96,6.16,3.31,12.9,2.99,19.29 c-0.45,9.21,6.35,16.71,15.73,16.97c7.94,0.21,9.27,0.78,10.69,8.61c5.23,28.73,19.4,53.73,32.21,79.33 c1.95,3.9,4.32,7.71,5.51,11.84c1.03,3.61,0.66,7.61,0.91,11.45c-0.73,0.14-1.45,0.28-2.18,0.42c-0.49-1.57-0.98-3.15-1.47-4.72 c-0.22,0.09-0.44,0.19-0.66,0.28c-0.85-2.62-1.7-5.24-2.74-8.45c-0.9,2.53-1.55,4.4-2.21,6.26c-0.41-0.03-0.83-0.06-1.24-0.08 c-0.19-2.78-0.35-5.56-0.56-8.34c-0.67-9.04-7.05-14.8-12.04-21.47c-5.2-6.95-10.31-14.09-14.36-21.73 c-3.56-6.7-5.59-14.21-9-21.29c-3.02,9.7-8.69,18.66-6.3,29.2c0.63,2.78,2.68,5.21,3.87,7.9c4.73,10.64,5.56,22.14,6.92,33.46 c1.21,10.13,1.88,20.38,1.96,30.59c0.06,7.02-1.67,14.04-1.85,21.08c-0.12,4.66,0.83,9.41,1.73,14.03 c1.21,6.22,2.81,12.36,4.28,18.52c0.3,1.26,0.69,2.51,1.23,3.69c3.92,8.54,7.79,17.1,11.88,25.55c1.3,2.67,3.24,5.04,5.07,7.83 c-2.19,0.86-3.64,1.76-5.17,1.97c-3.53,0.47-6.9,0.64-8.13-4.11c-1.71-6.58-3.78-13.07-5.87-19.54c-0.44-1.35-1.6-2.47-3.21-3.33 c0,16.17-7.35,32.86,6.17,48.11c-3.55,0-5.95,0.01-8.36,0c-7.59-0.03-7.66-0.54-7.72-7.64c-0.11-13.74-0.69-27.4-5.27-40.71 c-1.72-5.01-0.38-11.01-1.01-16.49c-0.67-5.79-2.11-11.48-3.08-17.24c-2.52-14.91-12.01-26.06-20.01-38.12 c-5.34-8.06-10.18-16.56-14.25-25.32c-5.18-11.16-5.52-22.61,1.24-33.57c3.68-5.96,3.12-12.27,1.17-18.55 c-2.5-8.03-5.22-16-8.05-24.61c-0.91,1.44-1.76,2.86-2.68,4.24C32.9-10.29,28.04-2.46,22.63,4.96c-5.34,7.34-14.22,8.45-22.08,10.9 c-8.48,2.65-17.2,4.46-23.03,12.01c-1.84,2.39-3.61,4.84-5.41,7.26c-0.39-0.17-0.78-0.34-1.16-0.51c0.81-2.38,1.62-4.76,2.43-7.14 c-0.2-0.22-0.39-0.44-0.59-0.66c-1.24,1.3-2.31,2.88-3.77,3.83c-2.54,1.66-5.33,2.94-8.02,4.37C-39,34.36-39,33.7-39,33.03z',
    'path://M80,100.49c0,5.23,0.13,10.46-0.03,15.69c-0.2,6.3-0.57,12.6-0.99,18.9 c-0.94,14.08-2.08,28.14-2.87,42.22c-0.41,7.29,4.95,14.31,12.03,16.62c1.22,0.4,2.43,0.84,3.65,2.16c-1.8,0.35-3.59,0.91-5.4,1 c-5.4,0.3-10.83,0.7-16.22,0.42c-1.44-0.07-3.7-2.25-3.95-3.74c-0.56-3.4,0.14-6.98-0.13-10.45c-0.77-9.67-0.8-19.56-3-28.92 c-1.97-8.39-2.18-16.07-0.02-24.35c1.28-4.91,1.34-10.48,0.5-15.52c-2.09-12.71-4.95-25.31-7.65-37.92 c-0.34-1.57-1.3-3.33-2.52-4.33c-3.71-3.01-7.37-6.38-11.62-8.38c-13.61-6.41-19.23-28.93-9.14-42.66 c5.41-7.36,5.32-13.85,0.74-21.4c-4.33-7.14-7.8-14.79-11.71-22.32C16.35-14.03,11.08-4.82,4.94,3.76 C1.8,8.13-2.43,12.19-7.04,14.93c-5.3,3.15-11.39,5.39-17.43,6.76c-9.05,2.05-14.31,7.59-17.67,15.68 c-0.43,1.05-1.13,1.99-1.76,2.95c-0.15,0.22-0.52,0.29-1.8,0.94c0.32-2.2,0.61-3.74,0.74-5.3c0.09-1.14-0.04-2.3-0.07-3.46 c-1.38,0.26-3.21,0.05-4.06,0.86c-2,1.91-3.5,4.33-5.27,6.49c-0.5,0.61-1.22,1.03-1.95,1.61c-1.02-5.19,1.42-10.27,7.11-13.9 C-36.09,19.24-22.82,11.2-9.77,2.82c2.12-1.36,3.99-3.6,5.17-5.85C1.52-14.72,7.44-26.52,13.29-38.35 c2.21-4.48,5.11-7.27,10.48-7.83c3.23-0.34,6.27-2.47,9.89-4.01c-4.23-4.83-8.31-8.74-11.49-13.28c-6.34-9.03-7.03-22.38,3.14-29.92 c6.9-5.12,13.79-4.47,20.85,0.69c6.15,4.5,6.15,11.2,7.55,17.13c1.32,5.6,0.82,11.84,0.1,17.67c-0.73,5.9-0.29,7.53,5.3,8.73 c0.96,0.21,1.99,0.17,2.98,0.19C72.51-48.76,74.44-47.06,76-36.52c1.83,12.35,2.1,25.03,6.99,36.77 c3.28,7.88,6.57,15.79,10.47,23.38c3.66,7.12,8.05,13.87,12.25,20.7c2.97,4.84,3.11,12.13-0.65,17c-1.8-2.05-3.45-3.92-5.01-5.7 c0.04-0.04-0.45,0.53-1.46,1.71C94.83,37.86,80.48,24.72,71.82,8.18c0.46,3.43,0.09,7.26,1.54,10.2c3.95,8.01,1.92,16.67,3.56,24.91 c1.63,8.22,1.87,16.74,3.79,24.88c0.88,3.73,4.32,6.84,6.58,10.25c1.09,1.65,2.2,3.29,3.17,5.01c4.84,8.58,9.09,17.55,14.58,25.69 c7.27,10.79,15.21,21.16,23.39,31.28c6.19,7.67,13.08,14.8,19.92,21.92c2.93,3.04,6.54,5.42,9.96,8.2 c-6.92,4.09-12.67,3.33-19.87-2.17c-1.82-1.39-3.76-2.79-5.87-3.62c-4.12-1.63-4.47-4.54-3.73-8.3c0.26-1.33,0.17-3.42-0.66-4.18 c-7.53-6.87-14.85-14.07-23.04-20.07c-7.75-5.68-12.26-13.2-16.11-21.54c-1.44-3.12-3.31-6.06-5.14-8.98 c-0.5-0.8-1.57-1.24-2.38-1.85C81.01,100.03,80.5,100.26,80,100.49z',
    'path://M-57,41.03c3.65-4.15,7.17-8.43,10.98-12.42c6.53-6.83,13.31-13.41,19.84-20.23 c1.76-1.84,3.51-3.98,4.4-6.31c3.8-9.99,6.99-20.23,10.99-30.14c2.74-6.79,5.65-13.62,12.37-17.95c4.17-2.68,5.12-7.31,4.29-11.96 c-0.3-1.67-2.02-3.08-3.35-4.97c-2.57,5.59-4.62,10.03-7.21,15.66c-4.79-6.43-9.76-10.83-11.68-16.31 c-1.77-5.04-1.18-11.44,0.04-16.86c1.27-5.62,5.24-9.71,12.03-9.7c1.55,0,3.1-1.68,4.66-2.55c9.3-5.22,20.47-1.53,25.73,7.59 c4.06,7.04,4.84,14.6,5.57,22.26c0.65,6.82-0.32,7.59-8.26,8.11c0,1.97,0,3.96,0,5.95c8.01-0.17,8.01,0.43,12.02,7.52 c2.09,3.69,6.34,6.1,9.41,9.29c2.48,2.58,7.04,3.14,7.24,8c0.29,6.79,0.46,6.78-6.43,11.08c0,15.78-0.02,31.49,0.03,47.2 c0,1.23,0.29,2.51,0.71,3.67c1.64,4.59,3.27,9.19,5.13,13.7c0.79,1.92,1.88,3.83,3.26,5.36c7.54,8.36,15.45,16.41,22.75,24.96 c5.09,5.97,9.05,12.9,14.18,18.84c9.73,11.26,19.47,22.59,30.08,33c8.84,8.67,18.88,16.13,28.51,23.98 c2.52,2.06,5.48,3.58,8.27,5.36c-4.02,3.54-10.94,4.01-16.34,1.62c-4.76-2.11-9.63-4.03-14.6-5.56c-5.6-1.72-6.59-3.72-4.42-9.32 c0.47-1.22-0.12-3.8-1.11-4.5c-7.36-5.15-14.66-10.53-22.55-14.78c-8.49-4.57-15.35-10.3-19.59-19.04 c-4.29-8.84-11.6-14.85-19.48-20.29c-3.2-2.21-6.43-4.4-9.64-6.6c-0.53,0.17-1.05,0.33-1.58,0.5c-0.11,11.17,0.12,22.36-0.45,33.51 c-0.29,5.72-2.33,11.33-3,17.05c-1.68,14.31-3.04,28.65-4.51,42.98c-0.34,3.34,0.94,5.76,4.12,7.18c6.09,2.73,12.14,5.56,18.61,9.26 c-3.96,0.36-7.93,0.72-11.89,1.08c-4.92,0.45-9.91,0.53-14.76,1.42c-6.96,1.28-9.68-0.99-8.69-8.02c1.73-12.28,0.67-24.36-1.4-36.56 c-1.08-6.36-2.02-14.02,0.49-19.47c5.62-12.19,2.4-23.48,0.01-35.2c-2.05-10.04-3.8-20.14-5.9-30.17c-0.32-1.52-1.72-2.91-2.87-4.13 c-3.6-3.83-8.03-7.09-10.85-11.41c-6.61-10.14-2.6-19.6,3.74-28.13c5.27-7.1,6.85-14.1,2.15-21.95c-3.79-6.34-7.53-12.7-11.38-19 c-0.46-0.75-1.41-1.2-2.77-2.3c-3.27,7.28-6.98,13.9-9.24,20.98c-3.58,11.2-12.11,17.05-21.53,22.3c-1.86,1.04-3.57,2.44-5.53,3.21 c-4.29,1.67-6.09,3.88-4.9,9.01c0.69,2.96-1.31,6.55-2.1,9.86c-0.5,0.03-0.99,0.06-1.49,0.08c-0.18-2.57-0.36-5.14-0.66-9.41 c-3.45,4.38-6.11,7.75-9.33,11.84c-1.07-2.08-1.61-3.13-2.15-4.18C-57,43.7-57,42.36-57,41.03z'
    ]


    d = []
    df1 = data[data['ÉLIGIBILITÉ_AU_DON.'] == 'Temporairement Non-eligible']
    for col in ['La DDR est mauvaise','Allaitement','A accouchée ces 6 derniers mois',
        'Interruption de grossesse ces 06 derniers mois', 'Est enceinte'] : 
        val1 = df1[col].value_counts().sum() - df1[col].value_counts()['Non']
        d.append(val1)
        
        
    e = []
    for col in ['Antécédent de transfusion', 'Porteur(HIV,hbs,hcv)','Opéré',
        'Tatoué', 'Diabétique'] : 
        val = dat[col].value_counts().sum() - dat[col].value_counts()['Non']
        e.append(val)

    d2 = []
    df1 = data[data['ÉLIGIBILITÉ_AU_DON.'] == 'Temporairement Non-eligible']
    for col in ['La DDR est mauvaise','Allaitement','A accouchée ces 6 derniers mois',
        'Interruption de grossesse ces 06 derniers mois', 'Est enceinte'] : 
        val1 = df1[col].value_counts()['Non']
        d2.append(val1)
        
        
    e2 = []
    for col in ['Antécédent de transfusion', 'Porteur(HIV,hbs,hcv)','Opéré',
        'Tatoué', 'Diabétique'] : 
        val = dat[col].value_counts()['Non']
        e2.append(val)

    d = [int(val) for val in d]  # Convert temporary ineligibility counts
    e = [int(val) for val in e] 
    d2 = [int(val) for val in d2]  # Convert temporary ineligibility counts
    e2 = [int(val) for val in e2] 




    labels_grossesse = [
        'DDR mauvaise', 
        'Allaitement', 
        'Accouchée (6 mois)',
        'Interruption grossesse', 
        'Enceinte'
    ]

    labels_medical = [
        'Antécédent de transfusion', 
        'Porteur (HIV,hbs,hcv)',
        'Opéré', 
        'Tatoué', 
        'Diabétique'
    ]


    # Définir les options ECharts
    options = {
        "tooltip": {},
        "legend": {
            "data": ['Definitely non-eligible', "Temporarily Non eligible"],
            "selectedMode": "single"
            #"selected": {"Critères Grossesse": True, "Critères Médicaux": False},
        },
        "formatter": 2,
        "xAxis": {
            "type": "category",
            "data": labels_medical,
            
        },
        "yAxis": {
            "max": 15,
            "offset": 20,
            "splitLine": {"show": False}
        },
        "grid": {
            "top": "center",
            "height": 230
        },
        "markLine": {
            "z": -100
        },
        "series": [
            
            {
                "name": 'Definitely non-eligible',
                "type": "pictorialBar",
                "symbolClip": True,
                "symbolBoundingData": 15,
                "label": {
                    "show": True,
                    "position": "top",
                    "offset": [0, -20],
                    "formatter": 5,
                    "fontSize": 12,
                    "fontFamily": "Arial"
                },
                "data": [
                    {"value": e[0], "symbol": symbols[0]},
                    {"value": e[1], "symbol": symbols[1]},
                    {"value": e[2], "symbol": symbols[2]},
                    {"value": e[3], "symbol": symbols[3]},
                    {"value": e[4], "symbol": symbols[4]}
                ],
                "markLine": {
                    "symbol": "none",
                    "lineStyle": {"opacity": 0.2},
                    "data": [
                        {"type": "max", "label": {"formatter": "max: {c}"}},
                        {"type": "min", "label": {"formatter": "min: {c}"}}
                    ]
                },
                "z": 10
            },
            {
                "name": "Temporarily Non eligible",
                "type": "pictorialBar",
                "symbolClip": True,
                "symbolBoundingData": 15,
                "label": {
                    "show": True,
                    "position": "top",
                    "offset": [0, -20],
                    "formatter": 5,
                    "fontSize": 12,
                    "fontFamily": "Arial"
                }, 
                "data": [
                    {"value": d[0], "symbol": symbols[0]},
                    {"value": d[1], "symbol": symbols[1]},
                    {"value": d[2], "symbol": symbols[2]},
                    {"value": d[3], "symbol": symbols[3]},
                    {"value": d[4], "symbol": symbols[4]}
                ],
                "markLine": {
                    "symbol": "none",
                    "lineStyle": {"opacity": 0.3},
                    "data": [
                        {"type": "max", "label": {"formatter": "max: {c}"}},
                        {"type": "min", "label": {"formatter": "min: {c}"}}
                    ]
                },
                "z": 10
            },
            {
                "name": 'Non',
                "type": "pictorialBar",
                "symbolBoundingData": 15,
                "animationDuration": 0,
                "itemStyle": {"color": "#ccc"},
                "data": [
                    {"value": d2[0], "symbol": symbols[0]},
                    {"value": d2[1], "symbol": symbols[1]},
                    {"value": d2[2], "symbol": symbols[2]},
                    {"value": d2[3], "symbol": symbols[3]},
                    {"value": d2[4], "symbol": symbols[4]}
                ]
            },
            {
                "name": "Non",
                "type": "pictorialBar",
                "symbolBoundingData": 15,
                "animationDuration": 0,
                "itemStyle": {"color": "#ccc"},
                "data": [
                    {"value": e2[0], "symbol": symbols[0]},
                    {"value": e2[1], "symbol": symbols[1]},
                    {"value": e2[2], "symbol": symbols[2]},
                    {"value": e2[3], "symbol": symbols[3]},
                    {"value": e2[4], "symbol": symbols[4]}
                ]
            }
        ]
    }
    return options
def women_reasons(df):
    op= options_women_reasons(df)
    st_echarts(options=op, height="350px")


@st.cache_resource
def prepas_horodateur(df):
    df['Horodateur'] = pd.to_datetime(df['Horodateur'])
    df['Jour'] = df['Horodateur'].dt.day_name()
    df['Heure'] = df['Horodateur'].dt.hour
    df['JourIndex'] = df['Horodateur'].dt.dayofweek  # Lundi=0, Dimanche=6

    # Préparation des données pour ECharts
    hours = [
        "12a", "1a", "2a", "3a", "4a", "5a", "6a",
        "7a", "8a", "9a", "10a", "11a",
        "12p", "1p", "2p", "3p", "4p", "5p",
        "6p", "7p", "8p", "9p", "10p", "11p"
    ]

    days = [
        "Monday", "Tuesday", "Wednesday", 
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    # Créer un dictionnaire pour compter les occurrences
    activity_count = {}
    for _, row in df.iterrows():
        day_idx = row['JourIndex']
        hour = row['Heure']
        key = (day_idx, hour)
        if key in activity_count:
            activity_count[key] += 1
        else:
            activity_count[key] = 1

    # Transformer en format attendu par ECharts
    heatmap_data = []
    for key, count in activity_count.items():
        day_idx, hour = key
        heatmap_data.append([day_idx, hour, count])


    st.markdown(
        "<h1 style='font-size: 36px; color: #2c3e50; text-align: center;'>Visualisation d'activité hebdomadaire</h1>",
        unsafe_allow_html=True
    )
    # Statistiques résumées
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total d'activités", len(df))
    with col2:
        st.metric("Jour le plus actif", df['Jour'].value_counts().idxmax())
    with col3:
        st.metric("Heure la plus active", f"{df['Heure'].value_counts().idxmax()}h")

    # Préparation des options ECharts
    title_options = []
    single_axis_options = []
    series_options = []

    for idx, day in enumerate(days):
        title_options.append({
            "textBaseline": "middle",
            "top": f"{((idx + 0.5) * 80 / 7)}%",
            "text": day
        })
        
        single_axis_options.append({
            "left": 150,
            "type": "category",
            "boundaryGap": False,
            "data": hours,
            "top": f"{(idx * 80 / 7 + 5)}%",
            "height": f"{80 / 7 - 10}%",
            "axisLabel": {"interval": 2}
        })
        
        # Préparer les données pour cette série
        series_data = []
        for item in heatmap_data:
            if item[0] == idx:
                series_data.append({
                    "value": [item[1], item[2]],
                    "symbolSize": item[2] / 1.5   # Taille basée sur la valeur
                })
        
        series_options.append({
            "singleAxisIndex": idx,
            "coordinateSystem": "singleAxis",
            "type": "scatter",
            "data": series_data,
            "symbolSize": 0.2  # Valeur par défaut (sera écrasée par symbolSize dans les données)
        })

    # Options finales
    options = {
        "tooltip": {
            "position": "top",
            "formatter": 0.2
        },
        "backgroundColor": "#fff",
        "title": title_options,
        "singleAxis": single_axis_options,
        "series": series_options
    }
    return options
    # Affichage dans Streamlit
    #st.title("Visualisation d'activité hebdomadaire")
def horodateur(df):
    st_echarts(
        options=prepas_horodateur(df),
        height="500px",
        key="heatmap"
    )
@st.cache_resource
def generate_wordcloud(column_data, colormap='Reds', mask_path="h.jpg"):
    # Téléchargement des stopwords français si nécessaire
    try:
        stopwords.words('french')
    except:
        nltk.download('stopwords')

    # Liste personnalisée de stopwords
    stopwords_custom = set(STOPWORDS)
    stopwords_custom.update(["autre", "raison", "non", "don", "de", "la", "le", "et"])
    # Nettoyer les données
    cleaned_data = column_data.dropna().astype(str)
    texte = " ".join(cleaned_data)
    texte_nettoye = texte.lower().translate(str.maketrans('', '', string.punctuation))
    
    # Filtrer les stopwords
    stop_words = set(stopwords.words('french'))
    stop_words.update(stopwords_custom)
    mots = texte_nettoye.split()
    mots_filtres = [mot for mot in mots if mot not in stop_words]
    texte_filtre = ' '.join(mots_filtres)
    
    # Utiliser le masque si fourni
    mask = None
    if mask_path:
        mask = np.array(Image.open(mask_path))
        mask = 255 - mask
    
    # Générer le nuage de mots
    wordcloud = WordCloud(
        width=1000,
        height=1000,
        background_color='white',
        mask=mask,
        colormap='Reds',  # Palette rouge fixe
        stopwords=stop_words
    ).generate(texte_filtre)
    
    # Créer la figure Plotly
    image_wc = wordcloud.to_array()
    fig = px.imshow(image_wc)
    fig.update_layout(
        xaxis_visible=False,
        yaxis_visible=False,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_resource
def generate_blood_drop_wordcloud(series):
    # -------------------------------------------------------------
    # Sous-fonction : Créer un masque en forme de goutte de sang
    # -------------------------------------------------------------
    def create_drop_mask(width=800, height=800):
        img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(img)

        x0, y0 = width // 2, height // 3
        drop_width = width * 0.6
        drop_height = height * 0.7

        # Partie arrondie
        draw.ellipse([x0 - drop_width/2, y0, x0 + drop_width/2, y0 + drop_height], fill=255)

        # Triangle supérieur
        triangle_height = height * 0.2
        draw.polygon([
            (x0, y0 - triangle_height),
            (x0 - drop_width/4, y0),
            (x0 + drop_width/4, y0)
        ], fill=255)

        return np.array(img)

    # -------------------------------------------------------------
    # Sous-fonction : Nettoyage et comptage des mots
    # -------------------------------------------------------------
    def get_word_frequencies(series):
        # Nettoyage basique
        text = ' '.join(series.dropna().astype(str).str.lower())
        text = re.sub(r"[^a-zàâäéèêëîïôöùûüç\s\-]", " ", text)
        words = text.split()
        return Counter(words)

    # -------------------------------------------------------------
    # Étapes principales
    # -------------------------------------------------------------
    # Masque goutte
    mask = create_drop_mask()

    # Fréquences de mots
    frequencies = get_word_frequencies(series)

    # Stopwords personnalisés
    stopwords_custom = set(STOPWORDS)
    stopwords_custom.update(["autre", "raison", "non", "don", "de", "la", "le", "et", "l", "les", "des", "un", "une", "du", "en", "pour"])

    # Génération du nuage
    wordcloud = WordCloud(
        width=1000,
        height=1000,
        background_color='white',
        mask=255 - mask,
        colormap='Reds',
        stopwords=stopwords_custom
    ).generate_from_frequencies(frequencies)

    # Affichage avec Plotly
    image_wc = wordcloud.to_array()
    fig = px.imshow(image_wc)
    fig.update_layout(
    
        xaxis_visible=False,
        yaxis_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)


import matplotlib.pyplot as plt
import io
from collections import Counter
import streamlit as st

def prepare_frequency_donut_plot(data_list):
    """
    Prepares and returns a donut chart from a list of values.
    Also displays it with Streamlit and returns the image buffer.
    """
    # Count the frequency of each unique value
    frequency = Counter(data_list)

    # Extract labels and sizes
    labels = list(frequency.keys())
    sizes = list(frequency.values())

    # Create the figure
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(width=0.4, edgecolor='w')  # Donut effect
    )
    ax.axis('equal')  # Ensures the pie is a circle
    plt.title('Frequency Distribution')

    # Save figure to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return buf
