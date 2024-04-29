import joblib
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words_en = stopwords.words('English')
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
import plotly_express as px
import re 
import streamlit as st
import pandas as pd
from PIL import Image


#Importamos DataFrame
df = pd.read_csv(r'D:\Usuarios\Ivan\Desktop\Disney\Diney_agency\df_Binario.csv', encoding='ISO-8859-1')

#Importamos modelos de IA
modeloSVC = joblib.load(r'D:\Usuarios\Ivan\Desktop\Disney\Diney_agency\ModeloSVC.joblib')
modelRegresion = joblib.load(r'D:\Usuarios\Ivan\Desktop\Disney\Diney_agency\RegresionL3.joblib')
tfidf = joblib.load(r'D:\Usuarios\Ivan\Desktop\Disney\Diney_agency\tfidf_vectorizer3.joblib')

#Preprocesamiento de la información
def limpiar(texto):
    res = texto.lower()
    res = re.sub(r'[^a-zA-Z0-9\s]', '', res)
    res = word_tokenize(res)
    res = [lemmatizer.lemmatize(token) for token in res]
    res = [token for token in res if token not in stop_words_en]
    res = ' '.join(res)
    return res

def predecir_sentimiento_R(texto):
    texto_limpo = limpiar(texto)
    vectorizado = tfidf.transform([texto_limpo])
    prediccion = modelRegresion.predict(vectorizado)
    return prediccion[0]

# Agrupar el DataFrame por la columna 'Reviewer_Location' y contar las ocurrencias
df_grouped = df['Reviewer_Location'].value_counts().reset_index()
df_grouped.columns = ['Reviewer_Location', 'Count']

# Crear un mapa del mundo con Plotly Express
fig = px.choropleth(
    df_grouped,
    locations='Reviewer_Location',
    locationmode='country names',
    color='Count',
    hover_name='Reviewer_Location',
    title='Number of Reviews per country',
    color_continuous_scale=px.colors.sequential.Plasma,  # Puedes cambiar la escala de color según tu preferencia
)

# Establecer el fondo como transparente
fig.update_layout(
    geo=dict(
        bgcolor='rgba(0,0,0,0)',
        showframe=False,
        showcoastlines=False,
        projection_type="natural earth",  # Puedes cambiar la proyección según tu preferencia
    )
)

# Agrupar el DataFrame por 'Year' y 'Branch' y contar las ocurrencias
df_grouped = df.groupby(['Year', 'Branch']).size().reset_index(name='Count')

# Crear la gráfica de barras
bar_year = px.bar(
    df_grouped,
    x='Year',
    y='Count',
    color='Branch',
    color_discrete_sequence=px.colors.qualitative.Dark24,
    title='Total Reviews per Year and Park',
    labels={'Count': 'Total Reviews', 'Year': 'Año'},
)

# Agrupar el DataFrame por 'Sentiment' y contar las ocurrencias
df_sentiment_count = df['Sentiment'].value_counts().reset_index()
df_sentiment_count.columns = ['Sentiment', 'Count']

# Crear la gráfica de barras
bar_sent = px.bar(
    df_sentiment_count,
    x='Sentiment',
    y='Count',
    color='Sentiment',
    color_discrete_sequence=px.colors.qualitative.G10,
    title='Total Reviews',
    labels={'Count': 'Total Reviews', 'Sentiment': 'Sentimiento'},
)


bigram_Good = Image.open(r'D:\Usuarios\Ivan\Desktop\Disney\Diney_agency\Bigramas_Good_Binario.png')
Trigram_Good = Image.open(r'D:\Usuarios\Ivan\Desktop\Disney\Diney_agency\Trigramas_Good_Binario.png')
bigram_Bad = Image.open(r'D:\Usuarios\Ivan\Desktop\Disney\Diney_agency\Bigramas_Bad_Binario.png')
Trigram_Bad = Image.open(r'D:\Usuarios\Ivan\Desktop\Disney\Diney_agency\Trigramas_Bad_Binario.png')


st.markdown("<h1 style='text-align: center; color: white;'>Disneyland Adventure Agency</h1>", unsafe_allow_html=True)
# st.write(df.head(5))
st.markdown("<h2 style='text-align: center; color: white;'>Binary Classification</h2>", unsafe_allow_html=True)
st.write(bar_sent)
st.write(fig)
st.write(bar_year)
col1,col2 = st.columns(2)
col1.write('### Bigrams')
col1.write('### \n ')

# Organizar las imágenes en una fila en la primera columna
col1.image(bigram_Good)
col1.image(bigram_Bad)

col2.write('### Trigrams')
col2.write('### \n ')

# Organizar las imágenes en una fila en la segunda columna
col2.image(Trigram_Good)
col2.image(Trigram_Bad)

st.write('### Say if a review is good or bad')
texto = st.text_input("Enter a review:")
predictionLR = predecir_sentimiento_R(texto)
predictionSVC = modeloSVC.predict([texto])

if (texto):
    with st.spinner('Esperate que ando chambeando...'):
        st.write(f'This is a {predictionLR} review. According to Logistic Regresion model')
        st.write(f'This is a {predictionSVC[0]} review. According to SVC model')


bigram_P = Image.open(r'D:\Usuarios\Ivan\Desktop\Disney\Diney_agency\Bigramas_Positive.png')
Trigram_P = Image.open(r'D:\Usuarios\Ivan\Desktop\Disney\Diney_agency\Trigramas_Positive.png')
bigram_Neg = Image.open(r'D:\Usuarios\Ivan\Desktop\Disney\Diney_agency\Bigramas_Negative.png')
Trigram_Neg = Image.open(r'D:\Usuarios\Ivan\Desktop\Disney\Diney_agency\Trigramas_Negative.png')
bigram_Neu = Image.open(r'D:\Usuarios\Ivan\Desktop\Disney\Diney_agency\Bigramas_Neutral.png')
Trigram_Neu = Image.open(r'D:\Usuarios\Ivan\Desktop\Disney\Diney_agency\Trigramas_Neutral.png')

st.markdown("<h2 style='text-align: center; color: white;'>Multi-class Classification</h2>", unsafe_allow_html=True)

col3,col4 = st.columns(2)
col3.write('### Bigrams')
col3.write('### \n ')

# Organizar las imágenes en una fila en la primera columna
col3.image(bigram_P)
col3.image(bigram_Neg)
col3.image(bigram_Neu)

col4.write('### Trigrams')
col4.write('### \n ')

# Organizar las imágenes en una fila en la segunda columna
col4.image(Trigram_P)
col4.image(Trigram_Neg)
col4.image(Trigram_Neu)

st.write('### Say if a review is good or bad')
texto = st.text_input("Enter a review:")
#AQUI ESCRIBE EL METODO PARA PREDECIR
if (texto):
    with st.spinner('Esperate que ando chambeando...'):
        st.write(f'This is a {predictionLR} review. According to CNN model')