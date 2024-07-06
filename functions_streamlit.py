
from pickle import TRUE
import streamlit as st
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.express as px
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle

from variables import df

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('datasets/Data_Reducida_Final.csv')
df_city = pd.read_csv('datasets/Data_City_Reducida_Final.csv')
df_cluster_scaled = pd.read_csv('datasets/Data_Reducida_Final_Cluster_Scaled.csv')
cluster_4 = pd.read_csv('datasets/Cluster_4.csv')
kmeans_4 = pickle.load(open('models/kmeans.pkl', 'rb'))
scaler_final = pickle.load(open('models/scaler.pkl', 'rb'))



# HOME

def eda():
    
    st.title('Sistema de recomendación, como aplicar machine learning y generativas a un negocio.')

    st.subheader('Autor:\nMiguel S. La Cruz.')
    st.image('img/data_science_19.png', caption='Clustering de datos')
    st.markdown('''Este proyecto combina análisis de datos, visualización interactiva y redes neuronales para la práctica de uso de apis. 
                Utilizando tecnologías como Streamlit, DALL-E y Whisper, el objetivo de este taller es ofrecer una plataforma donde 
                aquellos interesados puedan interactuar directamente con apis, exploración visual,segmentación de clientes y
                generación de imágenes personalizadas. Diseñado para ser accesible y participativo.''')

    st.header('Puesta en escena.')

    st.image('img/data_science_23.png', caption='E-Commerce')

    #Descripción de un problema de negocio de un e-commerce y cómo se puede resolver con un sistema de recomendación
    st.subheader('Problema de negocio')
    st.write('Imaginemos que somos propietarios de un e-commerce y queremos aumentar las ventas de nuestra tienda online. \
        Para ello, necesitamos ofrecer a nuestros clientes productos que sean de su interés, y que se ajusten a sus preferencias. \
        ¿Cómo podemos hacerlo?')
    
    st.subheader('Solución')

    st.write('Una solución a este problema es implementar un sistema de recomendación. Un sistema de recomendación es un algoritmo \
        que analiza los datos de los usuarios y sus preferencias, y sugiere productos que podrían interesarles. \
        De esta forma, podemos ofrecer a nuestros clientes productos que se ajusten a sus gustos y necesidades, lo que aumentará \
        la probabilidad de que realicen una compra.')
    
    st.image('img/data_science_20.png', caption='Sistema de recomendación')

    
    st.subheader('Beneficios de un sistema de recomendación')

    st.write('Un sistema de recomendación puede aportar numerosos beneficios a un e-commerce, entre los que destacan:')
    st.write('- Aumento de las ventas: Al ofrecer a los clientes productos que se ajusten a sus preferencias, aumentamos la probabilidad de que realicen una compra.')
    st.write('- Mejora de la experiencia del usuario: Los clientes se sentirán más satisfechos al recibir recomendaciones personalizadas.')
    st.write('- Fidelización de clientes: Al ofrecer una experiencia de compra personalizada, aumentamos la fidelidad de los clientes a nuestra tienda online.')

    st.subheader('Tipos de sistemas de recomendación')
    st.image('img/data_science_21.png', caption='Tipos de sistemas de recomendación')

    st.write('Existen varios tipos de sistemas de recomendación, nosotros por simplicidad utilizaremos un sistema de recomendación híbrido,\
            entre filtrado colaborativo y clustering')
    st.image('img/data_science_22.png', caption='Sistema de recomendación híbrido')

    st.header('Análisis de datos exploratorio.')
    st.write('Para implementar un sistema de recomendación, primero necesitamos analizar los datos de los clientes y los productos. \
             Haremos un análisis exploratorio de los datos corto para entender mejor la información con la que trabajamos.')
    st.subheader('Ubicación de los clientes en Europa.')

    st.write('Vamos a observar donde se ubican la mayoría de nuestros clientes, para ello vamos a utilizar un mapa interactivo.')
    include_names = st.checkbox('Incluir nombres de los países sobre el mapa')
    
    # Hacer un plot de ubicaciones de los clientes con plotly scatter_geo
    if include_names:

        fig_geo_eur = px.scatter_geo(df_city, lat='Latitud', lon='Longitud', color='GastoMensual',  
                                     hover_data = {'Ubicacion': True, 'GastoMensual': True, 
                                    'Latitud': False, 'Longitud': False},
                     hover_name='Ubicacion', size=df_city['GastoMensual'] * 1000, projection='natural earth', scope= 'europe',
                     color_continuous_scale="Cividis",
                     text = 'Ubicacion',
                     title= 'Ventas en Europa por ciudad')
        fig_geo_eur.update_geos(showcountries=True, countrycolor="Black", showland=True, showocean=True, oceancolor="LightBlue", landcolor="LightGreen")
        fig_geo_eur.update_layout(
        annotations=[dict(
            x=0.5, 
            xanchor='center',
            y=-0.1,  # Posición respecto al gráfico
            yanchor='bottom',
            text='Ventas en Europa por ciudad. En Madrid se gasta mucho y en Vienna poco.',
            showarrow=False,
            font=dict(
                size=12,
                color="grey"
                    )
                )]
            )
        st.plotly_chart(fig_geo_eur)
        
    else:
        fig_geo_eur = px.scatter_geo(df_city, lat='Latitud', lon='Longitud', color='GastoMensual', 
                                     hover_data = {'Ubicacion': True, 'GastoMensual': True, 
                                    'Latitud': False, 'Longitud': False},
                    hover_name='Ubicacion',
                    size=df_city['GastoMensual'] * 1000, projection='natural earth', scope= 'europe',
                     color_continuous_scale="Cividis",
                     title= 'Ventas en Europa por ciudad')
        
        fig_geo_eur.update_geos(showcountries=True, countrycolor="Black", showland=True, showocean=True, oceancolor="LightBlue", landcolor="LightGreen")
        fig_geo_eur.update_layout(
        annotations=[dict(
            x=0.5, 
            xanchor='center',
            y=-0.1,  # Posición respecto al gráfico
            yanchor='bottom',
            text='Ventas en Europa por ciudad. En Madrid se gasta mucho y en Vienna poco.',
            showarrow=False,
            font=dict(
                size=12,
                color="grey"
            )
        )]
        )
        st.plotly_chart(fig_geo_eur)

    st.write(''' \n\n''')

    
    st.subheader('Correlación entre variables.')

    st.write('Vamos a analizar la correlación entre las variables de nuestro dataset para entender mejor las relaciones entre ellas.')
    
    cmap = px.imshow(df.drop(columns=['Cluster', 'Latitud', 'Longitud']).corr().round(2), color_continuous_scale='Cividis', title='Mapa de correlación', text_auto=True)

    st.plotly_chart(cmap)

    st.write('En el mapa de correlación podemos observar variables correlacionadas, como gasto mensual con valor promedio de transacciones.')

    st.subheader('Distribución de variables.')

    st.write('Podemos obtener información inicial interesante como máximos y mínimos, desviación estándar y distribución de las variables.\
             con un análisis gráfico de las variables de nuestro dataset. Utilizaremos box plots y gráficos de densidad para esta tarea.')
    
    slider_distribuciones = st.select_slider('Selecciona variable a visualizar', options=['All', 'Días promedio entre compras', 'Gasto mensual', 'Desviación de gasto mensual', 'Promedio por transacción'])

    colors = {
    'Días promedio entre compras': '#00CC96',  # Rojo suave
    'Gasto mensual': '#EF553B',  # Verde suave
    'Desviación de gasto mensual': '#AB63FA',  # Morado suave
    'Promedio por transacción': '#FFA15A'  # Naranja suave
}

    if slider_distribuciones == 'All':
        df_long = df.drop(columns=['Cluster', 'Latitud', 'Longitud', 'Ubicacion', 'CategoriaProductoFavorito', 'CategoríaProductoFavorito']).melt(var_name='Variable', value_name='Value')
        boxplot_all = px.box(df_long, x='Value', y='Variable', color='Variable', title="Boxplot de Múltiples Variables", 
                             orientation='h', color_discrete_sequence=px.colors.qualitative.Prism)
        st.plotly_chart(boxplot_all)

    elif slider_distribuciones == 'Días promedio entre compras':
        density_plot = px.violin(df, x='Average_Days_Between_Purchases', box=True, points='all', title='Días promedio entre compras')
        density_plot.update_traces(marker_color=colors[slider_distribuciones], line_color=colors[slider_distribuciones])
        st.plotly_chart(density_plot)

    elif slider_distribuciones == 'Gasto mensual':
        density_plot = px.violin(df, x='GastoMensual', box=True, points='all', title='Gasto Mensual')
        density_plot.update_traces(marker_color=colors[slider_distribuciones], line_color=colors[slider_distribuciones])
        st.plotly_chart(density_plot)

    elif slider_distribuciones == 'Desviación de gasto mensual':
        density_plot = px.violin(df, x='Monthly_Spending_Std', box=True, points='all', title='Desviación de gasto mensual')
        density_plot.update_traces(marker_color=colors[slider_distribuciones], line_color=colors[slider_distribuciones])
        st.plotly_chart(density_plot)

    else:
        density_plot = px.violin(df, x='Average_Transaction_Value', box=True, points='all', title='Promedio por transacción')
        density_plot.update_traces(marker_color=colors[slider_distribuciones], line_color=colors[slider_distribuciones])
        st.plotly_chart(density_plot)
    
    st.write('En los gráficos de densidad podemos observar la distribución de las variables de nuestro dataset. \
             dandonos cuenta por ejemplo que hay algún grupo de clientes que gasta mucho dinero al mes.')
    
    st.subheader('Productos por categoría.')

    st.write('Vamos a mirar cuantas compras se han realizado por categoría de producto.')
    compras_por_categoria = px.histogram(df, x='CategoriaProductoFavorito', color='CategoriaProductoFavorito')

    st.plotly_chart(compras_por_categoria)

    st.write('Teniendo en cuenta las características de nuestra data, vamos a proceder a implementar un sistema de recomendación híbrido, \
             agrupando a los clientes en clusters y recomendando productos en base a los productos favoritos en cada cluster.')
    

def clustering():

    st.header('Implementando clustering para segmentar a los clientes.')
    st.write('Vamos a utilizar el algoritmo K-means para agrupar a los clientes en clusters. Esto nos permitirá segmentar a los clientes\
             para posteriormente poder ofrecerles productos en base a las preferencias del grupo al que fueron asignados.')
    

    st.image('img/data_science_24.png', caption='Algoritmo K-means.')

    st.subheader('Selección de número de clusters.')

    st.write('Para seleccionar el número de clusters óptimo, normalmente se utiliza el método del codo o el método de la silueta.\
             en estos métodos se prueba con distintos números de clusters y se evalúa la calidad de los clusters obtenidos\
             en base a los valores calculados de inercia o sillhouette score. No es el objetivo de este taller explicar las\
             métricas, pero crearemos agrupaciones con 2, 3, 4, y 5 clusters para ver cómo se comportan los datos.')
    
    st.image('img/data_science_25.png', caption='Método del codo.')

    st.header('Resultados de clustering.')

    st.write('Vamos a visualizar los resultados de clustering obtenidos con k-means para 2, 3, 4 y 5 clusters.')

    st.subheader('Individuos por cluster.')

    st.write('Vamos a ver cuántos individuos hay en cada cluster para cada uno de los números de clusters que hemos probado.')

    selector_n_clusters = st.select_slider('Selecciona número de clusters', options=[2, 3, 4, 5])

    if selector_n_clusters == 2:
        # Hacemos clusterizacion con 2 clusters

        kmeans = KMeans(n_clusters=2, random_state=42)

        df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

        # Mostrar número de individuos por cluster con un countplot

        n_individuos_2clusters = px.histogram(df, x='Cluster', title='Número de individuos por cluster con 2 clusters', color='Cluster')

        st.plotly_chart(n_individuos_2clusters)
    
    elif selector_n_clusters == 3:

        kmeans = KMeans(n_clusters=3, random_state=42)

        df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

        n_individuos_3clusters = px.histogram(df, x='Cluster', title='Número de individuos por cluster con 3 clusters', color='Cluster')

        st.plotly_chart(n_individuos_3clusters)
    
    elif selector_n_clusters == 4:

        kmeans = KMeans(n_clusters=4, random_state=42)

        df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

        n_individuos_4clusters = px.histogram(df, x='Cluster', title='Número de individuos por cluster con 4 clusters', color='Cluster')

        st.plotly_chart(n_individuos_4clusters)
    
    else:

        kmeans = KMeans(n_clusters=5, random_state=42)

        df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

        n_individuos_5clusters = px.histogram(df, x='Cluster', title='Número de individuos por cluster con 5 clusters', color='Cluster')

        st.plotly_chart(n_individuos_5clusters)
    
    st.write('No nos dice mucho, pero podemos ver que hay un grupo de elementos que siempre predomina, pues se parecen mucho entre ellos.')


    st.subheader('Visualización de clusters.')

    st.write('Vamos a visualizar los clusters obtenidos, para ver en detalle cómo se agrupan los individuos en cada uno de los clusters.\
             para ello utilizaremos gráficos 2D y 3D, así podremos entender cómo se diferencian los clientes.')
    
    selector_2d_3d = st.selectbox('Gráfico a visualizar', options=['2D', '3D'])

    selector_n_clusters_2 = st.select_slider('Selecciona número de clusters', options=[2, 3, 4, 5], key='2')

    if selector_2d_3d == '2D':
            
            if selector_n_clusters_2 == 2:

                kmeans = KMeans(n_clusters=2, random_state=42)

                df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

                # Hacemos pca para visualizar

                pca = PCA(n_components=2)

                df_pca = pca.fit_transform(df_cluster_scaled)

                df_pca = pd.DataFrame(data=df_pca, columns=['pca1', 'pca2'])

                df_pca['Cluster'] = df['Cluster']

                kmeans_2d_2clusters = px.scatter(df_pca, x='pca1', y='pca2', color='Cluster', title='K-Means con 2 clusters', color_continuous_scale='Cividis')

                st.plotly_chart(kmeans_2d_2clusters)
            
            elif selector_n_clusters_2 == 3:

                kmeans = KMeans(n_clusters=3, random_state=42)

                df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

                pca = PCA(n_components=2)

                df_pca = pca.fit_transform(df_cluster_scaled)

                df_pca = pd.DataFrame(data=df_pca, columns=['pca1', 'pca2'])

                df_pca['Cluster'] = df['Cluster']

                kmeans_2d_3clusters = px.scatter(df_pca, x='pca1', y='pca2', color='Cluster', title='K-Means con 3 clusters', color_continuous_scale='Cividis')

                st.plotly_chart(kmeans_2d_3clusters)
            
            elif selector_n_clusters_2 == 4:

                kmeans = KMeans(n_clusters=4, random_state=42)

                df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

                pca = PCA(n_components=2)

                df_pca = pca.fit_transform(df_cluster_scaled)

                df_pca = pd.DataFrame(data=df_pca, columns=['pca1', 'pca2'])

                df_pca['Cluster'] = df['Cluster']

                kmeans_2d_4clusters = px.scatter(df_pca, x='pca1', y='pca2', color='Cluster', title='K-Means con 4 clusters', color_continuous_scale='Cividis')

                st.plotly_chart(kmeans_2d_4clusters)
            
            else:
                    
                    kmeans = KMeans(n_clusters=5, random_state=42)
    
                    df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)
    
                    pca = PCA(n_components=2)
    
                    df_pca = pca.fit_transform(df_cluster_scaled)
    
                    df_pca = pd.DataFrame(data=df_pca, columns=['pca1', 'pca2'])
    
                    df_pca['Cluster'] = df['Cluster']
    
                    kmeans_2d_5clusters = px.scatter(df_pca, x='pca1', y='pca2', color='Cluster', title='K-Means con 5 clusters', color_continuous_scale='Cividis')
    
                    st.plotly_chart(kmeans_2d_5clusters)
            
    else:

        if selector_n_clusters_2 == 2:

            kmeans = KMeans(n_clusters=2, random_state=42)

            df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

            # Hacemos pca para visualizar

            pca = PCA(n_components=3)

            df_pca = pca.fit_transform(df_cluster_scaled)

            df_pca = pd.DataFrame(data=df_pca, columns=['pca1', 'pca2', 'pca3'])

            df_pca['Cluster'] = df['Cluster']

            kmeans_3d_2clusters = px.scatter_3d(df_pca, x='pca1', y='pca2', z='pca3', color='Cluster', title='K-Means con 2 clusters', color_continuous_scale='Cividis')

            st.plotly_chart(kmeans_3d_2clusters)
        
        elif selector_n_clusters_2 == 3:

            kmeans = KMeans(n_clusters=3, random_state=42)

            df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

            pca = PCA(n_components=3)

            df_pca = pca.fit_transform(df_cluster_scaled)

            df_pca = pd.DataFrame(data=df_pca, columns=['pca1', 'pca2', 'pca3'])

            df_pca['Cluster'] = df['Cluster']

            kmeans_3d_3clusters = px.scatter_3d(df_pca, x='pca1', y='pca2', z='pca3', color='Cluster', title='K-Means con 3 clusters', color_continuous_scale='Cividis')

            st.plotly_chart(kmeans_3d_3clusters)
        
        elif selector_n_clusters_2 == 4:

            kmeans = KMeans(n_clusters=4, random_state=42)

            df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

            pca = PCA(n_components=3)

            df_pca = pca.fit_transform(df_cluster_scaled)

            df_pca = pd.DataFrame(data=df_pca, columns=['pca1', 'pca2', 'pca3'])

            df_pca['Cluster'] = df['Cluster']

            kmeans_3d_4clusters = px.scatter_3d(df_pca, x='pca1', y='pca2', z='pca3', color='Cluster', title='K-Means con 4 clusters', color_continuous_scale='Cividis')

            st.plotly_chart(kmeans_3d_4clusters)
        
        else:
                    
            kmeans = KMeans(n_clusters=5, random_state=42)

            df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

            pca = PCA(n_components=3)

            df_pca = pca.fit_transform(df_cluster_scaled)

            df_pca = pd.DataFrame(data=df_pca, columns=['pca1', 'pca2', 'pca3'])

            df_pca['Cluster'] = df['Cluster']

            kmeans_3d_5clusters = px.scatter_3d(df_pca, x='pca1', y='pca2', z='pca3', color='Cluster', title='K-Means con 5 clusters', color_continuous_scale='Cividis')

            st.plotly_chart(kmeans_3d_5clusters)

    st.write('Los scatter plots son muy útiles para visualizar los clusters obtenidos con k-means. En estos gráficos \
             vemos claramente cómo se agrupan los individuos en cada uno de los clusters, y cómo se diferencian entre sí.')
    
    st.subheader('Distribución de variables por cluster.')

    st.write('Veamos la distibución de variables por cluster, con estas visualizaciones podremos entender mejor las características de cada \
             cluster para determinar qué tipo de cliente está encapsulando cada agrupación.')
    
    selector_variables = st.selectbox('Variable', options=['Días promedio entre compras', 'Gasto mensual', 'Desviación de gasto mensual', 'Promedio por transacción'])
    selector_n_clusters_3 = st.select_slider('Selecciona número de clusters', options=[2, 3, 4, 5], key='3')

    if selector_n_clusters_3 == 2 and selector_variables == 'Gasto mensual':

        kmeans = KMeans(n_clusters=2, random_state=42)
    
        df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

        # Mostrar distribución de variables por cluster con un boxplot

        boxplot_2clusters = px.box(df, x='Cluster', y='GastoMensual', title='Distribución de Gasto Mensual por Cluster con 2 clusters', color= 'Cluster')

        st.plotly_chart(boxplot_2clusters)
    
    elif selector_n_clusters_3 == 3 and selector_variables == 'Gasto mensual':

        kmeans = KMeans(n_clusters=3, random_state=42)
    
        df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

        boxplot_3clusters = px.box(df, x='Cluster', y='GastoMensual', title='Distribución de Gasto Mensual por Cluster con 3 clusters', color= 'Cluster')

        st.plotly_chart(boxplot_3clusters)
    
    elif selector_n_clusters_3 == 4 and selector_variables == 'Gasto mensual':

        kmeans = KMeans(n_clusters=4, random_state=42)
    
        df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

        boxplot_4clusters = px.box(df, x='Cluster', y='GastoMensual', title='Distribución de Gasto Mensual por Cluster con 4 clusters', color= 'Cluster')

        st.plotly_chart(boxplot_4clusters)

    elif selector_n_clusters_3 == 5 and selector_variables == 'Gasto mensual':

        kmeans = KMeans(n_clusters=5, random_state=42)
    
        df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

        boxplot_5clusters = px.box(df, x='Cluster', y='GastoMensual', title='Distribución de Gasto Mensual por Cluster con 5 clusters', color= 'Cluster')

        st.plotly_chart(boxplot_5clusters)
    
    elif selector_n_clusters_3 == 2 and selector_variables == 'Días promedio entre compras':

        kmeans = KMeans(n_clusters=2, random_state=42)
    
        df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

        boxplot_2clusters = px.box(df, x='Cluster', y='Average_Days_Between_Purchases', title='Distribución de Días promedio entre compras por Cluster con 2 clusters', color= 'Cluster')

        st.plotly_chart(boxplot_2clusters)
    
    elif selector_n_clusters_3 == 3 and selector_variables == 'Días promedio entre compras':

        kmeans = KMeans(n_clusters=3, random_state=42)
    
        df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

        boxplot_3clusters = px.box(df, x='Cluster', y='Average_Days_Between_Purchases', title='Distribución de Días promedio entre compras por Cluster con 3 clusters', color= 'Cluster')

        st.plotly_chart(boxplot_3clusters)

    elif selector_n_clusters_3 == 4 and selector_variables == 'Días promedio entre compras':

        kmeans = KMeans(n_clusters=4, random_state=42)
    
        df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

        boxplot_4clusters = px.box(df, x='Cluster', y='Average_Days_Between_Purchases', title='Distribución de Días promedio entre compras por Cluster con 4 clusters', color= 'Cluster')

        st.plotly_chart(boxplot_4clusters)
    
    elif selector_n_clusters_3 == 5 and selector_variables == 'Días promedio entre compras':

        kmeans = KMeans(n_clusters=5, random_state=42)
    
        df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

        boxplot_5clusters = px.box(df, x='Cluster', y='Average_Days_Between_Purchases', title='Distribución de Días promedio entre compras por Cluster con 5 clusters', color= 'Cluster')

        st.plotly_chart(boxplot_5clusters)
    
    elif selector_n_clusters_3 == 2 and selector_variables == 'Desviación de gasto mensual':

        kmeans = KMeans(n_clusters=2, random_state=42)
    
        df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

        boxplot_2clusters = px.box(df, x='Cluster', y='Monthly_Spending_Std', title='Distribución de Desviación de gasto mensual por Cluster con 2 clusters', color= 'Cluster')

        st.plotly_chart(boxplot_2clusters)
    
    elif selector_n_clusters_3 == 3 and selector_variables == 'Desviación de gasto mensual':

        kmeans = KMeans(n_clusters=3, random_state=42)
    
        df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

        boxplot_3clusters = px.box(df, x='Cluster', y='Monthly_Spending_Std', title='Distribución de Desviación de gasto mensual por Cluster con 3 clusters', color= 'Cluster')

        st.plotly_chart(boxplot_3clusters)
    
    elif selector_n_clusters_3 == 4 and selector_variables == 'Desviación de gasto mensual':

        kmeans = KMeans(n_clusters=4, random_state=42)
    
        df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

        boxplot_4clusters = px.box(df, x='Cluster', y='Monthly_Spending_Std', title='Distribución de Desviación de gasto mensual por Cluster con 4 clusters', color= 'Cluster')

        st.plotly_chart(boxplot_4clusters)
    
    elif selector_n_clusters_3 == 5 and selector_variables == 'Desviación de gasto mensual':

        kmeans = KMeans(n_clusters=5, random_state=42)
    
        df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

        boxplot_5clusters = px.box(df, x='Cluster', y='Monthly_Spending_Std', title='Distribución de Desviación de gasto mensual por Cluster con 5 clusters', color= 'Cluster')

        st.plotly_chart(boxplot_5clusters)
    
    elif selector_n_clusters_3 == 2 and selector_variables == 'Promedio por transacción':

        kmeans = KMeans(n_clusters=2, random_state=42)
    
        df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

        boxplot_2clusters = px.box(df, x='Cluster', y='Average_Transaction_Value', title='Distribución de Promedio por transacción por Cluster con 2 clusters', color= 'Cluster')

        st.plotly_chart(boxplot_2clusters)
    
    elif selector_n_clusters_3 == 3 and selector_variables == 'Promedio por transacción':

        kmeans = KMeans(n_clusters=3, random_state=42)
    
        df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

        boxplot_3clusters = px.box(df, x='Cluster', y='Average_Transaction_Value', title='Distribución de Promedio por transacción por Cluster con 3 clusters', color= 'Cluster')

        st.plotly_chart(boxplot_3clusters)
    
    elif selector_n_clusters_3 == 4 and selector_variables == 'Promedio por transacción':

        kmeans = KMeans(n_clusters=4, random_state=42)
    
        df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

        boxplot_4clusters = px.box(df, x='Cluster', y='Average_Transaction_Value', title='Distribución de Promedio por transacción por Cluster con 4 clusters', color= 'Cluster')

        st.plotly_chart(boxplot_4clusters)
    
    else:

        kmeans = KMeans(n_clusters=5, random_state=42)
    
        df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

        boxplot_5clusters = px.box(df, x='Cluster', y='Average_Transaction_Value', title='Distribución de Promedio por transacción por Cluster con 5 clusters', color= 'Cluster')

        st.plotly_chart(boxplot_5clusters)
    

    st.write('Con las diferencias entre distribuciones de variables por cluster podemos hacer un perfilado de los clientes, y así\
             saber qué tipo de cliente está encapsulado en cada cluster. Por ejemplo, podemos ver que hay un grupo de clientes que\
             gasta mucho dinero al mes, y otro grupo que gasta poco, u otros cuya freceuncia de compra es muy alta (Días promedio entre \
             compras bajo).')
        
            

    st.subheader('Qué representa cada cluster.')

    st.write('Vamos a analizar las características de cada cluster, para entender qué representa cada uno de ellos.\
             para ello utilizaremos un gráfico de radar, que nos permitirá comparar las características de los clusters. \
             Continuaremos con 4 clusters por ser los indicados como óptimos por el método de la silueta.')
    
    # Mostrar gráfico de radar con las características de cada cluster para 4 clusters

    st.image('img/radar_chart.png', caption='Gráfico de radar.')

    lista_de_elementos = ["Cliente conservador: No compra con demasiada frecuencia, no gasta mucho.", \
                          "Cliente ideal: Gasta bastante y con relativa frecuencia.", \
                            "Cliente impulsivo: Gasta poco y en artículos baratos pero muy frecuentemente.", \
                                "Cliente esporádico: Gasta mucho pero no con mucha frecuencia."]

    # Mostrar la lista en Streamlit
    st.write("Tipos de clientes de mi E-Commerce:")
    for elemento in lista_de_elementos:
        st.write("- ", elemento)
    st.header('Recomendación de productos.')

    st.write('Una vez que hemos agrupado a los clientes en clusters, podemos recomendar productos en base a los productos favoritos de cada cluster\
             así que veamos cuales son estos productos.')
    
    # Countplot de productos favoritos por cluster
    countplot_productos = px.histogram(cluster_4, x='Cluster_4', color='CategoríaProductoFavorito', barmode='group', 
                                       title='Productos favoritos por cluster', 
                                       labels={'CategoríaProductoFavorito': 'Categoría de Producto', 'Cluster_4': 'Cluster'})

    st.plotly_chart(countplot_productos)

    st.write('¡Todo listo! Ahora podemos recomendar productos a nuestros clientes en base a los productos favoritos de cada cluster.\
             para ello, simplemente identificamos a qué cluster pertenece cada cliente y le recomendamos los productos favoritos de \
             ese cluster.')

    # Escribir un formulario para introducir las características del nuevo cliente en streamlit
    st.subheader('Introduce las características del nuevo cliente para recomendarle un producto.')

    new_client = st.form(key='new_client_form')
    new_client_days = new_client.number_input('Días promedio entre compras', min_value=0, max_value=1000)
    new_client_monthly = new_client.number_input('Gasto mensual', min_value=0, max_value=10000)
    new_client_std = new_client.number_input('Desviación de gasto mensual', min_value=0, max_value=1000)
    new_client_average = new_client.number_input('Promedio por transacción', min_value=0, max_value=1000)

    submit_button = new_client.form_submit_button('Recomendar producto')

    if submit_button:
        # Hacer recomendación de producto en base a k-means y productos favoritos de cada cluster

        kmeans = KMeans(n_clusters=4, random_state=42)

        kmeans.fit(df_cluster_scaled)

        new_client_data = np.array([new_client_days, new_client_monthly, new_client_std, new_client_average]).reshape(1, -1)

        new_client_data = scaler_final.transform(new_client_data)

        new_client_data = pd.DataFrame(new_client_data, columns=['Average_Days_Between_Purchases', 'GastoMensual', 'Monthly_Spending_Std', 'Average_Transaction_Value'])

        new_client_cluster = kmeans.predict(new_client_data)

        new_client_cluster = int(new_client_cluster)

        new_client_recommendation = cluster_4[cluster_4['Cluster_4'] == new_client_cluster]['CategoríaProductoFavorito'].values[0]

        st.write('El nuevo cliente tiene las siguientes características:')
        st.write('- Días promedio entre compras:', new_client_days)
        st.write('- Gasto mensual:', new_client_monthly)
        st.write('- Desviación de gasto mensual:', new_client_std)
        st.write('- Promedio por transacción:', new_client_average)

        st.write('El nuevo cliente pertenece al cluster:', new_client_cluster)
        st.write('El producto recomendado para este cliente es:', new_client_recommendation)

    st.write('Con esto hemos terminado el clustering para sistema de recomendación, ahora vamos a pasar a la fase de generativas.')


