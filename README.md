# Sistema de Recomendación: Aplicación de Machine Learning y Técnicas Generativas en Negocios 🚀

## Descripción 📝
Este proyecto combina análisis de datos, visualización interactiva, y técnicas de inteligencia artificial para explorar la aplicación práctica de APIs como Streamlit, DALL-E, y Whisper en negocios. Dirigido a entusiastas de la tecnología, proporciona una plataforma interactiva para experimentar con recomendaciones personalizadas, segmentación de clientes, y generación de contenido visual.

## Acceso a la Aplicación 🌐
La aplicación está desplegada y accesible a través de [clusteringgenai.miguellacruz.es](http://clusteringgenai.miguellacruz.es)

## Características Principales ✨
- **Análisis Exploratorio de Datos:** Explora y visualiza datos para entender las tendencias y comportamientos de los clientes.
- **Sistema de Recomendación:** Utiliza algoritmos de Machine Learning para recomendar productos basados en las preferencias del usuario.
- **Segmentación de Clientes:** Implementa clustering para identificar grupos significativos de clientes.
- **Generación de Imágenes:** Crea imágenes personalizadas usando la red generativa DALL-E.

Para acceder a cada fase del proyecto usa el desplegable de la izquierda.

## Limitaciones del Despliegue 🚧
La funcionalidad de chatbot no está operativa en la versión desplegada. Esto se debe a que utiliza RAG (Retrieval Augmented Generation) con un documento que fue vectorizado por párrafos y almacenado en una base de datos de vectores ChromaDB en local, además, de un modelo LLM phi3 no soportado por el servidor debido a la limitada capacidad de RAM.

## Cómo Explorar la Creación de la Base de Datos de Vectores 📊
Para entender cómo se creó la base de datos de vectores, visita la carpeta `notebook_chroma`. Dentro encontrarás un notebook que detalla todo el procedimiento.

## Tecnologías Utilizadas 🛠️
- **Streamlit:** Para la creación de aplicaciones web interactivas.
- **Plotly:** Para visualizaciones interactivas de datos.
- **Pandas & NumPy:** Para manipulación de datos.
- **Scikit-Learn:** Para algoritmos de clustering y preprocesamiento.
- **Pickle:** Para guardar y cargar modelos entrenados.

## Cómo Empezar 🏁
Existen dos formas de poner en marcha el proyecto:

### Método 1: Uso de Docker (Recomendado)
1. Asegúrate de tener Docker y Docker Compose instalados en tu sistema.
2. Clona el repositorio y navega al directorio principal:

```bash
git clone https://github.com/Mlcruz9/BusinessClientSegmentationTesting.git
```

Luego entra en la carpeta principal del repo:

```bash
cd BusinessClientSegmentationTesting
```

Y ejecuta el siguiente comando:

```bash
docker compose up
```

Este método levantará todos los servicios necesarios en contenedores Docker, incluyendo todas las dependencias y variables de entorno configuradas automáticamente.

### Método 2: Configuración Manual
1. Clona el repositorio:
2. Instala [Ollama](https://ollama.com/) y activa el entorno virtual:

```bash
ollama serve
```

3. Descarga y configura `phi3`:

```bash
ollama run phi3
```

4. Navega al directorio principal e instala las dependencias adicionales:

```bash
pip install -r requirements.txt
```

5. Ejecuta la aplicación Streamlit:

```bash
streamlit run main.py
```

## Autor 📚
**Miguel S. La Cruz**
- Experto en ciencia de datos y desarrollo de soluciones basadas en ML.


## Contribuir 🤝
Si estás interesado en mejorar el sistema o agregar nuevas funcionalidades, por favor considera forkear el repositorio y crear un pull request con tus cambios.


