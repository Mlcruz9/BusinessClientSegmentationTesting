import requests
import time
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import chromadb
import ollama

def generate_image(prompt, API_KEY):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }

    data = {
        'prompt': prompt,
        'n': 1,  # Número de imágenes a generar
        'size': '1024x1024',  # Tamaño de la imagen
        'model': 'dall-e-3'
    }

    try:
        response = requests.post('https://api.openai.com/v1/images/generations', headers=headers, json=data)
        response.raise_for_status()  # Esto lanzará una excepción para códigos de error HTTP
        image_url = response.json()['data'][0]['url']
        return image_url
    except requests.exceptions.HTTPError as e:
        # Puedes especificar acciones diferentes basadas en el código de estado
        if response.status_code == 429:
            time.sleep(10)  # Espera 10 segundos y reintenta
            return generate_image(prompt)
        else:
            return f"Error HTTP: {e}"
    except requests.exceptions.RequestException as e:
        return f"Error de Red: {e}"
    
def audio_to_text(API_KEY, audio_path):
    url = 'https://api.openai.com/v1/audio/transcriptions'
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'language': 'es-ES'
    }

    with open(audio_path, 'rb') as audio_file:
        files = {'file': audio_file
                }
        data = {
            'model': 'whisper-1'  # Asegúrate de que el modelo se envíe correctamente
        }
        
        response = requests.post(url, headers=headers, files=files, data=data)
        
        if response.status_code == 200:
            return response.json()  # Retorna la transcripción del audio
        else:
                return response.text

def transcribe_audio(API_KEY):

    audio_data = audio_recorder("Descripción de tu producto:", icon_size= '1.8x', pause_threshold=2)
    if audio_data:
        # Check if audio is of sufficient length
        if len(audio_data) > 8000:
            st.success('Audio captured correctly')
            st.audio(audio_data, format="audio/wav")
            with open("audio/temp_audio_file.wav", "wb") as f:
                f.write(audio_data)
            
            if st.button('Generar imagen'):
                # Aquí llamarías a la función que envía el audio a la API de Whisper
                transcription = audio_to_text(API_KEY = API_KEY, audio_path= "audio/temp_audio_file.wav")
                return transcription['text']
        else:
            st.warning('Audio captured incorrectly, please try again.')


class ChatBotAI():
    def __init__(self, app):
        db = chromadb.PersistentClient()
        self.collection = db.get_or_create_collection("Amazon_Styleguide")
        self.app = app

    def query(self, q, top=10):
        res_db = self.collection.query(query_texts=[q])["documents"][0][0:top]
        context = ' '.join(res_db).replace("\n", " ")
        return context

    def respond(self, lst_messages, model="phi3", use_knowledge=False):
        q = lst_messages[-1]["content"]
        context = self.query(q)
        # st.write(context)
        if use_knowledge:
            prompt = "Give the most accurate answer using your knowledge and the folling additional information: \n"+context
        else:
            prompt = "Give the most accurate answer using only the folling information: \n"+context

        res_ai = ollama.chat(model=model, 
                                messages=[{"role":"system", "content":prompt}]+lst_messages,
                                stream=True)
        for res in res_ai:
            chunk = res["message"]["content"]
            self.app["full_response"] += chunk
            yield chunk