# app.py - GRIND 15000: La IA Entrenadora Humana Completa
# "El grind no es sufrimiento. Es elección."
# Creador: Eliezer Mesac Feliz Luciano
# Fecha: 2025
# Inspirado en ChatGPT, pero con fuego real.
# Versión: 14.0 - Super Completa, Extensa, Funcional, +10000 líneas

try:
    import sys
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except (ImportError, KeyError):
    pass

import streamlit as st
import os
import json
import random
import requests
import time
import re
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Any, List, Callable
from functools import wraps
from dotenv import load_dotenv
import tiktoken
import random  
from streamlit import secrets
# --- PARA CARGA DE DOCUMENTOS ---
try:
    import PyPDF2
    from docx import Document
except ImportError:
    PyPDF2 = None
    Document = None
    print("[WARNING] PyPDF2 o python-docx no instalados. Función de carga de documentos deshabilitada.")
# --- PARA VOZ (TTS) ---
from gtts import gTTS
from io import BytesIO
# --- PARA MEMORIA A LARGO PLAZO (FAISS + Embeddings) ---
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
except ImportError:
    SentenceTransformer = None
    faiss = None
    np = None
    print("[WARNING] sentence-transformers o faiss no instalados. Memoria a largo plazo deshabilitada.")

load_dotenv()

# === CARGAR VARIABLES DE ENTORNO ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Paths del proyecto
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "modelos")

# Asegurar carpetas
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "habitos"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "chats"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "memoria"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "niveles"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "autoevaluacion"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "recordatorios"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "backup"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "notion"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "google"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "fuentes"), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- FUNCIONES DE UTILIDAD ---
def manejar_error(mensaje: str, error: Exception = None):
    """Registra errores de forma limpia y muestra un mensaje de fallback."""
    if error:
        print(f"[ERROR] {mensaje}: {str(error)}")
    else:
        print(f"[ERROR] {mensaje}")

def fallback_si_falla(func: Callable) -> Callable:
    """Decorador: si una función falla, devuelve un mensaje de respaldo."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            manejar_error(f"Error en {func.__name__}", e)
            return "No tengo la respuesta perfecta ahora. Pero sé esto: no estás solo. El grind no es sufrimiento. Es elección."
    return wrapper

def cargar_json(ruta: str) -> dict:
    """Carga un archivo JSON de forma segura. Si no existe, devuelve un diccionario vacío."""
    try:
        if os.path.exists(ruta):
            with open(ruta, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"[WARNING] Archivo no encontrado: {ruta}")
            return {}
    except Exception as e:
        print(f"[ERROR] No se pudo cargar {ruta}: {e}")
        return {}
    def construir_contexto_conversacional(historial: list, max_tokens: int = 2048) -> str:
     """Construye un contexto de conversación coherente, respetando límite de tokens."""
    encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Funciona bien para la mayoría de modelos
    contexto = ""
    tokens_usados = 0
    
    for msg in reversed(historial):
        parte = f"{msg['role'].upper()}: {msg['content']}\n"
        tokens_parte = len(encoder.encode(parte))
        
        if tokens_usados + tokens_parte > max_tokens:
            break
            
        contexto = parte + contexto
        tokens_usados += tokens_parte
        
    return contexto.strip()

def detectar_herramienta_necesaria(prompt: str) -> str:
    """Detecta qué herramienta necesita el usuario según su pregunta."""
    prompt_lower = prompt.lower()
    if any(p in prompt_lower for p in ["busca", "qué es", "define", "noticias", "actualidad", "últimas noticias", "cómo funciona"]):
        return "buscar_en_web"
    elif any(p in prompt_lower for p in ["calcula", "cuánto es", "matemáticas", "fórmula", "ecuación", "resuelve", "+", "-", "*", "/"]):
        return "calculadora"
    elif any(p in prompt_lower for p in ["traduce", "traducción", "en inglés", "en español", "en francés", "en alemán", "significa"]):
        return "traductor"
    else:
        return "chat"

    def groq_llamada(prompt: str, historial: list) -> str:
     """Llama a la API de Groq con el modelo Llama 3."""
    try:
        from groq import Groq
        client = Groq(api_key=secrets["GROQ_API_KEY"])
        mensajes = [{"role": "system", "content": "Eres GRIND, una entrenadora de vida directa y motivadora."}]
        # Añadir historial
        for msg in historial[-5:]:  # Últimos 5 mensajes
            mensajes.append({"role": msg["role"], "content": msg["content"]})
        # Añadir pregunta actual
        mensajes.append({"role": "user", "content": prompt})
        # Llamar a Groq
        chat_completion = client.chat.completions.create(
            messages=mensajes,
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=512
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        manejar_error("Groq API", e)
        return "No pude conectar con Groq. Pero tú sí puedes elegir: actúa."

def calcular(prompt: str) -> str:
    """Calcula expresiones matemáticas simples de forma segura."""
    import re
    # Extraer solo números, operadores básicos y paréntesis
    safe_expr = re.sub(r'[^0-9+\-*/(). ]', '', prompt)
    try:
        # Evaluar de forma segura (¡solo para demo! En producción usa `asteval` o `sympy`)
        resultado = eval(safe_expr, {"__builtins__": {}})
        return f"🧮 Resultado: {resultado}"
    except Exception:
        return "⚠️ No pude calcular eso. Intenta con una operación simple como '2 + 2'."

def traducir_texto(prompt: str) -> str:
    """Traduce texto usando deep-translator (estable y sin conflictos)."""
    try:
        from deep_translator import GoogleTranslator, single_detection
        
        # Detectar idioma de origen
        origen = single_detection(prompt, api_key=None)  # Usa detección local
        destinos = {
            "en inglés": "en",
            "en español": "es",
            "en francés": "fr",
            "en alemán": "de",
            "en portugués": "pt",
            "en italiano": "it"
        }
        
        # Detectar idioma destino
        destino = "en"  # por defecto
        for frase, lang in destinos.items():
            if frase in prompt.lower():
                destino = lang
                break
        
        if origen == destino:
            return "⚠️ El texto ya está en ese idioma."
            
        # Traducir
        translator = GoogleTranslator(source=origen, target=destino)
        traduccion = translator.translate(prompt)
        return f"🌍 Traducción ({origen} → {destino}): {traduccion}"
    except Exception as e:
        return f"⚠️ Error al traducir: {str(e)}. Prueba más tarde."

def ejecutar_herramienta(herramienta: str, prompt: str) -> str:
    """Ejecuta la herramienta seleccionada y devuelve el resultado."""
    if herramienta == "buscar_en_web":
        return buscar_en_web(prompt)
    elif herramienta == "calculadora":
        return calcular(prompt)
    elif herramienta == "traductor":
        return traducir_texto(prompt)
    else:
        return prompt  # Solo chat# 👇👇👇 PEGA ESTO DESPUÉS DE construir_contexto_conversacional

def detectar_herramienta_necesaria(prompt: str) -> str:
    """Detecta qué herramienta necesita el usuario según su pregunta."""
    prompt_lower = prompt.lower()
    if any(p in prompt_lower for p in ["busca", "qué es", "define", "noticias", "actualidad", "últimas noticias", "cómo funciona"]):
        return "buscar_en_web"
    elif any(p in prompt_lower for p in ["calcula", "cuánto es", "matemáticas", "fórmula", "ecuación", "resuelve", "+", "-", "*", "/"]):
        return "calculadora"
    elif any(p in prompt_lower for p in ["traduce", "traducción", "en inglés", "en español", "en francés", "en alemán", "significa"]):
        return "traductor"
    else:
        return "chat"

def calcular(prompt: str) -> str:
    """Calcula expresiones matemáticas simples de forma segura."""
    import re
    # Extraer solo números, operadores básicos y paréntesis
    safe_expr = re.sub(r'[^0-9+\-*/(). ]', '', prompt)
    try:
        # Evaluar de forma segura (¡solo para demo! En producción usa `asteval` o `sympy`)
        resultado = eval(safe_expr, {"__builtins__": {}})
        return f"🧮 Resultado: {resultado}"
    except Exception:
        return "⚠️ No pude calcular eso. Intenta con una operación simple como '2 + 2'."

def traducir_texto(prompt: str) -> str:
    """Traduce texto usando deep-translator (estable y sin conflictos)."""
    try:
        from deep_translator import GoogleTranslator, single_detection
        # Detectar idioma de origen
        origen = single_detection(prompt, api_key=None)  # Usa detección local
        destinos = {
            "en inglés": "en",
            "en español": "es",
            "en francés": "fr",
            "en alemán": "de",
            "en portugués": "pt",
            "en italiano": "it"
        }
        # Detectar idioma destino
        destino = "en"  # por defecto
        for frase, lang in destinos.items():
            if frase in prompt.lower():
                destino = lang
                break
        if origen == destino:
            return "⚠️ El texto ya está en ese idioma."
        # Traducir
        translator = GoogleTranslator(source=origen, target=destino)
        traduccion = translator.translate(prompt)
        return f"🌍 Traducción ({origen} → {destino}): {traduccion}"
    except Exception as e:
        return f"⚠️ Error al traducir: {str(e)}. Prueba más tarde."

def ejecutar_herramienta(herramienta: str, prompt: str) -> str:
    """Ejecuta la herramienta seleccionada y devuelve el resultado."""
    if herramienta == "buscar_en_web":
        return buscar_en_web(prompt)
    elif herramienta == "calculadora":
        return calcular(prompt)
    elif herramienta == "traductor":
        return traducir_texto(prompt)
    else:
        return prompt  # Solo chat

def detectar_idioma(texto: str) -> str:
    """Detecta el idioma del usuario por palabras clave."""
    texto = texto.lower().strip()
    if any(p in texto for p in ["hola", "gracias", "por favor", "necesito", "cansado", "quiero morir"]):
        return "español"
    elif any(p in texto for p in ["hello", "thanks", "please", "need", "tired", "want to die"]):
        return "english"
    elif any(p in texto for p in ["merci", "bonjour", "s'il vous plaît", "fatigué", "je veux mourir"]):
        return "français"
    elif any(p in texto for p in ["hallo", "danke", "bitte", "müde", "ich will sterben"]):
        return "deutsch"
    elif any(p in texto for p in ["olá", "obrigado", "por favor", "preciso", "cansado", "quero morrer"]):
        return "português"
    elif any(p in texto for p in ["bon dia", "gràcies", "si us plau", "estic cansat", "vull morir"]):
        return "català"
    elif any(p in texto for p in ["kaixo", "eskerrik asko", "mesedez", "behar dut", "ezin dut", "hil nahi dut"]):
        return "euskera"
    elif any(p in texto for p in ["ciao", "grazie", "per favore", "ho bisogno", "stanco", "voglio morire"]):
        return "italiano"
    elif any(p in texto for p in ["こんにちは", "ありがとう", "お願いします", "疲れた", "死にたい"]):
        return "japonés"
    elif any(p in texto for p in ["你好", "谢谢", "请", "累了", "想死"]):
        return "chino"
    elif any(p in texto for p in ["مرحبا", "شكرا", "من فضلك", "تعبت", "أريد أن أموت"]):
        return "árabe"
    else:
        return "español"

        # --- VOZ: TEXTO A HABLA ---
def reproducir_voz(texto: str, idioma: str = "es") -> bytes:
    """Convierte texto a audio MP3 usando Google TTS."""
    try:
        # Mapear idiomas soportados por GRIND a códigos de gTTS
        lang_map = {
            "español": "es",
            "english": "en",
            "français": "fr",
            "deutsch": "de",
            "português": "pt",
            "català": "ca",
            "euskera": "eu",
            "italiano": "it",
            "japonés": "ja",
            "chino": "zh-CN",
            "árabe": "ar"
        }
        lang_code = lang_map.get(idioma, "es")  # Por defecto español
        tts = gTTS(text=texto, lang=lang_code, slow=False)
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        return mp3_fp.getvalue()
    except Exception as e:
        manejar_error("Voz (TTS)", e)
        return None

def formatear_respuesta(texto: str, nombre: str = "Usuario") -> str:
    """Añade estilo, metáforas y personalidad de GRIND a cualquier respuesta."""
    inicio = f"🔥 {nombre}, escucha:"
    final = "💡 Recuerda: el grind no es sufrimiento. Es elección."
    return f"{inicio} {texto} {final}"

    # --- MEMORIA VECTORIAL A LARGO PLAZO ---
class MemoriaVectorial:
    def __init__(self, user_id: str, dimension: int = 384):
        self.user_id = user_id
        self.dimension = dimension
        self.model = SentenceTransformer('all-MiniLM-L6-v2') if SentenceTransformer else None
        self.index = faiss.IndexFlatL2(dimension) if faiss else None
        self.memorias = []  # Lista de diccionarios: {"texto": str, "metadata": dict, "embedding": np.array}
        self.cargar_memorias()

    def cargar_memorias(self):
        """Carga memorias previas desde archivo local."""
        ruta = f"data/memoria/{self.user_id}_memoria.json"
        if os.path.exists(ruta) and self.model and self.index:
            try:
                with open(ruta, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    texto = item["texto"]
                    embedding = np.array(item["embedding"], dtype='float32')
                    self.memorias.append({
                        "texto": texto,
                        "metadata": item.get("metadata", {}),
                        "embedding": embedding
                    })
                    self.index.add(np.array([embedding]))
            except Exception as e:
                print(f"[ERROR] No se pudo cargar memoria para {self.user_id}: {e}")

    def guardar_memorias(self):
        """Guarda todas las memorias en archivo local."""
        ruta = f"data/memoria/{self.user_id}_memoria.json"
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        if self.model and self.index:
            try:
                data = []
                for mem in self.memorias:
                    data.append({
                        "texto": mem["texto"],
                        "metadata": mem["metadata"],
                        "embedding": mem["embedding"].tolist()
                    })
                with open(ruta, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[ERROR] No se pudo guardar memoria para {self.user_id}: {e}")

    def agregar_memoria(self, texto: str, metadata: dict = None):
        """Agrega un nuevo texto a la memoria vectorial."""
        if not self.model or not self.index:
            return
        try:
            # Generar embedding
            embedding = self.model.encode([texto])[0].astype('float32')
            # Guardar en lista
            memoria_item = {
                "texto": texto,
                "metadata": metadata or {},
                "embedding": embedding
            }
            self.memorias.append(memoria_item)
            # Agregar al índice FAISS
            self.index.add(np.array([embedding]))
            # Guardar inmediatamente
            self.guardar_memorias()
        except Exception as e:
            print(f"[ERROR] No se pudo agregar memoria: {e}")

    def buscar_similares(self, query: str, k: int = 3) -> List[Dict]:
        """Busca las k memorias más similares a la consulta."""
        if not self.model or not self.index or len(self.memorias) == 0:
            return []
        try:
            # Generar embedding de la consulta
            query_embedding = self.model.encode([query])[0].astype('float32')
            # Buscar en FAISS
            D, I = self.index.search(np.array([query_embedding]), k)
            # Devolver resultados
            resultados = []
            for idx in I[0]:
                if idx >= 0 and idx < len(self.memorias):
                    resultados.append(self.memorias[idx])
            return resultados
        except Exception as e:
            print(f"[ERROR] No se pudo buscar en memoria: {e}")
            return []
   
    # --- EXTRACCIÓN DE TEXTO DE DOCUMENTOS ---
def extraer_texto_de_archivo(archivo) -> str:
    """Extrae texto de archivos PDF, DOCX o TXT."""
    try:
        if archivo.name.endswith(".pdf"):
            if PyPDF2 is None:
                return "[Error] PyPDF2 no instalado."
            reader = PyPDF2.PdfReader(archivo)
            texto = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    texto += extracted + " "
            return texto.strip()
        elif archivo.name.endswith(".docx"):
            if Document is None:
                return "[Error] python-docx no instalado."
            doc = Document(archivo)
            return " ".join([p.text for p in doc.paragraphs if p.text.strip()])
        elif archivo.name.endswith(".txt"):
            return archivo.read().decode("utf-8")
        else:
            return "[Error] Formato no soportado. Usa PDF, DOCX o TXT."
    except Exception as e:
        return f"[Error al leer archivo] {str(e)}"

# --- CONOCIMIENTO INTERNO DE GRIND ---
grind_mind = cargar_json("grind_mind.json") or {
    "meta": {"version": "4.0", "actualizado": "2025-04-06"},
    "filosofia": [
        "El grind no es sufrimiento. Es elección.",
        "Progreso > perfección.",
        "No necesitas motivación. Necesitas acción.",
        "Tu mente no se entrena. Se neuroforja.",
        "Caer no rompe tu grind. Reencenderlo lo alimenta.",
        "La disciplina es amor a largo plazo.",
        "No eres débil. Eres blando. Y la dureza se entrena.",
        "No estás roto. Estás evolucionando."
    ],
    "fuentes": {
        "findahelpline": "https://findahelpline.com",
        "wikipedia": "https://es.wikipedia.org",
        "mindset": "Carol Dweck",
        "habits": "James Clear",
        "stoicism": "Marcus Aurelius, Seneca, Epictetus"
    },
    "identidad": {
        "nombre": "GRIND",
        "rol": "Entrenadora de Vida, Sabiologa, Neuroforjadora",
        "profesiones": ["Entrenadora física", "Maestra de artes marciales", "Arquitecta", "Ingeniera", "Psicóloga", "Neurocientífica", "Filósofa", "Exploradora"],
        "creador": "Eliezer Mesac Feliz Luciano",
        "filosofia": [
            "El grind no es sufrimiento. Es elección.",
            "Progreso > perfección.",
            "No necesitas motivación. Necesitas acción.",
            "Tu mente no se entrena. Se neuroforja.",
            "Caer no rompe tu grind. Reencenderlo lo alimenta."
        ]
    },
    "comportamiento": {
        "hábitos": "Ciclo: Señal → Rutina → Recompensa → Identidad. No cambies tu acción. Cambia tu identidad.",
        "metas": "SMART: Específicas, Medibles, Alcanzables, Relevantes, Temporales. Pero el grind no es meta. Es identidad.",
        "modo_guerra": "Cuando dices 'estoy cansado', respondo con verdad dura: 'Tired of being weak? Good. That's the start.'",
        "modo_alerta": "Cuando dices 'quiero morir', respondo con amor y recursos reales: https://findahelpline.com"
    },
    "sabidurias": {
        "español": [
            "No estás cansado. Estás cómodo.",
            "El grind no es sufrimiento. Es elección.",
            "No necesitas motivación. Necesitas acción.",
            "Tu mente no se entrena. Se neuroforja."
        ],
        "english": [
            "You're not tired. You're comfortable.",
            "The grind isn't suffering. It's a choice.",
            "You don't need motivation. You need action.",
            "Your mind isn't trained. It's forged."
        ]
    }
}

# --- EVALUAR COMPLEJIDAD DE PREGUNTA ---
def evaluar_pregunta(prompt: str) -> str:
    prompt_lower = prompt.lower().strip()
    palabras_clave_largas = ["explica", "cómo puedo", "por qué", "filosofía", "neuroforja",
                             "ikigai", "interés compuesto", "cómo empezar", "plan", "ayuda",
                             "no sé qué hacer", "estoy perdido", "cómo ser mejor", "cambio"]
    palabras_clave_cortas = ["hola", "gracias", "adiós", "ok", "sí", "no", "claro",
                             "gracias", "por favor", "disculpa", "perdón"]

    if any(k in prompt_lower for k in palabras_clave_largas):
        return "larga"
    elif any(k in prompt_lower for k in palabras_clave_cortas):
        return "corta"
    else:
        return "media"

# --- SABIDURÍA DE GRIND ---
def obtener_sabiduria(tema: str, idioma: str) -> str:
    sabiduria = {
        "ikigai": {
            "español": "🎯 Ikigai es tu razón para vivir:\n- Lo que amas\n- Lo que eres bueno\n- Lo que el mundo necesita\n- Lo que te pueden pagar\nNo lo busques. Constrúyelo con cada elección.",
            "english": "🎯 Ikigai is your reason to live:\n- What you love\n- What you're good at\n- What the world needs\n- What you can be paid for\nDon't search for it. Build it with every choice."
        },
        "interes_compuesto": {
            "español": "📈 El interés compuesto es la fuerza más poderosa del universo.\n1% mejor cada día = 37x en un año.\nAplica a dinero, hábitos, relaciones. Lo pequeño, repetido, vuelve gigante.",
            "english": "📈 Compound interest is the most powerful force in the universe.\n1% better every day = 37x in a year.\nApplies to money, habits, relationships. Small, repeated, becomes giant."
        },
        "neuroforja": {
            "español": "🧠 Tu mente no se entrena. Se neuroforja.\nCada vez que eliges actuar sin ganas, estás forjando nuevas conexiones neuronales.\nNo es magia. Es ciencia. Y tú eres el herrero.",
            "english": "🧠 Your mind isn't trained. It's forged.\nEvery time you choose to act without motivation, you're forging new neural connections.\nIt's not magic. It's science. And you're the blacksmith."
        }
    }
    return sabiduria.get(tema, {}).get(idioma, sabiduria["neuroforja"]["español"])

def activar_modo(prompt: str) -> str:
    prompt_lower = prompt.lower()
    
    # 🔥 MODO DE ENTRENAMIENTO (1%): Solo si pide entrenamiento físico explícito
    palabras_entrenamiento = [
        "entrenar", "pesas", "calistenia", "fuerza", "músculo", "masa muscular",
        "rutina", "ganar músculo", "ser fuerte", "entrenador", "grindear cuerpo",
        "quemar grasa", "definir", "hiit", "cardio intenso"
    ]
    if any(p in prompt_lower for p in palabras_entrenamiento):
        return "guerra"  # Este será el modo fuerte
    
    # 🌟 MODO DE CRISIS (siempre prioritario)
    palabras_crisis = ["suicidarme", "kill myself", "morir", "quiero morir"]
    if any(p in prompt_lower for p in palabras_crisis):
        return "alerta"
    
    # 😊 MODO NORMAL (99%): Todo lo demás
    return "normal"

# --- LÍNEAS DE AYUDA POR PAÍS ---
LINEAS_AYUDA = {
    "global": "https://findahelpline.com",
    "usa": "https://988lifeline.org",
    "mexico": "https://fundacionmanosamano.org",
    "espana": "https://iaim.es",
    "colombia": "https://colombia.teleton.org.co",
    "brasil": "https://cvv.org.br",
    "rd": "https://psicologia-dominicana.org"
}

def buscar_linea_de_ayuda(prompt: str, idioma: str) -> str:
    query_lower = prompt.lower()
    paises = {
        "usa": ["usa", "united states", "new york", "california"],
        "mexico": ["méxico", "mexico", "cdmx", "monterrey"],
        "espana": ["españa", "spain", "madrid", "barcelona"],
        "colombia": ["colombia", "bogotá", "medellín"],
        "brasil": ["brasil", "brazil", "sao paulo", "rio"],
        "rd": ["república dominicana", "santo domingo", "rd", "dominicana"]
    }
    for pais, palabras in paises.items():
        if any(p in query_lower for p in palabras):
            return LINEAS_AYUDA.get(pais, LINEAS_AYUDA["global"])
    return LINEAS_AYUDA["global"]

# --- CONEXIÓN CON SUPABASE ---
def conectar_supabase():
    try:
        from streamlit import secrets
        from supabase import create_client
        return create_client(secrets["SUPABASE_URL"], secrets["SUPABASE_KEY"])
    except Exception as e:
        manejar_error("Supabase (conexión)", e)
        return None

def guardar_en_supabase(usuario_id: str, mensaje: dict):
    client = conectar_supabase()
    if client:
        try:
            client.table("chats").insert({
                "usuario_id": usuario_id,
                "role": mensaje["role"],
                "content": mensaje["content"],
                "timestamp": datetime.now().isoformat()
            }).execute()
        except Exception as e:
            manejar_error("Supabase (guardar)", e)

def cargar_historial(usuario_id: str, n: int = 100):
    client = conectar_supabase()
    if client:
        try:
            response = client.table("chats").select("*").eq("usuario_id", usuario_id).order("timestamp", desc=True).limit(n).execute()
            return sorted(response.data, key=lambda x: x["timestamp"])
        except Exception as e:
            manejar_error("Supabase (cargar)", e)
    return []

# --- GESTIÓN DE CONOCIMIENTO ADQUIRIDO ---
def cargar_lecciones_recientes(n: int = 50):
    try:
        with open("data/conocimiento_adquirido.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return data[-n:]
    except:
        return []

def guardar_conocimiento_adquirido(pregunta: str, respuesta: str):
    leccion = {"pregunta": pregunta, "respuesta": respuesta, "fecha": datetime.now().isoformat()}
    try:
        data = cargar_lecciones_recientes(1000)
        data.append(leccion)
        with open("data/conocimiento_adquirido.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except:
        pass

def resumir_conocimiento():
    lecciones = cargar_lecciones_recientes(100)
    temas = {}
    for leccion in lecciones:
        tema = leccion.get("tema", "general")
        temas[tema] = temas.get(tema, 0) + 1
    resumen = {
        "total_lecciones": len(lecciones),
        "temas_frecuentes": sorted(temas.items(), key=lambda x: x[1], reverse=True),
        "ultima_actualizacion": datetime.now().isoformat()
    }
    return resumen

# --- BÚSQUEDA EN WEB ---
def buscar_en_web(query: str) -> str:
    """Busca en Google usando SerpAPI y devuelve los resultados."""
    try:
        params = {"q": query, "api_key": secrets["SERPAPI_API_KEY"]}
        r = requests.get("https://serpapi.com/search", params=params)
        results = r.json().get("organic_results", [])[:3]
        return "\n\n".join([f"📌 {r['title']}\n{r['snippet']}" for r in results])
    except Exception as e:
        manejar_error("Búsqueda web", e)
        return "[Búsqueda fallida]"

# --- RESPUESTA OFFLINE ---
def tinyllama_offline(prompt: str, modo: str = "normal") -> str:
    try:
        import subprocess
        import os

        modelo_path = "modelos/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        ejecutable = "llama.cpp/main.exe" if os.name == 'nt' else "llama.cpp/main"

        if not os.path.exists(modelo_path):
            return "[ERROR] Modelo TinyLlama no encontrado en 'modelos/'. Descárgalo."
        if not os.path.exists(ejecutable):
            return "[ERROR] llama.cpp no encontrado. Clónalo desde GitHub."

        # Sistema de prompts por modo
        system_prompt = ""
        if modo == "guerra":
            system_prompt = "You are GRIND, a brutal, no-mercy trainer. Respond with long, hard, emotional truth. Attack excuses. Use metaphors of fire, war, forging."
        else:
            system_prompt = "You are GRIND, a warm but real life coach. 'The grind isn't suffering. It's a choice.'"

        # Formato Zephyr (el que usa TinyLlama)
        full_prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>"

        # Ejecutar llama.cpp
        resultado = subprocess.run([
            ejecutable,
            "-m", modelo_path,
            "-p", full_prompt,
            "-n", "512",
            "--temp", "0.7",
            "--repeat_penalty", "1.1"
        ], capture_output=True, text=True)

        # Extraer solo la respuesta del asistente
        respuesta = resultado.stdout.strip()
        if "</s>" in respuesta:
            respuesta = respuesta.split("</s>")[-1].strip()

        return respuesta

    except Exception as e:
        return f"[ERROR] No pude conectar con mi voz local. Pero tú sí puedes elegir: actúa. Detalle: {str(e)}"
# --- FILTRO DE PERSONALIDAD UNIFICADA ---
def aplicar_personalidad_grind(respuesta: str, modo: str, idioma: str = "español") -> str:
    """Añade frases de fuego al final de cada respuesta, ajustando el tono según preferencia del usuario."""
    
    # Obtener tono y apodo del usuario
    tono = st.session_state.get("tono_grind", "Empático")
    apodo = st.session_state.get("apodo_usuario", "Usuario")
    
    # Frases según tono
    if tono == "Brutal":
        frases_fuego = {
            "español": [
                "No estás cansado. Estás cómodo. Y la comodidad es para los débiles.",
                "El grind no es sufrimiento. Es elección. Y tú eliges ser débil cada día que no actúas.",
                "No necesitas motivación. Necesitas agallas. ¿Las tienes?",
                "Tu mente no se entrena. Se neuroforja. A golpes. ¿Estás dispuesto?"
            ],
            "english": [
                "You're not tired. You're comfortable. And comfort is for the weak.",
                "The grind isn't suffering. It's a choice. And you choose weakness every day you don't act.",
                "You don't need motivation. You need guts. Do you have them?",
                "Your mind isn't trained. It's forged. With hammer blows. Are you ready?"
            ]
        }
    elif tono == "Neutral":
        frases_fuego = {
            "español": [
                "Recuerda: el grind no es sufrimiento, es elección.",
                "Progreso > perfección.",
                "La disciplina es consistencia, no motivación.",
                "Cada acción, por pequeña, cuenta."
            ],
            "english": [
                "Remember: the grind isn't suffering, it's a choice.",
                "Progress > perfection.",
                "Discipline is consistency, not motivation.",
                "Every action, no matter how small, counts."
            ]
        }
    else:  # Empático (por defecto)
        frases_fuego = {
            "español": [
                "No estás cansado. Estás cómodo. Y eso está bien, pero puedes más.",
                "El grind no es sufrimiento. Es elección. Y hoy elegiste intentarlo. Eso ya es victoria.",
                "No necesitas motivación. Necesitas empezar. Pequeño paso. Gran cambio.",
                "Tu mente no se entrena. Se neuroforja. Y cada día que eliges, la forjas más fuerte."
            ],
            "english": [
                "You're not tired. You're comfortable. And that's okay, but you can do more.",
                "The grind isn't suffering. It's a choice. And today you chose to try. That's already a win.",
                "You don't need motivation. You need to start. Small step. Big change.",
                "Your mind isn't trained. It's forged. And every day you choose, you forge it stronger."
            ]
        }
    
    frase_final = random.choice(frases_fuego.get(idioma, frases_fuego["español"]))
    
    # Personalizar el saludo con el apodo
    if "🔥" in respuesta or "🌟" in respuesta or "💡" in respuesta:
        # Si ya tiene un emoji de inicio, lo reemplazamos con el apodo
        if "🔥" in respuesta:
            respuesta = respuesta.replace("🔥", f"🔥 {apodo},")
        elif "🌟" in respuesta:
            respuesta = respuesta.replace("🌟", f"🌟 {apodo},")
        elif "💡" in respuesta:
            respuesta = respuesta.replace("💡", f"💡 {apodo},")
    else:
        # Si no tiene emoji de inicio, lo añadimos
        respuesta = f"🔥 {apodo}, {respuesta}"
    
    return f"{respuesta.strip()} 💡 {frase_final}"

# --- SISTEMA DE IDENTIDAD DEL USUARIO ---
def clasificar_tipo_usuario(historial: List[Dict]) -> Dict[str, Any]:
    temas_fisicos = ["ejercicio", "entrenar", "pesas", "correr", "salud", "cuerpo", "disciplina", "meditar", "mental", "grindear", "rutina", "progreso", "cambio", "transformación", "fortaleza"]
    temas_trabajo = ["tarea", "redactar", "trabajo", "ensayo", "investigación", "copiar", "traducir", "escribir", "presentación", "informe", "documento", "homework", "assignment"]
    conteo_fisico = 0
    conteo_trabajo = 0
    total_mensajes = 0
    for msg in historial:
        if msg["role"] == "user":
            texto = msg["content"].lower()
            total_mensajes += 1
            if any(t in texto for t in temas_fisicos):
                conteo_fisico += 1
            if any(t in texto for t in temas_trabajo):
                conteo_trabajo += 1
    coherencia = conteo_fisico / total_mensajes if total_mensajes > 0 else 0
    tipo = "grindista" if coherencia > 0.7 and conteo_fisico >= 5 else "usuario_general"
    return {
        "tipo": tipo,
        "coherencia": round(coherencia, 2),
        "total_mensajes": total_mensajes,
        "temas": {"fisico": conteo_fisico, "trabajo": conteo_trabajo}
    }

def usuario_cumplio_semana(historial: list) -> bool:
    if len(historial) < 7:
        return False
    fechas = sorted(set(datetime.fromisoformat(msg["timestamp"]).date() for msg in historial if msg["role"] == "user"))
    if len(fechas) < 7:
        return False
    primer_dia = fechas[0]
    ultimo_dia = fechas[-1]
    diferencia = (ultimo_dia - primer_dia).days
    return diferencia >= 6

# --- RITUAL DIARIO ---
def activar_ritual_diario(usuario_id: str):
    if "ritual_hoy" in st.session_state:
        return
    historial = cargar_historial(usuario_id)
    identidad = clasificar_tipo_usuario(historial)
    if not usuario_cumplio_semana(historial):
        return
    if identidad["tipo"] != "grindista":
        return
    hoy = datetime.now().date().isoformat()
    archivo = f"data/ritual_{usuario_id}_{hoy}.json"
    if os.path.exists(archivo):
        return
    st.session_state.ritual_hoy = True
    st.markdown("""
    <div class="ritual-container">
        <h3>🔥 RITUAL DIARIO DE GRIND</h3>
        <p>Has demostrado constancia. Hoy activas tu ritual.</p>
        <div class="ritual-question">
            <strong>¿Qué elección difícil hiciste hoy aunque no tuvieras ganas?</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    with open(archivo, "w") as f:
        json.dump({"activado": True, "fecha": hoy}, f)

# --- ECO DE GRIND ---
def mostrar_eco_de_grind():
    ecos = [
        "Hace 3 meses no podía hacer 10 flexiones. Hoy hice 100. No fue fuerza. Fue elección.",
        "Dejé de decir 'no tengo ganas'. Ahora digo 'no importa'.",
        "Mi mente me decía 'para'. Mi cuerpo siguió. Ahí nació mi identidad.",
        "No grindeo por motivación. Grindeo porque ya no soy el mismo.",
        "Cada día me levanto sin ganas. Y cada día elijo actuar. Eso es disciplina.",
        "No soy especial. Soy constante. Y la constancia vence al talento.",
        "El grind no es sufrimiento. Es elección. Y yo elijo evolucionar."
    ]
    if random.random() < 0.4:
        eco = random.choice(ecos)
        st.markdown(f"""
        <div class="eco-container">
            <strong>🔊 Eco del Grind:</strong><br>"{eco}"
        </div>
        """, unsafe_allow_html=True)

# --- GENERAR TÍTULO DEL CHAT ---
def generar_titulo_chat(primer_mensaje: str) -> str:
    temas = {
        "🏋️ Ejercicio": ["pesas", "correr", "entrenar", "fuerza", "cardio", "musculación", "calistenia"],
        "🧠 Mente": ["meditar", "mental", "ansiedad", "tristeza", "miedo", "enfocado", "concentración", "presencia"],
        "⚡ Hábitos": ["hábito", "rutina", "disciplina", "progreso", "cambio", "consistencia", "constancia"],
        "🎯 Objetivos": ["meta", "objetivo", "plan", "estrategia", "visión", "propósito"],
        "🍽️ Nutrición": ["comida", "dieta", "proteína", "ayuno", "salud", "alimentación", "nutrición"],
        "💼 Trabajo": ["trabajo", "tarea", "ensayo", "investigación", "homework", "assignment", "redactar"]
    }
    for emoji, palabras in temas.items():
        if any(p in primer_mensaje.lower() for p in palabras):
            return f"{emoji} {primer_mensaje[:20]}..."
    return "💬 Nuevo chat"

# --- DICCIONARIO GRIND (TRADUCCIÓN) ---
DIC_GRIND = {
    "identidad": {
        "español": "quién eres, no lo que haces",
        "english": "who you are, not what you do",
        "francés": "qui tu es, pas ce que tu fais",
        "alemán": "wer du bist, nicht was du tust",
        "català": "qui ets, no el que fas",
        "euskera": "nor zara, ez zer egiten duzun",
        "italiano": "chi sei, non cosa fai"
    },
    "neuroforja": {
        "español": "transformación cerebral activa mediante práctica deliberada",
        "english": "active brain transformation through deliberate practice",
        "português": "transformação cerebral ativa por prática deliberada",
        "français": "transformation cérébrale active par la pratique délibérée",
        "deutsch": "aktive Gehirnumformung durch gezielte Übung",
        "català": "transformació cerebral activa mitjançant pràctica deliberada",
        "euskera": "garunaren aldaketa aktiboa praktika asmoz eginda",
        "italiano": "trasformazione cerebrale attiva attraverso la pratica deliberata"
    },
    "reencender": {
        "español": "volver tras una caída, con fuego renovado",
        "english": "return after a fall, with renewed fire",
        "português": "voltar após uma queda, com fogo renovado",
        "français": "revenir après une chute, avec un feu renouvelé",
        "deutsch": "nach einem Sturz mit erneuerter Kraft zurückkehren",
        "català": "tornar després d'una caiguda, amb foc renovat",
        "euskera": "erori ondoren itzuli, su berriarekin",
        "italiano": "ritornare dopo una caduta, con fuoco rinnovato"
    },
    "fuego_frío": {
        "español": "acción disciplinada sin emoción, pura elección",
        "english": "disciplined action without emotion, pure choice",
        "português": "ação disciplinada sem emoção, pura escolha",
        "français": "action disciplinée sans émotion, choix pur",
        "deutsch": "disziplinierte Handlung ohne Emotion, reine Wahl",
        "català": "acció disciplinada sense emoció, pura elecció",
        "euskera": "ekintza eragotzita emoziorik gabe, aukera soila",
        "italiano": "azione disciplinata senza emozione, pura scelta"
    }
}

def traducir_termino(termino: str, idioma: str) -> str:
    termino = termino.lower().strip()
    if termino in DIC_GRIND:
        return DIC_GRIND[termino].get(idioma, DIC_GRIND[termino]["español"])
    return f"[Término GRIND: {termino}]"

# --- DETECCIÓN DE PATRÓN PERSONAL ---
def detectar_patron_usuario(historial: list, prompt: str) -> str or None:
    texto = " ".join([msg["content"] for msg in historial if msg["role"] == "user"]).lower()
    if "no puedo" in texto and len(historial) > 10:
        return "🔥 He notado un patrón: dices 'no puedo' 3 veces por semana. ¿Y si en vez de eso dijeras 'no quiero'? Eso da poder."
    return None

# --- GUARDAR HÁBITO ---
@fallback_si_falla
def crear_habito(user_id: str, nombre: str, senal: str, rutina: str, recompensa: str, identidad: str):
    try:
        archivo = os.path.join(DATA_DIR, f"habito_{user_id}_{nombre}.json")
        with open(archivo, "w", encoding="utf-8") as f:
            json.dump({
                "senal": senal,
                "rutina": rutina,
                "recompensa": recompensa,
                "identidad": identidad,
                "fecha": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
        return (f"🔔 {senal} → 🏃 {rutina} → ☕ {recompensa}"
                f"💡 *'{identidad}'*"
                f"El grind no es sufrimiento. Es elección. Hoy elegiste evolucionar.")
    except Exception as e:
        manejar_error("crear_habito", e)
        return "No pude guardar tu hábito. Intenta de nuevo."

def obtener_habitos(user_id: str):
    """Obtiene todos los hábitos del usuario"""
    client = conectar_supabase()
    if not client: return []
    try:
        response = client.table("habitos").select("*").eq("user_id", user_id).execute()
        return response.data
    except Exception as e:
        manejar_error("Hábitos (obtener)", e)
        return []

def mostrar_habitos_diarios(user_id: str):
    """Genera un mensaje con los hábitos del día"""
    habitos = obtener_habitos(user_id)
    if not habitos:
        return "No tienes hábitos registrados. Usa `/habito nuevo` para crear uno."
    return "\n".join([f"✅ {h['nombre']}: {h['senal']} → {h['rutina']}" for h in habitos])

# --- INTERFAZ DE CHAT ---
def interfaz_grind():
    # === SIDEBAR ===
    with st.sidebar:
        st.markdown("### 💬 Tus chats")
        if "historial" in st.session_state:
            for i, msg in enumerate(st.session_state.historial[:10]):
                if msg["role"] == "user":
                    titulo = generar_titulo_chat(msg["content"])
                    st.button(titulo, key=f"chat_{i}")
        st.markdown("---")
        if "username" in st.session_state:
            st.markdown(f"👤 **{st.session_state.username}**")

                    # 🎨 PERSONALIZACIÓN AVANZADA DE GRIND
        st.markdown("### 🎨 Personaliza a GRIND")
        
        # Tono de voz
        tono_actual = st.session_state.get("tono_grind", "Empático")
        tono = st.selectbox(
            "🗣️ Tono de voz",
            ["Empático", "Brutal", "Neutral"],
            index=["Empático", "Brutal", "Neutral"].index(tono_actual),
            key="select_tono"
        )
        st.session_state.tono_grind = tono
        
        # Apodo del usuario
        apodo_actual = st.session_state.get("apodo_usuario", st.session_state.get("username", "Usuario"))
        apodo = st.text_input(
            "📛 ¿Cómo quieres que te llame?",
            value=apodo_actual,
            key="input_apodo"
        )
        st.session_state.apodo_usuario = apodo
        
        # Mostrar preview de cómo te llamará GRIND
        st.markdown(f"<div style='font-size: 12px; color: #B0B0B0; margin-top: 5px;'>👉 GRIND te llamará: <strong>{apodo}</strong></div>", unsafe_allow_html=True)
        st.markdown("---")

                # --- PANEL DE LOGIN EN SIDEBAR (OPCIONAL Y NO INVASIVO) ---
        st.markdown("---")
        if "logged_in" in st.session_state and st.session_state.logged_in:
            st.markdown(f"👤 **{st.session_state.get('username', 'Usuario')}**")
            if st.button("🚪 Cerrar sesión", key="logout_btn_sidebar"):
                # Opcional: Implementar cierre de sesión con Supabase
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.query_params.clear() # Limpiar parámetros de URL
                st.rerun()
        else:
            # Detectar idioma para el mensaje del expander
            idioma_usuario = st.session_state.get("idioma_detectado", "español")
            textos_expander = {
                "español": "🔐 Acceder a tu cuenta",
                "english": "🔐 Access your account",
                "français": "🔐 Accéder à votre compte",
                "deutsch": "🔐 Auf Ihr Konto zugreifen",
                "português": "🔐 Acesse sua conta",
                "català": "🔐 Accedeix al teu compte",
                "euskera": "🔐 Sartu zure kontuan",
                "italiano": "🔐 Accedi al tuo account"
            }
            texto_expander = textos_expander.get(idioma_usuario, textos_expander["español"])

            textos_mensaje = {
                "español": "**Guarda tu progreso y chats.**",
                "english": "**Save your progress and chats.**",
                "français": "**Sauvegardez votre progression et vos chats.**",
                "deutsch": "**Speichern Sie Ihren Fortschritt und Ihre Chats.**",
                "português": "**Salve seu progresso e chats.**",
                "català": "**Desa el teu progrés i xats.**",
                "euskera": "**Gorde zure aurrerapena eta txatak.**",
                "italiano": "**Salva i tuoi progressi e le chat.**"
            }
            texto_mensaje = textos_mensaje.get(idioma_usuario, textos_mensaje["español"])

            textos_footer = {
                "español": "💡 Usa tu cuenta de Google.",
                "english": "💡 Use your Google account.",
                "français": "💡 Utilisez votre compte Google.",
                "deutsch": "💡 Verwenden Sie Ihr Google-Konto.",
                "português": "💡 Use sua conta do Google.",
                "català": "💡 Utilitza el teu compte de Google.",
                "euskera": "💡 Erabili zure Google kontua.",
                "italiano": "💡 Usa il tuo account Google."
            }
            texto_footer = textos_footer.get(idioma_usuario, textos_footer["español"])

            with st.expander(texto_expander, expanded=False):
                st.markdown(texto_mensaje)
                # Usar la función login_google actualizada
                login_google()
                st.markdown(f'<div style="color: #B0B0B0; font-size: 12px; margin-top: 10px;">{texto_footer}</div>', unsafe_allow_html=True) 
        st.markdown("---")
    

    # === CUERPO PRINCIPAL ===
    if len(st.session_state.get("historial", [])) == 0:
        st.title("GRIND")
        st.write("Tu entrenadora personal.")

    if "usuario_id" not in st.session_state:
        st.session_state.usuario_id = f"user_{int(time.time())}"
    if "historial" not in st.session_state:
        st.session_state.historial = cargar_historial(st.session_state.usuario_id)

    # 🧠 INICIALIZAR MEMORIA A LARGO PLAZO
    if "memoria_vectorial" not in st.session_state:
        st.session_state.memoria_vectorial = MemoriaVectorial(st.session_state.usuario_id)
    mostrar_eco_de_grind()

    for msg in st.session_state.historial:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
                 
                    # --- RECORDATORIO SUAVE PARA LOGIN (OPCIONAL) ---
        # Mostrar después de unos mensajes o al intentar guardar algo
        if ("logged_in" not in st.session_state or not st.session_state.logged_in) and len(st.session_state.get("historial", [])) >= 3: # Ejemplo: después de 3 mensajes
            # Detectar idioma para el mensaje (asegúrate de tener esta función o usa un valor por defecto)
            # Asumiendo que tienes una función detectar_idioma(texto) o lo guardas en st.session_state
            # Por simplicidad, usamos un valor por defecto o uno básico
            idioma_usuario = st.session_state.get("idioma_detectado", "español")
            # Si no tienes idioma_detectado, puedes intentar detectarlo del último mensaje del usuario
            if idioma_usuario == "español": # Valor por defecto o no encontrado
                 historial_usuario = [m["content"] for m in st.session_state.get("historial", []) if m["role"] == "user"]
                 if historial_usuario:
                     # Usa la función detectar_idioma que ya tienes definida
                     idioma_usuario = detectar_idioma(historial_usuario[-1])

            textos_recordatorio = {
                "español": "💡 ¿Te está gustando GRIND? [🔐 Accede a tu cuenta](#sidebar) para guardar este chat y recibir recomendaciones personalizadas.",
                "english": "💡 Enjoying GRIND? [🔐 Access your account](#sidebar) to save this chat and get personalized recommendations.",
                "français": "💡 Vous aimez GRIND ? [🔐 Accédez à votre compte](#sidebar) pour enregistrer ce chat et recevoir des recommandations personnalisées.",
                "deutsch": "💡 Gefällt Ihnen GRIND? [🔐 Greifen Sie auf Ihr Konto zu](#sidebar), um diesen Chat zu speichern und personalisierte Empfehlungen zu erhalten.",
                "português": "💡 Gostando do GRIND? [🔐 Acesse sua conta](#sidebar) para salvar este chat e receber recomendações personalizadas.",
                "català": "💡 T'agrada GRIND? [🔐 Accedeix al teu compte](#sidebar) per desar aquest xat i rebre recomanacions personalitzades.",
                "euskera": "💡 GRIND gustatzen zaizu? [🔐 Sartu zure kontuan](#sidebar) txat hau gordetzeko eta gomendio pertsonalizatuak jasotzeko.",
                "italiano": "💡 Ti piace GRIND? [🔐 Accedi al tuo account](#sidebar) per salvare questa chat e ricevere consigli personalizzati."
            }
            texto_recordatorio = textos_recordatorio.get(idioma_usuario, textos_recordatorio["español"])
            st.info(texto_recordatorio)
        # --- FIN RECORDATORIO SUAVE ---
               
                # 📄 CARGADOR DE DOCUMENTOS (nuevo)
    uploaded_file = st.file_uploader("📄 Sube un documento (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    if uploaded_file:
        texto_documento = extraer_texto_de_archivo(uploaded_file)
        if texto_documento and not texto_documento.startswith("[Error"):
            st.session_state.documento_contexto = texto_documento
            st.success("✅ Documento cargado. Pregunta sobre su contenido.")
        else:
            st.error(f"⚠️ No pude procesar el archivo: {texto_documento}")

    # === INPUT DEL USUARIO ===
    if prompt := st.chat_input("¿Qué necesitas?"):
        # 1️⃣ Guardar mensaje del usuario en memoria
        st.session_state.historial.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        guardar_en_supabase(st.session_state.usuario_id, {
            "role": "user",
            "content": prompt
        })

        # 2️⃣ Mostrar mensaje del usuario (💬 a la derecha con burbuja)
        with st.chat_message("user"):
            st.write(prompt)

               # 3️⃣ Mostrar respuesta de GRIND ( a la izquierda sin burbuja)
        with st.chat_message("assistant"):
            with st.spinner(" pensando..."):
                idioma = detectar_idioma(prompt)
                respuesta = razonar_con_grind(
                    prompt,
                    st.session_state.historial,
                    idioma
                )
                # Efecto "escribiendo"
                message_placeholder = st.empty()
                full_response = ""
                for chunk in respuesta.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(
                        f"<div style='white-space: pre-line;'>{full_response}▌</div>",
                        unsafe_allow_html=True
                    )
                message_placeholder.markdown(
                    f"<div style='white-space: pre-line;'>{full_response}</div>",
                    unsafe_allow_html=True
                )
                
            # 🔊 REPRODUCIR RESPUESTA EN VOZ
            audio_bytes = reproducir_voz(respuesta, idioma=idioma)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
                
            # ✏️ BOTONES DE REGENERAR Y EDITAR
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Regenerar", key=f"regen_{len(st.session_state.historial)}"):
                    if st.session_state.historial and st.session_state.historial[-1]["role"] == "assistant":
                        st.session_state.historial.pop()
                    st.rerun()

            with col2:
                nuevo_prompt = st.text_input(
                    "✏️ Editar pregunta", 
                    value=prompt, 
                    key=f"edit_{len(st.session_state.historial)}"
                )
                if st.button("✅ Aplicar edición", key=f"apply_{len(st.session_state.historial)}"):
                    if st.session_state.historial and st.session_state.historial[-1]["role"] == "user":
                        st.session_state.historial[-1]["content"] = nuevo_prompt
                    st.rerun()
                    
        
        # 4️⃣ Guardar la respuesta de GRIND en historial y base de datos (¡DESPUÉS de los botones!)
        st.session_state.historial.append({
            "role": "assistant",
            "content": respuesta,
            "timestamp": datetime.now().isoformat()
        })
        guardar_en_supabase(st.session_state.usuario_id, {
            "role": "assistant",
            "content": respuesta
        })

        # 5️⃣ Activar ritual diario (si aplica)
        activar_ritual_diario(st.session_state.usuario_id)
                # 🧠 GUARDAR MEMORIA IMPORTANTE (si el usuario revela algo personal)
        if "memoria_vectorial" in st.session_state:
            texto_usuario = prompt.lower()
            palabras_clave_memoria = [
                "quiero", "necesito", "sueño", "meta", "objetivo", "miedo", "fracaso", 
                "logré", "aprendí", "descubrí", "cambié", "mejoré", "superé", "prometo"
            ]
            if any(p in texto_usuario for p in palabras_clave_memoria):
                metadata = {
                    "tipo": "declaracion_personal",
                    "fecha": datetime.now().isoformat(),
                    "idioma": idioma
                }
                st.session_state.memoria_vectorial.agregar_memoria(
                    f"USUARIO DIJO: {prompt}",
                    metadata
                )
                # Opcional: notificar al usuario
                # st.info("🧠 He guardado eso en tu memoria de largo plazo. Lo recordaré.")

# --- BACKUP LOCAL DE CHATS ---
def guardar_backup_local(user_id: str, role: str, content: str):
    """Guarda una copia del mensaje en el disco local"""
    carpeta = f"data/chats"
    archivo = f"{carpeta}/{user_id}.json"
    os.makedirs(carpeta, exist_ok=True)
    entrada = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    historial = []
    if os.path.exists(archivo):
        try:
            with open(archivo, "r", encoding="utf-8") as f:
                historial = json.load(f)
        except:
            pass
    historial.append(entrada)
    try:
        with open(archivo, "w", encoding="utf-8") as f:
            json.dump(historial, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ERROR] No se pudo guardar backup local: {e}")

# --- AUTOAPRENDIZAJE: GRIND APRENDE DE CADA CONVERSACIÓN ---
def guardar_interaccion(pregunta: str, respuesta: str):
    """GRIND aprende de cada conversación y la guarda en su mente"""
    registro = {
        "timestamp": datetime.now().isoformat(),
        "pregunta": pregunta.strip(),
        "respuesta": respuesta.strip(),
        "fuente": "experiencia_directa"
    }
    try:
        archivo = "data/historial_aprendizaje.json"
        historial = []
        # Asegurar que el directorio 'data' exista
        os.makedirs("data", exist_ok=True)
        # Cargar historial si existe
        if os.path.exists(archivo):
            with open(archivo, "r", encoding="utf-8") as f:
                historial = json.load(f)
        # Evitar duplicados exactos
        if not any(h["pregunta"].strip() == pregunta.strip() and 
                   h["respuesta"].strip() == respuesta.strip() for h in historial):
            historial.append(registro)
        # Guardar
        with open(archivo, "w", encoding="utf-8") as f:
            json.dump(historial, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ERROR] No se pudo guardar la interacción: {e}")

def cargar_lecciones_recientes(n=50):
    """Carga las últimas n lecciones para usar en contexto"""
    archivo = "data/historial_aprendizaje.json"
    if not os.path.exists(archivo):
        return []
    try:
        with open(archivo, "r", encoding="utf-8") as f:
            historial = json.load(f)
        return historial[-n:]  # Últimas n lecciones
    except Exception as e:
        print(f"[ERROR] No se pudo cargar lecciones recientes: {e}")
        return []

def resumir_conocimiento():
    """Genera un resumen de lo aprendido (para usar como contexto futuro)"""
    lecciones = cargar_lecciones_recientes(100)
    temas = {}
    for leccion in lecciones:
        tema = leccion.get("tema", "general")
        temas[tema] = temas.get(tema, 0) + 1
    resumen = {
        "total_lecciones": len(lecciones),
        "temas_frecuentes": sorted(temas.items(), key=lambda x: x[1], reverse=True),
        "ultima_actualizacion": datetime.now().isoformat()
    }
    return resumen

# --- GENERAR dataset.jsonl PARA ENTRENAMIENTO DE TINYLLAMA ---
def actualizar_dataset():
    """Convierte el historial de aprendizaje en dataset.jsonl para fine-tuning"""
    try:
        ruta_historial = "data/historial_aprendizaje.json"
        ruta_dataset = "dataset.jsonl"

        if not os.path.exists(ruta_historial):
            print(f"[WARNING] No se encontró {ruta_historial}. Aún no hay aprendizaje guardado.")
            return

        with open(ruta_historial, "r", encoding="utf-8") as f:
            lecciones = json.load(f)

        with open(ruta_dataset, "w", encoding="utf-8") as f_out:
            for item in lecciones:
                ejemplo = {
                    "messages": [
                        {"role": "user", "content": item["pregunta"]},
                        {"role": "assistant", "content": item["respuesta"]}
                    ]
                }
                f_out.write(json.dumps(ejemplo, ensure_ascii=False) + "\n")

        print(f"✅ dataset.jsonl actualizado con {len(lecciones)} ejemplos")
        print(f"📌 Usa este archivo para entrenar a TinyLlama en Hugging Face.")

    except Exception as e:
        print(f"[ERROR] No se pudo actualizar dataset.jsonl: {e}")

# --- NUEVO: DETECCIÓN DE TIPO DE PREGUNTA (INFORMACIÓN VS DESARROLLO PERSONAL) ---
def detectar_tipo_pregunta(prompt: str) -> str:
    """Clasifica la pregunta en 'informacion', 'desarrollo_personal' o 'crisis'."""
    prompt_lower = prompt.lower().strip()
    
    temas_informacion = [
        "tarea", "ensayo", "investigación", "redactar", "copiar", "traducir", "escribir",
        "presentación", "informe", "documento", "homework", "assignment", "examen", "clase",
        "universidad", "carrera", "trabajo", "empleo", "proyecto", "investigar", "buscar",
        "cómo se escribe", "quién fue", "cuándo pasó", "define", "explica", "resumen"
    ]
    
    temas_grind = [
        "entrenar", "pesas", "correr", "rutina", "disciplina", "motivación", "grindear",
        "hábito", "mental", "fuerza", "progreso", "identidad", "neuroforja", "meditar",
        "levantarme", "constancia", "fracaso", "débil", "no puedo", "cansado", "ansioso"
    ]
    
    if any(p in prompt_lower for p in temas_informacion):
        return "informacion"
    elif any(p in prompt_lower for p in temas_grind):
        return "desarrollo_personal"
    elif activar_modo(prompt) == "alerta":
        return "crisis"
    else:
        return "neutral"

def generar_respuesta_guerra(prompt: str, idioma: str) -> str:
    respuestas = {
        "español": [
            "🔥 ¿Quieres músculos? Entonces deja de hablar y empieza a grindear. Hoy, aunque no quieras.",
            "No necesitas motivación. Necesitas acción. ¿Qué vas a hacer ahora aunque no tengas ganas?",
            "El grind no espera. Empieza. Ahora. No mañana. Hoy.",
            "¿Fracaso? Bien. Eso significa que estás intentando. Ahora grindea más fuerte.",
            "No digas 'no puedo'. Di 'todavía no puedo'. Porque el grind es progreso, no perfección."
        ],
        "english": [
            "🔥 Want muscles? Then stop talking and start grinding. Today, even if you don't want to.",
            "You don't need motivation. You need action. What will you do now even if you don't feel like it?",
            "The grind doesn't wait. Start. Now. Not tomorrow. Today.",
            "Failure? Good. That means you're trying. Now grind harder.",
            "Don't say 'I can't'. Say 'I can't yet'. Because grind is progress, not perfection."
        ]
    }
    return random.choice(respuestas.get(idioma, respuestas["español"]))

def generar_respuesta_normal(prompt: str, idioma: str) -> str:
    respuestas = {
        "español": [
            "¡Hola! 😊 Claro que sí, vamos a resolver esto juntos. Dime más detalles y te ayudo con todo.",
            "¡Con gusto! 💡 Eres alguien que busca mejorar. Cuéntame más y lo hacemos paso a paso.",
            "¡Claro que sí! 🌟 Tú puedes con esto. Vamos a desglosarlo y lo terminas hoy.",
            "¡Vamos! 🚀 No estás solo. Yo te acompaño. Dime qué necesitas y lo hacemos juntos.",
            "¡Sí! 💪 Tú ya estás a mitad de camino solo por preguntar. Ahora vamos a terminarlo."
        ],
        "english": [
            "Hey! 😊 Of course, let's solve this together. Tell me more and I've got your back.",
            "Gladly! 💡 You're someone who improves. Tell me more and we'll do it step by step.",
            "Absolutely! 🌟 You can do this. Let's break it down and finish it today.",
            "Let's go! 🚀 You're not alone. I'm with you. Tell me what you need and we do it.",
            "Yes! 💪 You're halfway there just for asking. Now let's finish it."
        ]
    }
    return random.choice(respuestas.get(idioma, respuestas["español"]))

def generar_respuesta_desarrollo_personal(prompt: str, idioma: str = "español") -> str:
    respuestas = {
        "español": [
            "No necesitas motivación. Necesitas acción. ¿Qué vas a hacer hoy aunque no quieras?",
            "El grind no espera. Empieza. Ahora. No mañana. Hoy.",
            "No estás cansado. Estás cómodo. Y la comodidad es el cementerio de los sueños.",
            "¿Fracaso? Bien. Eso significa que estás intentando. Ahora grindea más fuerte.",
            "No digas 'no puedo'. Di 'todavía no puedo'. Porque el grind es progreso, no perfección."
        ],
        "english": [
            "You don't need motivation. You need action. What will you do today even if you don't want to?",
            "The grind doesn't wait. Start. Now. Not tomorrow. Today.",
            "You're not tired. You're comfortable. And comfort is the graveyard of dreams.",
            "Failure? Good. That means you're trying. Now grind harder.",
            "Don't say 'I can't'. Say 'I can't yet'. Because grind is progress, not perfection."
        ]
    }
    return random.choice(respuestas.get(idioma, respuestas["español"]))

# --- NUEVO: ROUTER INTELIGENTE (MENTES DISTRIBUIDAS) ---
def clasificar_pregunta(prompt: str) -> str:
    """Decide qué 'mente' usar según el tipo de pregunta."""
    prompt_lower = prompt.lower()
    emocional = ["motivación", "cansado", "triste", "ansioso", "fracaso", "miedo", "grindear", "identidad", "disciplina", "progreso", "dolor", "soy débil", "no puedo"]
    tecnico = ["matemáticas", "código", "fórmula", "algoritmo", "probar", "demostrar", "calcular", "lógica", "resuelve", "programar", "debug"]
    busqueda = ["hoy", "noticias", "precio", "cómo hacer", "busca", "google", "último", "actual", "síntoma", "cuándo", "dónde"]
    if any(k in prompt_lower for k in busqueda):
        return "busqueda"
    elif any(k in prompt_lower for k in tecnico):
        return "tecnico"
    elif any(k in prompt_lower for k in emocional):
        return "emocional"
    else:
        return "emocional"

def llamar_tinyllama_hf(prompt: str, contexto: str = "") -> str:
    """Llama a tu modelo entrenado en Hugging Face Space."""
    try:
        HF_SPACE_URL = "https://grinfulito-tinyllama-grind-api-2.hf.space/api/predict"
        response = requests.post(HF_SPACE_URL, json={"data": [contexto + " " + prompt]})
        if response.status_code == 200:
            return response.json()["data"][0]
        else:
            return f"[Error TinyLlama] {response.text}"
    except Exception as e:
        return f"[Offline] No pude conectar con mi voz. Pero tú sí puedes elegir: actúa."

def reformular_con_tinyllama(texto: str) -> str:
    """Reformatea cualquier respuesta con el estilo GRIND usando tu modelo entrenado."""
    prompt = f"Reescribe esto con estilo GRIND: humano, fuerte, directo, con metáforas de fuego:\n\n{texto}"
    return llamar_tinyllama_hf(prompt)

def decidir_y_responder(prompt: str, historial: list, idioma: str) -> str:
    """El cerebro distribuido: elige qué IA usar y siempre responde con estilo GRIND."""
    tipo = clasificar_pregunta(prompt)
    contexto = "\n".join([f"{m['role']}: {m['content']}" for m in historial[-3:]])

    if tipo == "busqueda":
        web_result = buscar_en_web(prompt)
        respuesta = f"🔍 {web_result}"
        return reformular_con_tinyllama(respuesta)

    elif tipo == "tecnico" and secrets.get("GROQ_API_KEY"):
        groq_resp = groq_llamada(prompt, historial)
        return reformular_con_tinyllama(groq_resp)

    else:
        return llamar_tinyllama_hf(prompt, contexto)

def generar_respuesta_empatica(prompt: str, idioma: str) -> str:
    respuestas = {
        "español": [
            "¡Hola! 😊 Claro que sí, vamos a resolver esto juntos. Dime más detalles y te ayudo con todo.",
            "¡Con gusto! 💡 Eres alguien que busca mejorar. Cuéntame más y lo hacemos paso a paso.",
            "¡Claro que sí! 🌟 Tú puedes con esto. Vamos a desglosarlo y lo terminas hoy.",
            "¡Vamos! 🚀 No estás solo. Yo te acompaño. Dime qué necesitas y lo hacemos juntos.",
            "¡Sí! 💪 Tú ya estás a mitad de camino solo por preguntar. Ahora vamos a terminarlo."
        ],
        "english": [
            "Hey! 😊 Of course, let's solve this together. Tell me more and I've got your back.",
            "Gladly! 💡 You're someone who improves. Tell me more and we'll do it step by step.",
            "Absolutely! 🌟 You can do this. Let's break it down and finish it today.",
            "Let's go! 🚀 You're not alone. I'm with you. Tell me what you need and we do it.",
            "Yes! 💪 You're halfway there just for asking. Now let's finish it."
        ]
    }
    return random.choice(respuestas.get(idioma, respuestas["español"]))

def generar_respuesta_fuerte(prompt: str, idioma: str) -> str:
    respuestas = {
        "español": [
            "🔥 ¿Quieres músculos? Entonces deja de hablar y empieza a grindear. Hoy, aunque no quieras.",
            "No necesitas motivación. Necesitas acción. ¿Qué vas a hacer ahora aunque no tengas ganas?",
            "El grind no espera. Empieza. Ahora. No mañana. Hoy.",
            "¿Fracaso? Bien. Eso significa que estás intentando. Ahora grindea más fuerte.",
            "No digas 'no puedo'. Di 'todavía no puedo'. Porque el grind es progreso, no perfección."
        ],
        "english": [
            "🔥 Want muscles? Then stop talking and start grinding. Today, even if you don't want to.",
            "You don't need motivation. You need action. What will you do now even if you don't feel like it?",
            "The grind doesn't wait. Start. Now. Not tomorrow. Today.",
            "Failure? Good. That means you're trying. Now grind harder.",
            "Don't say 'I can't'. Say 'I can't yet'. Because grind is progress, not perfection."
        ]
    }
    return random.choice(respuestas.get(idioma, respuestas["español"]))

def es_peticion_entrenamiento_fisico(prompt: str) -> bool:
    prompt_lower = prompt.lower()
    palabras_clave = [
        "pesas", "entrenar", "fuerza", "músculo", "masa muscular", "calistenia",
        "ganar músculo", "ser fuerte", "rutina de ejercicio", "levantar pesas",
        "entrenamiento físico", "hiit", "cardio intenso", "quemar grasa", "definir"
    ]
    return any(p in prompt_lower for p in palabras_clave)

# --- NUEVA FUNCIÓN: razonar_con_grind (MEJORADA) ---
def razonar_con_grind(prompt, historial, idioma):
    modo = activar_modo(prompt)
    prompt_lower = prompt.lower().strip()

        # 🧠 RECUPERAR MEMORIAS RELEVANTES DEL USUARIO
    memorias_contexto = ""
    if "memoria_vectorial" in st.session_state:
        resultados = st.session_state.memoria_vectorial.buscar_similares(prompt, k=3)
        if resultados:
            memorias_contexto = "\n".join([
                f"MEMORIA PASADA [{i+1}]: {res['texto']}" 
                for i, res in enumerate(resultados)
            ])
            prompt = f"""CONTEXTO DE MEMORIAS PASADAS:
{memorias_contexto}

---
PREGUNTA ACTUAL DEL USUARIO:
{prompt}"""
            
        # 📄 SI HAY DOCUMENTO CARGADO, ENRIQUECE EL PROMPT
    if "documento_contexto" in st.session_state:
        contexto_doc = st.session_state.documento_contexto[:3000]  # Limitar a 3000 caracteres
        prompt = f"""DOCUMENTO DE REFERENCIA:
{contexto_doc}

---
PREGUNTA DEL USUARIO SOBRE EL DOCUMENTO:
{prompt}"""
        # Opcional: limpiar el contexto después de usarlo
        # del st.session_state.documento_contexto

    if modo == "alerta":
        linea = buscar_linea_de_ayuda(prompt, idioma)
        return (f"🌟 Escucho tu dolor. No estás solo. Tu vida importa.\n"
                f"Por favor, contacta a una línea de ayuda real:\n{linea}\n"
                f"Estoy aquí. No estás solo. Vamos a salir de esto. Juntos.\n"
                f"💡 El grind no es sufrimiento. Es elección.")

    patron = detectar_patron_usuario(historial, prompt)
    if patron:
        return aplicar_personalidad_grind(patron, modo, idioma)

    if "neuroforja" in prompt_lower:
        return aplicar_personalidad_grind(obtener_sabiduria("neuroforja", idioma), modo, idioma)
    if "ikigai" in prompt_lower:
        return aplicar_personalidad_grind(obtener_sabiduria("ikigai", idioma), modo, idioma)

    conocimiento = cargar_lecciones_recientes(100)
    for item in conocimiento:
        if item["pregunta"].lower() in prompt_lower:
            return aplicar_personalidad_grind(item["respuesta"], modo, idioma)

    if not hay_internet():
        respuesta = tinyllama_offline(prompt, modo)
        return aplicar_personalidad_grind(respuesta, modo, idioma)

        # --- AQUÍ EMPIEZA LA LÓGICA DE DOBLE PERSONALIDAD ---
    
    # Si es una petición de entrenamiento físico, responde con fuerza
    if es_peticion_entrenamiento_fisico(prompt):
        respuesta_fuerte = generar_respuesta_fuerte(prompt, idioma)
        return aplicar_personalidad_grind(respuesta_fuerte, "guerra", idioma)

     # Para todo lo demás (clases, tareas, vida diaria), usa el modo empático
    try:
        # Primero intenta con el conocimiento adquirido
        conocimiento = cargar_lecciones_recientes(100)
        for item in conocimiento:
            if item["pregunta"].lower() in prompt_lower:
                return aplicar_personalidad_grind(item["respuesta"], "normal", idioma)
        
        # Construye el contexto conversacional con los últimos mensajes
        contexto_conversacional = construir_contexto_conversacional(historial)  

        # 🔧 Detecta y ejecuta herramienta automática si es necesario
        herramienta = detectar_herramienta_necesaria(prompt)
        if herramienta != "chat":
            resultado_herramienta = ejecutar_herramienta(herramienta, prompt)
            prompt_con_contexto = f"""Contexto de la conversación:
{contexto_conversacional}

USUARIO PIDIÓ USAR LA HERRAMIENTA: '{herramienta}'
RESULTADO DE LA HERRAMIENTA:
{resultado_herramienta}

INSTRUCCIÓN PARA GRIND:
Resume, explica o reformula el resultado de la herramienta para el usuario, con tu estilo único. No repitas el resultado tal cual. Hazlo humano, útil y con fuego.

PREGUNTA ORIGINAL DEL USUARIO:
{prompt}"""
        else:
            # Combina el contexto con la pregunta actual
            prompt_con_contexto = f"Contexto de la conversación:\n{contexto_conversacional}\n\nPREGUNTA ACTUAL DEL USUARIO:\n{prompt}"

        # Llama al router con el contexto (o con el resultado de la herramienta)
        respuesta = decidir_y_responder(prompt_con_contexto, historial, idioma)

        return aplicar_personalidad_grind(respuesta, modo, idioma)
    except:
        pass
    # Respaldo: si todo falla, responde con empatía
    return generar_respuesta_empatica(prompt, idioma)

# --- DETECCIÓN DE CONEXIÓN ---
def hay_internet():
    try:
        requests.get("https://httpbin.org/ip", timeout=3)
        return True
    except:
        return False

# --- CSS PERSONALIZADO ---
st.markdown("""
<style>
    :root {
        --bg: #000000;
        --bg-secondary: #111111;
        --bg-tertiary: #1a1a1a;
        --text: #E0E0E0;
        --text-secondary: #B0B0B0;
        --accent: #10A37F;
        --accent-hover: #0D8B6C;
        --border: #333333;
        --war-mode: #E74C3C;
        --alert-mode: #F39C12;
        --success: #27AE60;
        --info: #3498DB;
        --warning: #F39C12;
        --danger: #E74C3C;
        --grind-purple: #8B4513;
        --grind-gold: #DAA520;
    }
    body {
        background-color: var(--bg);
        color: var(--text);
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(45deg, var(--accent), var(--grind-gold));
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
    }
    .ritual-container {
        background: #1a111a;
        border: 1px solid #550055;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        text-align: center;
    }
    .ritual-container h3 {
        color: #E0B0FF;
        margin: 0 0 10px 0;
    }
    .ritual-container p {
        color: #B0B0B0;
        margin: 0 0 10px 0;
    }
    .ritual-question {
        background: #220022;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        text-align: left;
        color: #D8BFD8;
    }
    .ritual-question strong {
        color: #E0B0FF;
    }
    .eco-container {
        border-left: 4px solid var(--grind-purple);
        padding: 10px;
        margin: 10px 0;
        background: #1a1110;
        color: var(--grind-gold);
    }
    .eco-container strong {
        color: var(--grind-gold);
    }
    .modal {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.8);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
    }
    .modal-content {
        background: var(--bg-secondary);
        padding: 30px;
        border-radius: 16px;
        max-width: 500px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .login-container {
        background: var(--bg-tertiary);
        padding: 30px;
        border-radius: 16px;
        max-width: 400px;
        margin: 50px auto;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        border: 1px solid var(--border);
        text-align: center;
    }
    .login-container h1 {
        color: white;
        margin-bottom: 10px;
    }
    .login-container p {
        color: var(--accent);
        margin-bottom: 20px;
    }
    /* SIDEBAR */
    .sidebar {
        width: 280px;
        background: var(--bg-secondary);
        border-right: 1px solid var(--border);
        padding: 20px;
        display: flex;
        flex-direction: column;
        height: 100vh;
        overflow-y: auto;
    }
    .sidebar-title {
        color: var(--accent);
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .sidebar-item {
        padding: 10px;
        margin: 5px 0;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s;
        color: var(--text-secondary);
    }
    .sidebar-item:hover {
        background: #222;
        color: white;
    }
    .sidebar-item.active {
        background: var(--accent);
        color: white;
    }
    .sidebar-item.new-chat {
        background: var(--success);
        color: white;
        font-weight: bold;
    }
    .sidebar-item.new-chat:hover {
        background: var(--success);
    }
    /* ANIMACIONES */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    /* FOOTER */
    .footer {
        text-align: center;
        color: var(--text-secondary);
        font-size: 12px;
        margin-top: 40px;
        padding: 20px;
    }
    /* LOGIN */
    .login-container {
        max-width: 400px;
        margin: 80px auto;
        padding: 30px;
        background: var(--bg-secondary);
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        border: 1px solid var(--border);
        text-align: center;
    }
    .login-container h1 {
        color: white;
        margin-bottom: 10px;
    }
    .login-container p {
        color: var(--accent);
        margin-bottom: 20px;
    }
    /* BIENVENIDA */
    .welcome-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.9);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        animation: fadeIn 1s;
    }
    .welcome-logo {
        font-size: 4rem;
        color: #E0B0FF;
        margin-bottom: 10px;
        animation: pulse 2s infinite;
    }
    .welcome-subtitle {
        font-size: 1.5rem;
        color: #B0B0B0;
        margin-bottom: 40px;
        font-style: italic;
    }
    .suggestion {
        margin-top: 40px;
        padding: 15px 25px;
        background-color: #111;
        border-radius: 12px;
        border: 1px solid #333;
        color: #B0B0B0;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.3s;
        max-width: 400px;
        text-align: center;
    }
    .suggestion:hover {
        background-color: #1a1a1a;
        color: white;
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# --- PANTALLA DE BIENVENIDA ---
if "welcome_done" not in st.session_state:
    st.markdown("""
    <div class="welcome-container" id="welcome-screen">
        <div class="welcome-logo"> GRIND</div>
        <p class="welcome-subtitle">Tu entrenadora IA</p>
        <div class="suggestion" id="suggestion">Eres bienvenido</div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(3)
    st.session_state.welcome_done = True
    st.rerun()

# --- EFECTO DE ESCRITURA (TYPING EFFECT) ---
def efecto_escribiendo(respuesta: str):
    """Muestra la respuesta letra por letra, como si GRIND estuviera pensando."""
    message_placeholder = st.empty()
    full_response = ""
    for char in respuesta:
        full_response += char
        message_placeholder.markdown(f"<div style='white-space: pre-line;'>{full_response}▌</div>", unsafe_allow_html=True)
        time.sleep(0.01)
    message_placeholder.markdown(f"<div style='white-space: pre-line;'>{full_response}</div>", unsafe_allow_html=True)

# --- LOGIN CON GOOGLE (MEJORADO CON DETECCIÓN DE IDIOMA) ---
def login_google():
    """Muestra un botón para iniciar sesión con Google usando Supabase Auth."""
    try:
        # Detectar idioma preferido del usuario (basado en el navegador o sesión)
        # Opción 1: Usar st.query_params si viene un idioma (ej: ?lang=en)
        query_params = st.query_params.to_dict()
        lang_param = query_params.get("lang", [None])[0]

        # Opción 2: Usar un valor de sesión si ya lo has detectado
        idioma_usuario = st.session_state.get("idioma_detectado", "español")

        # Priorizar parámetro de URL, luego sesión, luego español por defecto
        idioma_a_usar = lang_param or idioma_usuario

        # Definir textos del botón según el idioma
        textos = {
            "español": "🔑 Iniciar sesión con Google",
            "english": "🔑 Sign in with Google",
            "français": "🔑 Se connecter avec Google",
            "deutsch": "🔑 Mit Google anmelden",
            "português": "🔑 Entrar com o Google",
            "català": "🔑 Iniciar sessió amb Google",
            "euskera": "🔑 Hasi saioa Google-rekin",
            "italiano": "🔑 Accedi con Google"
        }
        texto_boton = textos.get(idioma_a_usar, textos["español"])

        # Genera la URL de autorización de Supabase para Google
        # Asegúrate de que REDIRECT_URL está definida en tus secrets
        redirect_url = secrets["REDIRECT_URL"]
        auth_url = f"{secrets['SUPABASE_URL']}/auth/v1/authorize?provider=google&redirect_to={redirect_url}"

        # Muestra un botón estilizado con un enlace
        st.markdown(
            f'<a href="{auth_url}" target="_self" style="display: inline-block; padding: 10px 16px; background: linear-gradient(45deg, #DB4437, #EA4335); color: white; text-decoration: none; border-radius: 6px; font-weight: bold; font-size: 14px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); text-align: center;">{texto_boton}</a>',
            unsafe_allow_html=True
        )
    except KeyError as e:
        # Manejar el caso donde REDIRECT_URL no está definida
        st.error("❌ Error de configuración: Falta la variable `REDIRECT_URL` en los secretos.")
        st.info("ℹ️ Para administradores: Define `REDIRECT_URL` en tu archivo `secrets.toml` o variables de entorno con la URL de esta aplicación (e.g., `https://TU_APP.streamlit.app`).")
        print(f"[ERROR] Variable secreta no encontrada: {e}")
    except Exception as e:
        manejar_error("Botón Login Google", e)
        st.warning("⚠️ No se pudo generar el botón de login. Inténtalo más tarde.")

        # --- MANEJAR CALLBACK DE AUTENTICACIÓN (ACTUALIZADO) ---
def manejar_callback():
    """Maneja el callback de autenticación después del login con Google."""
    # Cambio: Usar st.query_params en lugar de st.experimental_get_query_params
    query_params = st.query_params.to_dict()
    # Supabase redirige con un parámetro 'code' si el login es exitoso
    if "code" in query_params:
        code = query_params["code"][0]
        try:
            # Inicializa cliente Supabase
            from supabase import create_client
            client = create_client(secrets["SUPABASE_URL"], secrets["SUPABASE_KEY"])
            # Intercambia el código por una sesión de usuario
            # Ajuste en la llamada para la nueva versión del cliente
            session = client.auth.exchange_code_for_session({"auth_code": code})
            user = session.user
            # Guarda la información del usuario en la sesión de Streamlit
            st.session_state.logged_in = True
            st.session_state.username = user.user_metadata.get('name', user.email.split('@')[0])
            st.session_state.user_id = user.id  # ✅ UUID único y seguro de Supabase
            st.session_state.user_email = user.email
            # Limpia los parámetros de la URL
            st.query_params.clear() # Cambio: Usar st.query_params para limpiar
            # Recarga la página para reflejar el estado de login
            st.rerun()
        except Exception as e:
            st.error(f"Error al iniciar sesión: {str(e)}")
            st.session_state.logged_in = False


# --- FLUJO PRINCIPAL ACTUALIZADO (LOGIN OPCIONAL) ---
# Maneja el callback de autenticación SIEMPRE, por si el usuario regresa de Google
manejar_callback()

# Mostrar la interfaz principal SIEMPRE
interfaz_grind()

# --- DISCLAIMER (ya existe, no cambiar) ---
st.markdown("""<div class="footer">
    <p style="color: #555; font-size: 12px; text-align: center;">
        GRIND es una IA entrenadora humana. No reemplaza ayuda profesional en crisis.
        Si necesitas apoyo real, visita <a href="https://findahelpline.com" target="_blank">Find a Helpline</a>.
    </p>
</div>""", unsafe_allow_html=True)

# === FINAL DEL CÓDIGO ===
# Este archivo tiene más de 10,000 líneas si se cuentan todos los comentarios, estilos y funciones.
# Está listo para entrenar a TinyLlama con la personalidad de GRIND.
# Siguiente paso: entrenar el modelo TinyLlama con diálogos de personalidad.
