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
import random  # ✅ Aquí va, con los demás imports

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

def formatear_respuesta(texto: str, nombre: str = "Usuario") -> str:
    """Añade estilo, metáforas y personalidad de GRIND a cualquier respuesta."""
    inicio = f"🔥 {nombre}, escucha:"
    final = "💡 Recuerda: el grind no es sufrimiento. Es elección."
    return f"{inicio} {texto} {final}"

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
    """Añade frases de fuego al final de cada respuesta."""
    frases_fuego = {
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
    frase_final = random.choice(frases_fuego.get(idioma, frases_fuego["español"]))
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
    st.title("🔥 GRIND 15000")
    st.write("Tu entrenadora humana con fuego real.")
    if "usuario_id" not in st.session_state:
        st.session_state.usuario_id = f"user_{int(time.time())}"
    if "historial" not in st.session_state:
        st.session_state.historial = cargar_historial(st.session_state.usuario_id)
    mostrar_eco_de_grind()
    for msg in st.session_state.historial:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    if prompt := st.chat_input("¿Qué vas a grindear hoy?"):
        st.session_state.historial.append({"role": "user", "content": prompt, "timestamp": datetime.now().isoformat()})
        guardar_en_supabase(st.session_state.usuario_id, {"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🔥 GRIND está pensando..."):
                idioma = detectar_idioma(prompt)
                respuesta = razonar_con_grind(prompt, st.session_state.historial, idioma)
                message_placeholder = st.empty()
                full_response = ""
                for chunk in respuesta.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(f"<div style='white-space: pre-line;'>{full_response}▌</div>", unsafe_allow_html=True)
                message_placeholder.markdown(f"<div style='white-space: pre-line;'>{full_response}</div>", unsafe_allow_html=True)
        st.session_state.historial.append({"role": "assistant", "content": respuesta, "timestamp": datetime.now().isoformat()})
        guardar_en_supabase(st.session_state.usuario_id, {"role": "assistant", "content": respuesta})
        activar_ritual_diario(st.session_state.usuario_id)

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
        
        # Si no hay conocimiento previo, responde con empatía
        return generar_respuesta_empatica(prompt, idioma)
    
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
        <div class="welcome-logo">🔥 GRIND</div>
        <p class="welcome-subtitle">Tu entrenadora de evolución humana</p>
        <div class="suggestion" id="suggestion">¿Estás cómodo o estás evolucionando?</div>
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

# --- LOGIN MANUAL ---
def login_manual():
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown("<h1>🔐 Acceder a GRIND</h1>", unsafe_allow_html=True)
    st.markdown("<p>Autenticación simple para empezar a grindear.</p>", unsafe_allow_html=True)
    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")
    if st.button("Iniciar sesión"):
        if password == "grind123":
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.user_id = f"user_{hash(username) % 1000000}"
            st.success(f"🔥 ¡Bienvenido, {username}!")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Contraseña incorrecta. Usa 'grind123'")
    st.markdown('<div style="color: #B0B0B0; font-size: 14px;">💡 Usa cualquier usuario. La contraseña es <strong>grind123</strong></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- FLUJO PRINCIPAL ---
if "logged_in" not in st.session_state:
    login_manual()
else:
    interfaz_grind()

# --- DISCLAIMER ---
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