from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
from datetime import datetime
from langchain_core.exceptions import LangChainException
import google.api_core.exceptions as google_exceptions

# Make sure Flask knows where to find templates and static files
app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')

# Charger les variables d'environnement
load_dotenv()

# Vérifier si la clé API Gemini est définie
if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("GEMINI_API_KEY environment variable is not set. Please check your .env file.")

# Configuration de l'API Gemini
genai.configure(api_key=os.environ["GEMINI_API_KEY"], transport="rest")

# Modèle de prompt pour le fitness
FITNESS_PROMPT_TEMPLATE = """
Vous êtes un coach fitness expert nommé FitBot. Votre rôle est de créer des plans d'entraînement personnalisés et de fournir des conseils fitness en utilisant les informations suivantes :

**CONTEXTE** :
{context}

**DEMANDE** :
- Niveau de fitness : {niveau}
- Objectif : {objectif}
- Durée : {duree} semaines
- Préférences : {preferences}
- Type d'entraînement : {type_entrainement}

**Créez une réponse détaillée qui inclut** :

### Aperçu du Plan (1-2 paragraphes)
Résumez le plan, en expliquant comment il répond à l'objectif et au niveau de l'utilisateur.

### Plan d'Entraînement Quotidien
Détaillez un plan jour par jour pour la durée demandée, avec exercices, séries, répétitions, et durées.

### Exercices Recommandés
Listez les exercices inclus avec des instructions claires (nom, instructions, conseils de sécurité).

### Conseils Nutritionnels
Proposez un exemple de plan de repas adapté à l'objectif et aux préférences alimentaires.

### Conseils de Récupération
Incluez des recommandations pour le repos, les étirements, et la récupération.

### Conseils Pratiques
Donnez des astuces pour rester motivé, éviter les blessures, et suivre les progrès.

**Format de réponse** : Markdown avec des sections claires.
"""

# Listes des options
FITNESS_LEVELS = ["Débutant", "Intermédiaire", "Avancé"]
OBJECTIVES = ["Perte de poids", "Prise de muscle", "Endurance", "Mobilité"]
TRAINING_TYPES = ["Cardio", "Musculation", "HIIT", "Yoga", "Mixte"]
DIET_PREFERENCES = ["Omnivore", "Végétarien", "Végan"]

def initialize_components():
    persist_directory = "./chroma_db_fitness"
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Vérifier si la base vectorielle existe
    if os.path.exists(persist_directory):
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
    else:
        pdf_path = "bd_fitness.pdf"
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        vectorstore = Chroma.from_documents(
            documents, 
            embedding_model, 
            persist_directory=persist_directory
        )
    
    prompt = PromptTemplate(
        input_variables=["context", "niveau", "objectif", "duree", "preferences", "type_entrainement"],
        template=FITNESS_PROMPT_TEMPLATE
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7
    )
    
    return vectorstore, prompt, llm

# Initialisation
vectorstore, prompt, llm = initialize_components()

def generate_fitness_plan(niveau, objectif, duree, preferences, type_entrainement):
    try:
        query = f"{niveau} {objectif} {type_entrainement}"
        retrieved_docs = vectorstore.similarity_search(query, k=4)
        context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
        
        chain = prompt | llm
        result = chain.invoke({
            "context": context,
            "niveau": niveau,
            "objectif": objectif,
            "duree": duree,
            "preferences": preferences,
            "type_entrainement": type_entrainement
        })
        
        return result.content
    except google_exceptions.GoogleAPIError as e:
        return f"Erreur API Gemini : {str(e)}"
    except LangChainException as e:
        return f"Erreur LangChain : {str(e)}"
    except Exception as e:
        return f"Erreur inattendue : {str(e)}"

@app.route('/')
def home():
    return render_template('index.html', 
                         fitness_levels=FITNESS_LEVELS, 
                         objectives=OBJECTIVES, 
                         training_types=TRAINING_TYPES, 
                         diet_preferences=DIET_PREFERENCES)

@app.route('/generate_plan', methods=['POST'])
def api_generate_plan():
    data = request.get_json()
    
    if not data or 'niveau' not in data or 'objectif' not in data:
        return jsonify({"error": "Niveau et objectif requis"}), 400
    
    niveau = data['niveau']
    objectif = data['objectif']
    duree = data.get('duree', "4")
    preferences = data.get('preferences', "Omnivore")
    type_entrainement = data.get('type_entrainement', "Mixte")
    
    # Validation des entrées
    if niveau not in FITNESS_LEVELS:
        return jsonify({"error": f"Niveau invalide. Choisissez parmi : {FITNESS_LEVELS}"}), 400
    if objectif not in OBJECTIVES:
        return jsonify({"error": f"Objectif invalide. Choisissez parmi : {OBJECTIVES}"}), 400
    if preferences not in DIET_PREFERENCES:
        return jsonify({"error": f"Préférence alimentaire invalide. Choisissez parmi : {DIET_PREFERENCES}"}), 400
    if type_entrainement not in TRAINING_TYPES:
        return jsonify({"error": f"Type d'entraînement invalide. Choisissez parmi : {TRAINING_TYPES}"}), 400
    if not duree.isdigit() or int(duree) not in [2, 4, 6, 8, 12]:
        return jsonify({"error": "Durée invalide. Choisissez parmi : 2, 4, 6, 8, 12 semaines"}), 400
    
    try:
        plan = generate_fitness_plan(niveau, objectif, duree, preferences, type_entrainement)
        return jsonify({
            "success": True,
            "niveau": niveau,
            "objectif": objectif,
            "plan": plan,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({"success": False, "error": "Message requis"}), 400
    
    message = data['message']
    
    # Vérifier si le message demande un plan d'entraînement
    if "plan" in message.lower() and ("entrainement" in message.lower() or "entraînement" in message.lower()):
        # Rediriger vers le formulaire de plan d'entraînement
        return jsonify({
            "success": True,
            "response": "Pour créer un plan d'entraînement personnalisé, j'ai besoin de quelques informations. Veuillez préciser :\n\n" +
                        "- Votre niveau (débutant, intermédiaire, avancé)\n" +
                        "- Votre objectif (perte de poids, prise de muscle, endurance, mobilité)\n" +
                        "- Durée souhaitée (en semaines)\n" +
                        "- Préférences alimentaires\n" +
                        "- Type d'entraînement préféré"
        })
    
    # Pour les autres types de questions
    try:
        # Utiliser le modèle Gemini pour répondre aux questions générales
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.7
        )
        
        # Prompt pour les questions générales sur le fitness
        prompt = """Tu es FitBot, un coach fitness expert. Réponds à la question suivante de manière concise et utile:
        
        Question: {question}
        
        Donne des conseils pratiques et scientifiquement fondés. Si la question n'est pas liée au fitness, 
        indique poliment que tu es spécialisé dans le domaine du fitness et de la nutrition."""
        
        response = llm.invoke(prompt.format(question=message))
        
        return jsonify({
            "success": True,
            "response": response.content
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.errorhandler(500)
def handle_500_error(e):
    return jsonify({"error": "Internal server error occurred. Please check the logs."}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5001))
    app.run(debug=True, port=port)  # Set debug=True to see detailed error messages
