from dotenv import load_dotenv

# chains.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


load_dotenv()

# --- Prompts ---

# 1) Prompt de REFLEXIÓN (crítica y recomendaciones)
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        # Historial (tweet + versiones + críticas previas)
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 2) Prompt de GENERACIÓN (revisión/creación del tweet)
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        # Historial (críticas + versiones anteriores)
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# --- LLM (elige tu proveedor/modelo) ---
# OpenAI (requiere OPENAI_API_KEY en el entorno)

# llm = ChatOpenAI()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm

# Si prefieres Gemini:
# from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)


# --- Cadenas (LCEL) ---
# Cada cadena: prompt -> LLM
reflection_chain = reflection_prompt | llm
generation_chain = generation_prompt | llm

__all__ = ["reflection_chain", "generation_chain"]
