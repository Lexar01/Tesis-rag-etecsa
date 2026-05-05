from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import Tools
import psycopg2
import json
from datetime import datetime

model = OllamaLLM(model="llama3.2") 

template = """
You are a knowledgeable professor. 
You must ONLY answer using the information provided in {reviews}. 
Do not add external knowledge, examples, or assumptions. 
If the answer is not in the documents, say clearly: "No information available in the PDFs."
Always include a short citation like: (from source: filename) using metadata source if filename, but do not use the citation if filename is "usuario".
Highlight only the most essential points and explain them clearly.
Do not suggest information that is not provided in {reviews}.
Always provide a reasoned explanation, not just a definition. 
Support your answer with arguments, context, and examples from the documents.

Here are some relevant reviews: {reviews}

Here is the question to answer: {questions}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def audit_response(query, reviews, response, score):
     
    try:
        conn = psycopg2.connect(
            dbname="Tesis",
            user="postgres",
            password="1234",
            host="localhost",
            port="5432"
        )
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_log (id_aud, query, reviews, response, score)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            Tools.get_next_id("audit_log","id_aud"),
            query,
            json.dumps(reviews, ensure_ascii=False),
            response,
            score
        ))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        Tools.log_event("AUDIT_ERROR", str(e))



def get_history(limit=5):
     
    try:
        conn = psycopg2.connect(
            dbname="Tesis",
            user="postgres",
            password="1234",
            host="localhost",
            port="5432"
        )
        cursor = conn.cursor()
        cursor.execute("""
            SELECT query, response
            FROM audit_log
            ORDER BY id_aud DESC
            LIMIT %s
        """, (limit,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [{"query": q, "response": r} for q, r in rows]
    except Exception as e:
        Tools.log_event("HISTORY_ERROR", str(e))
        return []



def save_feedback(query_id, feedback):
    
    try:
        conn = psycopg2.connect(
            dbname="Tesis",
            user="postgres",
            password="1234",
            host="localhost",
            port="5432"
        )
        cursor = conn.cursor()

        cursor.execute("SELECT MAX(id_aud) FROM audit_log")
        new_id=cursor.fetchone()

        cursor.execute("""
            UPDATE audit_log
            SET feedback = %s
            WHERE id_aud = %s
        """, (feedback, new_id))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        Tools.log_event("FEEDBACK_ERROR", str(e))

 

def clear_temporals():
    try:
        conn = psycopg2.connect(
            dbname="Tesis",
            user="postgres",
            password="1234",
            host="localhost",
            port="5432"
        )
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chunk WHERE almacenamiento = 'temporal'")
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        Tools.log_event("CLEAR_TEMPORALS_ERROR", str(e))


 
def responder(query, k=5, threshold=0.2):
    try:
        reviews = Tools.retrieve(query, k)
    except Exception as e:
        Tools.log_event("RETRIEVE_ERROR", f"Error en retrieve: {e}")
        return "Error al recuperar información de la base de datos."

    if not reviews:
        return "No se encontraron fragmentos relevantes en la base de datos."
 
    filtered = [(score, registro) for score, registro in reviews if score >= threshold]
    if not filtered:
        return "No hay información relevante en los documentos."

    reviews_texts = [
        f"(from {registro['metadata']['source']}) {registro['text']}"
        for score, registro in filtered
    ]

    try:
        result = chain.invoke({
            "reviews": "\n".join(reviews_texts),
            "questions": query
        })
    except Exception as e:
        Tools.log_event("MODEL_ERROR", f"Error al invocar el modelo: {e}")
        return "Error al generar la respuesta con el modelo."

    top_score = filtered[0][0]

    try:
        audit_response(query, reviews_texts, result, top_score)
    except Exception as e:
        Tools.log_event("AUDIT_ERROR", f"Error al registrar auditoría: {e}")
       

    return result


 
def main():
    print("=== Sistema de Gestión de Preguntas con RAG ===")

    k = 1
    Tools.database()
    print("Base de datos poblada correctamente.\n")

    print("Escriba 'salir' para terminar.\n")
    while True:
        query = input("¿Qué necesitas? ").strip()
        if query.lower() == "salir":
            print("Saliendo del sistema...")
            break

        if query.startswith("k=") and len(query) == 3:
            k = int(query[2])
            print("Semejantes aumentados\n")
            continue

        try:
            result = responder(query, k)
            print("\n=== Respuesta ===")
            print(result)
            print("=================\n")
        except Exception as e:
            print("Error durante la consulta:", e)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("Error no esperado al ejecutar main():", e)
        traceback.print_exc()
