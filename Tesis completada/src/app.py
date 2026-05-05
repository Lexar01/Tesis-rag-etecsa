from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import io
from pypdf import PdfReader

from AgenteAI import responder, save_feedback, get_history, clear_temporals
import Tools

class QueryRequest(BaseModel):
    query: str
    k: int = 5

app = FastAPI(title="Asistente IA")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    clear_temporals()
    Tools.database()
    print("Base de datos poblada correctamente al iniciar FastAPI.")

@app.post("/reload_cleanup")
def reload_cleanup():
    clear_temporals()
    return {"status": "ok", "message": "Documentos temporales eliminados al recargar la página"}

@app.post("/ask")
def ask(request: QueryRequest):
    answer = responder(request.query, request.k)
    return {"answer": answer}

class FeedbackRequest(BaseModel):
    query_id: int
    feedback: str

@app.get("/history")
def history(limit: int = 5):
    return {"history": get_history(limit)}

@app.post("/feedback")
def feedback(request: FeedbackRequest):
    save_feedback(request.query_id, request.feedback)
    return {"status": "ok", "message": "Feedback registrado"}

class TempDocRequest(BaseModel):
    text: str

@app.post("/add_temp")
def add_temp_doc(request: TempDocRequest):
    
    chunks = Tools.chunk_text(request.text, chunk_size=500)
    corpus = [{"chunk_id": i+1, "text": chunk, "metadata": {"source": "usuario"}} for i, chunk in enumerate(chunks)]
    embeddings = Tools.generate_embeddings(corpus)

    for item in embeddings:
        Tools.insert_embedding(
            Tools.get_next_id("chunk", "chunk_id"),
            np.array(item["embedding"], dtype="float32"),
            item["metadata"],
            item["text"],
            almacenamiento="temporal"
        )
    return {"status": "ok", "message": "Documento temporal insertado"}

@app.post("/add_temp_file")
async def add_temp_file(file: UploadFile = File(...)):
    try:
        filename = (file.filename or "").lower()
        content = await file.read()

        if filename.endswith(".txt"):
            try:
                text = content.decode("utf-8", errors="ignore")
            except Exception:
                text = content.decode("latin-1", errors="ignore")
        elif filename.endswith(".pdf"):
            pdf_stream = io.BytesIO(content)
            reader = PdfReader(pdf_stream)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        elif filename.endswith(".docx"):
            
            text = Tools.extract_text_from_docx(content)
        else:
            raise HTTPException(status_code=400, detail="Formato no soportado")

        if not text or not text.strip():
            return {"status": "error", "message": "El archivo no contiene texto extraíble."}

       
        chunks = Tools.chunk_text(text, chunk_size=500)
        corpus = [{"chunk_id": i+1, "text": chunk, "metadata": {"source": file.filename}} for i, chunk in enumerate(chunks)]
        embeddings = Tools.generate_embeddings(corpus)

        for item in embeddings:
            Tools.insert_embedding(
                Tools.get_next_id("chunk", "chunk_id"),
                np.array(item["embedding"], dtype="float32"),
                item["metadata"],
                item["text"],
                almacenamiento="temporal"
            )

        return {"status": "ok", "message": f"Archivo {file.filename} insertado como temporal"}
    except Exception as e:
        Tools.log_event("ADD_TEMP_FILE_ERROR", str(e))

#abrir 2 consolas:

#consola 1 escribir
#cd C:\Osmay\Escuela\UCI\Tesis\IA para asistencia en soporte de negocios\Codigos Separados\Tesis completada\src
#uvicorn app:app --reload



#consola 2 escribir
#cd C:\Osmay\Escuela\UCI\Tesis\IA para asistencia en soporte de negocios\Codigos Separados\Tesis completada\src
#python -m http.server 5500


#abrir en navegador http://127.0.0.1:5500/index.html