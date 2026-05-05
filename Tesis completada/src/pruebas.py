# %%
import pytest              
from fastapi.testclient import TestClient  
import time                
import statistics          
import numpy as np         
import pandas as pd        
import psycopg2            
from sqlalchemy import create_engine  
import matplotlib.pyplot as plt   
import seaborn as sns        
import Tools     
import AgenteAI
import app
import concurrent.futures
import psutil

# %% [markdown]
# UNITARIAS:

# %%
def test_chunking():
    texto = "Este es un texto de prueba para verificar el chunking automático con varias palabras."
    chunks = Tools.chunk_text(texto, chunk_size=500)

    
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    
    assert all(len(c.split()) <= 500 for c in chunks)
    
    assert " ".join(chunks).strip() == texto.strip()


def test_embeddings_dimension():
    
    emb = Tools.chunk_embedding("texto de prueba", dim=200)

    
    assert emb is not None
    assert emb.shape == (200,)
    assert np.all(np.isfinite(emb))  

def test_cosine_similarity():
    v1 = np.array([1, 0])
    v2 = np.array([1, 0])
    v3 = np.array([0, 1])
    v4 = np.array([-1, 0])

    
    assert pytest.approx(Tools.cosine_similarity(v1, v2), 0.01) == 1.0   
    assert pytest.approx(Tools.cosine_similarity(v1, v3), 0.01) == 0.0   
    assert pytest.approx(Tools.cosine_similarity(v1, v4), 0.01) == -1.0  
    
    assert -1.0 <= Tools.cosine_similarity(v1, v2) <= 1.0

# %% [markdown]
# INTEGRACION:

# %%
client = TestClient(app.app)

def test_insert_and_retrieve_embeddings():
    texto = "Este es un documento de prueba para integración."
    chunks = Tools.chunk_text(texto, chunk_size=5)
    corpus = [{"chunk_id": i+1, "text": chunk, "metadata": {"source": "test_doc"}} for i, chunk in enumerate(chunks)]
    embeddings = Tools.generate_embeddings(corpus)

    
    for item in embeddings:
        Tools.insert_embedding(
            Tools.get_next_id("chunk", "chunk_id"),
            np.array(item["embedding"], dtype="float32"),
            item["metadata"],
            item["text"],
            almacenamiento="temporal"
        )


    registros = Tools.get_all_embeddings()
    assert any(r["metadata"]["source"] == "test_doc" for r in registros)
    assert all(r["embedding"].shape == (200,) for r in registros if r["metadata"]["source"] == "test_doc")


def test_full_flow_tools_agente_api():

    response_add = client.post("/add_temp", json={"text": "La inteligencia artificial se usa en soporte de negocios"})
    assert response_add.status_code == 200
    assert response_add.json()["status"] == "ok"


    response_ask = client.post("/ask", json={"query": "¿Dónde se aplica la inteligencia artificial?", "k": 1})
    assert response_ask.status_code == 200
    assert "answer" in response_ask.json()

    
    response_history = client.get("/history?limit=1")
    assert response_history.status_code == 200
    history = response_history.json()["history"]
    assert len(history) >= 1
    assert "query" in history[0] and "response" in history[0]

# %% [markdown]
# FLUJO:

# %%
def test_flow():
    
    response_add = client.post("/add_temp", json={"text": "La inteligencia artificial se aplica en soporte de negocios."})
    assert response_add.status_code == 200
    assert response_add.json()["status"] == "ok"

    
    response_ask = client.post("/ask", json={"query": "¿Dónde se aplica la inteligencia artificial?", "k": 1})
    assert response_ask.status_code == 200
    data = response_ask.json()
    assert "answer" in data

    
    response_history = client.get("/history?limit=1")
    assert response_history.status_code == 200
    history = response_history.json()["history"]
    assert len(history) >= 1
    assert "query" in history[0] and "response" in history[0]

def test_endpoints_rest():
    
    r1 = client.post("/add_temp", json={"text": "Documento temporal de prueba"})
    assert r1.status_code == 200

    
    files = {"file": ("test.txt", b"contenido de prueba")}
    r2 = client.post("/add_temp_file", files=files)
    assert r2.status_code == 200


    r3 = client.post("/ask", json={"query": "Pregunta de prueba", "k": 1})
    assert r3.status_code == 200
    assert "answer" in r3.json()

    
    r4 = client.get("/history?limit=2")
    assert r4.status_code == 200
    assert "history" in r4.json()

    
    r5 = client.post("/feedback", json={"query_id": 1, "feedback": "useful"})
    assert r5.status_code == 200

# %% [markdown]
# RENDIMIENTO:

# %%
def test_latency():
    tiempos = []
    for i in range(20):  
        inicio = time.time()
        response = client.post("/ask", json={"query": f"Consulta {i}", "k": 1})
        fin = time.time()
        assert response.status_code == 200
        tiempos.append(fin - inicio)

    print("Latencia promedio:", statistics.mean(tiempos))
    print("p50:", statistics.median(tiempos))
    print("p95:", statistics.quantiles(tiempos, n=100)[94])
    print("p99:", statistics.quantiles(tiempos, n=100)[98])


def test_throughput():
    def consulta(i):
        response = client.post("/ask", json={"query": f"Consulta {i}", "k": 1})
        return response.status_code

    inicio = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        resultados = list(executor.map(consulta, range(50)))  # 50 consultas concurrentes
    fin = time.time()

    duracion = fin - inicio
    throughput = len(resultados) / duracion
    print("Throughput (consultas/segundo):", throughput)
    assert all(r == 200 for r in resultados)


def test_resource_usage():
    proceso = psutil.Process()
    cpu_inicial = proceso.cpu_percent(interval=1)
    mem_inicial = proceso.memory_info().rss / (1024 * 1024)

    
    for i in range(30):
        client.post("/ask", json={"query": f"Consulta {i}", "k": 1})

    cpu_final = proceso.cpu_percent(interval=1)
    mem_final = proceso.memory_info().rss / (1024 * 1024)

    print("CPU inicial:", cpu_inicial, "%")
    print("CPU final:", cpu_final, "%")
    print("Memoria inicial:", mem_inicial, "MB")
    print("Memoria final:", mem_final, "MB")

# %% [markdown]
# SEGURIDAD:

# %%
def test_retencion_datos():
    
    r1 = client.post("/add_temp", json={"text": "Documento temporal de prueba"})
    assert r1.status_code == 200

    
    registros = Tools.get_all_embeddings()
    assert any("Documento temporal de prueba" in r["text"] for r in registros)

    
    client.post("/reload_cleanup")

    
    registros_post = Tools.get_all_embeddings()
    assert not any("Documento temporal de prueba" in r["text"] for r in registros_post)
    print("Retención de datos: documentos temporales eliminados correctamente")





