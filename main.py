from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4
from enum import Enum
import time
import threading

app = FastAPI()

# Enum para status das tarefas
class TarefaStatus(str, Enum):
    pendente = "pendente"
    processando = "processando"
    concluida = "concluida"
    erro = "erro"

# Modelo da requisição
class GerarImagemRequest(BaseModel):
    prompt: str
    quality: Optional[str] = "standard"
    orientation: Optional[str] = "portrait"
    quantity: Optional[int] = 1

# Modelo da tarefa
class Tarefa(BaseModel):
    id: str
    status: TarefaStatus
    prompt: str
    quality: str
    orientation: str
    quantity: int
    resultado: Optional[List[str]] = None

fila: List[Tarefa] = []

class PromptModel(BaseModel):
    prompt: str
    quality: str
    quantity: int
    results

def processar_tarefa(tarefa: Tarefa):
    tarefa.status = TarefaStatus.processando
    time.sleep(2)  # Simula tempo de geração da imagem
    try:
        # Aqui você integraria com um gerador de imagem real
        tarefa.resultado = [f"url_falsa_imagem_{i+1}.png" for i in range(tarefa.quantity)]
        tarefa.status = TarefaStatus.concluida
    except Exception as e:
        tarefa.status = TarefaStatus.erro

@app.post("/gerar-imagem", response_model=Tarefa)
def criar_tarefa(req: GerarImagemRequest):
    tarefa = Tarefa(
        id=str(uuid4()),
        status=TarefaStatus.pendente,
        prompt=req.prompt,
        quality=req.quality,
        orientation=req.orientation,
        quantity=req.quantity,
    )
    # fila.append(tarefa)
    # threading.Thread(target=processar_tarefa, args=(tarefa,), daemon=True).start()
    return {}

@app.get("/fila", response_model=List[Tarefa])
def listar_fila():
    return [t for t in fila if t.status in [TarefaStatus.pendente, TarefaStatus.processando]]

@app.get("/tarefa/{tarefa_id}", response_model=Tarefa)
def obter_tarefa(tarefa_id: str):
    for t in fila:
        if t.id == tarefa_id:
            return t
    raise HTTPException(status_code=404, detail="Tarefa não encontrada")
