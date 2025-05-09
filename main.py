from fastapi import FastAPI, HTTPException, APIRouter
from fastapi import Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4

import modules.FluxGeneration
import threading
import time

import enum

app = FastAPI()
router = APIRouter()

# Monta arquivos estáticos da SPA
# app.mount("/static", StaticFiles(directory="static/frontend"), name="static")
# app.mount("/assets", StaticFiles(directory="static/frontend/assets"), name="assets")
app.mount("/output", StaticFiles(directory="output"), name="output")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerationQuality(enum.Enum):    
    # width, height

    sm = (640, 960)
    md = (768, 1152)
    

class GenerateImageRequest(BaseModel):
    prompt: str
    quality: Optional[str] = "sm"
    orientation: Optional[str] = "portrait"
    quantity: Optional[int] = 1

# Armazena tarefas
task_queue = []
task_status = {} 

def process_task():

    while True:
        if task_queue:
            task_id, req = task_queue.pop(0)
            task_status[task_id]["status"] = "processing"
            task_status[task_id]["quantity"] = 5
            task_status[task_id]["results"] = []
            
            for i in range(req.quantity):
                
                match req.orientation:
                    case "portrait":
                        width, height = GenerationQuality[req.quality].value
                    case "landscape":
                        width, height = GenerationQuality[req.quality].value[::-1]
                
                
                img_object = FluxGeneration.generate(
                    positive_prompt=req.prompt,
                    width=width,
                    height=height,
                    seed=0,
                    steps=20,
                    sampler_name="euler",
                    scheduler="simple",
                    guidance=3.5,
                )
                
                # Salva imagem
                filename="output/{task_id}_{i}.png"
                img_object.save(filename)
                task_status[task_id]["results"].append(filename)

            # task_status[task_id]["results"].append("output/{imagem}.png")
            task_status[task_id]["status"] = "done"
            
        time.sleep(0.1)

# Inicia o worker em background
threading.Thread(target=process_task, daemon=True).start()

@router.post("/generate")
def generate_image(req: GenerateImageRequest):
    body = req.dict()

    task_id = str(uuid4())
    task_status[task_id] = {"status": "pending", "quantity": body["quantity"], "tags": [
        "Flux1.dev",
        "Baixa Qualidade" if body["quality"] == "sm" else "Qualidade Média",
        "Retrato" if body["orientation"] == "portrait" else "Horizontal"
    ], "prompt":body["prompt"]}

    task_queue.append((task_id, req.dict()))
    return {"msg": "task created successfully", "task_id": task_id}

@router.get("/task/{task_id}")
def check_task(task_id: str):
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    return task_status[task_id]

@router.get("/tasks")
def list_all_tasks(page: int = Query(1), limit: int = Query(5)):
    # Ordena colocando tarefas "done" primeiro
    sorted_items = sorted(
        task_status.items(),
        key=lambda item: item[1]["status"] == "done"  # False (done) vem antes de True (outros)
    )

    start = (page - 1) * limit
    end = start + limit
    paginated_items = sorted_items[start:end]

    return [
        {"task_id": task_id, **info}
        for task_id, info in paginated_items
    ]

# Rota raiz que retorna index.html
@app.get("/")
def spa():
    return FileResponse("static/frontend/index.html")
    # return {}

app.include_router(router, prefix="/api")
