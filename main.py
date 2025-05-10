from fastapi import FastAPI, HTTPException, APIRouter, Response
from fastapi.responses import StreamingResponse
from fastapi import Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4

from io import BytesIO
import zipfile
import os

from modules import FluxGeneration
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
            try:
                print(f"Processing task {task_id}")
                task_status[task_id]["status"] = "processing"
                task_status[task_id]["quantity"] = req.get("quantity", 1)
                
                for i in range(req.get("quantity", 1)):
                    
                    match req.get("orientation", "portrait"):
                        case "portrait":
                            width, height = GenerationQuality[req.get("quality", "sm")].value
                        case "landscape":
                            width, height = GenerationQuality[req.get("quality", "sm")].value[::-1]
                    
                    
                    img_object = FluxGeneration.generate(
                        positive_prompt=req.get("prompt"),
                        width=width,
                        height=height,
                        seed=0,
                        steps=20,
                        sampler_name="euler",
                        scheduler="simple",
                        guidance=3.5,
                    )
                    
                    # Salva imagem
                    filename=f"{task_id}_{i}.png"
                    img_object.save("output/"+filename)
                    task_status[task_id]["results"].append(filename)
                
                task_status[task_id]["status"] = "done"
                    
            except Exception as e:
                print(f"Error processing task {task_id}: {str(e)}")
                task_status[task_id]["status"] = "error"
                task_status[task_id]["error"] = str(e)
            
                
        time.sleep(0.1)

# Inicia o worker em background
threading.Thread(target=process_task, daemon=True).start()

@router.post("/generate")
def generate_image(req: GenerateImageRequest):
    body = req.dict()

    task_id = str(uuid4())
    task_status[task_id] = {"status": "pending", "results":[], "quantity": body["quantity"], "tags": [
        "Flux1.dev",
        "Baixa Qualidade" if body["quality"] == "sm" else "Qualidade Média",
        "Retrato" if body["orientation"] == "portrait" else "Horizontal"
    ], "prompt":body["prompt"]}

    task_queue.append((task_id, body))
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
@app.get("/download/{task_id}")
def download_task_result(task_id: str):
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="Task not found")

    task_results = task_status[task_id]["results"]
    
    if len(task_results) == 0:
        raise HTTPException(status_code=404, detail="Results not found")

    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for idx, file_path in enumerate(task_results):
            # Se os arquivos estiverem no disco
            if os.path.isfile(f"output/{file_path}"):
                file_name = os.path.basename(file_path)
                zip_file.write(f"output/{file_path}", arcname=file_name)
            else:
                raise HTTPException(status_code=404, detail=f"File {file_path} not found")

    def iter_file():
        yield zip_buffer.getvalue()
        
    # Preparar resposta com o arquivo ZIP
    headers = {
        "Content-Disposition": f"attachment; filename={task_id}.zip"
    }

    return StreamingResponse(iter_file(), media_type="application/zip", headers=headers)

@app.get("/gallery")
def gallery():
    results = []
    
    for task_id, info in task_status.items():
        if info["status"] == "done":
            results.extend(info["results"])  # Adiciona todas as imagens da tarefa
    
    return {"results": results}

# Rota raiz que retorna index.html
@app.get("/")
def spa():
    return FileResponse("static/frontend/index.html")
    # return {}

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    # Inicia o servidor FastAPI em uma thread separada
    threading.Thread(target=uvicorn.run, args=(app,), kwargs={"host": "localhost", "port": 8000}).start()

    # Conecta o ngrok e imprime a URL
    public_url = ngrok.connect(8000).public_url
    print(f"Servidor acessível em: {public_url}")