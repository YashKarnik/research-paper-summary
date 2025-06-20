import uuid

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile

from src.config import AppConfig
from src.vector_db.faiss_db import FaissVectorStore
from src.workflow.graph_builder import GraphBuilder

state = {"vector_store": None,
         "graph_builder": None,
         "graph": None}

load_dotenv("./.env")

app = FastAPI()


@app.post("/index-document")
def read_root(file: UploadFile):
    contents = file.file.read()
    filename = f"{uuid.uuid1()}.pdf"
    print(filename)
    try:
        with open(AppConfig.get_file_upload_path(filename), 'wb') as f:
            f.write(contents)
    except IOError as e:
        print("ERR :(")
    finally:
        file.file.close()

    vector_store = FaissVectorStore()
    index_name = vector_store.load(filename)
    return {"index_name": index_name}



@app.post("/invoke")
def read_item(query: str = "", index_name: str = ""):
    # graph = GraphB
    graph_builder = GraphBuilder(FaissVectorStore(), index_name)
    graph = graph_builder.build()
    result = graph.invoke({"messages": query})
    return {"result": result}


@app.get("/similarity-search")
def similarity_search(query: str = "", index_name: str = ""):
    local_store = FaissVectorStore.get_local_vector_db(index_name)
    result = local_store.similarity_search(query)
    return {"result": result}


if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8000)
