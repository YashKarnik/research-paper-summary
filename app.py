import uuid

from fastapi import FastAPI, UploadFile

from src.vector_db.faiss_db import FaissVectorStore
from src.workflow.graph_builder import GraphBuilder

state = {"vector_store": None,
         "graph_builder": None,
         "graph": None}

app = FastAPI()


@app.post("/")
def read_root(file: UploadFile):
    contents = file.file.read()
    filename = uuid.uuid1()
    try:
        with open(f'./uploads/{filename}.pdf', 'wb') as f:
            f.write(contents)
    except IOError as e:
        print("ERR :(")
    finally:
        file.file.close()

    state['vector_store'] = FaissVectorStore(f'./uploads/{filename}.pdf')
    state['graph_builder'] = GraphBuilder(state['vector_store'])
    state['graph'] = state['graph_builder'].build()
    return {"Hello": "World"}


@app.post("/invoke")
def read_item(query: str = ""):
    graph = state['graph']
    result = graph.invoke({"messages": query})
    return {"result": result}

# if __name__ == "__main__":
#     uvicorn.run("app:app", host="0.0.0.0", port=8000)
