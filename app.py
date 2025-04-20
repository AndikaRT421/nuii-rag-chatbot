from fastapi import FastAPI, File, UploadFile, status, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader, JSONLoader, UnstructuredExcelLoader
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Environment
QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
RUNPOD_URL = os.getenv("RUNPOD_SERVER_URL")

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Direktori untuk simpan upload
upload_dir = Path(__file__).parent / "upload_folder"
upload_dir.mkdir(exist_ok=True)

# Mount static untuk serve file
app.mount(
    "/uploads",
    StaticFiles(directory=upload_dir),
    name="uploads"
)

# Config koleksi & folders
folder_path = "db"
images_folder = "images/"
collections = [
    "jaringan_collection",
    "niaga_collection",
    "sdm_collection",
    "skki_skko_collection",
]

# Inisialisasi model & tools
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
fast_embedding = OllamaEmbeddings(model="nomic-embed-text")
llm = OllamaLLM(model="qwen2.5:7b")

model_st = SentenceTransformer("all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=300, length_function=len, is_separator_regex=False
)

# Qdrant client & koleksi
client = QdrantClient(QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)
def init_collections():
    embedding_dim = 768
    for col in collections:
        try:
            client.get_collection(collection_name=col)
        except:
            client.create_collection(
                collection_name=col,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
            )
init_collections()

vector_stores = {
    col: QdrantVectorStore(client=client, collection_name=col, embedding=fast_embedding)
    for col in collections
}

# Prompt template
raw_prompt = ChatPromptTemplate.from_template("""
Anda adalah asisten virtual bernama NUII yang memberikan jawaban langsung dan jelas terkait konstruksi jaringan PLN.
Gunakan informasi dalam Konteks sebagai acuan.
Jawablah dengan Bahasa Indonesia yang baku dan langsung ke inti jawaban tanpa frasa pembuka seperti 'Berdasarkan konteks yang diberikan'.
Jika user bertanya tentang hal umum, silakan jawab sesuai kemampuan Anda serta sertakan sumber yang Anda ketahui.
Jika user juga bertanya tentang hal yang diluar konteks konstruksi jaringan PLN dan PLN, silakan jawab dengan bijak dan tidak perlu memberikan jawaban terkait konstruksi jaringan PLN.

Pertanyaan: {input}
Konteks: {context}
Jawaban:
""")

# Helper image search
def search_image(folder_path: str, query: str, k: int = 1):
    files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    embeddings = model_st.encode(files, convert_to_tensor=True)
    q_emb = model_st.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, embeddings)[0]
    topk = scores.topk(k)
    threshold = 0.5
    return [
        files[idx.item()]
        for idx, sc in zip(topk.indices, topk.values)
        if sc.item() > threshold
    ]

# Request & upload schemas
class AskPDFRequest(BaseModel):
    query: str
    k_image: int = 1

class UploadRequest(BaseModel):
    collection_name: str = "jaringan_collection"

# Endpoint /tanya
@app.post("/tanya")
async def tanya(request: AskPDFRequest):
    try:
        query = request.query
        k_image = request.k_image

        # 1. Cari gambar
        image_path = search_image(images_folder, query, k_image)

        # 2. Retrieve dokumen dari semua koleksi
        all_docs = []
        for col, store in vector_stores.items():
            retr = store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "lambda_mult": 0.2}
            )
            docs = retr.invoke(query)
            for d in docs:
                d.metadata["collection"] = col
            all_docs.extend(docs)

        # 3. Rank dengan cross-encoder
        pairs = [(query, d.page_content) for d in all_docs]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(all_docs, scores), key=lambda x: x[1], reverse=True)
        top_k_docs = [d for d, _ in ranked[:3]]

        # 4. Format context & prompt
        context = "\n\n".join(d.page_content for d in top_k_docs)
        prompt = raw_prompt.format(input=query, context=context)
        result = llm.invoke(prompt)

        # 5. Siapkan sources + files
        out_sources = []
        file_names = []
        for doc in top_k_docs:
            # metadata["source"] harus berisi path file asal
            src_path = doc.metadata.get("source", "")
            fname = Path(src_path).name or "unknown.txt"
            file_names.append(fname)

            out_sources.append({
                "content": doc.page_content,
                "collection": doc.metadata.get("collection", "unknown"),
                "fileName": fname,
                "fileUrl": f"/uploads/{fname}"
            })

        # dedupe file URLs
        unique_files = list(dict.fromkeys(file_names))
        files_urls = [f"/uploads/{fn}" for fn in unique_files]

        return {
            "message": "Query processed successfully",
            "image": image_path,
            "answer": result,
            "context": context,
            "sources": out_sources,
            "files": files_urls
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )

# Endpoint /upload
@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    collection_name: str = Form("jaringan_collection")
):
    try:
        if collection_name not in collections:
            return JSONResponse(
                content={"error": "Invalid collection name"},
                status_code=400
            )

        file_name = file.filename
        save_path = upload_dir / file_name
        contents = await file.read()
        save_path.write_bytes(contents)

        # loader sesuai extension
        if file_name.endswith(".pdf"):
            loader = PDFPlumberLoader(str(save_path))
        elif file_name.endswith(".json"):
            loader = JSONLoader(
                file_path=str(save_path),
                jq_schema=".Konten[]",
                text_content=False
            )
        else:  # .xlsx
            loader = UnstructuredExcelLoader(str(save_path), mode="elements")

        docs = loader.load_and_split()
        chunks = text_splitter.split_documents(docs)
        vector_stores[collection_name].add_documents(documents=chunks)

        return {
            "message": "File uploaded successfully",
            "filename": file_name,
            "collection": collection_name,
            "documents": len(docs),
            "chunks": len(chunks)
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )

# Run server
def start():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11436)

if __name__ == "__main__":
    start()
