from fastapi import FastAPI, File, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import JSONLoader
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import UnstructuredExcelLoader
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
folder_path = "db"

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
llm = OllamaLLM(model="qwen2.5:7b")
fast_embedding = OllamaEmbeddings(model='nomic-embed-text')
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=300, length_function=len, is_separator_regex=False
)

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

raw_prompt_summary = ChatPromptTemplate.from_template("""
    Ekstrak kata kunci dari query berikut dan carilah 1 file gambar yang paling sesuai dengan kata kunci tersebut. 
    CUKUP JAWAB KATA KUNCINYA SAJA!!!
    Jika tidak ada, abaikan saja.:
    Query: "{query}"
    list gambar: {list_gambar}
    Nama File gambar yang paling sesuai:
""")

def write_list(folder_path: str, list_path: str):
    with open(list_path, "w") as f:
        for filename in os.listdir(folder_path):
            name, ext = os.path.splitext(filename)
            if ext.lower() in [".jpg", ".jpeg", ".png"]:
                f.write(name + "\n")

def search_image(folder_path: str, query: str, list_path: str):
    print("List Path:", list_path)
    if os.path.exists(list_path):
        with open(list_path, "r") as f:
            list_gambar = f.read().splitlines()
    else:
        list_gambar = []
    
    keyword = raw_prompt_summary.format(query = query, list_gambar = list_gambar)
    keyword = llm.invoke(keyword)
    
    if not keyword:
        print("Tidak dapat menemukan kata kunci dari query.")
        return None
    
    for filename in os.listdir(folder_path):
        name, ext = os.path.splitext(filename)
        keyword_words = keyword.lower().split()
        name_lower = name.lower()
        if all(word in name_lower for word in keyword_words):
            print(f"Gambar untuk '{keyword}' ditemukan.")
            return os.path.join(folder_path, filename)
    
    print(f"Gambar untuk '{keyword}' tidak ditemukan.")
    return None

class AskPDFRequest(BaseModel):
    query: str

@app.post("/tanya")
async def tanya(request: AskPDFRequest):
    print("Request /tanya received")
    
    try:
        query = request.query
        print("Original Query:", query)
        
        # Cari Gambar
        write_list(folder_path="images/", list_path="images/list.txt")
        image_path = search_image(folder_path="images/", query=query, list_path="images/list.txt")
        
        print("Loading vector database...")
        vector_db = Chroma(persist_directory=folder_path, embedding_function=fast_embedding)
        
        print("Creating Chain...")
        retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 20, "lambda_mult": 0.2},
        )
        retrieved_docs = retriever.invoke(query)
        print(f"Retrieved {len(retrieved_docs)} documents.")

        query_doc_pairs = [(query, doc.page_content) for doc in retrieved_docs]
        scores = cross_encoder.predict(query_doc_pairs)
        ranked_docs = sorted(
            zip(retrieved_docs, scores),
            key=lambda x: x[1],  
            reverse=True        
        )
        top_k_docs = [doc for doc, score in ranked_docs[:3]]
        context = "\n\n".join(doc.page_content for doc in top_k_docs)
        formatted_prompt = raw_prompt.format(input=query, context=context)
        result = llm.invoke(formatted_prompt)
        print("Result:", result)
        
        return {
            "message": "Query processed successfully",
            "image": image_path, 
            "answer": result,
            "context": context
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'error': str(e)}
        )

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    print("Request /upload received")
    
    if not file:
        return JSONResponse(content={'error': 'No file part'}, status_code=400)
    
    try:            
        file_name = file.filename
        if not os.path.exists("upload_folder"):
            os.makedirs("upload_folder")
        save_path = f"upload_folder/{file_name}"
        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)
        print(f"File saved at {save_path}")
        
        if file_name.endswith(".pdf"):
            loader = PDFPlumberLoader(save_path)
        elif file_name.endswith(".json"):
            loader = JSONLoader(file_path=save_path, jq_schema='.Konten[]', text_content=False)
        elif file_name.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(save_path, mode="elements")
        docs = loader.load_and_split()
        print(f"Number of documents: {len(docs)}")
        
        chunks = text_splitter.split_documents(docs)
        print(f"Number of chunks: {len(chunks)}")
        
        Chroma.from_documents(documents=chunks, embedding=fast_embedding, persist_directory=folder_path)
            
        return {
            "message": "File uploaded successfully", 
            "filename": file_name, 
            "documents": len(docs), 
            "chunks": len(chunks)
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'error': str(e)}
        )

def start():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=11436)

if __name__ == "__main__":
    start()