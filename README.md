# NUII RAG-Chatbot  

## 📌 Deskripsi Umum  
NUII RAG-Chatbot adalah sebuah chatbot berbasis **Retrieval-Augmented Generation (RAG)** yang menggunakan model **Large Language Model (LLM)** `QWEN2.5` (7B parameter).  

Teknologi utama yang digunakan dalam proyek ini:  
- **[Langchain](https://smith.langchain.com/)** – untuk mengelola alur kerja RAG.  
- **[Chroma](https://www.trychroma.com/)** – sebagai vektor database untuk penyimpanan dan pencarian dokumen.  

## 🚀 Cara Menggunakan  
### 1️⃣ Clone Repository  
```bash
git clone https://github.com/AndikaRT421/nuii-rag-chatbot.git
cd nuii-rag-chatbot
```
### 2️⃣ Buat Virtual Environment  
```bash
python -m venv venv
source venv/bin/activate   # Untuk Linux/macOS
venv\Scripts\activate      # Untuk Windows
```
### 3️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```
### 4️⃣ Konfigurasi Lingkungan  
Buat file **`.env`** di root proyek dan tambahkan konfigurasi berikut:  
```ini
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=""
LANGCHAIN_PROJECT=""
UNSTRUCTURED_API_KEY=""
UNSTRUCTURED_API_URL="https://api.unstructured.io/general/v0/general"
```
🔹 **LANGCHAIN_API_KEY** dan **LANGCHAIN_PROJECT** bisa didapatkan dengan mendaftar di [Langchain](https://smith.langchain.com/settings).  
🔹 **UNSTRUCTURED_API_KEY** bisa didapatkan dengan mendaftar di [Unstructured](https://unstructured.io/api-key-free).  
🔹 Untuk mendukung penggunaan lebih luas, pertimbangkan untuk meng-upgrade akun di Langchain dan Unstructured.  

### 5️⃣ Unduh Model LLM  
Unduh **QWEN2.5 (7B)** menggunakan [Ollama](https://ollama.com/library/qwen2.5):  
```bash
ollama pull qwen2.5:7b
```

### 6️⃣ Jalankan Aplikasi  
```bash
python app.py
```

### 7️⃣ Integrasikan API  
Gunakan endpoint dari `app.py` dalam aplikasi chatbot yang diinginkan.

## 📂 Struktur Folder  
| Folder | Deskripsi |
|--------|----------|
| **[db](/db)** | Menyimpan file database **Chroma**. |
| **[images](/images)** | Berisi gambar-gambar yang digunakan dalam chatbot. |
| **[upload_folder](/upload_folder)** | Menyimpan file yang diunggah ke database **Chroma**. |
