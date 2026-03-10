# **✨ Lumina RAG** 📝🤖  
🚀 **An Agentic Multi-Agent RAG system for intelligent document querying with fact verification**  

![Lumina RAG Cover Image](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/zSuj0yrlvjcVkkbW4frkNA/docchat-landing-page.png)

---

## **📌 Overview**  

**Lumina RAG** is a **multi-agent Retrieval-Augmented Generation (RAG) system** designed to help users query **long, complex documents** with **accurate, fact-verified answers**. Unlike traditional chatbots like **ChatGPT or DeepSeek**, which **hallucinate responses and struggle with structured data**, Lumina RAG **retrieves, verifies, and corrects** answers before delivering them.  

💡 **Key Features:**  
✅ **Multi-Agent System** – A **Research Agent** generates answers, while a **Verification Agent** fact-checks responses.  
✅ **Hybrid Retrieval** – Uses **BM25 and vector search** to find the most relevant content.  
✅ **Self-Correction (Query Transformation)** – Iterates and refines search queries if fact-checking fails.  
✅ **Handles Multiple Documents** – Selects the most relevant document even when multiple files are uploaded.  
✅ **Scope Detection** – Prevents hallucinations by **rejecting irrelevant queries**.  
✅ **Fact Verification** – Ensures responses are accurate before presenting them to the user.  
✅ **Web Interface with Gradio** – Allowing seamless document upload and question-answering with real-time UI progress updates.  

---

## **🎥 Demo Video**  

📹 **[Click here to watch the demo](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/zyARt3f3bnm5T-6C4AE3mw/docchat-demo.mp4)**  
*(Opens in a new tab)*

---

## **🛠️ How Lumina RAG Works**  

### **1️⃣ Query Processing & Scope Analysis**  
- Users **upload documents** and **ask a question**.  
- Lumina RAG **analyzes query relevance** and determines if the question is **within scope**.  
- If the query is **irrelevant**, it **rejects it** instead of generating hallucinated responses.  

### **2️⃣ Multi-Agent Research & Retrieval**  
- **LlamaParse (Cloud) & PyPDF (Local Fallback)** parse documents into a structured Markdown format.  
- **LangChain & ChromaDB** handle **hybrid retrieval** (BM25 + vector embeddings).  
- Even when **multiple documents** are uploaded concurrently by different users, **Lumina RAG safely isolates sessions** and finds the most relevant sections dynamically.  

### **3️⃣ Answer Generation & Verification**  
- **Research Agent** generates an answer using retrieved content.  
- **Verification Agent** cross-checks the response against the source document.  
- If **verification fails**, a **self-correction loop (Query Transformer)** rewrites the search query and re-runs retrieval and research.  

### **4️⃣ Response Finalization**  
- **If the answer passes verification**, it is displayed to the user.  
- **If the question is out of scope**, it informs the user instead of hallucinating.  

---

## **🎯 Why Use Lumina RAG Instead of ChatGPT or DeepSeek?**  

| Feature | **ChatGPT/DeepSeek** ❌ | **Lumina RAG** ✅ |
|---------|-----------------|---------|
| Retrieves from uploaded documents | ❌ No | ✅ Yes |
| Handles multiple documents | ❌ No | ✅ Yes |
| Extracts structured data from PDFs | ❌ No | ✅ Yes |
| Prevents hallucinations | ❌ No | ✅ Yes |
| Fact-checks answers | ❌ No | ✅ Yes |
| Detects out-of-scope queries | ❌ No | ✅ Yes |
| Self-corrects bad searches | ❌ No | ✅ Yes |

🚀 **Lumina RAG is built for enterprise-grade document intelligence, research, and compliance workflows.**  

---

## **📦 Installation**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/hungle123-dev/lumina-rag.git lumina-rag
cd lumina-rag
```

### **2️⃣ Set Up Virtual Environment**  
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **4️⃣ Set Up API Keys**  
Lumina RAG requires OpenRouter, Azure (for embeddings), and LlamaCloud API keys. Add them to a `.env` file:
```bash
OPENROUTER_API_KEY=your-openrouter-key
LLAMA_CLOUD_API_KEY=your-llama-parse-key
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=your-azure-endpoint
AZURE_OPENAI_API_VERSION=your-azure-version
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your-embedding-deployment
```

### **5️⃣ Run the Application** 
```bash
python app.py
```

Lumina RAG will be accessible at local address displayed in the terminal (e.g. `http://127.0.0.1:7860`).

---

## 🖥️ Usage Guide  

1️⃣ **Upload one or more documents** (PDF, DOCX, TXT, Markdown).  

2️⃣ **Enter a question** related to the document.  

3️⃣ **Click "Submit Query 🚀"** – Lumina RAG retrieves, analyzes, and verifies the response while showing real-time progress.  

4️⃣ **Review the answer & Verification Report** for confidence.  

5️⃣ **If the question is out of scope or hallucinates**, Lumina will self-correct or inform you directly!  




## 📜 License  

This project is licensed under a Custom Non-Commercial License – check LICENSE for more details.

---

## 💬 Contact & Support  

📧 **Email:** [hunglecrkh2k5@gmail.com]  
