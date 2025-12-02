

# 📰 RockyBot: AI-Powered Research & News Analyzer

RockyBot is a lightweight AI research assistant that extracts content from online news articles, chunks them, generates embeddings, stores them in a local FAISS database, and allows you to query the processed content using a local LLM such as **Llama 3.2 (Ollama)**.

No cloud dependency — everything runs locally. 💻🔒

---

## 🚀 Features

* 🔗 Load and scrape multiple article URLs
* ✂️ Smart text chunking
* 🧠 Semantic embeddings using HuggingFace models
* 📁 Local vector storage using FAISS
* 🤖 Query content through local LLM (Ollama)
* 🎨 Simple UI built with Streamlit
* 🔒 `.env` support for secure credentials

---

## 🛠 Tech Stack

| Component  | Technology            |
| ---------- | --------------------- |
| UI         | Streamlit             |
| Embeddings | Sentence Transformers |
| Vector DB  | FAISS                 |
| LLM        | Ollama (Llama3.2)     |
| Framework  | LangChain             |

---

## 📦 Installation

### 1️⃣ Clone the repo

```bash
git clone https://github.com/NiteshVyas1/Research-Tool.git
cd Research-Tool
```

### 2️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧰 Setup Ollama

Download Ollama from:
[https://ollama.com/download](https://ollama.com/download)

Then pull a supported model:

```bash
ollama pull llama3.2
```

---

## 🔑 Environment Variables

Create a `.env` file in project root:

```
# Optional (only if using restricted HuggingFace models)
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

> `.env` is already ignored and **will not be pushed to GitHub**.

---

## ▶️ Running the App

```bash
streamlit run app.py
```

---

## 🧪 Usage

1. Paste news article URLs in the sidebar
2. Click **“Process URLs”**
3. Ask a question
4. RockyBot answers based on extracted content

---

## 📂 Project Structure

```
📁 Research-Tool
│-- app.py
│-- requirements.txt
│-- README.md
│-- .gitignore
│-- .env  (ignored)
```

---



## 🤝 Contributing

PRs are welcome.
For major changes, please open an issue first.

---

## ⭐ Support

If this project helps you, please give it a ⭐ on GitHub.

---



