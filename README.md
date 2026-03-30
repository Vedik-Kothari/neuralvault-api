# ⚙️ NeuralVault API (Backend)

> Enterprise-grade Retrieval-Augmented Generation (RAG) backend with RBAC, secure document ingestion, and scalable architecture.

---

## 🚀 Overview

NeuralVault API is a production-ready backend that powers secure knowledge retrieval using RAG. It supports document ingestion, role-based access control (RBAC), and intelligent querying over private data.

---

## ✨ Features

* 🔐 **JWT Authentication**
* 🛡️ **Role-Based Access Control (RBAC)**
* 📄 **Document Upload + Storage (Supabase Storage)**
* 🧠 **RAG Pipeline (retrieval + generation)**
* ⚡ **FastAPI high-performance endpoints**
* 📊 **Structured logging (Loguru)**
* 🌐 **CORS-enabled for frontend integration**

---

## 🧱 Tech Stack

* **Framework:** FastAPI
* **Language:** Python
* **Database:** Supabase (PostgreSQL)
* **Storage:** Supabase Storage
* **Auth:** JWT
* **Deployment:** Render
* **Logging:** Loguru

---

## 🧠 Architecture

```text
Client → FastAPI → Auth (JWT)
                    ↓
              RBAC Middleware
                    ↓
        Document Storage (Supabase)
                    ↓
         Retrieval + Filtering
                    ↓
              LLM / RAG Layer
                    ↓
                 Response
```

---

## 📂 Project Structure

```
backend/
├── routers/        # API endpoints
├── middleware/     # RBAC & auth
├── core/           # config, supabase client
├── models/         # schemas
└── main.py         # entry point
```

---

## 📦 Installation

```bash
git clone https://github.com/Vedik-Kothari/neuralvault-api
cd neuralvault-api
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate (Windows)
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

---

## 🔧 Environment Variables

Create a `.env` file:

```env
SUPABASE_URL=your_url
SUPABASE_KEY=your_key
JWT_SECRET=your_secret
APP_ENV=development
```

---

## 🔐 Security Model

* JWT-based authentication
* Role hierarchy:

  * intern < employee < manager < admin
* Row-Level Security (Supabase)
* Backend-level RBAC enforcement

---

## 📡 API Endpoints

### Auth

* `POST /auth/login`
* `POST /auth/register`

### Documents

* `POST /documents/upload`
* `GET /documents`
* `DELETE /documents/{id}`

### Query

* `POST /query`

### System

* `GET /health`

---

## 🚀 Deployment

Deployed on **Render**

Start command:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 10000
```

---

## ⚠️ Important Notes

* File upload limit enforced per user
* RLS policies must be configured in Supabase
* Storage bucket must be named `documents`

---

## 🧩 Future Improvements

* Vector embeddings (pgvector)
* Streaming responses
* Background ingestion workers
* Rate limiting
* Audit logging

---

## 📄 License

MIT License

---

## 👨‍💻 Author

**Vedik Kothari**

---

⭐ Star the repo if you found it useful!
