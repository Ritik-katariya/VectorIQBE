# VectorIQBE

VectorIQ Backend - A FastAPI application for document processing and vector storage.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the root directory with the following variables:

```
CHROMA_API_KEY=your_chroma_api_key_here
CHROMA_TENANT=your_chroma_tenant_here
CHROMA_DATABASE=your_chroma_database_here
OPENAI_API_KEY=your_openai_api_key_here
```

3. Run the server:

```bash
uvicorn main:app --reload
```

The server will be available at http://127.0.0.1:8000

## Fixed Issues

- Fixed import error: Changed `langchain_unstructured` to `langchain_community` imports
- Updated `UnstructuredLoader` to `UnstructuredFileLoader` for proper compatibility
- Fixed SessionStore imports in data_loader_service
