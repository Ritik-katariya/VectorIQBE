from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from app.routes import rag_routes

app = FastAPI()

# CORS setup (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Welcome to the VectorIQ Backend!"}
# app.include_router(rag_routes.router)
