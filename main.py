from dotenv import load_dotenv
load_dotenv()  # Add this at the very top
# from routes.routes import routers
from routes.allroutes import routers as rag_routes
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="VectorIQ Backend", version="0.1.0")

# CORS setup (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(rag_routes)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Welcome to the VectorIQ Backend!"}
