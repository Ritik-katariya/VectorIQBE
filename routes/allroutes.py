from fastapi import APIRouter
from modules.data_loader.data_loader_service import router as data_loader_router

routers = APIRouter()

# include the data_loader router under a clear prefix
routers.include_router(data_loader_router)


