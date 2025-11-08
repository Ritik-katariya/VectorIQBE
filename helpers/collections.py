from fastapi import HTTPException
from lib.chroma_connection import get_chroma_client, get_permanent_collection

def get_existing_collection(base_collection: str, namespace: str | None):
    """
    Get an existing collection from Chroma Cloud (same client as ingestion).
    Uses the same naming convention as get_permanent_collection: base_collection_namespace
    """
    client = get_chroma_client()
    
    # Use the same naming convention as ingestion: base_collection_namespace
    if namespace:
        coll_name = f"{base_collection}_{namespace}"
    else:
        coll_name = base_collection
    
    # First, try to get the collection directly (doesn't auto-create)
    try:
        collection = client.get_collection(name=coll_name)
        return collection
    except Exception as get_error:
        # If get_collection fails, list all collections to help debug
        try:
            existing_collections = client.list_collections()
            collection_names = [col.name for col in existing_collections]
            
            # Check for exact match (case-sensitive)
            if coll_name in collection_names:
                # Collection exists but get_collection failed - try using get_permanent_collection
                # which uses get_or_create_collection (will return existing)
                return get_permanent_collection(base_collection, namespace)
            
            # Check for similar collection names
            matching_collections = [name for name in collection_names if coll_name.lower() in name.lower() or name.lower() in coll_name.lower()]
            
            error_detail = f"Collection not found: '{coll_name}'. "
            if matching_collections:
                error_detail += f"Similar collections found: {matching_collections}. "
            if collection_names:
                error_detail += f"Available collections (showing first 20): {collection_names[:20]}. "
            else:
                error_detail += "No collections found in database. "
            error_detail += f"Original error: {str(get_error)}"
            
            raise HTTPException(status_code=404, detail=error_detail)
        except Exception as list_error:
            # If listing also fails, return the original error
            raise HTTPException(
                status_code=404,
                detail=f"Collection not found: '{coll_name}'. Error: {str(get_error)}. Failed to list collections: {str(list_error)}"
            )
