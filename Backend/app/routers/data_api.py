# app/routers/data_api.py
from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
import pandas as pd
from pathlib import Path
from ..deps import BASE_DIR

router = APIRouter(prefix="/data", tags=["data"])

DATA_PATH = BASE_DIR / "DataAndCleaning" / "Data" / "CleanedData" / "Crop_production_cleaned.csv"

def load_data():
    """Load the cleaned crop production dataset."""
    if not DATA_PATH.exists():
        raise HTTPException(500, f"Dataset not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

@router.get("/preview")
def get_data_preview(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=1000, description="Items per page"),
    state: Optional[str] = Query(None, description="Filter by State_Name"),
    season: Optional[str] = Query(None, description="Filter by Season"),
    crop: Optional[str] = Query(None, description="Filter by Crop"),
    sort_by: Optional[str] = Query(None, description="Column to sort by"),
    sort_order: Optional[str] = Query("asc", description="Sort order: asc or desc")
):
    """
    Get paginated data with optional filtering and sorting.
    """
    df = load_data()
    
    # Apply filters
    if state:
        df = df[df['State_Name'].str.lower() == state.lower()]
    if season:
        df = df[df['Season'].str.lower() == season.lower()]
    if crop:
        df = df[df['Crop'].str.lower() == crop.lower()]
    
    # Apply sorting
    if sort_by and sort_by in df.columns:
        ascending = (sort_order.lower() == "asc")
        df = df.sort_values(by=sort_by, ascending=ascending)
    
    # Calculate pagination
    total_records = len(df)
    total_pages = (total_records + page_size - 1) // page_size
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_records)
    
    # Get page data
    page_data = df.iloc[start_idx:end_idx]
    
    return {
        "data": page_data.to_dict(orient="records"),
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_records": total_records,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    }

@router.get("/statistics")
def get_data_statistics():
    """Get summary statistics about the dataset."""
    df = load_data()
    
    numeric_cols = ["N", "P", "K", "pH", "rainfall", "temperature", "Yield_ton_per_hec"]
    
    stats = {
        "total_records": len(df),
        "numeric_stats": df[numeric_cols].describe().to_dict(),
        "unique_states": df['State_Name'].nunique(),
        "unique_seasons": df['Season'].nunique(),
        "unique_crops": df['Crop'].nunique(),
        "states": sorted(df['State_Name'].unique().tolist()),
        "seasons": sorted(df['Season'].unique().tolist()),
        "crops": sorted(df['Crop'].unique().tolist())
    }
    
    return stats

@router.get("/filters")
def get_filter_options():
    """Get available filter options for dropdowns."""
    df = load_data()
    
    return {
        "states": sorted(df['State_Name'].unique().tolist()),
        "seasons": sorted(df['Season'].unique().tolist()),
        "crops": sorted(df['Crop'].unique().tolist()),
        "columns": df.columns.tolist()
    }