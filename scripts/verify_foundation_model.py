import pandas as pd
import sys
import os
from loguru import logger

# Add root directory to sys.path
sys.path.append(os.getcwd())

from src.strategies.ai_predictor import AIPredictor

def verify_foundation():
    path = "data/foundation_vn_3y.parquet"
    if not os.path.exists(path):
        logger.error(f"Data file not found: {path}. Run build_vn_foundation_cache.py first.")
        return
        
    logger.info("--- STARTING FOUNDATION MODEL VERIFICATION ---")
    logger.info("Loading Big Data Matrix...")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows across {df['Symbol'].nunique()} stocks.")
    
    predictor = AIPredictor()
    
    logger.info("Training Global Foundation Model (Panel Data)...")
    predictor.train(df)
    
    if predictor.train_metrics["r2"] > 0:
        logger.success(f"--- FOUNDATION MODEL CERTIFIED ---")
        logger.info(f"Median R2: {predictor.train_metrics['r2']:.4f}")
        logger.info(f"MAE: {predictor.train_metrics['mae']:.6f}")
        logger.info("Top Cross-Asset Drivers:")
        for feat, imp in list(predictor.feature_importance.items())[:8]:
            logger.info(f" - {feat}: {imp:.2f}")
    else:
        logger.error("Foundation Model training failed or R2 is non-positive.")

if __name__ == "__main__":
    verify_foundation()
