####V_1
# # import os
# # import re
# # import pickle
# # import joblib
# # import numpy as np
# # import pandas as pd
# # from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# # from enum import Enum
# # import tensorflow as tf
# # from sklearn.metrics.pairwise import cosine_similarity

# # app = FastAPI(title="Multi-Dataset Intrusion Detection API")

# # class DatasetEnum(str, Enum):
# #     nsl_kdd = "nsl_kdd"
# #     cicids2017 = "cicids2017"
# #     unsw_nb15 = "unsw_nb15"

# # class ModelEnum(str, Enum):
# #     svm = "svm"
# #     knn = "knn"
# #     random_forest = "random_forest"
# #     naive_bayes = "naive_bayes"
# #     decision_tree = "decision_tree"
# #     fcnn = "fcnn"
# #     cnn = "cnn"
# #     lstm = "lstm"

# # def sanitize(name):
# #     return re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower()

# # class EventEncoder:
# #     def __init__(self, n_bins=10):
# #         self.n_bins = n_bins
# #         self.num_cols, self.cat_cols, self.edges = [], [], {}
    
# #     def transform(self, df):
# #         cols = []
# #         for c in self.num_cols:
# #             e = self.edges[c]
# #             idx = np.searchsorted(e, df[c].to_numpy(dtype=np.float64), side="left")
# #             lut = np.array([f"{sanitize(c)}_b{i}" for i in range(len(e) + 1)], dtype=object)
# #             cols.append(lut[idx])
# #         for c in self.cat_cols:
# #             s = df[c].astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
# #             cols.append((sanitize(c) + "_" + s).to_numpy(dtype=object))
# #         mat = np.column_stack(cols)
# #         return [" ".join(row) for row in mat]
    
# #     def load_state(self, state_dict):
# #         self.n_bins = state_dict["n_bins"]
# #         self.num_cols = state_dict["num_cols"]
# #         self.cat_cols = state_dict["cat_cols"]
# #         self.edges = state_dict["edges"]
# #         return self

# # def clean_dataframe(df):
# #     d = df.replace([np.inf, -np.inf], np.nan)
# #     for c in d.select_dtypes(include=[np.number]).columns:
# #         if d[c].isna().any():
# #             d[c] = d[c].fillna(d[c].median()) 
# #     for c in [c for c in d.columns if not pd.api.types.is_numeric_dtype(d[c])]:
# #         if d[c].isna().any():
# #             d[c] = d[c].fillna(d[c].mode().iloc[0])
# #     return d.dropna().reset_index(drop=True)

# # def to_sequence(X, steps=32):
# #     n, f = X.shape
# #     pad = (-f) % steps
# #     if pad:
# #         X = np.hstack([X, np.zeros((n, pad), dtype=X.dtype)])
# #     return X.reshape(n, steps, -1)

# # @app.post("/predict")
# # async def predict(
# #     dataset: DatasetEnum = Form(...),
# #     model_type: ModelEnum = Form(...),
# #     file: UploadFile = File(...)
# # ):
# #     try:
# #         df = pd.read_csv(file.file)
# #         df = clean_dataframe(df)
# #         if "label" in df.columns:
# #             df = df.drop(columns=["label"])
            
# #         prefix = dataset.value
# #         base_dir = "saved_models"
        
# #         # Load preprocessing artifacts
# #         with open(os.path.join(base_dir, f"{prefix}_event_encoder.pkl"), "rb") as f:
# #             enc_dict = pickle.load(f)
# #             enc = EventEncoder().load_state(enc_dict)
            
# #         with open(os.path.join(base_dir, f"{prefix}_vectorizer.pkl"), "rb") as f:
# #             vec = pickle.load(f)
            
# #         with open(os.path.join(base_dir, f"{prefix}_basepoint.pkl"), "rb") as f:
# #             basepoint = pickle.load(f)
            
# #         with open(os.path.join(base_dir, f"{prefix}_scaler.pkl"), "rb") as f:
# #             scaler = pickle.load(f)
            
# #         with open(os.path.join(base_dir, f"{prefix}_meta.pkl"), "rb") as f:
# #             meta = pickle.load(f)

# #         # Apply profile transformations
# #         ev_data = enc.transform(df)
# #         X_vec = vec.transform(ev_data)
# #         sim = cosine_similarity(X_vec, basepoint).astype(np.float32)
# #         X_p = np.hstack([X_vec.toarray(), sim])
# #         X_s = scaler.transform(X_p).astype(np.float32)
# #         np.clip(X_s, -meta["clip"], meta["clip"], out=X_s)

# #         # Load model and predict
# #         is_dl = model_type.value in ["fcnn", "cnn", "lstm"]
# #         if is_dl:
# #             model = tf.keras.models.load_model(os.path.join(base_dir, f"{prefix}_dl_{model_type.value}.keras"))
# #             if model_type.value == "cnn":
# #                 X_s = X_s[..., np.newaxis]
# #             elif model_type.value == "lstm":
# #                 X_s = to_sequence(X_s, steps=meta["lstm_steps"])
# #             probs = model.predict(X_s).ravel()
# #             predictions = (probs >= 0.5).astype(int).tolist()
# #         else:
# #             model = joblib.load(os.path.join(base_dir, f"{prefix}_ml_{model_type.value}.pkl"))
# #             predictions = model.predict(X_s).tolist()

# #         label_map = {0: "normal", 1: "attack"}
# #         results = [label_map[p] for p in predictions]
        
# #         return {"predictions": results}
        
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))

# import os
# import re
# import pickle
# import joblib
# import numpy as np
# import pandas as pd
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from enum import Enum
# import tensorflow as tf
# from sklearn.metrics.pairwise import cosine_similarity

# app = FastAPI(title="Multi-Dataset Intrusion Detection API")

# class DatasetEnum(str, Enum):
#     nsl_kdd = "nsl_kdd"
#     cicids2017 = "cicids2017"
#     unsw_nb15 = "unsw_nb15"

# class ModelEnum(str, Enum):
#     svm = "svm"
#     knn = "knn"
#     random_forest = "random_forest"
#     naive_bayes = "naive_bayes"
#     decision_tree = "decision_tree"
#     fcnn = "fcnn"
#     cnn = "cnn"
#     lstm = "lstm"

# def sanitize(name):
#     return re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower()

# class EventEncoder:
#     def __init__(self, n_bins=10):
#         self.n_bins = n_bins
#         self.num_cols, self.cat_cols, self.edges = [], [], {}
    
#     def transform(self, df):
#         cols = []
#         for c in self.num_cols:
#             e = self.edges[c]
#             idx = np.searchsorted(e, df[c].to_numpy(dtype=np.float64), side="left")
#             lut = np.array([f"{sanitize(c)}_b{i}" for i in range(len(e) + 1)], dtype=object)
#             cols.append(lut[idx])
#         for c in self.cat_cols:
#             s = df[c].astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
#             cols.append((sanitize(c) + "_" + s).to_numpy(dtype=object))
#         mat = np.column_stack(cols)
#         return [" ".join(row) for row in mat]
    
#     def load_state(self, state_dict):
#         self.n_bins = state_dict["n_bins"]
#         self.num_cols = state_dict["num_cols"]
#         self.cat_cols = state_dict["cat_cols"]
#         self.edges = state_dict["edges"]
#         return self

# def clean_dataframe(df):
#     d = df.replace([np.inf, -np.inf], np.nan)
#     for c in d.select_dtypes(include=[np.number]).columns:
#         if d[c].isna().any():
#             d[c] = d[c].fillna(d[c].median()) 
#     for c in [c for c in d.columns if not pd.api.types.is_numeric_dtype(d[c])]:
#         if d[c].isna().any():
#             mode_val = d[c].mode()
#             d[c] = d[c].fillna(mode_val.iloc[0] if not mode_val.empty else "unknown")
#     return d

# def to_sequence(X, steps=32):
#     n, f = X.shape
#     pad = (-f) % steps
#     if pad:
#         X = np.hstack([X, np.zeros((n, pad), dtype=X.dtype)])
#     return X.reshape(n, steps, -1)

# @app.post("/predict")
# async def predict(
#     dataset: DatasetEnum = Form(...),
#     model_type: ModelEnum = Form(...),
#     file: UploadFile = File(...)
# ):
#     try:
#         # Load data and keep an original copy for the analyst report
#         df = pd.read_csv(file.file)
#         df_original = df.copy() 
        
#         df = clean_dataframe(df)
#         if "label" in df.columns:
#             df = df.drop(columns=["label"])
            
#         prefix = dataset.value
#         base_dir = "saved_models"
        
#         # Load preprocessing artifacts
#         with open(os.path.join(base_dir, f"{prefix}_event_encoder.pkl"), "rb") as f:
#             enc_dict = pickle.load(f)
#             enc = EventEncoder().load_state(enc_dict)
            
#         with open(os.path.join(base_dir, f"{prefix}_vectorizer.pkl"), "rb") as f:
#             vec = pickle.load(f)
            
#         with open(os.path.join(base_dir, f"{prefix}_basepoint.pkl"), "rb") as f:
#             basepoint = pickle.load(f)
            
#         with open(os.path.join(base_dir, f"{prefix}_scaler.pkl"), "rb") as f:
#             scaler = pickle.load(f)
            
#         with open(os.path.join(base_dir, f"{prefix}_meta.pkl"), "rb") as f:
#             meta = pickle.load(f)

#         # Apply profile transformations
#         ev_data = enc.transform(df)
#         X_vec = vec.transform(ev_data)
#         sim = cosine_similarity(X_vec, basepoint).astype(np.float32)
#         X_p = np.hstack([X_vec.toarray(), sim])
#         X_s = scaler.transform(X_p).astype(np.float32)
#         np.clip(X_s, -meta["clip"], meta["clip"], out=X_s)

#         # Load model and predict
#         is_dl = model_type.value in ["fcnn", "cnn", "lstm"]
#         if is_dl:
#             model = tf.keras.models.load_model(os.path.join(base_dir, f"{prefix}_dl_{model_type.value}.keras"))
#             if model_type.value == "cnn":
#                 X_s = X_s[..., np.newaxis]
#             elif model_type.value == "lstm":
#                 X_s = to_sequence(X_s, steps=meta["lstm_steps"])
#             probs = model.predict(X_s).ravel()
#             predictions = (probs >= 0.5).astype(int).tolist()
#         else:
#             model = joblib.load(os.path.join(base_dir, f"{prefix}_ml_{model_type.value}.pkl"))
#             predictions = model.predict(X_s).tolist()

#         label_map = {0: "normal", 1: "attack"}
#         results = [label_map[p] for p in predictions]
        
#         # 1. Generate Summary Metrics
#         total_records = len(results)
#         threat_count = results.count("attack")
#         normal_count = total_records - threat_count
        
#         # 2. Extract Threat Details
#         df_original["predicted_status"] = results
#         df_threats = df_original[df_original["predicted_status"] == "attack"]
        
#         # FastAPI/JSON cannot serialize NaN values, so we replace them with None
#         df_threats = df_threats.replace({np.nan: None})
#         threat_details = df_threats.to_dict(orient="records")
        
#         return {
#             "summary": {
#                 "total_records": total_records,
#                 "normal_traffic": normal_count,
#                 "detected_threats": threat_count
#             },
#             "threat_details": threat_details
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# import os
# import re
# import pickle
# import joblib
# import numpy as np
# import pandas as pd
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from enum import Enum
# import tensorflow as tf
# from sklearn.metrics.pairwise import cosine_similarity

# app = FastAPI(title="Multi-Dataset Intrusion Detection API")

# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI(title="Multi-Dataset Intrusion Detection API")

# # --- ADD THIS CORS CONFIGURATION ---
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows requests from any frontend port (Vite/React)
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class DatasetEnum(str, Enum):
#     nsl_kdd = "nsl_kdd"
#     cicids2017 = "cicids2017"
#     unsw_nb15 = "unsw_nb15"

# class ModelEnum(str, Enum):
#     svm = "svm"
#     knn = "knn"
#     random_forest = "random_forest"
#     naive_bayes = "naive_bayes"
#     decision_tree = "decision_tree"
#     fcnn = "fcnn"
#     cnn = "cnn"
#     lstm = "lstm"

# def sanitize(name):
#     return re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower()

# class EventEncoder:
#     def __init__(self, n_bins=10):
#         self.n_bins = n_bins
#         self.num_cols, self.cat_cols, self.edges = [], [], {}
    
#     def transform(self, df):
#         cols = []
#         for c in self.num_cols:
#             e = self.edges[c]
#             idx = np.searchsorted(e, df[c].to_numpy(dtype=np.float64), side="left")
#             lut = np.array([f"{sanitize(c)}_b{i}" for i in range(len(e) + 1)], dtype=object)
#             cols.append(lut[idx])
#         for c in self.cat_cols:
#             s = df[c].astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
#             cols.append((sanitize(c) + "_" + s).to_numpy(dtype=object))
#         mat = np.column_stack(cols)
#         return [" ".join(row) for row in mat]
    
#     def load_state(self, state_dict):
#         self.n_bins = state_dict["n_bins"]
#         self.num_cols = state_dict["num_cols"]
#         self.cat_cols = state_dict["cat_cols"]
#         self.edges = state_dict["edges"]
#         return self

# def clean_dataframe(df):
#     d = df.replace([np.inf, -np.inf], np.nan)
#     for c in d.select_dtypes(include=[np.number]).columns:
#         if d[c].isna().any():
#             d[c] = d[c].fillna(d[c].median()) 
#     for c in [c for c in d.columns if not pd.api.types.is_numeric_dtype(d[c])]:
#         if d[c].isna().any():
#             mode_val = d[c].mode()
#             d[c] = d[c].fillna(mode_val.iloc[0] if not mode_val.empty else "unknown")
#     return d

# def to_sequence(X, steps=32):
#     n, f = X.shape
#     pad = (-f) % steps
#     if pad:
#         X = np.hstack([X, np.zeros((n, pad), dtype=X.dtype)])
#     return X.reshape(n, steps, -1)

# # ==========================================
# # DEDICATED MODEL PREDICTION HANDLERS
# # ==========================================

# def predict_fcnn(model, X_s, dataset_name):
#     """Specialized prediction handler for FCNN matching Kaggle's high-recall dynamic thresholding."""
#     probs = model.predict(X_s, batch_size=1024, verbose=0).ravel()
    
#     if dataset_name == "cicids2017":
#         # Apply probability stretching to lift suppressed threat confidence scores
#         stretched_probs = np.power(probs, 0.65)
        
#         # Match Kaggle's aggressive dynamic threshold scan (down to 0.01)
#         best_th = 0.01
#         predictions = (stretched_probs >= best_th).astype(int)
#     else:
#         # Standard threshold for NSL-KDD and UNSW-NB15
#         predictions = (probs >= 0.5).astype(int)
        
#     return predictions.tolist()

# def predict_cnn_lstm(model, X_s, model_type, meta):
#     """Specialized prediction handler for CNN and LSTM deep learning architectures."""
#     if model_type == "cnn":
#         X_input = X_s[..., np.newaxis]
#     elif model_type == "lstm":
#         X_input = to_sequence(X_s, steps=meta.get("lstm_steps", 32))
#     else:
#         X_input = X_s
        
#     probs = model.predict(X_input, batch_size=1024, verbose=0).ravel()
#     predictions = (probs >= 0.5).astype(int)
#     return predictions.tolist()

# def predict_ml_model(model, X_s):
#     """Specialized prediction handler for classical machine learning models (SVM, RF, KNN, etc.)."""
#     predictions = model.predict(X_s)
#     return predictions.astype(int).tolist()


# # ==========================================
# # MAIN API ROUTE
# # ==========================================

# @app.post("/predict")
# async def predict(
#     dataset: DatasetEnum = Form(...),
#     model_type: ModelEnum = Form(...),
#     file: UploadFile = File(...)
# ):
#     try:
#         # Load data and keep an original copy for the analyst report
#         df = pd.read_csv(file.file)
#         df_original = df.copy() 
        
#         df = clean_dataframe(df)
#         if "label" in df.columns:
#             df = df.drop(columns=["label"])
            
#         prefix = dataset.value
#         base_dir = "saved_models"
        
#         # 1. Load preprocessing artifacts
#         with open(os.path.join(base_dir, f"{prefix}_event_encoder.pkl"), "rb") as f:
#             enc_dict = pickle.load(f)
#             enc = EventEncoder().load_state(enc_dict)
            
#         with open(os.path.join(base_dir, f"{prefix}_vectorizer.pkl"), "rb") as f:
#             vec = pickle.load(f)
            
#         with open(os.path.join(base_dir, f"{prefix}_basepoint.pkl"), "rb") as f:
#             basepoint = pickle.load(f)
            
#         with open(os.path.join(base_dir, f"{prefix}_scaler.pkl"), "rb") as f:
#             scaler = pickle.load(f)
            
#         with open(os.path.join(base_dir, f"{prefix}_meta.pkl"), "rb") as f:
#             meta = pickle.load(f)

#         # 2. Apply profile transformations (Event Encoding -> TF-IDF -> Cosine Sim -> Scaling -> Clipping)
#         ev_data = enc.transform(df)
#         X_vec = vec.transform(ev_data)
#         sim = cosine_similarity(X_vec, basepoint).astype(np.float32)
#         X_p = np.hstack([X_vec.toarray(), sim])
#         X_s = scaler.transform(X_p).astype(np.float32)
#         np.clip(X_s, -meta["clip"], meta["clip"], out=X_s)

#         # 3. Route to Dedicated Model Predictors
#         if model_type.value == "fcnn":
#             model = tf.keras.models.load_model(os.path.join(base_dir, f"{prefix}_dl_fcnn.keras"))
#             predictions = predict_fcnn(model, X_s, prefix)
            
#         elif model_type.value in ["cnn", "lstm"]:
#             model = tf.keras.models.load_model(os.path.join(base_dir, f"{prefix}_dl_{model_type.value}.keras"))
#             predictions = predict_cnn_lstm(model, X_s, model_type.value, meta)
            
#         else:
#             model = joblib.load(os.path.join(base_dir, f"{prefix}_ml_{model_type.value}.pkl"))
#             predictions = predict_ml_model(model, X_s)

#         # 4. Map Predictions and Build Response
#         label_map = {0: "normal", 1: "attack"}
#         results = [label_map[p] for p in predictions]
        
#         total_records = len(results)
#         threat_count = results.count("attack")
#         normal_count = total_records - threat_count
        
#         df_original["predicted_status"] = results
#         df_threats = df_original[df_original["predicted_status"] == "attack"]
        
#         # FastAPI/JSON cannot serialize NaN values, replace with None
#         df_threats = df_threats.replace({np.nan: None})
#         threat_details = df_threats.to_dict(orient="records")
        
#         return {
#             "summary": {
#                 "total_records": total_records,
#                 "normal_traffic": normal_count,
#                 "detected_threats": threat_count
#             },
#             "threat_details": threat_details
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# # main.py 2222222
# import io
# import os
# import re
# import pickle
# import joblib
# import logging
# from enum import Enum
# from typing import List, Dict, Any, Optional

# import numpy as np
# import pandas as pd
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse, JSONResponse
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import tensorflow as tf

# app = FastAPI(title="Multi-Dataset Intrusion Detection API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# logger = logging.getLogger("uvicorn.error")

# class DatasetEnum(str, Enum):
#     nsl_kdd = "nsl_kdd"
#     cicids2017 = "cicids2017"
#     unsw_nb15 = "unsw_nb15"

# class ModelEnum(str, Enum):
#     svm = "svm"
#     knn = "knn"
#     random_forest = "random_forest"
#     naive_bayes = "naive_bayes"
#     decision_tree = "decision_tree"
#     fcnn = "fcnn"
#     cnn = "cnn"
#     lstm = "lstm"

# def sanitize(name):
#     return re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower()

# class EventEncoder:
#     def __init__(self, n_bins=10):
#         self.n_bins = n_bins
#         self.num_cols, self.cat_cols, self.edges = [], [], {}
#     def transform(self, df):
#         cols = []
#         for c in self.num_cols:
#             e = self.edges[c]
#             idx = np.searchsorted(e, df[c].to_numpy(dtype=np.float64), side="left")
#             lut = np.array([f"{sanitize(c)}_b{i}" for i in range(len(e) + 1)], dtype=object)
#             cols.append(lut[idx])
#         for c in self.cat_cols:
#             s = df[c].astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
#             cols.append((sanitize(c) + "_" + s).to_numpy(dtype=object))
#         mat = np.column_stack(cols)
#         return [" ".join(row) for row in mat]
#     def load_state(self, state_dict):
#         self.n_bins = state_dict["n_bins"]
#         self.num_cols = state_dict["num_cols"]
#         self.cat_cols = state_dict["cat_cols"]
#         self.edges = state_dict["edges"]
#         return self

# def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
#     d = df.replace([np.inf, -np.inf], np.nan).copy()
#     for c in d.select_dtypes(include=[np.number]).columns:
#         if d[c].isna().any():
#             d[c] = d[c].fillna(d[c].median())
#     for c in [c for c in d.columns if not pd.api.types.is_numeric_dtype(d[c])]:
#         if d[c].isna().any():
#             mode_val = d[c].mode()
#             d[c] = d[c].fillna(mode_val.iloc[0] if not mode_val.empty else "unknown")
#     return d

# def to_sequence(X, steps=32):
#     n, f = X.shape
#     pad = (-f) % steps
#     if pad:
#         X = np.hstack([X, np.zeros((n, pad), dtype=X.dtype)])
#     return X.reshape(n, steps, -1)

# def predict_fcnn(model, X_s, dataset_name):
#     probs = model.predict(X_s, batch_size=1024, verbose=0).ravel()
#     if dataset_name == "cicids2017":
#         probs = np.power(probs, 0.65)
#         best_th = 0.01
#         predictions = (probs >= best_th).astype(int)
#     else:
#         predictions = (probs >= 0.5).astype(int)
#     return predictions, probs

# def predict_cnn_lstm(model, X_s, model_type, meta):
#     if model_type == "cnn":
#         X_input = X_s[..., np.newaxis]
#     elif model_type == "lstm":
#         X_input = to_sequence(X_s, steps=meta.get("lstm_steps", 32))
#     else:
#         X_input = X_s
#     probs = model.predict(X_input, batch_size=1024, verbose=0).ravel()
#     predictions = (probs >= 0.5).astype(int)
#     return predictions, probs

# def predict_ml_model(model, X_s):
#     if hasattr(model, "predict_proba"):
#         probs = model.predict_proba(X_s)
#         if probs.ndim == 2 and probs.shape[1] >= 2:
#             confs = probs[:, 1]
#         else:
#             confs = np.max(probs, axis=1)
#         preds = (confs >= 0.5).astype(int)
#         return preds, confs
#     else:
#         preds = model.predict(X_s).astype(int)
#         confs = np.where(preds == 1, 0.9, 0.1)
#         return preds, confs

# BASE_DIR = "saved_models"
# ARTIFACTS: Dict[str, Dict[str, Any]] = {}
# MODEL_REGISTRY: Dict[tuple, Any] = {}

# @app.on_event("startup")
# def load_all_artifacts_and_models():
#     datasets = [d.value for d in DatasetEnum]
#     ml_models = ["svm","knn","random_forest","naive_bayes","decision_tree"]
#     dl_models = ["fcnn","cnn","lstm"]
#     for ds in datasets:
#         try:
#             enc_path = os.path.join(BASE_DIR, f"{ds}_event_encoder.pkl")
#             vec_path = os.path.join(BASE_DIR, f"{ds}_vectorizer.pkl")
#             basepoint_path = os.path.join(BASE_DIR, f"{ds}_basepoint.pkl")
#             scaler_path = os.path.join(BASE_DIR, f"{ds}_scaler.pkl")
#             meta_path = os.path.join(BASE_DIR, f"{ds}_meta.pkl")

#             ARTIFACTS.setdefault(ds, {})
#             if os.path.exists(enc_path):
#                 with open(enc_path, "rb") as f:
#                     ARTIFACTS[ds]["enc"] = EventEncoder().load_state(pickle.load(f))
#             if os.path.exists(vec_path):
#                 with open(vec_path, "rb") as f:
#                     ARTIFACTS[ds]["vec"] = pickle.load(f)
#             if os.path.exists(basepoint_path):
#                 with open(basepoint_path, "rb") as f:
#                     ARTIFACTS[ds]["basepoint"] = pickle.load(f)
#             if os.path.exists(scaler_path):
#                 with open(scaler_path, "rb") as f:
#                     ARTIFACTS[ds]["scaler"] = pickle.load(f)
#             if os.path.exists(meta_path):
#                 with open(meta_path, "rb") as f:
#                     ARTIFACTS[ds]["meta"] = pickle.load(f)
#         except Exception:
#             logger.exception("Failed loading artifacts for %s", ds)

#         for m in ml_models:
#             path = os.path.join(BASE_DIR, f"{ds}_ml_{m}.pkl")
#             if os.path.exists(path):
#                 try:
#                     MODEL_REGISTRY[(ds, m)] = joblib.load(path)
#                 except Exception:
#                     logger.exception("Failed loading ML model %s for %s", m, ds)

#         for m in dl_models:
#             path = os.path.join(BASE_DIR, f"{ds}_dl_{m}.keras")
#             if os.path.exists(path):
#                 try:
#                     MODEL_REGISTRY[(ds, m)] = tf.keras.models.load_model(path)
#                 except Exception:
#                     logger.exception("Failed loading DL model %s for %s", m, ds)

#     logger.info("Startup load complete. Datasets loaded: %s", ", ".join(ARTIFACTS.keys()))

# @app.post("/predict")
# async def predict(
#     dataset: DatasetEnum = Form(...),
#     model_types: str = Form(...),
#     file: UploadFile = File(...),
#     page: int = Form(1),
#     page_size: int = Form(50)
# ):
#     try:
#         contents = await file.read()
#         df = pd.read_csv(io.BytesIO(contents))
#         df_original = df.copy()
#         has_labels = "label" in df.columns
#         labels = df["label"].to_numpy() if has_labels else None

#         df = clean_dataframe(df)
#         if has_labels:
#             df = df.drop(columns=["label"])

#         prefix = dataset.value
#         artifacts = ARTIFACTS.get(prefix)
#         if artifacts is None or any(k not in artifacts for k in ("enc","vec","basepoint","scaler","meta")):
#             raise RuntimeError(f"Artifacts not loaded for dataset: {prefix}")

#         enc = artifacts["enc"]
#         vec = artifacts["vec"]
#         basepoint = artifacts["basepoint"]
#         scaler = artifacts["scaler"]
#         meta = artifacts["meta"]

#         ev_data = enc.transform(df)
#         X_vec = vec.transform(ev_data)
#         sim = cosine_similarity(X_vec, basepoint).astype(np.float32)
#         X_p = np.hstack([X_vec.toarray(), sim])
#         X_s = scaler.transform(X_p).astype(np.float32)
#         np.clip(X_s, -meta.get("clip", 10.0), meta.get("clip", 10.0), out=X_s)

#         requested = [m.strip() for m in model_types.split(",") if m.strip()]
#         if not requested:
#             raise HTTPException(status_code=400, detail="No model types requested")

#         per_model_preds: Dict[str, np.ndarray] = {}
#         per_model_conf: Dict[str, np.ndarray] = {}

#         for m in requested:
#             key = (prefix, m)
#             model = MODEL_REGISTRY.get(key)
#             if model is None:
#                 try:
#                     if m in ["fcnn","cnn","lstm"]:
#                         path = os.path.join(BASE_DIR, f"{prefix}_dl_{m}.keras")
#                         model = tf.keras.models.load_model(path)
#                     else:
#                         path = os.path.join(BASE_DIR, f"{prefix}_ml_{m}.pkl")
#                         model = joblib.load(path)
#                     MODEL_REGISTRY[key] = model
#                 except Exception:
#                     logger.exception("Model load failed for %s %s", prefix, m)
#                     continue

#             if m == "fcnn":
#                 preds, confs = predict_fcnn(model, X_s, prefix)
#             elif m in ["cnn", "lstm"]:
#                 preds, confs = predict_cnn_lstm(model, X_s, m, meta)
#             else:
#                 preds, confs = predict_ml_model(model, X_s)

#             per_model_preds[m] = np.asarray(preds, dtype=int)
#             per_model_conf[m] = np.asarray(confs, dtype=float)

#         if not per_model_preds:
#             raise RuntimeError("No models available to run for requested model_types")

#         n = X_s.shape[0]
#         label_map = {0: "normal", 1: "attack"}
#         results: List[Dict[str, Any]] = []

#         for i in range(n):
#             votes = {}
#             vote_vals = []
#             conf_vals = []
#             for m in per_model_preds:
#                 p = int(per_model_preds[m][i])
#                 votes[m] = label_map.get(p, str(p))
#                 vote_vals.append(p)
#                 conf_vals.append(float(per_model_conf[m][i]))

#             maj = int(round(np.mean(vote_vals)))
#             confidence = float(np.max(conf_vals)) if conf_vals else 0.0

#             attack_type: Optional[str] = None
#             for m in per_model_preds:
#                 model = MODEL_REGISTRY.get((prefix, m))
#                 if model is not None and hasattr(model, "classes_"):
#                     classes = getattr(model, "classes_", None)
#                     if classes is not None and len(classes) > 2:
#                         cls_idx = int(per_model_preds[m][i])
#                         try:
#                             attack_type = str(classes[cls_idx])
#                             break
#                         except Exception:
#                             continue

#             rec = {
#                 "record_index": int(i),
#                 "predicted_status": label_map.get(maj, "attack" if maj == 1 else "normal"),
#                 "confidence": confidence,
#                 "attack_type": attack_type,
#                 "model_votes": votes,
#                 "original_record": df_original.iloc[i].replace({np.nan: None}).to_dict()
#             }
#             results.append(rec)

#         total_records = n
#         threat_count = sum(1 for r in results if r["predicted_status"] == "attack")
#         normal_count = total_records - threat_count
#         comparisons = {m: {"detected_threats": int(np.sum(per_model_preds[m] == 1))} for m in per_model_preds}

#         metrics = None
#         if has_labels:
#             final_preds = np.array([1 if r["predicted_status"] == "attack" else 0 for r in results])
#             try:
#                 metrics = {
#                     "accuracy": float(accuracy_score(labels, final_preds)),
#                     "precision": float(precision_score(labels, final_preds, zero_division=0)),
#                     "recall": float(recall_score(labels, final_preds, zero_division=0)),
#                     "f1": float(f1_score(labels, final_preds, zero_division=0)),
#                     "confusion_matrix": confusion_matrix(labels, final_preds).tolist()
#                 }
#             except Exception:
#                 logger.exception("Failed computing metrics")

#         threats = [r for r in results if r["predicted_status"] == "attack"]
#         full_threat_count = len(threats)
#         page = max(1, int(page))
#         page_size = max(1, int(page_size))
#         start = (page - 1) * page_size
#         end = start + page_size
#         paged_threats = threats[start:end]

#         response = {
#             "summary": {
#                 "total_records": total_records,
#                 "normal_traffic": normal_count,
#                 "detected_threats": threat_count
#             },
#             "comparisons": comparisons,
#             "threat_details": paged_threats,
#             "threat_details_full_count": full_threat_count,
#             "page": page,
#             "page_size": page_size,
#             "metrics": metrics
#         }

#         return JSONResponse(response)

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.exception("Prediction failed")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/export")
# def export_threats(dataset: DatasetEnum = Query(...), model_types: str = Query(...), file_path: Optional[str] = Query(None)):
#     """
#     Export endpoint that streams a CSV of the last run's threats.
#     If file_path is provided, the server will attempt to read that CSV and run the same pipeline.
#     Otherwise, this endpoint expects the frontend to call /predict and then request export with the same params.
#     """
#     # For simplicity, this implementation expects the frontend to POST to /predict and then call /export
#     # with the same dataset and model_types. The server will attempt to find a cached file in /tmp named
#     # by dataset+models. If not found, return 404.
#     cache_key = f"/tmp/siem_export_{dataset.value}_{sanitize(model_types)}.csv"
#     if file_path:
#         cache_key = file_path
#     if not os.path.exists(cache_key):
#         raise HTTPException(status_code=404, detail="Export not available. Run a scan first or provide file_path.")
#     def iterfile():
#         with open(cache_key, "rb") as f:
#             while True:
#                 chunk = f.read(8192)
#                 if not chunk:
#                     break
#                 yield chunk
#     return StreamingResponse(iterfile(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=threats_{dataset.value}.csv"})

# ####V_2
# import os
# import re
# import time
# import pickle
# from enum import Enum
# from functools import lru_cache

# import joblib
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
# from sklearn.metrics.pairwise import cosine_similarity

# app = FastAPI(title="AI-SIEM Multi-Dataset Intrusion Detection API")

# # Allows requests from any frontend port (Vite/React)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# BASE_DIR = "saved_models"

# # Severity is derived from the model's attack score (0-1). Tune the bands here.
# SEVERITY_BANDS = ((0.90, "high"), (0.70, "medium"))  # anything lower is "low"

# # Max flagged rows sent to the browser per scan (highest scores first).
# # Summary counts always cover every record. Set MAX_THREAT_ROWS=0 for no limit.
# MAX_THREAT_ROWS = int(os.getenv("MAX_THREAT_ROWS", "5000"))


# class DatasetEnum(str, Enum):
#     nsl_kdd = "nsl_kdd"
#     cicids2017 = "cicids2017"
#     unsw_nb15 = "unsw_nb15"


# class ModelEnum(str, Enum):
#     svm = "svm"
#     knn = "knn"
#     random_forest = "random_forest"
#     naive_bayes = "naive_bayes"
#     decision_tree = "decision_tree"
#     fcnn = "fcnn"
#     cnn = "cnn"
#     lstm = "lstm"


# def sanitize(name):
#     return re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower()


# class EventEncoder:
#     def __init__(self, n_bins=10):
#         self.n_bins = n_bins
#         self.num_cols, self.cat_cols, self.edges = [], [], {}

#     def transform(self, df):
#         cols = []
#         for c in self.num_cols:
#             e = self.edges[c]
#             idx = np.searchsorted(e, df[c].to_numpy(dtype=np.float64), side="left")
#             lut = np.array([f"{sanitize(c)}_b{i}" for i in range(len(e) + 1)], dtype=object)
#             cols.append(lut[idx])
#         for c in self.cat_cols:
#             s = df[c].astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
#             cols.append((sanitize(c) + "_" + s).to_numpy(dtype=object))
#         mat = np.column_stack(cols)
#         return [" ".join(row) for row in mat]

#     def load_state(self, state_dict):
#         self.n_bins = state_dict["n_bins"]
#         self.num_cols = state_dict["num_cols"]
#         self.cat_cols = state_dict["cat_cols"]
#         self.edges = state_dict["edges"]
#         return self


# def clean_dataframe(df):
#     d = df.replace([np.inf, -np.inf], np.nan)
#     for c in d.select_dtypes(include=[np.number]).columns:
#         if d[c].isna().any():
#             d[c] = d[c].fillna(d[c].median())
#     for c in [c for c in d.columns if not pd.api.types.is_numeric_dtype(d[c])]:
#         if d[c].isna().any():
#             mode_val = d[c].mode()
#             d[c] = d[c].fillna(mode_val.iloc[0] if not mode_val.empty else "unknown")
#     return d


# def to_sequence(X, steps=32):
#     n, f = X.shape
#     pad = (-f) % steps
#     if pad:
#         X = np.hstack([X, np.zeros((n, pad), dtype=X.dtype)])
#     return X.reshape(n, steps, -1)


# # ==========================================
# # CACHED LOADERS (restart the server after retraining)
# # ==========================================

# @lru_cache(maxsize=8)
# def load_artifacts(prefix):
#     """Preprocessing artifacts for one dataset, loaded once and reused."""
#     def _load(suffix):
#         with open(os.path.join(BASE_DIR, f"{prefix}_{suffix}.pkl"), "rb") as f:
#             return pickle.load(f)

#     enc = EventEncoder().load_state(_load("event_encoder"))
#     vec = _load("vectorizer")
#     basepoint = _load("basepoint")
#     scaler = _load("scaler")
#     meta = _load("meta")

#     try:
#         feature_names = np.asarray(vec.get_feature_names_out())
#     except Exception:
#         feature_names = None  # explanations are skipped if the vectorizer can't list its features

#     bp = basepoint.toarray() if hasattr(basepoint, "toarray") else basepoint
#     bp_vec = np.asarray(bp, dtype=np.float64).reshape(-1)

#     return {
#         "enc": enc, "vec": vec, "basepoint": basepoint, "scaler": scaler,
#         "meta": meta, "feature_names": feature_names, "bp_vec": bp_vec,
#     }


# @lru_cache(maxsize=12)
# def load_dl_model(path):
#     return tf.keras.models.load_model(path)


# @lru_cache(maxsize=12)
# def load_ml_model(path):
#     return joblib.load(path)


# def preprocess(df, art):
#     """Event Encoding -> TF-IDF -> Cosine Sim -> Scaling -> Clipping (unchanged pipeline)."""
#     ev_data = art["enc"].transform(df)
#     X_vec = art["vec"].transform(ev_data)
#     sim = cosine_similarity(X_vec, art["basepoint"]).astype(np.float32)
#     X_dense = X_vec.toarray()
#     X_p = np.hstack([X_dense, sim])
#     X_s = art["scaler"].transform(X_p).astype(np.float32)
#     np.clip(X_s, -art["meta"]["clip"], art["meta"]["clip"], out=X_s)
#     return X_s, X_dense, sim.max(axis=1)


# class InputError(ValueError):
#     """The uploaded file can't be scanned; reported to the user as HTTP 422."""


# def required_columns(art):
#     return [str(c) for c in (*art["enc"].num_cols, *art["enc"].cat_cols)]


# def check_required_columns(df, art, prefix):
#     required = required_columns(art)
#     present = {str(c) for c in df.columns}
#     missing = [c for c in required if c not in present]
#     if missing:
#         shown = ", ".join(repr(c) for c in missing[:10])  # repr exposes stray spaces
#         more = f" (and {len(missing) - 10} more)" if len(missing) > 10 else ""
#         raise InputError(
#             f"Your CSV is missing {len(missing)} of the {len(required)} columns the {prefix} "
#             f"model needs: {shown}{more}. Check that you picked the right dataset."
#         )


# def read_and_prepare(file, prefix):
#     try:
#         df = pd.read_csv(file.file)
#     except Exception as e:
#         raise InputError(f"Could not read the file as a CSV: {e}")
#     if df.empty:
#         raise InputError("The uploaded CSV has no rows.")
#     df_original = df.copy()

#     art = load_artifacts(prefix)
#     check_required_columns(df, art, prefix)

#     df = clean_dataframe(df)
#     if "label" in df.columns:
#         df = df.drop(columns=["label"])

#     X_s, X_dense, sim = preprocess(df, art)
#     return df_original, art, X_s, X_dense, sim


# # ==========================================
# # DEDICATED MODEL PREDICTION HANDLERS
# # Each returns (predictions as 0/1 array, attack score array in 0-1)
# # ==========================================

# def predict_fcnn(model, X_s, dataset_name):
#     """FCNN handler matching Kaggle's high-recall dynamic thresholding."""
#     probs = model.predict(X_s, batch_size=1024, verbose=0).ravel()

#     if dataset_name == "cicids2017":
#         # Probability stretching lifts suppressed threat confidence scores
#         scores = np.power(probs, 0.65)
#         # Matches Kaggle's aggressive dynamic threshold scan (down to 0.01)
#         predictions = (scores >= 0.01).astype(int)
#     else:
#         scores = probs
#         predictions = (probs >= 0.5).astype(int)

#     return predictions, scores


# def predict_cnn_lstm(model, X_s, model_type, meta):
#     """CNN and LSTM handler."""
#     if model_type == "cnn":
#         X_input = X_s[..., np.newaxis]
#     elif model_type == "lstm":
#         X_input = to_sequence(X_s, steps=meta.get("lstm_steps", 32))
#     else:
#         X_input = X_s

#     probs = model.predict(X_input, batch_size=1024, verbose=0).ravel()
#     return (probs >= 0.5).astype(int), probs


# def predict_ml_model(model, X_s, want_scores=True):
#     """Classical ML handler (SVM, RF, KNN, ...). Predictions still come from model.predict."""
#     predictions = np.asarray(model.predict(X_s)).astype(int)
#     scores = None

#     if want_scores and hasattr(model, "predict_proba"):
#         try:
#             proba = model.predict_proba(X_s)
#             classes = list(getattr(model, "classes_", [0, 1]))
#             col = classes.index(1) if 1 in classes else proba.shape[1] - 1
#             scores = proba[:, col]
#         except Exception:
#             scores = None

#     if want_scores and scores is None and hasattr(model, "decision_function"):
#         try:
#             d = np.asarray(model.decision_function(X_s)).ravel()
#             scores = 1.0 / (1.0 + np.exp(-np.clip(d, -30, 30)))  # monotonic, not calibrated
#         except Exception:
#             scores = None

#     if scores is None:
#         scores = predictions.astype(float)
#     return predictions, scores


# def run_model(prefix, model_name, X_s, meta, want_scores=True):
#     if model_name == "fcnn":
#         model = load_dl_model(os.path.join(BASE_DIR, f"{prefix}_dl_fcnn.keras"))
#         return predict_fcnn(model, X_s, prefix)
#     if model_name in ("cnn", "lstm"):
#         model = load_dl_model(os.path.join(BASE_DIR, f"{prefix}_dl_{model_name}.keras"))
#         return predict_cnn_lstm(model, X_s, model_name, meta)
#     model = load_ml_model(os.path.join(BASE_DIR, f"{prefix}_ml_{model_name}.pkl"))
#     return predict_ml_model(model, X_s, want_scores)


# # ==========================================
# # ANALYST HELPERS (score, severity, evidence, metrics)
# # ==========================================

# def severity_of(score):
#     for limit, name in SEVERITY_BANDS:
#         if score >= limit:
#             return name
#     return "low"


# def severity_counts(scores):
#     """Counts per severity band for an array of attack scores."""
#     counts = {name: 0 for _, name in SEVERITY_BANDS}
#     remaining = np.ones(len(scores), dtype=bool)
#     for limit, name in SEVERITY_BANDS:
#         hit = remaining & (scores >= limit)
#         counts[name] = int(hit.sum())
#         remaining &= ~hit
#     counts["low"] = int(remaining.sum())
#     return counts


# def build_threat_meta(threat_idx, scores, sim, X_dense, art, top_k=3, chunk=5000):
#     """One entry per flagged row, in the same order as threat_details.

#     unusual_events = the event bins this record shows far more than the average
#     normal record does. It describes how the record differs from the normal
#     profile; it is not the model's internal explanation.
#     """
#     names, bp = art["feature_names"], art["bp_vec"]
#     can_explain = names is not None and bp.shape[0] == X_dense.shape[1]

#     out = []
#     for s in range(0, len(threat_idx), chunk):
#         part = threat_idx[s:s + chunk]
#         events = [[] for _ in part]
#         if can_explain:
#             delta = X_dense[part] - bp
#             top = np.argsort(-delta, axis=1)[:, :top_k]
#             for r, cols in enumerate(top):
#                 events[r] = [str(names[c]) for c in cols if delta[r, c] > 0]
#         for r, i in enumerate(part):
#             score = float(scores[i])
#             out.append({
#                 "row": int(i) + 1,  # 1-based data row in the uploaded CSV
#                 "score": round(score, 4),
#                 "severity": severity_of(score),
#                 "similarity_to_normal": round(float(sim[i]), 4),
#                 "unusual_events": events[r],
#             })
#     return out


# NORMAL_LABELS = {"normal", "benign", "0", "0.0", "false", "none"}


# def compute_metrics(df_original, predictions):
#     """Accuracy/precision/recall/F1 when the CSV has a 'label' column.

#     Numeric labels: non-zero = attack. Text labels: anything except
#     normal/benign/0 counts as attack.
#     """
#     try:
#         col = next((c for c in df_original.columns if str(c).strip().lower() == "label"), None)
#         if col is None:
#             return None
#         series = df_original[col]
#         mask = series.notna().to_numpy()
#         if mask.sum() == 0:
#             return None

#         labelled = series[mask]
#         if pd.api.types.is_numeric_dtype(labelled):
#             y_true = (labelled.astype(float) != 0).astype(int).to_numpy()
#         else:
#             y_true = (~labelled.astype(str).str.strip().str.lower().isin(NORMAL_LABELS)).astype(int).to_numpy()
#         y_pred = np.asarray(predictions).astype(int)[mask]

#         tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
#         return {
#             "label_column": str(col),
#             "labelled_records": int(mask.sum()),
#             "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
#             "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
#             "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
#             "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
#             "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
#         }
#     except Exception:
#         return None  # metrics are optional and must never break a scan


# # ==========================================
# # API ROUTES
# # ==========================================

# @app.get("/health")
# async def health():
#     return {"status": "ok"}


# @app.get("/schema/{dataset}")
# async def schema(dataset: DatasetEnum):
#     """Columns the chosen dataset's model needs, so the UI can check a CSV before upload."""
#     try:
#         art = load_artifacts(dataset.value)
#         return {
#             "dataset": dataset.value,
#             "required_columns": required_columns(art),
#             "optional_columns": ["label"],
#         }
#     except FileNotFoundError:
#         raise HTTPException(status_code=404, detail=f"No saved artifacts found for {dataset.value}.")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/predict")
# async def predict(
#     dataset: DatasetEnum = Form(...),
#     model_type: ModelEnum = Form(...),
#     file: UploadFile = File(...)
# ):
#     try:
#         started = time.time()
#         prefix = dataset.value

#         df_original, art, X_s, X_dense, sim = read_and_prepare(file, prefix)

#         predictions, scores = run_model(prefix, model_type.value, X_s, art["meta"])
#         predictions = np.asarray(predictions).astype(int)
#         scores = np.nan_to_num(np.asarray(scores, dtype=np.float64))

#         # Map predictions and build response (original fields unchanged)
#         label_map = {0: "normal", 1: "attack"}
#         results = [label_map[p] for p in predictions.tolist()]

#         total_records = len(results)
#         threat_count = results.count("attack")
#         normal_count = total_records - threat_count

#         df_original["predicted_status"] = results

#         # Highest-scoring threats first if there are more than the browser should receive
#         threat_idx = np.flatnonzero(predictions == 1)
#         keep = threat_idx
#         if MAX_THREAT_ROWS and len(threat_idx) > MAX_THREAT_ROWS:
#             keep = np.sort(threat_idx[np.argsort(-scores[threat_idx])[:MAX_THREAT_ROWS]])

#         df_threats = df_original.iloc[keep]

#         # FastAPI/JSON cannot serialize NaN values, replace with None
#         df_threats = df_threats.replace({np.nan: None})
#         threat_details = df_threats.to_dict(orient="records")

#         # Per-threat score, severity and evidence (same order as threat_details)
#         threat_meta = build_threat_meta(keep, scores, sim, X_dense, art)

#         return {
#             "summary": {
#                 "total_records": total_records,
#                 "normal_traffic": normal_count,
#                 "detected_threats": threat_count,
#                 "severity_counts": severity_counts(scores[threat_idx]),
#                 "threats_returned": len(threat_details),
#                 "threats_truncated": len(threat_details) < threat_count,
#                 "dataset": prefix,
#                 "model_type": model_type.value,
#                 "elapsed_seconds": round(time.time() - started, 2),
#             },
#             "threat_details": threat_details,
#             "threat_meta": threat_meta,
#             "metrics": compute_metrics(df_original, predictions),
#         }

#     except InputError as e:
#         raise HTTPException(status_code=422, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/compare")
# async def compare(
#     dataset: DatasetEnum = Form(...),
#     file: UploadFile = File(...)
# ):
#     """Run one CSV through every model and report how they differ and agree."""
#     try:
#         prefix = dataset.value
#         df_original, art, X_s, _, _ = read_and_prepare(file, prefix)
#         total = len(df_original)

#         rows, preds_by_model = [], {}
#         for m in ModelEnum:
#             started = time.time()
#             try:
#                 preds, _ = run_model(prefix, m.value, X_s, art["meta"], want_scores=False)
#                 preds = np.asarray(preds).astype(int)
#                 preds_by_model[m.value] = preds

#                 threats = int(preds.sum())
#                 row = {
#                     "model": m.value,
#                     "status": "ok",
#                     "threats": threats,
#                     "normal": total - threats,
#                     "threat_rate": round(100.0 * threats / total, 2),
#                     "seconds": round(time.time() - started, 2),
#                 }
#                 metrics = compute_metrics(df_original, preds)
#                 if metrics:
#                     row["metrics"] = metrics
#             except Exception as e:  # one missing/broken model must not stop the rest
#                 row = {"model": m.value, "status": "error", "detail": str(e)}
#             rows.append(row)

#         agreement = None
#         ok = len(preds_by_model)
#         if ok >= 2:
#             votes = np.sum(list(preds_by_model.values()), axis=0)
#             majority = (votes * 2 > ok).astype(int)
#             agreement = {
#                 "models_compared": ok,
#                 "unanimous_threat": int((votes == ok).sum()),
#                 "unanimous_normal": int((votes == 0).sum()),
#                 "split": int(((votes > 0) & (votes < ok)).sum()),
#             }
#             for row in rows:
#                 if row["status"] == "ok":
#                     same = preds_by_model[row["model"]] == majority
#                     row["agrees_with_majority"] = round(100.0 * float(same.mean()), 2)

#         return {"dataset": prefix, "total_records": total, "models": rows, "agreement": agreement}

#     except InputError as e:
#         raise HTTPException(status_code=422, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# ###V3
# import os
# import re
# import time
# import pickle
# from enum import Enum
# from functools import lru_cache
# from huggingface_hub import snapshot_download

# import joblib
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
# from sklearn.metrics.pairwise import cosine_similarity

# BASE_DIR = "saved_models"

# # ==========================================
# # AUTO-DOWNLOAD MODELS FROM HUGGING FACE
# # ==========================================
# def ensure_models_downloaded():
#     """Downloads models from Hugging Face if they don't exist locally (e.g., on Render)."""
#     os.makedirs(BASE_DIR, exist_ok=True)
#     if not os.listdir(BASE_DIR):
#         print("Models directory is empty. Downloading artifacts from Hugging Face...")
#         try:
#             snapshot_download(
#                 repo_id="Rithvik-3103/cyber-ann-models",  # <--- REPLACE WITH YOUR ACTUAL HF USERNAME & REPO
#                 repo_type="dataset",                         
#                 local_dir=BASE_DIR
#             )
#             print("Successfully downloaded all models!")
#         except Exception as e:
#             print(f"Error downloading models: {e}")

# # Run the check immediately when the app starts up
# ensure_models_downloaded()

# # 1. Initialize the FastAPI app exactly once
# app = FastAPI(title="AI-SIEM Multi-Dataset Intrusion Detection API")

# # Allows requests from any frontend port (Vite/React)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Severity is derived from the model's attack score (0-1). Tune the bands here.
# SEVERITY_BANDS = ((0.90, "high"), (0.70, "medium"))  # anything lower is "low"

# # Max flagged rows sent to the browser per scan (highest scores first).
# # Summary counts always cover every record. Set MAX_THREAT_ROWS=0 for no limit.
# MAX_THREAT_ROWS = int(os.getenv("MAX_THREAT_ROWS", "5000"))


# class DatasetEnum(str, Enum):
#     nsl_kdd = "nsl_kdd"
#     cicids2017 = "cicids2017"
#     unsw_nb15 = "unsw_nb15"


# class ModelEnum(str, Enum):
#     svm = "svm"
#     knn = "knn"
#     random_forest = "random_forest"
#     naive_bayes = "naive_bayes"
#     decision_tree = "decision_tree"
#     fcnn = "fcnn"
#     cnn = "cnn"
#     lstm = "lstm"


# def sanitize(name):
#     return re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower()


# class EventEncoder:
#     def __init__(self, n_bins=10):
#         self.n_bins = n_bins
#         self.num_cols, self.cat_cols, self.edges = [], [], {}

#     def transform(self, df):
#         cols = []
#         for c in self.num_cols:
#             e = self.edges[c]
#             idx = np.searchsorted(e, df[c].to_numpy(dtype=np.float64), side="left")
#             lut = np.array([f"{sanitize(c)}_b{i}" for i in range(len(e) + 1)], dtype=object)
#             cols.append(lut[idx])
#         for c in self.cat_cols:
#             s = df[c].astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
#             cols.append((sanitize(c) + "_" + s).to_numpy(dtype=object))
#         mat = np.column_stack(cols)
#         return [" ".join(row) for row in mat]

#     def load_state(self, state_dict):
#         self.n_bins = state_dict["n_bins"]
#         self.num_cols = state_dict["num_cols"]
#         self.cat_cols = state_dict["cat_cols"]
#         self.edges = state_dict["edges"]
#         return self


# def clean_dataframe(df):
#     """Replaces Inf/-Inf with NaN, then fills NaNs to prevent JSON serialization errors."""
#     d = df.replace([np.inf, -np.inf], np.nan)
#     for c in d.select_dtypes(include=[np.number]).columns:
#         if d[c].isna().any():
#             d[c] = d[c].fillna(d[c].median())
#     for c in [c for c in d.columns if not pd.api.types.is_numeric_dtype(d[c])]:
#         if d[c].isna().any():
#             mode_val = d[c].mode()
#             d[c] = d[c].fillna(mode_val.iloc[0] if not mode_val.empty else "unknown")
#     return d


# def to_sequence(X, steps=32):
#     n, f = X.shape
#     pad = (-f) % steps
#     if pad:
#         X = np.hstack([X, np.zeros((n, pad), dtype=X.dtype)])
#     return X.reshape(n, steps, -1)


# # ==========================================
# # CACHED LOADERS (restart the server after retraining)
# # ==========================================

# @lru_cache(maxsize=8)
# def load_artifacts(prefix):
#     """Preprocessing artifacts for one dataset, loaded once and reused."""
#     def _load(suffix):
#         with open(os.path.join(BASE_DIR, f"{prefix}_{suffix}.pkl"), "rb") as f:
#             return pickle.load(f)

#     enc = EventEncoder().load_state(_load("event_encoder"))
#     vec = _load("vectorizer")
#     basepoint = _load("basepoint")
#     scaler = _load("scaler")
#     meta = _load("meta")

#     try:
#         feature_names = np.asarray(vec.get_feature_names_out())
#     except Exception:
#         feature_names = None  # explanations are skipped if the vectorizer can't list its features

#     bp = basepoint.toarray() if hasattr(basepoint, "toarray") else basepoint
#     bp_vec = np.asarray(bp, dtype=np.float64).reshape(-1)

#     return {
#         "enc": enc, "vec": vec, "basepoint": basepoint, "scaler": scaler,
#         "meta": meta, "feature_names": feature_names, "bp_vec": bp_vec,
#     }


# @lru_cache(maxsize=12)
# def load_dl_model(path):
#     return tf.keras.models.load_model(path)


# @lru_cache(maxsize=12)
# def load_ml_model(path):
#     return joblib.load(path)


# def preprocess(df, art):
#     """Event Encoding -> TF-IDF -> Cosine Sim -> Scaling -> Clipping."""
#     ev_data = art["enc"].transform(df)
#     X_vec = art["vec"].transform(ev_data)
#     sim = cosine_similarity(X_vec, art["basepoint"]).astype(np.float32)
#     X_dense = X_vec.toarray()
#     X_p = np.hstack([X_dense, sim])
#     X_s = art["scaler"].transform(X_p).astype(np.float32)
#     np.clip(X_s, -art["meta"]["clip"], art["meta"]["clip"], out=X_s)
#     return X_s, X_dense, sim.max(axis=1)


# class InputError(ValueError):
#     """The uploaded file can't be scanned; reported to the user as HTTP 422."""


# def required_columns(art):
#     return [str(c) for c in (*art["enc"].num_cols, *art["enc"].cat_cols)]


# def check_required_columns(df, art, prefix):
#     required = required_columns(art)
#     present = {str(c) for c in df.columns}
#     missing = [c for c in required if c not in present]
#     if missing:
#         shown = ", ".join(repr(c) for c in missing[:10])  # repr exposes stray spaces
#         more = f" (and {len(missing) - 10} more)" if len(missing) > 10 else ""
#         raise InputError(
#             f"Your CSV is missing {len(missing)} of the {len(required)} columns the {prefix} "
#             f"model needs: {shown}{more}. Check that you picked the right dataset."
#         )


# def read_and_prepare(file, prefix):
#     try:
#         df = pd.read_csv(file.file)
#     except Exception as e:
#         raise InputError(f"Could not read the file as a CSV: {e}")
#     if df.empty:
#         raise InputError("The uploaded CSV has no rows.")
        
#     # We clean the df of NaN/Infinity before copying it so df_original is safe to serialize
#     df = clean_dataframe(df)
#     df_original = df.copy()

#     art = load_artifacts(prefix)
#     check_required_columns(df, art, prefix)

#     if "label" in df.columns:
#         df = df.drop(columns=["label"])

#     X_s, X_dense, sim = preprocess(df, art)
#     return df_original, art, X_s, X_dense, sim


# # ==========================================
# # DEDICATED MODEL PREDICTION HANDLERS
# # ==========================================

# def predict_fcnn(model, X_s, dataset_name):
#     """FCNN handler matching Kaggle's high-recall dynamic thresholding."""
#     probs = model.predict(X_s, batch_size=1024, verbose=0).ravel()

#     if dataset_name == "cicids2017":
#         scores = np.power(probs, 0.65)
#         predictions = (scores >= 0.01).astype(int)
#     else:
#         scores = probs
#         predictions = (probs >= 0.5).astype(int)

#     return predictions, scores


# def predict_cnn_lstm(model, X_s, model_type, meta):
#     """CNN and LSTM handler."""
#     if model_type == "cnn":
#         X_input = X_s[..., np.newaxis]
#     elif model_type == "lstm":
#         X_input = to_sequence(X_s, steps=meta.get("lstm_steps", 32))
#     else:
#         X_input = X_s

#     probs = model.predict(X_input, batch_size=1024, verbose=0).ravel()
#     return (probs >= 0.5).astype(int), probs


# def predict_ml_model(model, X_s, want_scores=True):
#     """Classical ML handler (SVM, RF, KNN, ...)."""
#     predictions = np.asarray(model.predict(X_s)).astype(int)
#     scores = None

#     if want_scores and hasattr(model, "predict_proba"):
#         try:
#             proba = model.predict_proba(X_s)
#             classes = list(getattr(model, "classes_", [0, 1]))
#             col = classes.index(1) if 1 in classes else proba.shape[1] - 1
#             scores = proba[:, col]
#         except Exception:
#             scores = None

#     if want_scores and scores is None and hasattr(model, "decision_function"):
#         try:
#             d = np.asarray(model.decision_function(X_s)).ravel()
#             scores = 1.0 / (1.0 + np.exp(-np.clip(d, -30, 30)))  # monotonic, not calibrated
#         except Exception:
#             scores = None

#     if scores is None:
#         scores = predictions.astype(float)
#     return predictions, scores


# def run_model(prefix, model_name, X_s, meta, want_scores=True):
#     if model_name == "fcnn":
#         model = load_dl_model(os.path.join(BASE_DIR, f"{prefix}_dl_fcnn.keras"))
#         return predict_fcnn(model, X_s, prefix)
#     if model_name in ("cnn", "lstm"):
#         model = load_dl_model(os.path.join(BASE_DIR, f"{prefix}_dl_{model_name}.keras"))
#         return predict_cnn_lstm(model, X_s, model_name, meta)
#     model = load_ml_model(os.path.join(BASE_DIR, f"{prefix}_ml_{model_name}.pkl"))
#     return predict_ml_model(model, X_s, want_scores)


# # ==========================================
# # ANALYST HELPERS (score, severity, evidence, metrics)
# # ==========================================

# def severity_of(score):
#     for limit, name in SEVERITY_BANDS:
#         if score >= limit:
#             return name
#     return "low"


# def severity_counts(scores):
#     """Counts per severity band for an array of attack scores."""
#     counts = {name: 0 for _, name in SEVERITY_BANDS}
#     remaining = np.ones(len(scores), dtype=bool)
#     for limit, name in SEVERITY_BANDS:
#         hit = remaining & (scores >= limit)
#         counts[name] = int(hit.sum())
#         remaining &= ~hit
#     counts["low"] = int(remaining.sum())
#     return counts


# def build_threat_meta(threat_idx, scores, sim, X_dense, art, top_k=3, chunk=5000):
#     """One entry per flagged row, in the same order as threat_details."""
#     names, bp = art["feature_names"], art["bp_vec"]
#     can_explain = names is not None and bp.shape[0] == X_dense.shape[1]

#     out = []
#     for s in range(0, len(threat_idx), chunk):
#         part = threat_idx[s:s + chunk]
#         events = [[] for _ in part]
#         if can_explain:
#             delta = X_dense[part] - bp
#             top = np.argsort(-delta, axis=1)[:, :top_k]
#             for r, cols in enumerate(top):
#                 events[r] = [str(names[c]) for c in cols if delta[r, c] > 0]
#         for r, i in enumerate(part):
#             score = float(scores[i])
#             out.append({
#                 "row": int(i) + 1,  # 1-based data row in the uploaded CSV
#                 "score": round(score, 4),
#                 "severity": severity_of(score),
#                 "similarity_to_normal": round(float(sim[i]), 4),
#                 "unusual_events": events[r],
#             })
#     return out


# NORMAL_LABELS = {"normal", "benign", "0", "0.0", "false", "none"}


# def compute_metrics(df_original, predictions):
#     """Accuracy/precision/recall/F1 when the CSV has a 'label' column."""
#     try:
#         col = next((c for c in df_original.columns if str(c).strip().lower() == "label"), None)
#         if col is None:
#             return None
#         series = df_original[col]
#         mask = series.notna().to_numpy()
#         if mask.sum() == 0:
#             return None

#         labelled = series[mask]
#         if pd.api.types.is_numeric_dtype(labelled):
#             y_true = (labelled.astype(float) != 0).astype(int).to_numpy()
#         else:
#             y_true = (~labelled.astype(str).str.strip().str.lower().isin(NORMAL_LABELS)).astype(int).to_numpy()
#         y_pred = np.asarray(predictions).astype(int)[mask]

#         tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
#         return {
#             "label_column": str(col),
#             "labelled_records": int(mask.sum()),
#             "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
#             "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
#             "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
#             "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
#             "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
#         }
#     except Exception:
#         return None  # metrics are optional and must never break a scan


# # ==========================================
# # API ROUTES
# # ==========================================

# @app.get("/health")
# async def health():
#     return {"status": "ok"}


# @app.get("/schema/{dataset}")
# async def schema(dataset: DatasetEnum):
#     """Columns the chosen dataset's model needs, so the UI can check a CSV before upload."""
#     try:
#         art = load_artifacts(dataset.value)
#         return {
#             "dataset": dataset.value,
#             "required_columns": required_columns(art),
#             "optional_columns": ["label"],
#         }
#     except FileNotFoundError:
#         raise HTTPException(status_code=404, detail=f"No saved artifacts found for {dataset.value}.")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/predict")
# async def predict(
#     dataset: DatasetEnum = Form(...),
#     model_type: ModelEnum = Form(...),
#     file: UploadFile = File(...)
# ):
#     try:
#         started = time.time()
#         prefix = dataset.value

#         df_original, art, X_s, X_dense, sim = read_and_prepare(file, prefix)

#         predictions, scores = run_model(prefix, model_type.value, X_s, art["meta"])
#         predictions = np.asarray(predictions).astype(int)
#         scores = np.nan_to_num(np.asarray(scores, dtype=np.float64))

#         # Map predictions and build response (original fields unchanged)
#         label_map = {0: "normal", 1: "attack"}
#         results = [label_map[p] for p in predictions.tolist()]

#         total_records = len(results)
#         threat_count = results.count("attack")
#         normal_count = total_records - threat_count

#         df_original["predicted_status"] = results

#         # Highest-scoring threats first if there are more than the browser should receive
#         threat_idx = np.flatnonzero(predictions == 1)
#         keep = threat_idx
#         if MAX_THREAT_ROWS and len(threat_idx) > MAX_THREAT_ROWS:
#             keep = np.sort(threat_idx[np.argsort(-scores[threat_idx])[:MAX_THREAT_ROWS]])

#         df_threats = df_original.iloc[keep]

#         # FastAPI/JSON cannot serialize NaN values, replace with None
#         df_threats = df_threats.replace({np.nan: None})
#         threat_details = df_threats.to_dict(orient="records")

#         # Per-threat score, severity and evidence (same order as threat_details)
#         threat_meta = build_threat_meta(keep, scores, sim, X_dense, art)

#         return {
#             "summary": {
#                 "total_records": total_records,
#                 "normal_traffic": normal_count,
#                 "detected_threats": threat_count,
#                 "severity_counts": severity_counts(scores[threat_idx]),
#                 "threats_returned": len(threat_details),
#                 "threats_truncated": len(threat_details) < threat_count,
#                 "dataset": prefix,
#                 "model_type": model_type.value,
#                 "elapsed_seconds": round(time.time() - started, 2),
#             },
#             "threat_details": threat_details,
#             "threat_meta": threat_meta,
#             "metrics": compute_metrics(df_original, predictions),
#         }

#     except InputError as e:
#         raise HTTPException(status_code=422, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/compare")
# async def compare(
#     dataset: DatasetEnum = Form(...),
#     file: UploadFile = File(...)
# ):
#     """Run one CSV through every model and report how they differ and agree."""
#     try:
#         prefix = dataset.value
#         df_original, art, X_s, _, _ = read_and_prepare(file, prefix)
#         total = len(df_original)

#         rows, preds_by_model = [], {}
#         for m in ModelEnum:
#             started = time.time()
#             try:
#                 preds, _ = run_model(prefix, m.value, X_s, art["meta"], want_scores=False)
#                 preds = np.asarray(preds).astype(int)
#                 preds_by_model[m.value] = preds

#                 threats = int(preds.sum())
#                 row = {
#                     "model": m.value,
#                     "status": "ok",
#                     "threats": threats,
#                     "normal": total - threats,
#                     "threat_rate": round(100.0 * threats / total, 2),
#                     "seconds": round(time.time() - started, 2),
#                 }
#                 metrics = compute_metrics(df_original, preds)
#                 if metrics:
#                     row["metrics"] = metrics
#             except Exception as e:  # one missing/broken model must not stop the rest
#                 row = {"model": m.value, "status": "error", "detail": str(e)}
#             rows.append(row)

#         agreement = None
#         ok = len(preds_by_model)
#         if ok >= 2:
#             votes = np.sum(list(preds_by_model.values()), axis=0)
#             majority = (votes * 2 > ok).astype(int)
#             agreement = {
#                 "models_compared": ok,
#                 "unanimous_threat": int((votes == ok).sum()),
#                 "unanimous_normal": int((votes == 0).sum()),
#                 "split": int(((votes > 0) & (votes < ok)).sum()),
#             }
#             for row in rows:
#                 if row["status"] == "ok":
#                     same = preds_by_model[row["model"]] == majority
#                     row["agrees_with_majority"] = round(100.0 * float(same.mean()), 2)

#         return {"dataset": prefix, "total_records": total, "models": rows, "agreement": agreement}

#     except InputError as e:
#         raise HTTPException(status_code=422, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))



import os
import re
import time
import pickle
from enum import Enum
from functools import lru_cache
from huggingface_hub import snapshot_download

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = "saved_models"

# ==========================================
# AUTO-DOWNLOAD MODELS FROM HUGGING FACE
# ==========================================
def ensure_models_downloaded():
    """Downloads models from Hugging Face if they don't exist locally (e.g., on Render)."""
    os.makedirs(BASE_DIR, exist_ok=True)
    if not os.listdir(BASE_DIR):
        print("Models directory is empty. Downloading artifacts from Hugging Face...")
        try:
            snapshot_download(
                repo_id="Rithvik-3103/cyber-ann-models",
                repo_type="dataset",
                local_dir=BASE_DIR
            )
            print("Successfully downloaded all models!")
        except Exception as e:
            print(f"Error downloading models: {e}")

# Run the check immediately when the app starts up
ensure_models_downloaded()

# Initialize FastAPI app
app = FastAPI(title="AI-SIEM Multi-Dataset Intrusion Detection API")

# CORS Middleware for React frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SEVERITY_BANDS = ((0.90, "high"), (0.70, "medium"))
MAX_THREAT_ROWS = int(os.getenv("MAX_THREAT_ROWS", "5000"))

class DatasetEnum(str, Enum):
    nsl_kdd = "nsl_kdd"
    cicids2017 = "cicids2017"
    unsw_nb15 = "unsw_nb15"

class ModelEnum(str, Enum):
    svm = "svm"
    knn = "knn"
    random_forest = "random_forest"
    naive_bayes = "naive_bayes"
    decision_tree = "decision_tree"
    fcnn = "fcnn"
    cnn = "cnn"
    lstm = "lstm"

def sanitize(name):
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower()

class EventEncoder:
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.num_cols, self.cat_cols, self.edges = [], [], {}

    def transform(self, df):
        cols = []
        for c in self.num_cols:
            e = self.edges[c]
            idx = np.searchsorted(e, df[c].to_numpy(dtype=np.float64), side="left")
            lut = np.array([f"{sanitize(c)}_b{i}" for i in range(len(e) + 1)], dtype=object)
            cols.append(lut[idx])
        for c in self.cat_cols:
            s = df[c].astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
            cols.append((sanitize(c) + "_" + s).to_numpy(dtype=object))
        mat = np.column_stack(cols)
        return [" ".join(row) for row in mat]

    def load_state(self, state_dict):
        self.n_bins = state_dict["n_bins"]
        self.num_cols = state_dict["num_cols"]
        self.cat_cols = state_dict["cat_cols"]
        self.edges = state_dict["edges"]
        return self

def clean_dataframe(df):
    d = df.replace([np.inf, -np.inf], np.nan)
    for c in d.select_dtypes(include=[np.number]).columns:
        if d[c].isna().any():
            d[c] = d[c].fillna(d[c].median())
    for c in [c for c in d.columns if not pd.api.types.is_numeric_dtype(d[c])]:
        if d[c].isna().any():
            mode_val = d[c].mode()
            d[c] = d[c].fillna(mode_val.iloc[0] if not mode_val.empty else "unknown")
    return d

def to_sequence(X, steps=32):
    n, f = X.shape
    pad = (-f) % steps
    if pad:
        X = np.hstack([X, np.zeros((n, pad), dtype=X.dtype)])
    return X.reshape(n, steps, -1)

@lru_cache(maxsize=8)
def load_artifacts(prefix):
    def _load(suffix):
        with open(os.path.join(BASE_DIR, f"{prefix}_{suffix}.pkl"), "rb") as f:
            return pickle.load(f)

    enc = EventEncoder().load_state(_load("event_encoder"))
    vec = _load("vectorizer")
    basepoint = _load("basepoint")
    scaler = _load("scaler")
    meta = _load("meta")

    try:
        feature_names = np.asarray(vec.get_feature_names_out())
    except Exception:
        feature_names = None

    bp = basepoint.toarray() if hasattr(basepoint, "toarray") else basepoint
    bp_vec = np.asarray(bp, dtype=np.float64).reshape(-1)

    return {
        "enc": enc, "vec": vec, "basepoint": basepoint, "scaler": scaler,
        "meta": meta, "feature_names": feature_names, "bp_vec": bp_vec,
    }

@lru_cache(maxsize=12)
def load_dl_model(path):
    return tf.keras.models.load_model(path)

@lru_cache(maxsize=12)
def load_ml_model(path):
    return joblib.load(path)

def preprocess(df, art):
    ev_data = art["enc"].transform(df)
    X_vec = art["vec"].transform(ev_data)
    sim = cosine_similarity(X_vec, art["basepoint"]).astype(np.float32)
    X_dense = X_vec.toarray()
    X_p = np.hstack([X_dense, sim])
    X_s = art["scaler"].transform(X_p).astype(np.float32)
    np.clip(X_s, -art["meta"]["clip"], art["meta"]["clip"], out=X_s)
    return X_s, X_dense, sim.max(axis=1)

class InputError(ValueError):
    pass

def required_columns(art):
    return [str(c) for c in (*art["enc"].num_cols, *art["enc"].cat_cols)]

def check_required_columns(df, art, prefix):
    required = required_columns(art)
    present = {str(c) for c in df.columns}
    missing = [c for c in required if c not in present]
    if missing:
        shown = ", ".join(repr(c) for c in missing[:10])
        more = f" (and {len(missing) - 10} more)" if len(missing) > 10 else ""
        raise InputError(
            f"Your CSV is missing {len(missing)} of the {len(required)} columns the {prefix} "
            f"model needs: {shown}{more}. Check that you picked the right dataset."
        )

def read_and_prepare(file, prefix):
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise InputError(f"Could not read the file as a CSV: {e}")
    if df.empty:
        raise InputError("The uploaded CSV has no rows.")
        
    df = clean_dataframe(df)
    df_original = df.copy()

    art = load_artifacts(prefix)
    check_required_columns(df, art, prefix)

    if "label" in df.columns:
        df = df.drop(columns=["label"])

    X_s, X_dense, sim = preprocess(df, art)
    return df_original, art, X_s, X_dense, sim

def predict_fcnn(model, X_s, dataset_name):
    probs = model.predict(X_s, batch_size=1024, verbose=0).ravel()
    if dataset_name == "cicids2017":
        scores = np.power(probs, 0.65)
        predictions = (scores >= 0.01).astype(int)
    else:
        scores = probs
        predictions = (probs >= 0.5).astype(int)
    return predictions, scores

def predict_cnn_lstm(model, X_s, model_type, meta):
    if model_type == "cnn":
        X_input = X_s[..., np.newaxis]
    elif model_type == "lstm":
        X_input = to_sequence(X_s, steps=meta.get("lstm_steps", 32))
    else:
        X_input = X_s

    probs = model.predict(X_input, batch_size=1024, verbose=0).ravel()
    return (probs >= 0.5).astype(int), probs

def predict_ml_model(model, X_s, want_scores=True):
    predictions = np.asarray(model.predict(X_s)).astype(int)
    scores = None

    if want_scores and hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_s)
            classes = list(getattr(model, "classes_", [0, 1]))
            col = classes.index(1) if 1 in classes else proba.shape[1] - 1
            scores = proba[:, col]
        except Exception:
            scores = None

    if want_scores and scores is None and hasattr(model, "decision_function"):
        try:
            d = np.asarray(model.decision_function(X_s)).ravel()
            scores = 1.0 / (1.0 + np.exp(-np.clip(d, -30, 30)))
        except Exception:
            scores = None

    if scores is None:
        scores = predictions.astype(float)
    return predictions, scores

def run_model(prefix, model_name, X_s, meta, want_scores=True):
    if model_name == "fcnn":
        model = load_dl_model(os.path.join(BASE_DIR, f"{prefix}_dl_fcnn.keras"))
        return predict_fcnn(model, X_s, prefix)
    if model_name in ("cnn", "lstm"):
        model = load_dl_model(os.path.join(BASE_DIR, f"{prefix}_dl_{model_name}.keras"))
        return predict_cnn_lstm(model, X_s, model_name, meta)
    model = load_ml_model(os.path.join(BASE_DIR, f"{prefix}_ml_{model_name}.pkl"))
    return predict_ml_model(model, X_s, want_scores)

def severity_of(score):
    for limit, name in SEVERITY_BANDS:
        if score >= limit:
            return name
    return "low"

def severity_counts(scores):
    counts = {name: 0 for _, name in SEVERITY_BANDS}
    remaining = np.ones(len(scores), dtype=bool)
    for limit, name in SEVERITY_BANDS:
        hit = remaining & (scores >= limit)
        counts[name] = int(hit.sum())
        remaining &= ~hit
    counts["low"] = int(remaining.sum())
    return counts

def build_threat_meta(threat_idx, scores, sim, X_dense, art, top_k=3, chunk=5000):
    names, bp = art["feature_names"], art["bp_vec"]
    can_explain = names is not None and bp.shape[0] == X_dense.shape[1]

    out = []
    for s in range(0, len(threat_idx), chunk):
        part = threat_idx[s:s + chunk]
        events = [[] for _ in part]
        if can_explain:
            delta = X_dense[part] - bp
            top = np.argsort(-delta, axis=1)[:, :top_k]
            for r, cols in enumerate(top):
                events[r] = [str(names[c]) for c in cols if delta[r, c] > 0]
        for r, i in enumerate(part):
            score = float(scores[i])
            out.append({
                "row": int(i) + 1,
                "score": round(score, 4),
                "severity": severity_of(score),
                "similarity_to_normal": round(float(sim[i]), 4),
                "unusual_events": events[r],
            })
    return out

NORMAL_LABELS = {"normal", "benign", "0", "0.0", "false", "none"}

def compute_metrics(df_original, predictions):
    try:
        col = next((c for c in df_original.columns if str(c).strip().lower() == "label"), None)
        if col is None:
            return None
        series = df_original[col]
        mask = series.notna().to_numpy()
        if mask.sum() == 0:
            return None

        labelled = series[mask]
        if pd.api.types.is_numeric_dtype(labelled):
            y_true = (labelled.astype(float) != 0).astype(int).to_numpy()
        else:
            y_true = (~labelled.astype(str).str.strip().str.lower().isin(NORMAL_LABELS)).astype(int).to_numpy()
        y_pred = np.asarray(predictions).astype(int)[mask]

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return {
            "label_column": str(col),
            "labelled_records": int(mask.sum()),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }
    except Exception:
        return None

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/schema/{dataset}")
async def schema(dataset: DatasetEnum):
    try:
        art = load_artifacts(dataset.value)
        return {
            "dataset": dataset.value,
            "required_columns": required_columns(art),
            "optional_columns": ["label"],
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No saved artifacts found for {dataset.value}.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(
    dataset: DatasetEnum = Form(...),
    model_type: ModelEnum = Form(...),
    file: UploadFile = File(...)
):
    try:
        started = time.time()
        prefix = dataset.value

        df_original, art, X_s, X_dense, sim = read_and_prepare(file, prefix)

        predictions, scores = run_model(prefix, model_type.value, X_s, art["meta"])
        predictions = np.asarray(predictions).astype(int)
        scores = np.nan_to_num(np.asarray(scores, dtype=np.float64))

        label_map = {0: "normal", 1: "attack"}
        results = [label_map[p] for p in predictions.tolist()]

        total_records = len(results)
        threat_count = results.count("attack")
        normal_count = total_records - threat_count

        df_original["predicted_status"] = results

        threat_idx = np.flatnonzero(predictions == 1)
        keep = threat_idx
        if MAX_THREAT_ROWS and len(threat_idx) > MAX_THREAT_ROWS:
            keep = np.sort(threat_idx[np.argsort(-scores[threat_idx])[:MAX_THREAT_ROWS]])

        df_threats = df_original.iloc[keep]
        df_threats = df_threats.replace({np.nan: None})
        threat_details = df_threats.to_dict(orient="records")

        threat_meta = build_threat_meta(keep, scores, sim, X_dense, art)

        return {
            "summary": {
                "total_records": total_records,
                "normal_traffic": normal_count,
                "detected_threats": threat_count,
                "severity_counts": severity_counts(scores[threat_idx]),
                "threats_returned": len(threat_details),
                "threats_truncated": len(threat_details) < threat_count,
                "dataset": prefix,
                "model_type": model_type.value,
                "elapsed_seconds": round(time.time() - started, 2),
            },
            "threat_details": threat_details,
            "threat_meta": threat_meta,
            "metrics": compute_metrics(df_original, predictions),
        }

    except InputError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare")
async def compare(
    dataset: DatasetEnum = Form(...),
    file: UploadFile = File(...)
):
    try:
        prefix = dataset.value
        df_original, art, X_s, _, _ = read_and_prepare(file, prefix)
        total = len(df_original)

        rows, preds_by_model = [], {}
        for m in ModelEnum:
            started = time.time()
            try:
                preds, _ = run_model(prefix, m.value, X_s, art["meta"], want_scores=False)
                preds = np.asarray(preds).astype(int)
                preds_by_model[m.value] = preds

                threats = int(preds.sum())
                row = {
                    "model": m.value,
                    "status": "ok",
                    "threats": threats,
                    "normal": total - threats,
                    "threat_rate": round(100.0 * threats / total, 2),
                    "seconds": round(time.time() - started, 2),
                }
                metrics = compute_metrics(df_original, preds)
                if metrics:
                    row["metrics"] = metrics
            except Exception as e:
                row = {"model": m.value, "status": "error", "detail": str(e)}
            rows.append(row)

        agreement = None
        ok = len(preds_by_model)
        if ok >= 2:
            votes = np.sum(list(preds_by_model.values()), axis=0)
            majority = (votes * 2 > ok).astype(int)
            agreement = {
                "models_compared": ok,
                "unanimous_threat": int((votes == ok).sum()),
                "unanimous_normal": int((votes == 0).sum()),
                "split": int(((votes > 0) & (votes < ok)).sum()),
            }
            for row in rows:
                if row["status"] == "ok":
                    same = preds_by_model[row["model"]] == majority
                    row["agrees_with_majority"] = round(100.0 * float(same.mean()), 2)

        return {"dataset": prefix, "total_records": total, "models": rows, "agreement": agreement}

    except InputError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))