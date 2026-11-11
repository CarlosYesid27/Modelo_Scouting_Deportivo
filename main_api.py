from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, StrictInt
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = (
    Path(__file__).resolve().parent / "models" / "modelo_scouting_rf.joblib"
)


def preprocesar_y_predecir(nuevos_jugadores_df, ruta_paquete=MODEL_PATH):
    # 1. Cargar el paquete serializado
    paquete = joblib.load(ruta_paquete)
    modelo = paquete['modelo']
    columnas_entrenamiento = paquete['columnas_X']

    df = nuevos_jugadores_df.copy()

    # 2. Aplicar las MISMAS reglas de imputación aprendidas en entrenamiento
    df['estatura_cm'] = pd.to_numeric(df['estatura_cm'], errors='coerce')
    df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
    df['posicion'] = df['posicion'].fillna('Sin Posicion')

    df['estatura_cm'] = df['estatura_cm'].fillna(df['posicion'].map(paquete['medianas_estatura']))
    df['estatura_cm'] = df['estatura_cm'].fillna(paquete['mediana_global_estatura'])
    df['edad'] = df['edad'].fillna(paquete['mediana_global_edad'])

    # 3. Formatear variables categóricas alineadas con la matriz original
    # CORRECCIÓN: SIN drop_first. Con drop_first=True, pandas descartaría la primera
    # categoría del lote nuevo (no la del entrenamiento) y un Defender quedaría
    # codificado como Attack. El reindex de abajo descarta las columnas sobrantes.
    df_dummies = pd.get_dummies(df[['posicion']], dtype=int)
    X_nuevo = pd.concat([df[['edad', 'estatura_cm']], df_dummies], axis=1)

    # Reindexar para asegurar que contenga exactamente las mismas columnas que vio el modelo
    X_nuevo = X_nuevo.reindex(columns=columnas_entrenamiento, fill_value=0)

    # 4. Generar inferencias y probabilidades
    predicciones = modelo.predict(X_nuevo)
    probabilidades = modelo.predict_proba(X_nuevo)[:, 1]

    df['Prediccion_Clase'] = np.where(predicciones == 1, 'Élite', 'Estándar')
    df['Probabilidad_Elite'] = np.round(probabilidades * 100, 2)

    return df


class PlayerRequest(BaseModel):
    jugador: str = Field(..., min_length=1, description="Nombre del jugador")
    posicion: Literal["Attack", "Defender", "Goalkeeper", "Midfield"] = Field(
        ..., description="Posición conocida por el modelo"
    )
    estatura_cm: Optional[float] = Field(
        default=None, ge=140, le=230, description="Estatura en centímetros (140 a 230 cm)"
    )
    edad: StrictInt = Field(..., ge=17, le=45, description="Edad entera en años (17 a 45)")

    @field_validator("estatura_cm", mode="before")
    @classmethod
    def validar_estatura(cls, valor):
        if valor is None or valor == "":
            return None
        if isinstance(valor, bool):
            raise ValueError("La estatura debe ser un número válido.")
        if isinstance(valor, str):
            texto = valor.strip()
            # Aceptar 180,5 y 180.5; no aceptar mezclas o miles.
            import re
            if not re.fullmatch(r"[0-9]+(?:[.,][0-9]+)?", texto):
                raise ValueError("La estatura debe ser numérica; usa coma o punto para decimales.")
            return float(texto.replace(",", "."))
        return valor


class PredictionResponse(BaseModel):
    jugador: str
    prediccion: str
    porcentaje_probabilidad_elite: float


app = FastAPI(
    title="API de Scouting Deportivo",
    description="Predice si un jugador pertenece al segmento de élite.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Confirma que la API está disponible y que el modelo existe."""
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"El modelo no está disponible en {MODEL_PATH}",
        )
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(player: PlayerRequest) -> PredictionResponse:
    """Predice la clase y la probabilidad de élite para un jugador."""
    data = pd.DataFrame([player.model_dump()])
    result = preprocesar_y_predecir(data, ruta_paquete=str(MODEL_PATH)).iloc[0]

    return PredictionResponse(
        jugador=str(result["jugador"]),
        prediccion=str(result["Prediccion_Clase"]),
        porcentaje_probabilidad_elite=float(result["Probabilidad_Elite"]),
    )
