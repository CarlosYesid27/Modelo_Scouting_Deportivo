# Modelo de Scouting Deportivo

Aplicación que estima si un futbolista pertenece al segmento **Élite** o **Estándar** a partir de su edad, estatura y posición. El modelo Random Forest se entrenó con registros de jugadores de la base MySQL `transfermarket`. Durante el entrenamiento, la clase Élite se asigna a los jugadores cuyo valor de mercado alcanza o supera el percentil 75 del conjunto de datos.

## Ejecutar en local

Necesitas Git y Python 3.11 o posterior. Desde PowerShell, clona el repositorio y prepara el entorno de Python.

```powershell
git clone https://github.com/CarlosYesid27/Modelo_Scouting_Deportivo.git
cd Modelo_Scouting_Deportivo
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install fastapi uvicorn pandas numpy joblib scikit-learn==1.9.0
.\.venv\Scripts\python.exe -m uvicorn main_api:app --host 127.0.0.1 --port 8000
```

Con la API en ejecución, abre `index.html` en el navegador. También puedes probar las solicitudes en la [documentación interactiva de FastAPI](http://127.0.0.1:8000/docs). La API usa el modelo incluido en `models/modelo_scouting_rf.joblib` y funciona sin conexión a MySQL. La versión de scikit-learn indicada coincide con la que quedó registrada en el modelo.

## API

| Método y ruta | Descripción |
| --- | --- |
| `GET /health` | Comprueba que el archivo del modelo está disponible. |
| `POST /predict` | Devuelve el nombre, la clase estimada y la probabilidad de pertenecer al segmento Élite en una escala de 0 a 100. |

Ejemplo de solicitud a `POST /predict`.

```json
{
  "jugador": "Jugador de ejemplo",
  "posicion": "Midfield",
  "estatura_cm": 180,
  "edad": 24
}
```

La edad debe ser un entero entre 17 y 45 años. Las posiciones admitidas son `Attack`, `Defender`, `Goalkeeper` y `Midfield`. La estatura es opcional y, si se indica, debe estar entre 140 y 230 cm. Si se omite, la API usa la mediana de la posición calculada durante el entrenamiento y recurre a la mediana general si hace falta.

## Volver a entrenar el modelo

Instala las dependencias adicionales.

```powershell
.\.venv\Scripts\python.exe -m pip install matplotlib python-dotenv sqlalchemy pymysql
```

Configura un archivo `.env` en la raíz con los datos de conexión de tu instancia de MySQL.

```dotenv
DB_USER=tu_usuario
DB_PASSWORD=tu_contraseña
DB_HOST=localhost
DB_PORT=3306
DB_NAME=transfermarket
```

La base debe tener una tabla `players` con las columnas `player_id`, `name`, `position`, `height_in_cm`, `date_of_birth` y `market_value_in_eur`. Ejecuta el script desde la raíz del repositorio.

```powershell
.\.venv\Scripts\python.exe exportar_modelo.py
```

El script reemplaza `models/modelo_scouting_rf.joblib` con el modelo recién entrenado.
