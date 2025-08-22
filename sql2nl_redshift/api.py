from fastapi import FastAPI
from pydantic import BaseModel
from .heuristics import explain_sql as baseline_explain
from .model import Seq2SeqExplainer

app = FastAPI(title="SQL→NL (Redshift) Translator")

class Payload(BaseModel):
    sql: str
    model_dir: str | None = None

@app.post("/translate")
def translate(payload: Payload):
    sql = payload.sql
    if payload.model_dir:
        try:
            explainer = Seq2SeqExplainer(model_dir=payload.model_dir)
            return {"explanation": explainer.predict(sql), "mode": "llm"}
        except Exception as e:
            # Fallback
            return {"explanation": baseline_explain(sql), "mode": "baseline", "warning": str(e)}
    return {"explanation": baseline_explain(sql), "mode": "baseline"}
