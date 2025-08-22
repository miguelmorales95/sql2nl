# SQL → Natural Language Translator (Redshift-focused)

A tiny, portfolio-ready project that turns **Amazon Redshift SQL** into plain-English explanations.

It includes two inference modes:
1. **Heuristic Baseline (no ML)** – quick, dependency-light, and works out-of-the-box.
2. **Seq2Seq LLM (T5-small)** – optional fine-tuning with Hugging Face Transformers to learn fluent, context-aware descriptions.

Perfect for a GitHub repo that *demonstrates you can build and ship an LLM-backed tool* while also being immediately usable without training.

---

## Why this project?
- Real-world usefulness for **analysts & data engineers** reviewing queries.
- Showcases **Redshift-specific** concepts like `DISTKEY`, `SORTKEY`, `COPY`, `UNLOAD`, `QUALIFY`, `DISTSTYLE`, Spectrum external schemas, and window functions.
- Clear engineering: CLI, API, tests, training loop, eval, docs.

## Quickstart (Baseline – no GPU, no training)

```bash
pip install -r requirements.txt
python -m sql2nl_redshift.infer --sql "SELECT user_id, COUNT(*) AS orders FROM public.orders GROUP BY 1 ORDER BY orders DESC LIMIT 10;"
```

Output (example):
```
Returns the top 10 users by number of orders from public.orders.
```

## LLM Training (Optional)

```bash
# 1) Install deps
pip install -r requirements.txt

# 2) Fine-tune a small model (T5-small by default)
python -m sql2nl_redshift.train   --train sql2nl_redshift/data/sample_pairs.jsonl   --val sql2nl_redshift/data/sample_pairs.jsonl   --outdir artifacts/t5-small-redshift   --epochs 2   --batch_size 8

# 3) Inference with the fine-tuned model
python -m sql2nl_redshift.infer   --sql "SELECT COUNT(*) FROM public.events WHERE event_type = 'purchase';"   --model_dir artifacts/t5-small-redshift
```

## REST API

```bash
uvicorn sql2nl_redshift.api:app --reload --port 8000
# Then: POST http://localhost:8000/translate
# Body: { "sql": "SELECT * FROM public.users LIMIT 5;" }
```

## Redshift Focus
- Understands/mentions **`QUALIFY`**, **window functions**, **Spectrum** external schemas, **DISTKEY/SORTKEY** implications (where appropriate), **DATEADD/DATE_TRUNC**, **COPY/UNLOAD** (described when seen), and **system tables** like `STL_`/`SVV_`.
- Not a full SQL parser—keeps it pragmatic and explainable for resumes and interviews.

## Project Structure
```
sql2nl_redshift/
  ├── sql2nl_redshift/
  │   ├── __init__.py
  │   ├── heuristics.py
  │   ├── model.py
  │   ├── infer.py
  │   ├── train.py
  │   ├── eval.py
  │   ├── api.py
  │   ├── prompts.py
  │   ├── redshift_terms.py
  │   └── data/
  │       └── sample_pairs.jsonl
  ├── tests/
  │   └── test_heuristics.py
  ├── requirements.txt
  ├── setup.cfg
  ├── .gitignore
  ├── LICENSE
  └── README.md
```

## Notes
- You can expand the dataset with your own SQL/NL pairs from your analytics work.
- The baseline translator provides safe/consistent output even if the LLM isn't available.

**License:** MIT
