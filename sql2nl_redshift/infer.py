import argparse
from .heuristics import explain_sql as baseline_explain
from .model import Seq2SeqExplainer

def main():
    p = argparse.ArgumentParser(description="Translate Redshift SQL to natural language")
    p.add_argument("--sql", required=True, help="SQL string to explain")
    p.add_argument("--model_dir", default=None, help="Path to fine-tuned seq2seq model directory (optional)")
    args = p.parse_args()

    if args.model_dir:
        try:
            explainer = Seq2SeqExplainer(model_dir=args.model_dir)
            print(explainer.predict(args.sql))
            return
        except Exception as e:
            print(f"[LLM mode unavailable: {e}] Falling back to heuristic baseline.\n")

    print(baseline_explain(args.sql))

if __name__ == "__main__":
    main()
