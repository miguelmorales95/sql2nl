import argparse, json
from datasets import load_dataset
import evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=False, help="JSONL with fields: nl_pred, nl_true")
    ap.add_argument("--pairs", required=False, help="JSONL with fields: sql, nl (evaluates heuristics)")
    args = ap.parse_args()

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")

    if args.pairs:
        # Evaluate heuristic baseline directly
        from .heuristics import explain_sql
        ds = load_dataset("json", data_files=args.pairs, split="train")
        preds, refs = [], []
        for ex in ds:
            preds.append(explain_sql(ex["sql"]))
            refs.append(ex["nl"])
    else:
        ds = load_dataset("json", data_files=args.preds, split="train")
        preds = [ex["nl_pred"] for ex in ds]
        refs = [ex["nl_true"] for ex in ds]

    rouge_res = rouge.compute(predictions=preds, references=refs)
    bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs])

    print("ROUGE:", rouge_res)
    print("BLEU:", bleu_res)
