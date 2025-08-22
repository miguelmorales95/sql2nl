import argparse, os, json, math
from typing import Dict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from .prompts import INSTRUCTION_TEMPLATE

def _format_example(example: Dict):
    sql = example["sql"]
    nl = example["nl"]
    return {
        "input_text": INSTRUCTION_TEMPLATE.format(sql=sql),
        "target_text": nl
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to JSONL with fields: sql, nl")
    ap.add_argument("--val", required=True, help="Path to JSONL with fields: sql, nl")
    ap.add_argument("--outdir", required=True, help="Where to save the model")
    ap.add_argument("--base_model", default="t5-small")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    train_ds = load_dataset("json", data_files=args.train, split="train")
    val_ds = load_dataset("json", data_files=args.val, split="train")

    train_ds = train_ds.map(_format_example)
    val_ds = val_ds.map(_format_example)

    def tokenize(ex):
        model_inputs = tokenizer(ex["input_text"], truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(ex["target_text"], truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = TrainingArguments(
        output_dir=os.path.join(args.outdir, "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        report_to=[],
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.outdir)
    tokenizer.save_pretrained(args.outdir)
    print(f"Saved model to {args.outdir}")

if __name__ == "__main__":
    main()
