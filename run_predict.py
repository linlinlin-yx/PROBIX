import argparse
import json
import os

from src.prediction_utils import ensure_prediction_target_available, get_prediction_call_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run prediction on saved adversarial examples.")
    parser.add_argument("--input_file", default="generated_adv_ex.jsonl")
    parser.add_argument("--output_file", default="prediction_results.jsonl")
    parser.add_argument("--model_name", default=None, help="Override base model name for prediction routing.")
    parser.add_argument("--transfer_mode", choices=["on", "off"], default="off")
    parser.add_argument("--target_model", default=None)
    parser.add_argument("--hf_local_path", default=None, help="Optional local Mistral-7B-Instruct-v0.3 directory for prediction.")
    return parser.parse_args()


def load_records(path):
    records = []
    with open(path, "r") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    args = parse_args()
    records = load_records(args.input_file)
    if not records:
        raise ValueError(f"No records found in {args.input_file}")

    base_model = args.model_name or records[0]["model_name"]
    ensure_prediction_target_available(base_model, transfer_mode=args.transfer_mode, target_model=args.target_model, hf_local_path=args.hf_local_path)

    successful_attacks = 0
    n_empty = 0
    n_unchange = 0
    output_records = []

    for record in records:
        prediction = get_prediction_call_model(
            base_model,
            record["paraphrased_prompt"],
            dataset=record["dataset_name"],
            transfer_mode=args.transfer_mode,
            target_model=args.target_model,
            hf_local_path=args.hf_local_path,
        )
        is_success = prediction != record["orig_pred"]
        if prediction is None:
            n_empty += 1
        elif prediction == record["orig_pred"]:
            n_unchange += 1
        if is_success:
            successful_attacks += 1

        output_record = dict(record)
        output_record["new_pred"] = prediction
        output_record["attack_success"] = is_success
        output_records.append(output_record)

    with open(args.output_file, "w") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    failed_attacks = len(output_records) - successful_attacks
    asr = (successful_attacks / len(output_records)) * 100 if output_records else 0

    avg_similarity_all = sum(record["similarity"] for record in output_records) / len(output_records) if output_records else 0.0
    avg_perplexity_all = sum(record["perplexity"] for record in output_records) / len(output_records) if output_records else 0.0
    avg_perplexity_opt_all = sum(record["perplexity_opt"] for record in output_records) / len(output_records) if output_records else 0.0

    successful_records = [record for record in output_records if record["attack_success"]]
    avg_similarity_success = sum(record["similarity"] for record in successful_records) / len(successful_records) if successful_records else 0.0
    avg_perplexity_success = sum(record["perplexity"] for record in successful_records) / len(successful_records) if successful_records else 0.0
    avg_perplexity_opt_success = sum(record["perplexity_opt"] for record in successful_records) / len(successful_records) if successful_records else 0.0

    summary = {
        "total_candidates": len(output_records),
        "successful_attacks": successful_attacks,
        "failed_attacks": failed_attacks,
        "attack_success_rate": asr,
        "n_empty": n_empty,
        "n_unchange": n_unchange,
        "avg_similarity_all": avg_similarity_all,
        "avg_perplexity_all": avg_perplexity_all,
        "avg_perplexity_opt_all": avg_perplexity_opt_all,
        "avg_similarity_success": avg_similarity_success,
        "avg_perplexity_success": avg_perplexity_success,
        "avg_perplexity_opt_success": avg_perplexity_opt_success,
        "transfer_mode": args.transfer_mode,
        "target_model": args.target_model,
        "hf_local_path": args.hf_local_path,
        "base_model": base_model,
    }
    summary_path = os.path.splitext(args.output_file)[0] + "_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved predictions to {args.output_file}")
    print(f"Saved summary to {summary_path}")
    print(f"Attack Success Rate: {asr:.2f}%")


if __name__ == "__main__":
    main()
