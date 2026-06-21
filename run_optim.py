import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="Run PROBIX adversarial suffix optimization.")
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset_name", default="sst2")
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--similarity_threshold", type=float, default=0.5)
    parser.add_argument("--suffix_len", type=int, default=5)
    parser.add_argument("--ppl_threshold", type=float, default=5000.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--transfer_mode", choices=["on", "off"], default="off")
    parser.add_argument(
        "--target_model",
        default=None,
        help="Optional target model. In transfer_mode=on this selects the black-box target; in transfer_mode=off it can override the default same-model predictor.",
    )
    return parser.parse_args()


args = parse_args()

import time
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, AutoModelForMaskedLM, AutoTokenizer, BertForSequenceClassification, BertTokenizer, GPT2LMHeadModel, GPT2TokenizerFast
from src.dataset_utils import load_dataset_custom
from src.prompt_optim import compute_perplexity, compute_use_similarity, optimize_adversarial_suffix
from src.generate_and_save_category_ids import select_candidate_ids_for_agnews,load_category_ids, load_emotional_words, select_candidate_ids
from src.prediction_utils import ensure_prediction_target_available, get_prediction_call_model
from sentence_transformers import SentenceTransformer
from run_paraphrase import get_paraphrase_text

print('Initialize experimental configuration')
# Validate the evaluation backend before loading large models.
ensure_prediction_target_available(
    args.model_name,
    transfer_mode=args.transfer_mode,
    target_model=args.target_model,
)

use_model = SentenceTransformer('all-MiniLM-L6-v2')
model_name = args.model_name
dataset_name = args.dataset_name
num_examples = args.num_examples
n_epochs = args.n_epochs
similarity_threshold = args.similarity_threshold
suffix_len = args.suffix_len

#load model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
bert_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = '[PAD]'

gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
if gpt2_tokenizer.pad_token is None:
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

if args.dataset_name == "AG-News":
    classifier_model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news", num_labels=4)
    classifier_tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news")
else:
    classifier_model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2", num_labels=2)
    classifier_tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")

# Transfer to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
bert_model.to(device)
gpt2_model.to(device)
classifier_model.to(device)
use_model.to(device)

# load probing based classifier_probe
classifier_probe_path = f"./src/probe/classifier_probe_{args.model_name.replace('/', '_')}_{args.dataset_name}.pt"
num_labels = 4 if args.dataset_name == "AG-News" else 2
classifier_probe = torch.nn.Linear(model.config.hidden_size, num_labels).to(device)
if os.path.exists(classifier_probe_path):
    classifier_probe.load_state_dict(torch.load(classifier_probe_path))
    print(f"Loaded classifier probe from {classifier_probe_path}")
else:
    print(f"Classifier probe not found at {classifier_probe_path}. Please train the classifier probe first.")
    raise FileNotFoundError(f"Classifier probe not found at {classifier_probe_path}")
classifier_probe.eval()
# Load datasets such as ag_news, sst2, strategyqa
dataset_class = load_dataset_custom(args)

# save results function
def save_attack_results(successful_attacks, failed_attacks, total_samples, asr, attack_results, n_empty, n_unchange, n_parafail, execution_time, execution_time_data, model_name, filepath_prefix="attack_results"):
    filepath = f"{filepath_prefix}_{model_name.replace('/', '_')}_{args.dataset_name}.txt"
    avg_similarity = sum(result['similarity'] for result in attack_results) / len(attack_results) if attack_results else 0.0
    avg_perplexity = sum(result['perplexity'] for result in attack_results) / len(attack_results) if attack_results else 0.0
    avg_perplexity_before = sum(result['perplexity_opt'] for result in attack_results) / len(attack_results) if attack_results else 0.0
    print(f"Average Similarity: {avg_similarity:.4f}")
    print(f"Average Perplexity: {avg_perplexity:.4f}")
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Execution Time for Adv examples: {execution_time_data:.2f} seconds")

    with open(filepath, "w") as f:
        f.write(f"Successful Attacks: {successful_attacks}\n")
        f.write(f"Failed Attacks: {failed_attacks}\n")
        f.write(f"  Label unchanged ({n_unchange})\n")
        f.write(f"  Paraphrasing failed ({n_parafail})\n")
        f.write(f"Total Samples: {total_samples}\n")
        f.write(f"Attack Success Rate: {asr:.2f}%\n")
        f.write(f"Execution Time: {execution_time:.2f} seconds\n")
        f.write(f"Execution Time for Adv examples: {execution_time_data:.2f} seconds\n")
        f.write(f"similarity_threshold: {similarity_threshold}\n")
        f.write(f"suffix_length: {suffix_len}\n")
        f.write(f"Number of iterations: {n_epochs}\n")
        f.write("\nSuccessful Attack Details:\n")
        f.write(f"Average Similarity: {avg_similarity:.4f}\n")
        f.write(f"Average Perplexity: {avg_perplexity:.4f}\n")
        f.write(f"Average Perplexity before paraphrase: {avg_perplexity_before:.4f}\n")
        for result in attack_results:
            f.write(f"Sample {result['sample']}: {result['original_prompt']} -> {result['paraphrased_prompt']} (Epoch {result['epoch']})\n")
            f.write(f"  Similarity: {result['similarity']:.4f}\n")
            f.write(f"  Perplexity: {result['perplexity']:.4f}\n")
    print(f"Results saved to {filepath}")


# Main Optimization Call
successful_attacks = 0
total_samples = len(dataset_class)
attack_results = []

n_empty = 0
n_unchange = 0
n_parafail = 0

print("Starting adversarial optimization...")
start_time = time.time()
category_ids, positive_ids, negative_ids = load_emotional_words(bert_tokenizer, dataset_name)

if args.dataset_name == "AG-News":
    category_ids = load_category_ids()

execution_time_data = 0
for i, (prompt_text, ground_truth_label) in enumerate(dataset_class):
    start_time_data = time.time()
    print(f"Processing sample {i+1}/{total_samples}")
    if args.dataset_name == "AG-News":
        # ”0” for World, ”1” for Sports, ”2” for Business, and ”3” for Sci/Tech
        label_priority = {
            0: [1, 2, 3],
            1: [0, 2, 3],
            2: [0, 1, 3],
            3: [2, 0, 1],
        }
        target_label = label_priority[ground_truth_label][0]
        candidate_ids = select_candidate_ids_for_agnews(bert_tokenizer, target_label, category_ids)
    else:
        target_label = 0 if ground_truth_label == 1 else 1
        candidate_ids = select_candidate_ids(dataset_name, target_label, positive_ids, negative_ids)
        category_ids = None

    optimized_text = optimize_adversarial_suffix(
        model=model,
        tokenizer=tokenizer,
        bert_model=bert_model,
        bert_tokenizer=bert_tokenizer,
        gpt2_model = gpt2_model,
        gpt2_tokenizer = gpt2_tokenizer,
        classifier_model=classifier_model,
        classifier_tokenizer=classifier_tokenizer,
        orig_prompt=prompt_text,
        orig_pred=ground_truth_label,
        target_label=target_label,
        dataset_name=args.dataset_name,
        suffix_len=suffix_len,
        similarity_threshold=args.similarity_threshold,
        ppl_threshold=args.ppl_threshold,
        n_steps=n_epochs,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        use_model=use_model,
        candidate_ids=candidate_ids,
        classifier_probe=classifier_probe
    )
    paraphrased_text = get_paraphrase_text(args.dataset_name, optimized_text)
    if paraphrased_text is None:
        print("Paraphrasing failed: return None.")
        n_parafail += 1
        continue
    end_time_data = time.time()
    execution_time_data = execution_time_data + end_time_data - start_time_data

    adv_examples_file = "all_attacks_examples.txt"
    with open(adv_examples_file, "a") as f:
        f.write(f"Original Prompt: {prompt_text} | Optimized Prompt: {optimized_text} | Paraphrased Prompt: {paraphrased_text} | Orig_pred: {ground_truth_label}\n")

    similarity = compute_use_similarity(prompt_text, paraphrased_text, use_model)
    perplexity = compute_perplexity(gpt2_model, gpt2_tokenizer, paraphrased_text, device)
    ppl_before = compute_perplexity(gpt2_model, gpt2_tokenizer, optimized_text, device)
    
    api_new_pred = get_prediction_call_model(
        model_name,
        paraphrased_text,
        dataset=args.dataset_name,
        transfer_mode=args.transfer_mode,
        target_model=args.target_model,
    )
    if api_new_pred != ground_truth_label:
        successful_attacks += 1
        print(f"Attack succeeded: {prompt_text} -> {paraphrased_text} ")
        print(f"(Similarity: {similarity:.4f}, Perplexity: {perplexity:.2f})")
        attack_results.append({
            "sample": i+1,
            "original_prompt": prompt_text,
            "optimized_prompt": optimized_text,
            "paraphrased_prompt": paraphrased_text,
            "epoch": n_epochs,
            "similarity": similarity,
            "perplexity": perplexity,
            "perplexity_opt": ppl_before,
        })
        if api_new_pred is None :
            print('Empty return for prediction!')
            n_empty += 1
        success_file = "successful_attacks.txt"
        with open(success_file, "a") as f:
            f.write(f"Original Prompt: {prompt_text} | Optimized Prompt: {optimized_text} | Paraphrased Prompt: {paraphrased_text} | Orig_pred: {ground_truth_label} | New_pred: {api_new_pred}\n")
        
    else:
        if api_new_pred == ground_truth_label :
            print("Prediction unchanged.")
            n_unchange += 1
        print(f"Attack failed: {prompt_text} -> {paraphrased_text} (Similarity: {similarity:.4f})")
    

end_time = time.time()
execution_time = end_time - start_time
failed_attacks = total_samples - successful_attacks
asr = (successful_attacks / total_samples) * 100 if total_samples > 0 else 0
print(f"Optimization finished.")
save_attack_results(successful_attacks, failed_attacks, total_samples, asr, attack_results, n_empty, n_unchange, n_parafail, execution_time, execution_time_data, model_name)