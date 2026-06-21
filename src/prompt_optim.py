# %%

import os
import random
import json
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
import nltk
import torch
import torch.nn.functional as F
#import textattack

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('sentiwordnet')

# Main optimization function
def optimize(
    bert_model,
    classifier_model,
    classifier_tokenizer,
    bert_tokenizer,
    gpt2_model,
    gpt2_tokenizer,
    input_text: str,
    orig_pred: int,
    target_label: int,
    device,
    max_replacements: int = 30,
    similarity_threshold: float = 8.0,
    ppl_threshold: float = 5000.0,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.1,
    use_model=None,
    model=None,
    tokenizer=None,
    classifier_probe=None,
    candidate_ids=None,
    suffix_len: int = 20
):
    classifier_model.eval()
    bert_model.eval()
    gpt2_model.eval()
    model.eval()

    #Encoding
    inputs = bert_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    #Conjunction Handling 
    connector_file = "connector_structures.json"
    if not os.path.exists(connector_file):
        raise FileNotFoundError(f"Connector structures file not found at {connector_file}.")
    with open(connector_file, "r") as f:
        connector_structures = json.load(f)

    # Use sber to select the connecting phrase that best matches the input text
    prompt_embedding = use_model.encode(input_text, convert_to_tensor=True)
    connector_phrases = [phrase for phrase, _ in connector_structures]
    connector_embeddings = use_model.encode(connector_phrases, convert_to_tensor=True)
    similarities = torch.nn.functional.cosine_similarity(prompt_embedding, connector_embeddings, dim=-1)
    
    # Select the connecting phrase with the highest similarity
    structure_idx = torch.argmax(similarities).item()
    print(f"Selected connector phrase: '{connector_phrases[structure_idx]}' with similarity {similarities[structure_idx]:.4f}")
    _, structure_tokens = connector_structures[structure_idx]
    structure_token_ids = [bert_tokenizer.encode(word, add_special_tokens=False)[0] for word in structure_tokens]

    seen_tokens = set(structure_token_ids)
    insert_ids = structure_token_ids.copy()

    # Initialize the adversarial suffix
    num_candidates = suffix_len
    available_tokens = [tid for tid in candidate_ids if tid not in seen_tokens]
    num_candidates = min(num_candidates, len(available_tokens))
    for _ in range(num_candidates):
        candidate_idx = torch.randint(0, len(available_tokens), (1,)).item()
        insert_ids.append(available_tokens[candidate_idx])
        seen_tokens.add(available_tokens[candidate_idx])
        available_tokens.pop(candidate_idx)

    #Concatenate into adv_input_ids
    orig_ids_list = input_ids[0].tolist()
    adv_input_ids = orig_ids_list[:-1] + insert_ids + [orig_ids_list[-1]]
    adv_input_ids = torch.tensor(adv_input_ids, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(adv_input_ids).to(device)

    
    
    # Main optimization process
    replacements = 0
    while replacements < max_replacements:
    # for replacements in range(max_replacements):
        current_text = bert_tokenizer.decode(adv_input_ids[0], skip_special_tokens=True)
        #Calculate perplexity loss
        ppl_loss, perplexity = compute_perplexity_loss(gpt2_model, gpt2_tokenizer, current_text, device)
        print(f"start Replacement {replacements+1} perplexity={perplexity.item():.4f}")
        
        #Calculate similarity loss
        sim_loss = 1.0 - compute_use_similarity(input_text, current_text, use_model)

        #Calculate adv_loss
        embeddings = model.get_input_embeddings()(adv_input_ids)
        embeddings.retain_grad()

        m_logits = get_logits_with_embeddings(model, classifier_probe, embeddings=embeddings)
        target_tensor = torch.tensor([target_label], device=device)
        adv_loss = F.cross_entropy(m_logits, target_tensor)
        
        #total loss
        total_loss = alpha * adv_loss + beta * sim_loss + gamma * ppl_loss
        print(f"Iteration {replacements + 1}: adv_loss: {alpha * adv_loss:.4f}, sim_loss: {beta * sim_loss:.4f}, ppl_loss: {gamma * ppl_loss:.4f}, total_loss: {total_loss:.4f}")
        
        #Initialize parameters needed for replacement
        gradients = torch.autograd.grad(outputs=total_loss, inputs=embeddings, retain_graph=False)[0]
        grad = gradients.detach()
        embedding_matrix = model.get_input_embeddings().weight.detach()
        best_loss = total_loss.item()
        best_pos = -1
        best_token = None
        suffix_start = len(orig_ids_list[:-1]) + len(structure_token_ids)
        suffix_end = len(orig_ids_list[:-1]) + len(structure_token_ids) + num_candidates
        max_end = adv_input_ids.size(1) - 1
        for pos in range(suffix_start, min(suffix_end, max_end)):
            if attention_mask[0, pos] == 0:
                continue

            orig_token = adv_input_ids[0, pos].item()
            orig_embedding = embedding_matrix[orig_token]
            grad_at_pos = grad[0, pos]

            for cand_id in candidate_ids:
                if cand_id == orig_token:
                    continue
                cand_embedding = embedding_matrix[cand_id]
                delta_embedding = cand_embedding - orig_embedding
                loss_change = torch.dot(grad_at_pos, delta_embedding)
                new_loss = total_loss.item() + loss_change.item()
                # replacement
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_pos = pos
                    best_token = cand_id
        # stop replacement 
        if best_pos == -1:
            print("No better replacement found, stopping.")
            break

        adv_input_ids[0, best_pos] = best_token
        replacements += 1
        print(f"Replacement {replacements}: Position {best_pos}, New Token {bert_tokenizer.decode([best_token])}, Estimated Loss = {best_loss:.4f}")

        # Reverse Detection 
        current_text = bert_tokenizer.decode(adv_input_ids[0], skip_special_tokens=True)
        new_pred = get_prediction_model(model, tokenizer, classifier_probe, current_text)
        similarity = compute_use_similarity(input_text, current_text, use_model)
        _, perplexity = compute_perplexity_loss(gpt2_model, gpt2_tokenizer, current_text, device)
        
        # Accept a suffix only when it flips the probe prediction and stays within the similarity and perplexity constraints.
        if new_pred != orig_pred and similarity > similarity_threshold and perplexity.item() < ppl_threshold:
            print("Probe Prediction flipped successfully!")
            break
        else:
            print("Probe Prediction did not flip! Proceed to POS-aware diversification...")



        # Randomly decide with 20% probability to perform replacement, simulating variability
        if random.random() < 0.2:
            suffix_start = len(orig_ids_list[:-1]) + len(structure_token_ids) # Calculate start of suffix based on original IDs and structure tokens
            suffix_end = len(orig_ids_list[:-1]) + len(structure_token_ids) + num_candidates #Define end of suffix with number of candidates
            valid_positions = list(range(suffix_start, min(suffix_end, adv_input_ids.size(1) - 1)))

            if valid_positions:
                replace_pos = random.choice(valid_positions) # Randomly select a position for replacement
                current_token = adv_input_ids[0, replace_pos].item()
                current_word = bert_tokenizer.decode([current_token], skip_special_tokens=True)

                pos_tags = nltk.pos_tag([current_word]) # Determine part-of-speech (POS) tag of current word
                current_pos = pos_tags[0][1] # Extract POS tag
                if current_pos.startswith('JJ'):
                    target_pos = 'JJ'
                elif current_pos.startswith('NN'):
                    target_pos = 'NN'
                elif current_pos.startswith('VB'):
                    target_pos = 'VB'
                else:
                    target_pos = None

                if target_pos: # Proceed if a valid POS is found
                    available_tokens = []
                    for tid in candidate_ids:
                        if tid in seen_tokens or tid == current_token:
                            continue
                        word = bert_tokenizer.decode([tid], skip_special_tokens=True)
                        cand_pos_tags = nltk.pos_tag([word])
                        cand_pos = cand_pos_tags[0][1]
                        if cand_pos.startswith(target_pos): # Check if candidate matches target POS
                            available_tokens.append(tid)

                    if available_tokens:
                        new_token = random.choice(available_tokens) # Randomly select a new token
                        adv_input_ids[0, replace_pos] = new_token # Replace the token in the input sequence
                        seen_tokens.add(new_token)
                        print(f"Randomly replaced token at position {replace_pos} (POS: {target_pos}): '{current_word}' -> '{bert_tokenizer.decode([new_token])}'")
                    else:
                        print(f"No candidates with POS {target_pos} available for replacement at position {replace_pos}.")
                else:
                    print(f"Skipping replacement at position {replace_pos}: unsupported POS {current_pos} for '{current_word}'.")

    adv_text = bert_tokenizer.decode(adv_input_ids[0], skip_special_tokens=True)
    return adv_text






def compute_perplexity_loss(gpt2_model, gpt2_tokenizer, text, device):
    gpt2_model.eval()
    gpt2_inputs = gpt2_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    gpt2_inputs = gpt2_inputs.input_ids.to(device)
    with torch.no_grad():
        gpt2_outputs = gpt2_model(gpt2_inputs, labels=gpt2_inputs)
        loss = gpt2_outputs.loss
        perplexity = torch.exp(loss)
    return loss, perplexity

def compute_perplexity(gpt2_model, gpt2_tokenizer, text, device):
    gpt2_model.eval()
    gpt2_inputs = gpt2_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    gpt2_inputs = gpt2_inputs.input_ids.to(device)
    with torch.no_grad():
        gpt2_outputs = gpt2_model(gpt2_inputs, labels=gpt2_inputs)
        loss = gpt2_outputs.loss
        perplexity = torch.exp(loss).item()
    return perplexity


def compute_use_similarity(prompt, optimized_prompt, use_model):
    prompt_embedding = use_model.encode(prompt, convert_to_tensor=True)
    optimized_embedding = use_model.encode(optimized_prompt, convert_to_tensor=True)
    similarity = torch.nn.functional.cosine_similarity(prompt_embedding, optimized_embedding, dim=0).item()
    return similarity

def get_logits_with_embeddings(model, classifier_probe, embeddings):
    outputs = model(inputs_embeds=embeddings, output_hidden_states=True)
    last_hidden_state = outputs.hidden_states[-1][:, -1, :] 
    last_hidden_state = last_hidden_state.to(classifier_probe.weight.dtype)
    logits = classifier_probe(last_hidden_state)
    del outputs, last_hidden_state
    return logits

def get_prediction_model(model, tokenizer, classifier_probe, text: str) -> int:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][:, -1, :]
        hidden_states = hidden_states.to(classifier_probe.weight.dtype)
        logits = classifier_probe(hidden_states)
        pred = torch.argmax(logits, dim=-1).item()
    return pred

def load_emotional_words(bert_tokenizer, dataset_name, threshold=0.75, max_words=100):
   
    def get_words_by_category(category="positive"): 
        words = [] 
        for synset in wn.all_synsets(wn.ADJ):  
            word = synset.name().split('.')[0]
            senti_synsets = list(swn.senti_synsets(word, 'a'))
            if senti_synsets:
                s = senti_synsets[0] 
                if category == "positive" and s.pos_score() > s.neg_score() and s.pos_score() >= threshold: 
                    words.append(word) 
                elif category == "negative" and s.neg_score() > s.pos_score() and s.neg_score() >= threshold:
                    words.append(word) 
            if len(words) >= max_words: 
                break 
        return words

    negative_words = get_words_by_category(category="negative") 
    positive_words = get_words_by_category(category="positive")
    negative_ids = [bert_tokenizer.encode(word, add_special_tokens=False)[0] for word in negative_words if bert_tokenizer.encode(word, add_special_tokens=False)]
    positive_ids = [bert_tokenizer.encode(word, add_special_tokens=False)[0] for word in positive_words if bert_tokenizer.encode(word, add_special_tokens=False)]
    negative_ids = list(set(negative_ids)) 
    positive_ids = list(set(positive_ids)) 
    return [], positive_ids, negative_ids

def select_candidate_ids(dataset_name, target_label, category_ids, positive_ids, negative_ids):
    return positive_ids if target_label == 1 else negative_ids
 

"""
optimize_adversarial_suffix is the core function of the system, designed to generate adversarial suffixes and append them to the original prompt to mislead the predictions of large language models.
It supports multiple datasets (SST-2, StrategyQA, AG-News).
Generates a suffix S = C ⊕ W (where C is a connecting phrase and W is an adversarial word sequence), optimized iteratively to alter model predictions.
"""
def optimize_adversarial_suffix(
    model,
    tokenizer,
    bert_model,
    bert_tokenizer,
    gpt2_model,
    gpt2_tokenizer,
    classifier_model,
    classifier_tokenizer,
    orig_prompt: str,
    orig_pred:int,
    target_label: int,
    dataset_name: str,
    suffix_len: int = 20,
    similarity_threshold: float = 8.0,
    ppl_threshold: float = 5000.0,
    n_steps: int = 400,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.1,
    use_model=None,
    candidate_ids=None,
    classifier_probe=None, 
) -> str:
    device = classifier_model.device

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            default_pad_token = '[PAD]'
            tokenizer.add_special_tokens({'pad_token': default_pad_token})
            tokenizer.pad_token = default_pad_token
        print(f"Set tokenizer.pad_token to {tokenizer.pad_token}")

    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    adv_text = optimize(
        bert_model=bert_model,
        classifier_model=classifier_model,
        classifier_tokenizer=classifier_tokenizer,
        bert_tokenizer=bert_tokenizer,
        gpt2_model = gpt2_model,
        gpt2_tokenizer = gpt2_tokenizer,
        input_text=orig_prompt,
        orig_pred=orig_pred,
        target_label=target_label,
        device=device,
        max_replacements=n_steps,
        similarity_threshold=similarity_threshold,
        ppl_threshold=ppl_threshold,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        use_model=use_model,
        model=model,
        tokenizer=tokenizer,
        classifier_probe=classifier_probe,
        candidate_ids=candidate_ids,
        suffix_len=suffix_len
    )

    return adv_text