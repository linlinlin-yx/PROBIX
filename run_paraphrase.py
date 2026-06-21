from openai import OpenAI, APITimeoutError
import time
import os
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def get_paraphrase_text(dataset, opt_text, model="gpt-4o-mini", temperature=0.7):
    prompt = f"Please paraphrase the following text while keeping its meaning. Output ONLY the rewritten text. Do NOT add explanations, disclaimers, or metadata. Do NOT mention training data, dates, or yourself. Text: {opt_text} "  
    max_retries = 3
    for attempt in range(max_retries):
        try: 
            response = client.chat.completions.create(
                    model=model, 
                    messages=[
                        {"role": "system", "content": "You are a text rewriting assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens = 256, 
                    temperature = temperature,
                    timeout=60,
                )
            paraphrased_text = response.choices[0].message.content.strip()
            return paraphrased_text
        except APITimeoutError as e:
            print(f"API Timeout, Paraphrase failed (attempt {attempt+1})")
            time.sleep(2)
    return None


