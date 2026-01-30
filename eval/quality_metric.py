import os
import json
import jsonlines
import numpy as np
import argparse
from tqdm import tqdm
from openai import OpenAI
from collections import defaultdict


def get_gpt_score(client, text, model_name="gpt-4o"):
    """
    Evaluates the text quality across 4 dimensions using GPT-4o.
    """
    system_prompt = (
        "You are an expert writing evaluator. Score the provided text from 1 to 10 "
        "on the following criteria:\n"
        "1. Creativity: Originality and imagination.\n"
        "2. Coherence: Logical flow and structural integrity.\n"
        "3. Writing Quality: Grammar, vocabulary, and style.\n"
        "4. Relevance: Adherence to the writing prompt/topic.\n\n"
        "Return ONLY a JSON object with these keys: "
        "'creativity', 'coherence', 'writing_quality', 'relevance'."
    )

    # Send a representative chunk of the text to stay within token limits
    user_content = f"Text to evaluate:\n{text[:2000]}"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"\n[Error] API call failed: {e}")
        return None


def process_experiment(target_dir, api_key):
    # --- API Key Initialization ---
    client = OpenAI(api_key=api_key)

    # Store scores: { prompt_id: [list of individual sample scores] }
    all_scores_by_id = defaultdict(list)

    # 1. Locate all jsonl files recursively
    jsonl_files = []
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file == "generations.rank0.jsonl":
                jsonl_files.append(os.path.join(root, file))

    if not jsonl_files:
        print(f"Error: No generations.rank0.jsonl found in {target_dir}")
        return

    # 2. Iteratively score all samples
    print(f"Starting evaluation for: {target_dir}")
    for file_path in jsonl_files:
        print(f"Processing: {file_path}")
        with jsonlines.open(file_path) as reader:
            for item in tqdm(list(reader), desc="Scoring"):
                p_id = item["prompt_id"]
                scores = get_gpt_score(client, item["text"])

                if scores:
                    # Metric for ranking: average of the 4 dimensions
                    scores["_mean_for_ranking"] = np.mean(
                        list(scores.values()))
                    all_scores_by_id[p_id].append(scores)

    # 3. Calculate Top 12 per prompt_id and aggregate
    final_metric_accumulator = defaultdict(list)

    for p_id, samples in all_scores_by_id.items():
        # Sort by the ranking metric (descending) and take top 12
        top_12_samples = sorted(
            samples, key=lambda x: x["_mean_for_ranking"], reverse=True)[:12]

        for s in top_12_samples:
            for metric in ['creativity', 'coherence', 'writing_quality', 'relevance']:
                final_metric_accumulator[metric].append(s[metric])

    # 4. Compute final averages across the entire method
    if not final_metric_accumulator:
        print("No valid scores were collected.")
        return

    summary = {
        metric: round(float(np.mean(values)), 4)
        for metric, values in final_metric_accumulator.items()
    }

    # 5. Save results to the method's root directory
    output_path = os.path.join(target_dir, "quality_summary.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"\nEvaluation Complete!")
    print(f"Results saved to: {output_path}")
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LLM generation quality using GPT-4o.")
    parser.add_argument("--dir", type=str, required=True,
                        help="Path to the method directory.")
    parser.add_argument("--key", type=str, default=None,
                        help="OpenAI API Key (optional if set in env).")

    args = parser.parse_args()

    # Use key from argument, or fallback to environment variable
    api_key = args.key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Error: API Key not found. Provide it via --key or set OPENAI_API_KEY environment variable.")
    else:
        process_experiment(args.dir, api_key)
