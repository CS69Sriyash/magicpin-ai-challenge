import json
import time
import sys
from pathlib import Path
from judge_simulator import BotClient

BOT_URL = "http://127.0.0.1:8080"
DATASET_DIR = Path("./dataset")
OUTPUT_FILE = "submission.jsonl"

def load_json_dir(directory: Path, key_field: str):
    data = {}
    if directory.exists():
        for f in directory.glob("*.json"):
            try:
                with open(f) as fp:
                    item = json.load(fp)
                if key_field in item:
                    data[item[key_field]] = item
                elif "slug" in item:
                    data[item["slug"]] = item
                else:
                    data[f.stem] = item
            except Exception as e:
                print(f"Error reading {f}: {e}")
    return data

def load_full_dataset():
    categories = load_json_dir(DATASET_DIR / "categories", "slug")
    merchants = load_json_dir(DATASET_DIR / "merchants", "merchant_id")
    customers = load_json_dir(DATASET_DIR / "customers", "customer_id")
    triggers = load_json_dir(DATASET_DIR / "triggers", "id")
    return categories, merchants, customers, triggers

def main():
    print(f"Loading full dataset from {DATASET_DIR}...")
    categories, merchants, customers, triggers = load_full_dataset()
    print(f"Loaded {len(categories)} categories, {len(merchants)} merchants, {len(customers)} customers, {len(triggers)} triggers.")
    
    test_pairs_path = DATASET_DIR / "test_pairs.json"
    if not test_pairs_path.exists():
        print("Error: test_pairs.json not found! Run 'python dataset/generate_dataset.py --seed-dir dataset --out dataset' first.")
        sys.exit(1)
        
    with open(test_pairs_path) as f:
        test_pairs_data = json.load(f)
    
    test_pairs = test_pairs_data.get("pairs", [])
    if len(test_pairs) != 30:
        print(f"Warning: Expected 30 test pairs, found {len(test_pairs)}")
        
    print(f"\nConnecting to bot at {BOT_URL}...")
    client = BotClient(BOT_URL)
    _, err, _ = client.healthz()
    if err:
        print(f"Bot unreachable: {err}")
        print("Please ensure your FastAPI server is running.")
        sys.exit(1)
        
    print("Pushing all contexts to bot...")
    for slug, cat in categories.items():
        client.push_context("category", slug, 1, cat)
    for mid, m in merchants.items():
        client.push_context("merchant", mid, 1, m)
    for cid, c in customers.items():
        client.push_context("customer", cid, 1, c)
        
    results = []
    
    print(f"\nProcessing {len(test_pairs)} specific test pairs...")
    
    # Process in batches of 1 to rigorously respect Groq 8000 TPM limits
    batch_size = 1
    for i in range(0, len(test_pairs), batch_size):
        batch = test_pairs[i:i+batch_size]
        trigger_ids = [pair["trigger_id"] for pair in batch]
        
        # Push triggers for this batch
        for tid in trigger_ids:
            trigger_payload = triggers.get(tid)
            if trigger_payload:
                client.push_context("trigger", tid, 1, trigger_payload)
            else:
                print(f"Warning: Trigger {tid} not found in dataset!")
                
        print(f"  Sending tick for batch {i//batch_size + 1} ({len(batch)} triggers)...")
        data, err, lat = client.tick(trigger_ids)
        if err:
            print(f"Error during tick: {err}")
            continue
            
        actions = data.get("actions", [])
        
        for pair in batch:
            # Match action to our specific test pair
            action = next((a for a in actions if a.get("merchant_id") == pair["merchant_id"] and a.get("trigger_id") == pair["trigger_id"]), None)
            
            if action:
                results.append({
                    "merchant_id": action.get("merchant_id"),
                    "trigger_id": action.get("trigger_id"),
                    "body": action.get("body", ""),
                    "cta": action.get("cta", "none"),
                    "send_as": action.get("send_as", "vera")
                })
            else:
                print(f"    Warning: No action generated for {pair['merchant_id']} / {pair['trigger_id']}")
                
        # Pause for 12s to prevent Groq 8000 TPM limit (1500 tokens * 5 req/min = 7500)
        time.sleep(12)
        
    print(f"\nGenerated {len(results)} responses. Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    if len(results) == 30:
        print(f"Success! Exactly 30 lines written to {OUTPUT_FILE}. You are ready to submit!")
    else:
        print(f"Warning: Wrote {len(results)} lines instead of 30. Please review the logs.")

if __name__ == "__main__":
    main()
