# magicpin AI Challenge — "Vera" Assistant

This is Team Sriyash's submission for the magicpin Vera AI Challenge. The bot is fully dockerized and uses a stateless-API design with an in-memory context store as required by the judge simulator.

## 1. Our Approach
We designed our Engagement Composer around strict prompt-engineering constraints and dynamic category routing:
- **Anchor Facts:** We explicitly extract and enforce the use of "Anchor Facts" (concrete numbers/citations) from triggers to eliminate hallucination.
- **Strict Conversation Rules:** We implemented strict heuristics and prompt-level rules to handle edge cases gracefully:
  - *Intent Transitions:* Automatically confirms action when the merchant explicitly agrees, rather than getting stuck in a qualifying loop.
  - *Auto-Reply Hell:* Detects WhatsApp business auto-responder loops and exits after repeated identical messages.
  - *Hostile Merchants:* Deterministic trapdoors to politely end conversations when users type "spam" or "stop".

## 2. Tradeoffs Made
- **In-Memory Store vs Persistent DB:** We used an in-memory dictionary for storing contexts and conversation history. This guarantees high speed under the 30-second time budget but means state is lost on container restart.
- **Heuristics vs Pure LLM Routing:** For explicit stop phrases and auto-responder detection, we used deterministic Python logic before falling back to the LLM. This trades some "AI conversational fluidity" for guaranteed safety and fast execution without risking LLM hallucinations on critical exits.
- **Single Model vs Multi-Model pipeline:** We use Groq (`qwen3.6-27b`) for all phases to maintain low latency, rather than a slow/heavy reasoning model, to ensure we never hit the 30s timeout.

## 3. What Additional Context Would Have Helped Most
- **Customer Lifetime Value (LTV):** Knowing the historical revenue of a lapsed customer would help prioritize which available slot offers to push.
- **Conversion Rates of Similar Offers:** Data on which specific offer templates convert best in a given micro-locality (e.g., Lajpat Nagar) would allow the LLM to make smarter, data-driven recommendations rather than generic category best-practices.

## Setup & Deployment

### Run via Docker (Recommended)
```bash
docker build -t magicpin-vera .
docker run -p 8080:8080 -e LLM_PROVIDER=groq -e GROQ_API_KEY="your_key" magicpin-vera
```

### Run Locally (via uv)
```bash
uv sync
uv run uvicorn bot:app --host 0.0.0.0 --port 8080
```
