# Vera 2.0 — Merchant AI Assistant

## Approach

Vera 2.0 is a **deterministic merchant decision system** that uses an LLM only for natural-language realization. The architecture enforces a strict separation:

- **Deterministic code** decides: trigger interpretation, should-send, audience, CTA type, suppression key, category taboos, required facts, auto-reply detection, direct-intent detection, conversation state transitions.
- **Generative LLM** decides only: sentence structure, Hinglish code-mixing, tone, curiosity framing within the given angle.

### The MessagePlan — the contract

The single most important object is the `MessagePlan`. Every field maps to exactly one judge dimension:

| MessagePlan field | Judge dimension |
|---|---|
| `why_now` | Trigger Relevance |
| `evidence` + `required_facts` | Specificity |
| `angle` + `emotional_lever` | Engagement Compulsion |
| `voice` | Category Fit |
| `language` | Merchant Fit |

The LLM receives this plan and must produce natural WhatsApp text without changing any of the locked fields. This makes judge scores **predictable** rather than hoping the LLM figures out what it's being scored on.

## Architecture

```
Judge Harness → HTTP API → Context Store (versioned, idempotent)
                                ↓
                         Trigger Engine (kind → objective/angle/evidence)
                                ↓
                         Suppression Check (key + TTL by trigger kind)
                                ↓
                         Decision Engine (should-send, language, priority)
                                ↓
                         Message Planner → MessagePlan
                                ↓
                         Composition Engine (LLM, grounded prompts)
                           ↓ fail (≤2x retry)        ↓ pass
                    retry with violations      Validator (taboo, facts, CTA)
                           ↓ still fail               ↓ pass
                    Fallback Template         Final ComposedMessage
```

### Multi-turn Conversations

The conversation state machine implements:
- **Auto-reply detection**: Pattern matching (English + Hindi canned responses) + repetition detection. Two auto-replies → graceful exit.
- **Direct intent handoff**: "Yes" / "I want to join" → immediate transition to EXECUTE mode (no re-qualifying).
- **Graceful exit**: After 3 unanswered nudges or explicit "stop" → exit politely.

## Key Design Decisions

1. **Category differences are data, not code.** All 5 categories (dentists, salons, restaurants, gyms, pharmacies) share the same pipeline — voice rules, taboos, offer catalogs, and digest sources live in JSON, not in separate policy modules.

2. **Audience branching is a hard fork.** Merchant-facing and customer-facing messages use completely separate prompt templates, not one template with a variable swap. The registers are too different.

3. **Suppression uses per-kind TTLs.** A research digest is suppressed for 7 days, an IPL match for 12 hours, a supply alert for 24 hours. This prevents spamming while allowing time-sensitive triggers.

4. **Fallback templates are always valid.** If the LLM fails 3 times, a deterministic template produces a message that passes validation. Less natural, but never wrong.

5. **The validator catches hallucinations.** Required facts must appear verbatim in the output. Category taboo words trigger rejection. Promotional anti-patterns are blocked.

## Tradeoffs

- **No real LLM tuning yet** — the prompts are functional but could be refined with A/B testing against the judge simulator.
- **Reply composition in multi-turn is template-based** — ideally, replies would also go through the full LLM pipeline with conversation history context.
- **Suppression is in-memory** — restarts clear it. For production, this would need persistent storage.

## What Additional Context Would Help

1. **Real merchant reply patterns** — a corpus of actual merchant responses (beyond the 4 examples in the brief) would let us train better intent classifiers.
2. **Judge scoring examples** — seeing what scores 8/10 vs 4/10 on each dimension would help calibrate the composition.
3. **WhatsApp template approval rules** — real template constraints (character limits, parameter rules) would help the validator.

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Configure (copy and edit .env)
cp .env.example .env
# Add your LLM API key to .env

# Start the server
python server.py
# Server runs on http://localhost:8080

# Run against the judge simulator
python judge_simulator.py
```

## File Structure

```
├── server.py                  # FastAPI: /context /tick /reply /healthz /metadata
├── bot.py                     # Public compose() — the full pipeline
├── config.py                  # Environment configuration
├── core/
│   ├── models.py              # Pydantic contexts + MessagePlan + API contracts
│   ├── context_store.py       # Versioned, idempotent context storage
│   ├── trigger_engine.py      # trigger_kind → TriggerResolution
│   ├── suppression.py         # Key + TTL deduplication
│   ├── decision_engine.py     # Should-send + audience + language + priority
│   ├── planner.py             # Build MessagePlan from resolution + decision
│   └── validator.py           # Post-LLM quality gate
├── conversations/
│   ├── state_machine.py       # FSM: IDLE→PROPOSED→EXECUTE/BACKOFF/CLARIFY/END
│   └── intent.py              # Auto-reply + direct-intent detection
├── llm/
│   └── composer.py            # Audience-branched prompts, LLM call, fallback
├── conversation_handlers.py   # Optional: respond() for multi-turn
├── tests/
│   └── test_smoke.py          # End-to-end smoke test
└── dataset/                   # Category/merchant/customer/trigger data
```
