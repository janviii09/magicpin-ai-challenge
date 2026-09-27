"""
Vera 2.0 — End-to-end smoke test
Tests the full pipeline without LLM (uses fallback templates).
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.context_store import ContextStore
from core.trigger_engine import resolve
from core.decision_engine import decide
from core.planner import build_message_plan
from core.suppression import SuppressionEngine
from core.validator import validate
from llm.composer import _fallback_template
from conversations.intent import detect_intent, DetectedIntent
from conversations.state_machine import ConversationManager
from bot import compose


def test_full_pipeline():
    """Test the complete pipeline end-to-end with dentist data."""
    print("=" * 60)
    print("VERA 2.0 — End-to-End Smoke Test")
    print("=" * 60)

    # 1. Load dataset
    store = ContextStore()
    store.load_from_dataset(Path(__file__).parent.parent / "dataset")
    print(f"\n✅ Dataset loaded: {store.counts()}")

    # 2. Test versioned context store
    accepted, _ = store.upsert("category", "dentists", 1, {"slug": "dentists", "voice": {"tone": "peer_clinical"}})
    assert accepted  # Same version = idempotent
    accepted, cv = store.upsert("category", "dentists", 0, {"slug": "dentists", "voice": {"tone": "old"}})
    assert not accepted  # Stale version = rejected
    print(f"✅ Context store versioning works (stale rejected, current_version={cv})")

    # 3. Test trigger resolution — research digest for Dr. Meera
    trigger = store.get("trigger", "trg_001_research_digest_dentists")
    merchant = store.get("merchant", "m_001_drmeera_dentist_delhi")
    category = store.get("category", "dentists")

    resolution = resolve(trigger, merchant, category, None)
    print(f"\n✅ Trigger resolved: kind={trigger.kind}")
    print(f"   objective: {resolution.objective}")
    print(f"   angle: {resolution.angle}")
    print(f"   evidence: {resolution.evidence[:2]}...")

    # 4. Test decision engine
    decision = decide(trigger, resolution, merchant, category, None)
    assert decision.should_send
    print(f"✅ Decision: should_send={decision.should_send}, lang={decision.language}")

    # 5. Test planner
    plan = build_message_plan(trigger, resolution, decision, merchant, category, None)
    print(f"✅ Plan built: audience={plan.audience}, cta={plan.cta}, voice={plan.voice}")
    print(f"   why_now: {plan.why_now}")
    print(f"   required_facts: {plan.required_facts}")

    # 6. Test fallback template (no LLM key needed)
    message = _fallback_template(plan, category)
    print(f"\n✅ Fallback message composed:")
    print(f"   body: {message.body}")
    print(f"   cta: {message.cta}")
    print(f"   send_as: {message.send_as}")

    # 7. Test validator
    result = validate(message, plan, category)
    print(f"✅ Validation: valid={result.valid}, violations={result.violations}")

    # 8. Test suppression
    suppression = SuppressionEngine()
    assert not suppression.is_suppressed("m_001", "test_key", "research_digest")
    suppression.record_sent("m_001", "test_key")
    assert suppression.is_suppressed("m_001", "test_key", "research_digest")
    print(f"✅ Suppression: correctly blocks after send")

    # 9. Test intent detection
    test_intents = [
        ("Thank you for contacting us. Our team will get back.", DetectedIntent.AUTO_REPLY),
        ("Yes", DetectedIntent.DIRECT_YES),
        ("STOP", DetectedIntent.DIRECT_NO),
        ("Mujhe magicpin judna hai", DetectedIntent.JOIN_INTENT),
        ("Update my profile please", DetectedIntent.ACTION_REQUEST),
        ("What's the price?", DetectedIntent.QUESTION),
        ("Interesting, tell me more", DetectedIntent.GENERAL_REPLY),
    ]
    all_pass = True
    for msg, expected in test_intents:
        detected = detect_intent(msg)
        status = "✅" if detected == expected else "❌"
        if detected != expected:
            all_pass = False
        print(f"   {status} \"{msg[:40]}...\" → {detected.value} (expected {expected.value})")
    print(f"{'✅' if all_pass else '❌'} Intent detection: {'all passed' if all_pass else 'SOME FAILED'}")

    # 10. Test conversation state machine
    conv_mgr = ConversationManager()
    state = conv_mgr.create_conversation("conv_test_1", "m_001", "trg_001")
    assert state.phase.value == "proposed"

    # Simulate auto-reply
    state, action, reason = conv_mgr.process_reply("conv_test_1", "merchant", "Thank you for contacting us. Our team will respond.")
    print(f"\n✅ FSM auto-reply: phase={state.phase.value}, action={action}")

    # Simulate another auto-reply → should end
    state, action, reason = conv_mgr.process_reply("conv_test_1", "merchant", "We will get back to you soon.")
    print(f"✅ FSM double auto-reply: phase={state.phase.value}, action={action}")
    assert state.phase.value == "end"

    # Test direct YES flow
    state2 = conv_mgr.create_conversation("conv_test_2", "m_001", "trg_001")
    state2, action, reason = conv_mgr.process_reply("conv_test_2", "merchant", "Yes")
    print(f"✅ FSM direct YES: phase={state2.phase.value}, action={action}")
    assert state2.phase.value == "execute"

    # 11. Test compose() — the full public API (without LLM)
    print("\n" + "=" * 60)
    print("FULL PIPELINE TEST (compose())")
    print("=" * 60)

    # Load raw dicts for compose()
    with open(Path(__file__).parent.parent / "dataset" / "categories" / "dentists.json") as f:
        cat_dict = json.load(f)
    merchants_data = json.loads((Path(__file__).parent.parent / "dataset" / "merchants_seed.json").read_text())
    mer_dict = merchants_data["merchants"][0]
    triggers_data = json.loads((Path(__file__).parent.parent / "dataset" / "triggers_seed.json").read_text())
    trg_dict = triggers_data["triggers"][0]  # research digest

    result = compose(cat_dict, mer_dict, trg_dict)
    print(f"\n✅ compose() returned:")
    for k, v in result.items():
        print(f"   {k}: {v}")

    # Test customer-facing compose
    trg_recall = triggers_data["triggers"][2]  # recall_due for Priya
    customers_data = json.loads((Path(__file__).parent.parent / "dataset" / "customers_seed.json").read_text())
    cus_dict = customers_data["customers"][0]  # Priya

    result_cust = compose(cat_dict, mer_dict, trg_recall, cus_dict)
    print(f"\n✅ Customer-facing compose() returned:")
    for k, v in result_cust.items():
        print(f"   {k}: {v}")

    # Test multiple trigger types
    print("\n" + "-" * 60)
    print("TESTING ALL TRIGGER TYPES:")
    print("-" * 60)

    for trg in triggers_data["triggers"]:
        mid = trg.get("merchant_id", "")
        mer = None
        for m in merchants_data["merchants"]:
            if m["merchant_id"] == mid:
                mer = m
                break
        if mer is None:
            continue

        cat_slug = mer.get("category_slug", "dentists")
        cat_file = Path(__file__).parent.parent / "dataset" / "categories" / f"{cat_slug}.json"
        if not cat_file.exists():
            continue
        cat = json.loads(cat_file.read_text())

        cus = None
        if trg.get("customer_id"):
            for c in customers_data["customers"]:
                if c["customer_id"] == trg["customer_id"]:
                    cus = c
                    break

        try:
            result = compose(cat, mer, trg, cus)
            body_preview = result["body"][:60] + "..." if len(result["body"]) > 60 else result["body"]
            print(f"  ✅ {trg['id']}: [{result['send_as']}] {body_preview}")
        except Exception as e:
            print(f"  ❌ {trg['id']}: ERROR — {e}")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    test_full_pipeline()
