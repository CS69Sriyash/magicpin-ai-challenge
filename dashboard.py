"""
dashboard.py — Streamlit control panel for the Vera bot backend.

Run:
    pip install streamlit requests
    streamlit run dashboard.py

Expects the FastAPI bot (bot.py) already running at BASE_URL below.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import requests
import streamlit as st

import os

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8080")
REQUEST_TIMEOUT = 15  # seconds — tick can legitimately take longer; overridden below

st.set_page_config(page_title="Vera Control Panel", page_icon="🤖", layout="wide")


# ------------------------------------------------------------------------- #
# Small API client helpers — every call returns (data_or_None, error_or_None)
# so the UI code never has to sprinkle try/except everywhere.
# ------------------------------------------------------------------------- #


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def api_get(path: str, timeout: int = REQUEST_TIMEOUT):
    try:
        resp = requests.get(f"{BASE_URL}{path}", timeout=timeout)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.RequestException as exc:
        return None, str(exc)


def api_post(path: str, payload: dict, timeout: int = REQUEST_TIMEOUT):
    try:
        resp = requests.post(f"{BASE_URL}{path}", json=payload, timeout=timeout)
        # Don't raise_for_status blindly — /v1/context returns a real JSON
        # body (accepted: False, reason: ...) on 4xx that's more useful to
        # the user than a generic HTTPError string.
        try:
            data = resp.json()
        except ValueError:
            resp.raise_for_status()
            return None, f"Non-JSON response (HTTP {resp.status_code})"
        if resp.status_code >= 400:
            return data, f"HTTP {resp.status_code}"
        return data, None
    except requests.exceptions.RequestException as exc:
        return None, str(exc)


# ------------------------------------------------------------------------- #
# Sidebar — system status
# ------------------------------------------------------------------------- #

with st.sidebar:
    st.title("🤖 Vera Control Panel")
    st.caption(BASE_URL)

    if st.button("🔄 Refresh status", use_container_width=True):
        st.rerun()

    health, health_err = api_get("/v1/healthz", timeout=5)
    is_online = health is not None and health_err is None

    if is_online:
        st.markdown("**Server Status:** 🟢 Online")
    else:
        st.markdown("**Server Status:** 🔴 Offline")
        if health_err:
            st.caption(f"`{health_err}`")

    st.divider()

    if is_online:
        st.subheader("Health")
        st.metric("Uptime (s)", health.get("uptime_seconds", "—"))
        counts = health.get("contexts_loaded", {})
        if counts:
            st.write("Contexts loaded:")
            st.table(
                {
                    "scope": list(counts.keys()),
                    "count": list(counts.values()),
                }
            )
        else:
            st.caption("No context counts returned.")

        st.subheader("Metadata")
        meta, meta_err = api_get("/v1/metadata", timeout=5)
        if meta and not meta_err:
            st.write(f"**Name:** {meta.get('name', '—')}")
            st.write(f"**Model:** {meta.get('model', '—')}")
            st.write(f"**Version:** {meta.get('version', '—')}")
            with st.expander("Full metadata JSON"):
                st.json(meta)
        else:
            st.caption(f"Could not fetch metadata: {meta_err}")
    else:
        st.info("Start the bot with `uvicorn bot:app --port 8080`, then refresh.")


# ------------------------------------------------------------------------- #
# Main tabs
# ------------------------------------------------------------------------- #

tab_context, tab_tick, tab_chat = st.tabs(
    ["📥 Context Management", "⚡ Proactive Trigger (Tick)", "💬 Chat Simulator"]
)

# --- Tab 1: Context Management -------------------------------------------- #

with tab_context:
    st.header("Push context to the bot")
    st.caption("POST /v1/context — pushes a category, merchant, customer, or trigger payload.")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        scope = st.selectbox("Scope", ["category", "merchant", "customer", "trigger"])
        context_id = st.text_input("Context ID", placeholder="e.g. m_001 / t_dentist_001")
        version = st.number_input("Version", min_value=1, value=1, step=1)
        push_btn = st.button("🚀 Push to /v1/context", type="primary", use_container_width=True)

    with col_right:
        default_payload = json.dumps({"example_field": "replace me"}, indent=2)
        raw_payload = st.text_area(
            "Payload JSON",
            value=default_payload,
            height=280,
            help="Paste the raw JSON payload for this scope (category / merchant / customer / trigger).",
        )

    if push_btn:
        if not context_id.strip():
            st.warning("Please provide a Context ID.")
        else:
            try:
                payload_dict = json.loads(raw_payload)
            except json.JSONDecodeError as exc:
                st.error(f"Payload is not valid JSON: {exc}")
            else:
                body = {
                    "scope": scope,
                    "context_id": context_id.strip(),
                    "version": int(version),
                    "payload": payload_dict,
                    "delivered_at": _utc_now_iso(),
                }
                data, err = api_post("/v1/context", body)
                if err:
                    st.error(f"Request failed: {err}")
                    if data:
                        st.json(data)
                elif data and data.get("accepted"):
                    st.success(f"Accepted — ack_id: {data.get('ack_id')}")
                    st.json(data)
                else:
                    st.error("Bot rejected the payload.")
                    st.json(data)

# --- Tab 2: Proactive Trigger (Tick) --------------------------------------- #

with tab_tick:
    st.header("Fire a proactive trigger")
    st.caption("POST /v1/tick — asks the bot to compose a proactive message for one or more triggers.")

    trigger_id = st.text_input("Trigger ID", placeholder="e.g. t_dentist_research_001")
    fire_btn = st.button("⚡ Fire tick", type="primary")

    if fire_btn:
        if not trigger_id.strip():
            st.warning("Please provide a Trigger ID.")
        else:
            body = {"now": _utc_now_iso(), "available_triggers": [trigger_id.strip()]}
            with st.spinner("Composing…"):
                data, err = api_post("/v1/tick", body, timeout=35)

            if err:
                st.error(f"Request failed: {err}")
                if data:
                    st.json(data)
            else:
                actions = (data or {}).get("actions", [])
                if not actions:
                    st.info("Bot returned no actions — it chose not to send (or the trigger wasn't found).")
                else:
                    for action in actions:
                        with st.container(border=True):
                            top_left, top_right = st.columns([3, 1])
                            with top_left:
                                st.subheader(action.get("merchant_id", "unknown merchant"))
                                st.caption(
                                    f"conversation_id: `{action.get('conversation_id', '—')}` · "
                                    f"send_as: `{action.get('send_as', '—')}`"
                                )
                            with top_right:
                                st.metric("CTA", action.get("cta", "—"))

                            st.markdown("**Message body:**")
                            st.info(action.get("body", "(empty)"))

                            if action.get("rationale"):
                                with st.expander("Rationale (internal)"):
                                    st.write(action["rationale"])

                            with st.expander("Full action JSON"):
                                st.json(action)

# --- Tab 3: Chat Simulator (Reply) ----------------------------------------- #

with tab_chat:
    st.header("Chat simulator")
    st.caption("POST /v1/reply — simulate a merchant replying to Vera, turn by turn.")

    init_col1, init_col2, init_col3 = st.columns([2, 2, 1])
    with init_col1:
        conversation_id_input = st.text_input(
            "Conversation ID", value=st.session_state.get("conversation_id", "conv_dashboard_1")
        )
    with init_col2:
        merchant_id_input = st.text_input(
            "Merchant ID", value=st.session_state.get("merchant_id", "")
        )
    with init_col3:
        st.write("")
        st.write("")
        if st.button("🔁 Reset chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.turn_number = 1
            st.rerun()

    # Initialize / re-key session state when the conversation id changes,
    # so switching conversations doesn't bleed history across them.
    if "conversation_id" not in st.session_state or st.session_state.conversation_id != conversation_id_input:
        st.session_state.conversation_id = conversation_id_input
        st.session_state.chat_history = []
        st.session_state.turn_number = 1

    st.session_state.merchant_id = merchant_id_input

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "turn_number" not in st.session_state:
        st.session_state.turn_number = 1

    st.divider()

    # Render existing history
    for entry in st.session_state.chat_history:
        role = "user" if entry["role"] == "merchant" else "assistant"
        avatar = "🧑💼" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.write(entry["content"])
            if entry.get("meta"):
                st.caption(entry["meta"])

    user_message = st.chat_input("Type a merchant reply…")

    if user_message:
        if not merchant_id_input.strip():
            st.warning("Please provide a Merchant ID before chatting.")
        else:
            turn = st.session_state.turn_number

            st.session_state.chat_history.append(
                {"role": "merchant", "content": user_message, "meta": f"turn {turn}"}
            )
            with st.chat_message("user", avatar="🧑💼"):
                st.write(user_message)

            body = {
                "conversation_id": st.session_state.conversation_id,
                "merchant_id": merchant_id_input.strip(),
                "message": user_message,
                "turn_number": turn,
                "from_role": "merchant",
                "received_at": _utc_now_iso(),
            }

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Vera is thinking…"):
                    data, err = api_post("/v1/reply", body, timeout=20)

                if err:
                    st.error(f"Request failed: {err}")
                    if data:
                        st.json(data)
                else:
                    action = data.get("action", "?")
                    if action == "send":
                        reply_text = data.get("body", "(empty)")
                        st.write(reply_text)
                        st.caption(f"action: send · cta: {data.get('cta', '—')}")
                        st.session_state.chat_history.append(
                            {
                                "role": "vera",
                                "content": reply_text,
                                "meta": f"action: send · cta: {data.get('cta', '—')}",
                            }
                        )
                    elif action == "wait":
                        wait_msg = f"⏳ Vera is waiting ({data.get('wait_seconds', '?')}s) before replying again."
                        st.info(wait_msg)
                        st.session_state.chat_history.append(
                            {"role": "vera", "content": wait_msg, "meta": "action: wait"}
                        )
                    elif action == "end":
                        end_msg = "🚫 Vera ended the conversation."
                        st.warning(end_msg)
                        if data.get("rationale"):
                            st.caption(data["rationale"])
                        st.session_state.chat_history.append(
                            {"role": "vera", "content": end_msg, "meta": data.get("rationale", "action: end")}
                        )
                    else:
                        st.write(f"Unrecognized action: `{action}`")
                        st.json(data)

                    with st.expander("Raw response JSON"):
                        st.json(data)

            st.session_state.turn_number = turn + 1
