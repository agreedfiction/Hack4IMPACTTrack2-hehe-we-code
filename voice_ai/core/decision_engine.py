from core.data_loader import load_prices
from llm.ollama_client import query_llm
from utils.mapper import to_hindi
from utils.tts import is_online  # Import the internet checker!


def clean_output(text):
    text = text.strip()

    unwanted = ["I'm ready", "Here", "Sure", "Response", ":", "The rate for", "Based on"]
    for u in unwanted:
        if text.lower().startswith(u.lower()):
            text = text.split("\n")[-1]

    parts = text.split(".")
    return ".".join(parts[:2]).strip() + "."


def generate_decision(parsed):
    item = parsed["item"]
    mandi_prices = load_prices()
    
    # Check if we have internet right now
    online = is_online()

    if mandi_prices is None:
        if online:
            return "Data collect ho raha hai. Thodi der baad try karo."
        else:
            return "System is gathering data. Please try again later."

    if item not in mandi_prices:
        if online:
            return "Is commodity ka data available nahi hai."
        else:
            return "Data for this commodity is not available."

    price_quintal = int(mandi_prices[item])
    quality_score = 75

    # ==========================================
    # 🧠 DYNAMIC PROMPT (Hinglish vs English)
    # ==========================================
    if online:
        item_hindi = to_hindi(item)
        prompt = f"""
You are a mandi pricing assistant.

STRICT RULES:
- Output ONLY final answer in Hinglish (Hindi written in English)
- No introduction
- Max 2 sentences

FORMAT:
{item_hindi} ka rate {price_quintal} rupees/quintal rakho. <Action>

Actions:
Aaj bech do
Thoda hold karo
Jaldi becho, quality gir rahi hai

DATA:
Item: {item_hindi}
Price quintal: {price_quintal}
Quality: {quality_score}

Output ONLY:
"""
    else:
        prompt = f"""
You are a mandi pricing assistant.

STRICT RULES:
- Output ONLY final answer in pure English
- No introduction
- Max 2 sentences

FORMAT:
Set the price of {item} to {price_quintal} rupees per quintal. <Action>

Actions:
Sell it today.
Hold it for now.
Sell it quickly, quality is dropping.

DATA:
Item: {item}
Price quintal: {price_quintal}
Quality: {quality_score}

Output ONLY:
"""

    res = query_llm(prompt)

    if not res:
        if online:
            return f"{to_hindi(item)} ka rate {price_quintal} rupees/quintal rakho. Aaj bech do."
        else:
            return f"Set the price of {item} to {price_quintal} rupees per quintal. Sell it today."

    return clean_output(res)