def parse_input(text: str):
    text = text.lower()

    if any(word in text for word in ["aloo", "aaloo", "potato"]):
        item = "potato"
    elif any(word in text for word in ["tamatar", "tomato"]):
        item = "tomato"
    elif any(word in text for word in ["pyaz", "pyaaz", "onion"]):
        item = "onion"
    else:
        item = None

    return {"item": item}