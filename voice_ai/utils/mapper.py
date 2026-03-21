def to_hindi(item):
    mapping = {
        "potato": "aloo",
        "tomato": "tamatar",
        "onion": "pyaz"
    }
    return mapping.get(item, item)