import base64
import json
from argparse import ArgumentParser

import requests

API_URL = "http://localhost:8123/classify"


def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def classify_image(image_path: str, x1=None, y1=None, x2=None, y2=None):
    img_b64 = image_to_base64(image_path)
    payload = {"image_base64": img_b64}

    if None not in (x1, y1, x2, y2):
        payload.update({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    resp = requests.post(API_URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the image file")

    args = parser.parse_args()
    image_path = args.image

    result = classify_image(image_path)

    print("Результат:", json.dumps(result, ensure_ascii=False, indent=2))
