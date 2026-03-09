"""
PlantGuard AI — Drone/Phone Image Watcher
Watches a folder for new plant images and auto-analyzes them via PlantGuard API.
Results are uploaded to Supabase history.

Usage:
    python scripts/drone_watcher.py --watch-dir "D:/DCIM" --email your@email.com --password yourpassword
    python scripts/drone_watcher.py --watch-dir "./drone_photos" --api-url http://localhost:8000

Press Ctrl+C to stop.
"""

import os
import sys
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime

DEFAULT_API_URL = "https://plantguard-api.onrender.com"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
POLL_INTERVAL = 3


def login(api_url: str, email: str, password: str) -> str:
    resp = requests.post(
        f"{api_url}/api/auth/signin",
        params={"email": email, "password": password},
    )
    resp.raise_for_status()
    data = resp.json()
    token = data.get("access_token") or data.get("token")
    if not token:
        raise ValueError(f"No token in response: {data}")
    return token


def predict_image(api_url: str, image_path: str, token: str = None) -> dict:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        resp = requests.post(
            f"{api_url}/api/predict?top_k=5",
            files=files,
            headers=headers,
            timeout=120,
        )
    resp.raise_for_status()
    return resp.json()


def format_result(result: dict, image_path: str) -> str:
    lines = [
        f"\n{'='*60}",
        f"  Image: {os.path.basename(image_path)}",
        f"  Disease: {result['class'].replace('___', ' - ').replace('_', ' ')}",
        f"  Confidence: {result['probability']*100:.1f}%",
        f"  Uncertainty: {result['uncertainty']*100:.1f}%",
    ]
    validation = result.get("gemini_validation")
    if validation:
        lines.append(f"  Gemini: {'Agrees' if validation['agrees'] else 'Disagrees'}")
        lines.append(f"  Summary: {validation['summary']}")
        if validation.get("treatment_advice"):
            advice = validation["treatment_advice"]
            lines.append(f"  Treatment: {advice[:200]}{'...' if len(advice) > 200 else ''}")
    else:
        lines.append("  Gemini: Skipped (rate limited or unavailable)")
    lines.append(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    return "\n".join(lines)


def watch_folder(watch_dir: str, api_url: str, token: str = None):
    watch_path = Path(watch_dir).resolve()
    if not watch_path.exists():
        watch_path.mkdir(parents=True, exist_ok=True)
        print(f"Created watch directory: {watch_path}")

    processed = set()
    for f in watch_path.rglob("*"):
        if f.suffix.lower() in SUPPORTED_EXTENSIONS:
            processed.add(str(f))

    print(f"\nWatching: {watch_path}")
    print(f"API: {api_url}")
    print(f"{len(processed)} existing images found (skipped)")
    print(f"Polling every {POLL_INTERVAL}s - drop images into the folder!")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            for f in watch_path.rglob("*"):
                fpath = str(f)
                if f.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                if fpath in processed:
                    continue

                # Wait for file to finish writing
                prev_size = -1
                while True:
                    curr_size = f.stat().st_size
                    if curr_size == prev_size and curr_size > 0:
                        break
                    prev_size = curr_size
                    time.sleep(0.5)

                processed.add(fpath)
                print(f"\nNew image detected: {f.name}")

                try:
                    result = predict_image(api_url, fpath, token)
                    print(format_result(result, fpath))
                except requests.exceptions.RequestException as e:
                    print(f"API error for {f.name}: {e}")
                except Exception as e:
                    print(f"Error processing {f.name}: {e}")

            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print(f"\n\nStopped. Processed {len(processed)} images total.")


def main():
    parser = argparse.ArgumentParser(
        description="PlantGuard AI - Watch folder for drone/phone images and auto-analyze"
    )
    parser.add_argument(
        "--watch-dir", "-w",
        default="./drone_photos",
        help="Folder to watch for new images (default: ./drone_photos)"
    )
    parser.add_argument(
        "--api-url", "-u",
        default=DEFAULT_API_URL,
        help=f"PlantGuard API URL (default: {DEFAULT_API_URL})"
    )
    parser.add_argument(
        "--email", "-e",
        help="Supabase account email (for saving to history)"
    )
    parser.add_argument(
        "--password", "-p",
        help="Supabase account password"
    )
    args = parser.parse_args()

    token = None
    if args.email and args.password:
        print(f"Logging in as {args.email}...")
        try:
            token = login(args.api_url, args.email, args.password)
            print("Authenticated - results will be saved to history")
        except Exception as e:
            print(f"Login failed: {e}")
            print("Continuing without auth (results won't be saved to history)")

    watch_folder(args.watch_dir, args.api_url, token)


if __name__ == "__main__":
    main()
