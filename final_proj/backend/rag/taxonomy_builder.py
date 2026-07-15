import os
import json
import urllib.request
from ..config import settings

SEED_SCENES = [
    "parking lot at night",
    "indoor hospital or office corridor",
    "building entrance / lobby",
    "retail store floor",
    "outdoor pedestrian pathway",
    "loading dock / warehouse area",
]

UCF_CRIME_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
    "Fighting", "Road Accident", "Robbery", "Shooting", "Shoplifting",
    "Stealing", "Vandalism",
]

PROMPT_TEMPLATE = """You are building a structured anomaly-knowledge entry for a
video surveillance system. Ground your answer in real-world plausibility,
not fiction.

Scene: {scene}
Reference anomaly categories (pick only the ones plausible for this scene): {categories}

Return ONLY valid JSON in this exact shape, no other text:
{{
  "scene": "{scene}",
  "normal_behaviors": ["...", "..."],
  "anomaly_types": ["...", "..."],
  "evidence_needed": ["...", "..."],
  "recommended_tools": ["...", "..."]
}}
"""

def generate_entry_via_ollama(scene: str) -> dict:
    url = f"{settings.OLLAMA_BASE_URL}/api/generate"
    prompt = PROMPT_TEMPLATE.format(
        scene=scene,
        categories=", ".join(UCF_CRIME_CATEGORIES)
    )
    
    payload = {
        "model": settings.OLLAMA_MODEL_REASONING,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.1}
    }
    
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    try:
        # Use http client with timeout to prevent hang
        with urllib.request.urlopen(req, timeout=15) as res:
            res_data = json.loads(res.read().decode("utf-8"))
            content = res_data.get("response", "")
            return json.loads(content)
    except Exception as e:
        print(f"[TaxonomyBuilder] Ollama call failed for scene '{scene}': {e}. Using robust static fallbacks.")
        
        # Safe offline fallback mapping
        fallbacks = {
            "parking lot at night": {
                "scene": "parking lot at night",
                "normal_behaviors": ["Cars parking", "People walking to cars", "Security patrols"],
                "anomaly_types": ["Assault", "Burglary", "Stealing", "Vandalism"],
                "evidence_needed": ["Sudden physical movements", "Car window glass breakage", "Person lying on ground", "Loitering near vehicle"],
                "recommended_tools": ["OCR", "DynamicYOLO", "CrowdAnalytics"]
            },
            "indoor hospital or office corridor": {
                "scene": "indoor hospital or office corridor",
                "normal_behaviors": ["Nurses walking", "Patients in wheel chairs", "Staff carrying folders"],
                "anomaly_types": ["Abuse", "Assault", "Arrest"],
                "evidence_needed": ["Person lying on floor", "Aggressive posturing", "Rapid running in hallway"],
                "recommended_tools": ["CrowdAnalytics", "OCR", "AttentionRollout"]
            },
            "building entrance / lobby": {
                "scene": "building entrance / lobby",
                "normal_behaviors": ["People entering turnstiles", "Receptionist working", "Visitors checking in"],
                "anomaly_types": ["Arrest", "Assault", "Robbery", "Vandalism"],
                "evidence_needed": ["Clustering near entrance", "Unattended bags", "Forced entrance", "Physical struggle"],
                "recommended_tools": ["OCR", "DynamicYOLO", "CrowdAnalytics"]
            },
            "retail store floor": {
                "scene": "retail store floor",
                "normal_behaviors": ["Browsing shelves", "Placing items in cart", "Checkout queues"],
                "anomaly_types": ["Shoplifting", "Stealing", "Robbery", "Assault"],
                "evidence_needed": ["Concealing items", "Running towards exit", "Cashier distress hand signals"],
                "recommended_tools": ["OCR", "DynamicYOLO", "CrowdAnalytics"]
            },
            "outdoor pedestrian pathway": {
                "scene": "outdoor pedestrian pathway",
                "normal_behaviors": ["Joggers running", "Bicyclists riding", "Pedestrians walking dogs"],
                "anomaly_types": ["Assault", "Robbery", "Fighting", "Vandalism"],
                "evidence_needed": ["People clustering or scattering", "Physical combat", "Screaming facial expressions"],
                "recommended_tools": ["CrowdAnalytics", "DynamicYOLO", "AttentionRollout"]
            },
            "loading dock / warehouse area": {
                "scene": "loading dock / warehouse area",
                "normal_behaviors": ["Forklifts moving pallets", "Trucks backing up", "Workers wearing safety vests"],
                "anomaly_types": ["Burglary", "Stealing", "Vandalism", "Arson"],
                "evidence_needed": ["Smoke rising", "Loitering near shipping docks at night", "Forced container locks"],
                "recommended_tools": ["DynamicYOLO", "OCR", "CrowdAnalytics"]
            }
        }
        return fallbacks.get(scene, {
            "scene": scene,
            "normal_behaviors": ["Standard activities"],
            "anomaly_types": ["General Anomaly"],
            "evidence_needed": ["Unusual activity signature"],
            "recommended_tools": ["AttentionRollout"]
        })

def build_taxonomy_json(output_path: str):
    print("[TaxonomyBuilder] Seeding taxonomy entries...")
    entries = []
    for scene in SEED_SCENES:
        entry = generate_entry_via_ollama(scene)
        entries.append(entry)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    print(f"[TaxonomyBuilder] Wrote {len(entries)} entries to {output_path}")
