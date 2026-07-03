# IBM Granite telemetry analysis: per-corner reports (entry/apex/exit speeds)
# and natural-language driving critiques generated from telemetry CSVs.

import os
import sys
import csv
import json
import argparse
import glob
import time
from datetime import datetime
from typing import Optional

import numpy as np

# ==================================================================
# Configuration
# ==================================================================

GRANITE_MODEL_ID  = "ibm/granite-3-3-8b-instruct"   # adjust to available model
OLLAMA_MODEL      = "granite4:350m"
MAX_TOKENS        = 1500
TEMPERATURE       = 0.2    # low temperature for factual analysis

# ==================================================================
# Telemetry summarisation (produce compact stats for the prompt)
# ==================================================================

def summarise_car_telemetry(csv_path: str, max_rows: int = 200_000) -> dict:
    # Read a Car_telemetry CSV and compute summary statistics.
    # Designed to be compact enough to fit in a Granite context window.
    #
    # Returns a dict with: n_episodes, n_steps, episode_dist_stats,
    # top_offtrack_dist, max_speed, lap_times, reward_stats, termination_causes.
    episodes     = {}
    lap_times    = []
    all_rewards  = []
    all_speeds   = []
    n_steps      = 0
    term_causes  = {"backwards": 0, "offtrack": 0, "stuck": 0}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        prev_ep = None
        ep_rows = []

        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            n_steps += 1

            ep   = row.get("episode", "0")
            dist = float(row.get("distRaced", 0.0))
            spd  = float(row.get("speedX", 0.0))
            llt  = float(row.get("lastLapTime", 0.0))
            rwd  = float(row.get("reward", 0.0))
            ang  = float(row.get("angle", 0.0))
            tp   = float(row.get("trackPos", 0.0))

            all_speeds.append(spd)
            all_rewards.append(rwd)

            if ep not in episodes:
                episodes[ep] = {"max_dist": 0.0, "last_row": None}
            if dist > episodes[ep]["max_dist"]:
                episodes[ep]["max_dist"] = dist
            episodes[ep]["last_row"] = row

            if llt > 0:
                lap_times.append(llt)

        # Termination cause per episode (classify by final-row state)
        for ep_data in episodes.values():
            r = ep_data["last_row"]
            if r is None:
                continue
            ang = float(r.get("angle", 0.0))
            tp  = float(r.get("trackPos", 0.0))
            import math
            if math.cos(ang) < 0:
                term_causes["backwards"] += 1
            elif abs(tp) > 1.1:
                term_causes["offtrack"] += 1
            else:
                term_causes["stuck"] += 1

    ep_dists = sorted([e["max_dist"] for e in episodes.values()], reverse=True)

    return {
        "n_episodes":        len(episodes),
        "n_steps":           n_steps,
        "max_dist_m":        float(ep_dists[0]) if ep_dists else 0.0,
        "avg_ep_dist_m":     float(np.mean(ep_dists)) if ep_dists else 0.0,
        "top5_ep_dists_m":   [round(d, 1) for d in ep_dists[:5]],
        "lap_times_s":       sorted(lap_times),
        "n_laps":            len(lap_times),
        "best_lap_s":        float(min(lap_times)) if lap_times else None,
        "max_speed_kmh":     float(max(all_speeds)) if all_speeds else 0.0,
        "avg_reward":        float(np.mean(all_rewards)) if all_rewards else 0.0,
        "reward_std":        float(np.std(all_rewards)) if all_rewards else 0.0,
        "termination_causes": term_causes,
        "source_file":       os.path.basename(csv_path),
    }


def summarise_neuron_telemetry(csv_path: str) -> dict:
    # Read a Neuron_telemetry CSV and compute training diagnostic statistics.
    critic_losses  = []
    actor_losses   = []
    entropy_coefs  = []
    mean_q_values  = []

    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    cl = float(row.get("critic_loss", "nan"))
                    al = float(row.get("actor_loss",  "nan"))
                    ec = float(row.get("entropy_coef","nan"))
                    qv = float(row.get("mean_q_value","nan"))
                    if not (cl != cl): critic_losses.append(cl)   # nan check
                    if not (al != al): actor_losses.append(al)
                    if not (ec != ec): entropy_coefs.append(ec)
                    if not (qv != qv): mean_q_values.append(qv)
                except (ValueError, TypeError):
                    pass
    except FileNotFoundError:
        return {"error": f"File not found: csv_path"}

    def safe_stats(arr):
        if not arr:
            return {"mean": None, "max": None, "min": None}
        return {"mean": round(float(np.mean(arr)), 4),
                "max":  round(float(np.max(arr)), 4),
                "min":  round(float(np.min(arr)), 4)}

    return {
        "n_records":     max(len(critic_losses), len(actor_losses), 1),
        "critic_loss":   safe_stats(critic_losses),
        "actor_loss":    safe_stats(actor_losses),
        "entropy_coef":  safe_stats(entropy_coefs),
        "mean_q_value":  safe_stats(mean_q_values),
        "source_file":   os.path.basename(csv_path),
    }


# ==================================================================
# Prompt construction
# ==================================================================

SYSTEM_PROMPT = """You are an expert in reinforcement learning for autonomous racing.
You analyze TORCS simulator telemetry data from a Soft Actor-Critic (SAC) agent
training on the Laguna Seca Corkscrew circuit (3602 m lap). Your role is to identify
training problems, diagnose failure modes, and provide specific, actionable
recommendations for improving the reward function, controller parameters, or training
configuration.

Be specific. Reference numbers from the data. Prioritize the most impactful fixes.
Format your output as structured JSON with keys: "diagnosis", "top_issues" (list),
"reward_recommendations" (list), "controller_recommendations" (list), "training_recommendations" (list).
"""


def build_analysis_prompt(car_summary: dict, neuron_summary: Optional[dict] = None) -> str:
    # Build the Granite analysis prompt from telemetry summaries.

    car_json    = json.dumps(car_summary,    indent=2)
    neuron_json = json.dumps(neuron_summary, indent=2) if neuron_summary else "Not available."

    return f"""Analyze the following TORCS SAC training telemetry and provide a structured diagnosis.

## Car Telemetry Summary
{car_json}

## Neural Network Training Stats
{neuron_json}

## Context
- Target: complete a lap under 1:20 (80 seconds) on Laguna Seca Corkscrew (3602 m)
- Algorithm: SAC with behavior-cloning initialization from a rule-based teacher
- Current problem: car is not completing laps; average episode distance is very short
- The track has a famous blind double-apex downhill Corkscrew section at around 1200 m

Based on this telemetry data, provide your structured analysis in JSON format with:
- "diagnosis": 1-2 sentence summary of the main problem
- "top_issues": list of the 3-5 most critical issues identified from the data
- "reward_recommendations": specific changes to reward weights/thresholds
- "controller_recommendations": specific changes to the teacher controller parameters
- "training_recommendations": specific changes to SAC hyperparameters or training setup
"""


# ==================================================================
# Granite inference backends
# ==================================================================

def _call_watsonx(prompt: str) -> str:
    # Call IBM WatsonX Granite API.
    api_key    = os.environ.get("WATSONX_API_KEY")
    url        = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    project_id = os.environ.get("WATSONX_PROJECT_ID")

    if not api_key or not project_id:
        raise ValueError(
            "Set WATSONX_API_KEY and WATSONX_PROJECT_ID environment variables. "
            "Get them from cloud.ibm.com → WatsonX → Credentials."
        )

    try:
        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    except ImportError:
        raise ImportError(
            "Install WatsonX SDK: pip install ibm-watsonx-ai"
        )

    credentials = Credentials(url=url, api_key=api_key)
    client      = APIClient(credentials=credentials, project_id=project_id)

    model = ModelInference(
        model_id=GRANITE_MODEL_ID,
        api_client=client,
        params={
            GenParams.MAX_NEW_TOKENS: MAX_TOKENS,
            GenParams.TEMPERATURE:    TEMPERATURE,
        },
    )

    response = model.generate_text(
        prompt=f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n"
    )
    return response


def _call_ollama(prompt: str) -> str:
    # Call a local Granite model via Ollama.
    try:
        import requests
    except ImportError:
        raise ImportError("pip install requests")

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
    }

    resp = requests.post("http://localhost:11434/api/chat", json=payload, timeout=120)
    if not resp.ok:
        raise ValueError(f"Ollama API Error ({resp.status_code}): {resp.text}")
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def call_granite(prompt: str) -> str:
    # Route to the correct Granite backend based on env vars.
    backend = os.environ.get("GRANITE_BACKEND", "ollama").lower()

    if backend == "ollama":
        print("[Granite] Using local Ollama backend")
        return _call_ollama(prompt)
    else:
        print("[Granite] Using IBM WatsonX backend")
        return _call_watsonx(prompt)


# ==================================================================
# Main analysis flow
# ==================================================================

def analyse_session(
    car_csv: str,
    neuron_csv: Optional[str] = None,
    output_dir: str = None,
    dry_run: bool = False,
) -> dict:
    # Run a full Granite analysis of a training session.
    #
    # Parameters
    # ----------
    # car_csv : str
    # Path to Car_telemetry CSV.
    # neuron_csv : str or None
    # Path to Neuron_telemetry CSV (optional).
    # output_dir : str or None
    # Directory to save the report JSON. Defaults to same dir as car_csv.
    # dry_run : bool
    # If True, build and print the prompt without calling Granite (for testing).
    #
    # Returns
    # -------
    # dict
    # Analysis result with summaries and Granite response.
    print(f"\n[Granite] Analysing: {os.path.basename(car_csv)}")

    print("  Summarising car telemetry...")
    car_summary = summarise_car_telemetry(car_csv)
    print(f"  {car_summary['n_episodes']} episodes | "
          f"max_dist={car_summary['max_dist_m']:.0f}m | "
          f"{car_summary['n_laps']} laps")

    neuron_summary = None
    if neuron_csv and os.path.exists(neuron_csv):
        print("  Summarising neuron telemetry...")
        neuron_summary = summarise_neuron_telemetry(neuron_csv)

    prompt = build_analysis_prompt(car_summary, neuron_summary)

    if dry_run:
        print("\n[Granite DRY RUN] Prompt that would be sent:")
        print("-" * 60)
        print(prompt[:2000], "..." if len(prompt) > 2000 else "")
        print("-" * 60)
        granite_response = "[DRY RUN — no API call made]"
    else:
        print("  Calling Granite API...")
        t0 = time.time()
        granite_response = call_granite(prompt)
        elapsed = time.time() - t0
        print(f"  Response received in {elapsed:.1f}s")

    # Try to parse JSON from the response
    parsed_response = None
    try:
        # Find the first '{' in the response and parse from there
        start = granite_response.find("{")
        if start >= 0:
            parsed_response = json.loads(granite_response[start:])
    except (json.JSONDecodeError, ValueError):
        parsed_response = {"raw_response": granite_response}

    result = {
        "timestamp":       datetime.now().isoformat(),
        "car_csv":         car_csv,
        "neuron_csv":      neuron_csv,
        "car_summary":     car_summary,
        "neuron_summary":  neuron_summary,
        "granite_analysis": parsed_response or {"raw": granite_response},
    }

    # Save report
    if output_dir is None:
        output_dir = os.path.dirname(car_csv)
    os.makedirs(output_dir, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"granite_report_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Granite] Report saved → {out_path}")

    # Pretty-print key findings
    if parsed_response and "diagnosis" in parsed_response:
        print(f"\n  DIAGNOSIS: {parsed_response.get('diagnosis', '')}")
        issues = parsed_response.get("top_issues", [])
        if issues:
            print("  TOP ISSUES:")
            for i, issue in enumerate(issues[:5], 1):
                print(f"    {i}. {issue}")

    return result


# ==================================================================
# CLI
# ==================================================================

def find_latest_telemetry(telemetry_dir: str):
    # Find the most recently written car and neuron telemetry files.
    car_files    = sorted(glob.glob(os.path.join(telemetry_dir, "Car_telemetry", "*.csv")),
                          key=os.path.getmtime)
    neuron_files = sorted(glob.glob(os.path.join(telemetry_dir, "Neuron_telemetry", "*.csv")),
                          key=os.path.getmtime)
    return (car_files[-1] if car_files else None,
            neuron_files[-1] if neuron_files else None)


def main():
    parser = argparse.ArgumentParser(
        description="IBM Granite AI telemetry analysis for SAC Racing"
    )
    parser.add_argument("--car-telemetry",    type=str, default=None,
                        help="Path to Car_telemetry CSV")
    parser.add_argument("--neuron-telemetry", type=str, default=None,
                        help="Path to Neuron_telemetry CSV")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-detect and analyse the most recent telemetry files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save the analysis report")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build and print the prompt without calling Granite")
    args = parser.parse_args()

    if args.auto:
        from config import Config
        car_csv, neuron_csv = find_latest_telemetry(Config.TELEMETRY_DIR)
        if not car_csv:
            print("No telemetry files found in", Config.TELEMETRY_DIR)
            sys.exit(1)
        print(f"[Granite] Auto-detected: {os.path.basename(car_csv)}")
    else:
        if not args.car_telemetry:
            parser.error("Provide --car-telemetry PATH or use --auto")
        car_csv    = args.car_telemetry
        neuron_csv = args.neuron_telemetry

    analyse_session(
        car_csv=car_csv,
        neuron_csv=neuron_csv,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
