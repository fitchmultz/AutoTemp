import json
import logging
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Tuple

import gradio as gr
import openai
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class PromptAnalyzer:
    PROMPT_TYPES = {
        "creative_writing": {
            "keywords": ["write", "story", "poem", "creative", "imagine", "describe"],
            "indicators": ["write a", "tell a", "create a", "compose"],
            "temps": [0.7, 0.9, 1.1],
            "top_p": 0.9,
            "freq_penalty": 0.3,
        },
        "technical_explanation": {
            "keywords": ["explain", "how", "what", "why", "technical", "concept"],
            "indicators": ["explain", "what is", "how does", "define"],
            "temps": [0.3, 0.5, 0.7],
            "top_p": 0.7,
            "freq_penalty": 0.0,
        },
        "business_formal": {
            "keywords": ["email", "draft", "business", "formal", "professional"],
            "indicators": ["write an email", "draft a", "compose a letter"],
            "temps": [0.4, 0.6, 0.8],
            "top_p": 0.8,
            "freq_penalty": 0.1,
        },
        "brainstorming": {
            "keywords": [
                "ideas",
                "brainstorm",
                "suggest",
                "innovative",
                "possibilities",
            ],
            "indicators": ["generate ideas", "come up with", "think of"],
            "temps": [0.8, 1.0, 1.2],
            "top_p": 1.0,
            "freq_penalty": 0.5,
        },
        "humor_casual": {
            "keywords": ["joke", "funny", "humor", "laugh", "chicken", "knock knock"],
            "indicators": ["tell a joke", "why did", "what do you call"],
            "temps": [0.6, 0.8, 1.0],  # Balanced for humor
            "top_p": 0.9,  # High for creative responses
            "freq_penalty": 0.3,  # Moderate to avoid repetition
        },
    }

    @classmethod
    def analyze_prompt(cls, prompt: str) -> Tuple[str, Dict]:
        """
        Analyze the prompt and return the most likely prompt type + recommended parameters.
        This version does slightly improved scoring logic.
        """
        prompt_lower = prompt.lower()
        scores = {}

        for prompt_type, config in cls.PROMPT_TYPES.items():
            # Score by presence of keywords
            keyword_score = sum(
                2 for keyword in config["keywords"] if keyword in prompt_lower
            )
            # Score by presence of indicators
            indicator_score = sum(
                3 for indicator in config["indicators"] if indicator in prompt_lower
            )

            # Basic length-based weighting
            token_count = len(prompt.split())
            length_score = 0
            if token_count < 8:
                # Short prompts = possibly casual/humor
                if prompt_type == "humor_casual":
                    length_score += 2
            else:
                # For bigger prompts, prefer categories like tech/business/creative
                if prompt_type in [
                    "technical_explanation",
                    "creative_writing",
                    "business_formal",
                ]:
                    length_score += 1

            total_score = keyword_score + indicator_score + length_score
            scores[prompt_type] = total_score

        best_type = max(scores.items(), key=lambda x: x[1])[0]
        if scores[best_type] == 0:
            best_type = "creative_writing"  # fallback

        # Round out the config floats
        config = cls.PROMPT_TYPES[best_type].copy()
        config["temps"] = [round(t, 1) for t in config["temps"]]
        config["top_p"] = round(config["top_p"], 1)
        config["freq_penalty"] = round(config["freq_penalty"], 1)

        return best_type, config


class ParameterHistory:
    """
    Enhanced to do exponential weighting so that we prefer recent successes
    but still keep older data around for diversity.
    """

    def __init__(self, history_file="flagged/parameter_history.json"):
        self.history_file = history_file
        self.ensure_history_file()

    def ensure_history_file(self):
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        if not os.path.exists(self.history_file):
            with open(self.history_file, "w") as f:
                json.dump({"prompt_types": {}}, f)

    def update_history(self, prompt_type: str, params: Dict, score: float):
        try:
            with open(self.history_file, "r") as f:
                history = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            history = {"prompt_types": {}}

        if prompt_type not in history["prompt_types"]:
            history["prompt_types"][prompt_type] = []

        history["prompt_types"][prompt_type].append(
            {
                "params": params,
                "score": score,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Keep only the last 100 entries per type
        history["prompt_types"][prompt_type] = history["prompt_types"][prompt_type][
            -100:
        ]

        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2)

    def get_best_params(self, prompt_type: str) -> Dict:
        """
        Returns a dict with the average best parameters, weighted more heavily by newer entries.
        """
        try:
            with open(self.history_file, "r") as f:
                history = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None

        if (
            prompt_type not in history["prompt_types"]
            or not history["prompt_types"][prompt_type]
        ):
            return None

        entries = history["prompt_types"][prompt_type]

        # Sort by timestamp ascending, so we can exponentially weight
        # older entries get less weight
        def parse_time(e):
            return datetime.fromisoformat(e["timestamp"])

        entries.sort(key=lambda x: parse_time(x))
        weighted_sum_temp = 0.0
        weighted_sum_top_p = 0.0
        weighted_sum_freq = 0.0
        weighted_sum_weight = 0.0

        # Exponential weighting factor
        # e.g. alpha=0.85 means older entries quickly degrade in influence
        alpha = 0.85
        weight = 1.0
        for entry in entries[::-1]:  # newest first
            p = entry["params"]
            s = entry["score"]
            weighted_sum_temp += p["temperature"] * weight
            weighted_sum_top_p += p["top_p"] * weight
            weighted_sum_freq += p["frequency_penalty"] * weight
            weighted_sum_weight += weight
            weight *= alpha  # degrade as we go older

        return {
            "temperature": weighted_sum_temp / weighted_sum_weight,
            "top_p": weighted_sum_top_p / weighted_sum_weight,
            "frequency_penalty": weighted_sum_freq / weighted_sum_weight,
        }


class CustomFlaggingCallback(gr.FlaggingCallback):
    def setup(self, components, flagging_dir: str):
        self.flagging_dir = "flagged"
        os.makedirs(self.flagging_dir, exist_ok=True)
        self.log_file = os.path.join(self.flagging_dir, "flagged_data.json")
        logging.info(f"Flagging directory set up: {self.flagging_dir}")

    def flag(self, flag_data, flag_option=None, flag_index=None, username=None):
        logging.debug(f"Received flag_data: {flag_data}")
        logging.debug(
            f"flag_option: {flag_option}, flag_index: {flag_index}, username: {username}"
        )

        try:
            entry = {
                "metadata": {
                    "prompt": flag_data[0] if len(flag_data) > 0 else "",
                    "temperature_string": flag_data[1] if len(flag_data) > 1 else "",
                    "top_p": flag_data[2] if len(flag_data) > 2 else None,
                    "auto_select": flag_data[3] if len(flag_data) > 3 else None,
                    "model_version": flag_data[5] if len(flag_data) > 5 else "unknown",
                    "timestamp": datetime.now().isoformat(),
                },
                "results": self.parse_output(
                    flag_data[4] if len(flag_data) > 4 else ""
                ),
            }

            logging.debug(f"Created entry: {entry}")

            if os.path.exists(self.log_file):
                with open(self.log_file, "r+") as file:
                    try:
                        file_data = json.load(file)
                    except json.JSONDecodeError:
                        logging.warning(
                            "Could not decode existing JSON. Starting fresh."
                        )
                        file_data = []
                    file_data.append(entry)
                    file.seek(0)
                    json.dump(file_data, file, indent=2)
            else:
                with open(self.log_file, "w") as file:
                    json.dump([entry], file, indent=2)

            logging.info(f"Flagged and saved to {self.log_file}")
            return f"Flagged and saved to {self.log_file}"

        except Exception as e:
            logging.error(f"Error while flagging: {str(e)}", exc_info=True)
            return f"Error while flagging: {str(e)}"

    def parse_output(self, output):
        """
        Attempt to parse the multiple outputs that were displayed.
        This regex might need improvement if your output format changes.
        """
        results = []
        pattern = (
            r"Temp (\d+\.\d+) \| Top-p (\d+\.\d+) \| Score: (\d+\.\d+):(.*?)(?=Temp|\Z)"
        )
        matches = re.findall(pattern, output, re.DOTALL)

        for match in matches:
            temperature, top_p, score, text = match
            results.append(
                {
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "score": float(score),
                    "output": text.strip(),
                }
            )

        return results


class RateLimiter:
    def __init__(self, max_requests_per_minute=60):
        self.max_requests_per_minute = max_requests_per_minute
        self.requests = []
        self.min_delay = 1.0 / (max_requests_per_minute / 60.0)
        self.last_request_time = 0

    def wait_if_needed(self):
        """
        Wait if we're exceeding our rate limit. Also enforces a minimal delay
        between consecutive requests.
        """
        current_time = time.time()
        # Remove requests older than 60s
        self.requests = [t for t in self.requests if current_time - t < 60]
        # If at limit, wait
        if len(self.requests) >= self.max_requests_per_minute:
            sleep_time = self.requests[0] + 60 - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        # Also enforce a minimal gap
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_delay:
            time.sleep(self.min_delay - time_since_last)

        self.requests.append(time.time())
        self.last_request_time = time.time()


class RequestTracker:
    def __init__(self):
        self.request_times = []
        self.error_counts = {}

    def track_request(self, duration: float, error: str = None):
        self.request_times.append(duration)
        if error:
            self.error_counts[error] = self.error_counts.get(error, 0) + 1

    def get_stats(self) -> Dict:
        if not self.request_times:
            return {"avg_time": 0, "max_time": 0, "min_time": 0, "total_requests": 0}

        return {
            "avg_time": sum(self.request_times) / len(self.request_times),
            "max_time": max(self.request_times),
            "min_time": min(self.request_times),
            "total_requests": len(self.request_times),
            "error_counts": self.error_counts,
        }


class AutoTemp:
    def __init__(
        self,
        auto_select=False,
        max_workers=12,
        model_version=None,
        max_retries=5,
        initial_retry_delay=1.0,
        max_retry_delay=32.0,
        max_requests_per_minute=90,
        max_generations=9,
        short_circuit_score=95.0,  # if we find a combo scoring above this, we can short-circuit
    ):
        # Setup API key
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set in environment.")
        openai.api_key = self.api_key

        # Model version
        self.model_version = model_version or os.getenv("OPENAI_MODEL", "gpt-4o")

        self.auto_select = auto_select
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay
        self.max_generations = max_generations
        self.short_circuit_score = short_circuit_score

        self.prompt_analyzer = PromptAnalyzer()
        self.parameter_history = ParameterHistory()
        self.rate_limiter = RateLimiter(max_requests_per_minute)
        self.request_tracker = RequestTracker()

    def get_recommended_params(self, prompt: str, max_combos: int = None):
        prompt_type, base_config = self.prompt_analyzer.analyze_prompt(prompt)
        historical_params = self.parameter_history.get_best_params(prompt_type)

        # If user has a max_generations limit, interpret that as the max combos we want to test
        if not max_combos:
            max_combos = self.max_generations

        # Basic fallback approach if no historical data
        if not historical_params:
            base_temps = base_config["temps"]
            base_top_p = [round(base_config["top_p"], 1)]
            base_freq = [round(base_config["freq_penalty"], 1)]

            # We'll just do a small grid around these defaults
            # This is an example of a naive approach with a +/- 0.2 offset if possible
            temp_candidates = set()
            for t in base_temps:
                temp_candidates.add(round(t, 1))
                temp_candidates.add(round(min(2.0, t + 0.2), 1))
                temp_candidates.add(round(max(0.0, t - 0.2), 1))

            top_p_candidates = set()
            for p in base_top_p:
                top_p_candidates.add(p)
                up = round(min(1.0, p + 0.1), 1)
                down = round(max(0.0, p - 0.1), 1)
                top_p_candidates.add(up)
                top_p_candidates.add(down)

            freq_candidates = set()
            for f in base_freq:
                freq_candidates.add(f)
                freq_candidates.add(round(min(2.0, f + 0.2), 1))
                freq_candidates.add(round(max(-2.0, f - 0.2), 1))

        else:
            # Generate candidates around the historical best with random or +/- offsets
            bt = round(historical_params["temperature"], 1)
            bp = round(historical_params["top_p"], 1)
            bf = round(historical_params["frequency_penalty"], 1)

            def clamp(val, low, high):
                return round(min(high, max(low, val)), 1)

            temp_candidates = {clamp(bt + i * 0.2, 0.0, 2.0) for i in [-2, -1, 0, 1, 2]}
            top_p_candidates = {
                clamp(bp + i * 0.1, 0.0, 1.0) for i in [-2, -1, 0, 1, 2]
            }
            freq_candidates = {clamp(bf + i * 0.2, -2.0, 2.0) for i in [-1, 0, 1]}

        # Final sets
        temps_sorted = sorted(temp_candidates)
        top_p_sorted = sorted(top_p_candidates)
        freq_sorted = sorted(freq_candidates)

        # If total combos exceed max_combos, randomly sample to keep cost in check
        import random

        combos = [
            (t, p, f) for t in temps_sorted for p in top_p_sorted for f in freq_sorted
        ]
        if len(combos) > max_combos:
            combos = random.sample(combos, max_combos)

        return prompt_type, combos

    def generate_with_openai(self, prompt, temp, top_p, freq_pen, retries=None):
        if retries is None:
            retries = self.max_retries

        start_time = time.time()
        last_error = None
        retry_delay = self.initial_retry_delay
        client = OpenAI(api_key=self.api_key)

        while retries > 0:
            try:
                self.rate_limiter.wait_if_needed()
                logging.info(f"Generating with T={temp}, P={top_p}, F={freq_pen}")
                response = client.chat.completions.create(
                    model=self.model_version,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temp,
                    top_p=top_p,
                    frequency_penalty=freq_pen,
                )

                duration = time.time() - start_time
                self.request_tracker.track_request(duration)
                message = response.choices[0].message.content
                return message.strip()

            except openai.RateLimitError as e:
                last_error = "RateLimitError"
                logging.warning(f"Rate limit error: {e}")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self.max_retry_delay)

            except openai.APIError as e:
                last_error = "APIError"
                logging.error(f"API error: {e}")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self.max_retry_delay)

            except Exception as e:
                last_error = "UnexpectedError"
                logging.error(f"Unexpected error: {e}", exc_info=True)
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self.max_retry_delay)

            retries -= 1
            self.request_tracker.track_request(time.time() - start_time, last_error)

        # If we exhausted retries
        return f"Error: {last_error or 'Unknown'} generating text at T={temp}, P={top_p}, F={freq_pen}"

    def evaluate_output(self, output, temp, top_p, freq_pen):
        """
        Evaluate the output with a separate call. If the output is blank or an error, return a low score.
        """
        if not output or output.startswith("Error"):
            return 0.0

        eval_prompt = f"""
You are an extremely strict evaluator. Given the text below, produce a single float (0.0 to 100.0),
with one decimal place, that measures how relevant, clear, useful, pride-worthy, and delightful
the text is, all equally weighted. Do NOT add any explanation. Just the numeric score.

Text:
---
{output}
---
"""

        # We'll fix top_p at 1.0 to avoid messing with the numeric generation
        # We'll keep the same freq_pen so it doesn't drastically alter repetition behavior.
        eval_response = self.generate_with_openai(eval_prompt, 1.0, 1.0, freq_pen)

        # Extract a numeric score from 0.0 to 100.0
        # We'll do a more robust pattern: look for up to 1 decimal place
        match = re.search(r"\b(\d{1,3}\.\d)\b|\b(\d{1,3})\b", eval_response)
        if not match:
            return 0.0

        # match.group() might be "95.4" or "85"
        try:
            possible_score = match.group().strip()
            score_val = float(possible_score)
            if score_val < 0.0:
                return 0.0
            if score_val > 100.0:
                return 100.0
            return round(score_val, 1)
        except:
            return 0.0

    def run(self, prompt, temperature_string, top_p_string, freq_string):
        # Analyze prompt & get combos
        prompt_type, combos = self.get_recommended_params(prompt, self.max_generations)
        logging.info(f"Detected prompt type: {prompt_type}")
        logging.info(f"Testing {len(combos)} param combos")

        # If user forcibly entered param combos, parse them instead
        if temperature_string.strip():
            t_list = [float(s.strip()) for s in temperature_string.split(",")]
            t_list = [round(x, 1) for x in t_list]
        else:
            # All distinct temps from combos
            t_list = sorted(list({c[0] for c in combos}))

        if top_p_string.strip():
            p_list = [float(s.strip()) for s in top_p_string.split(",")]
            p_list = [round(x, 1) for x in p_list]
        else:
            p_list = sorted(list({c[1] for c in combos}))

        if freq_string.strip():
            f_list = [float(s.strip()) for s in freq_string.split(",")]
            f_list = [round(x, 1) for x in f_list]
        else:
            f_list = sorted(list({c[2] for c in combos}))

        # Build the final combos from user-specified or recommended sets
        final_combos = [(t, p, f) for t in t_list for p in p_list for f in f_list]
        logging.info(f"Final # of combos after user override: {len(final_combos)}")

        # 1) Generate text for each combo in parallel
        outputs = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {
                executor.submit(self.generate_with_openai, prompt, t, p, f): (t, p, f)
                for (t, p, f) in final_combos
            }
            for fut in as_completed(future_map):
                (temp, top_p, freq) = future_map[fut]
                try:
                    result = fut.result()
                    outputs[(temp, top_p, freq)] = result
                except Exception as e:
                    logging.error(f"Generation error for {temp},{top_p},{freq}: {e}")
                    outputs[(temp, top_p, freq)] = f"Error: Generation exception"

        # 2) Evaluate each output in parallel
        #    We can short-circuit if we find a super high score
        scores = {}
        stop_early = False
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {}
            for (temp, top_p, freq), gen_text in outputs.items():
                future_map[
                    executor.submit(self.evaluate_output, gen_text, temp, top_p, freq)
                ] = (temp, top_p, freq)

            for fut in as_completed(future_map):
                (temp, top_p, freq) = future_map[fut]
                try:
                    score_val = fut.result()
                    scores[(temp, top_p, freq)] = score_val

                    # If good enough, we can short-circuit if desired
                    if (
                        self.short_circuit_score
                        and score_val >= self.short_circuit_score
                    ):
                        logging.info(f"Short-circuit triggered at {score_val} score.")
                        stop_early = True
                except Exception as e:
                    logging.error(f"Evaluation error for {temp},{top_p},{freq}: {e}")
                    scores[(temp, top_p, freq)] = 0.0

                if stop_early:
                    break

        # 3) If we short-circuited, fill missing combos with minimal or zero scores
        if stop_early:
            for combo in final_combos:
                if combo not in scores:
                    scores[combo] = -1.0  # or some sentinel

        # 4) Update parameter history with any combos that scored >80
        for combo, sc in scores.items():
            if sc > 80.0:
                t, p, f = combo
                self.parameter_history.update_history(
                    prompt_type,
                    {"temperature": t, "top_p": p, "frequency_penalty": f},
                    sc,
                )

        # Summaries
        stats = self.request_tracker.get_stats()
        logging.info(f"Request stats: {stats}")

        if not scores:
            return "No outputs produced. Something went wrong."

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_combo, best_score = sorted_results[0]
        best_output = outputs[best_combo]

        # 5) Auto-select or show all
        if self.auto_select:
            return (
                f"***Auto-Selected Best Output***\n\n"
                f"**Score**: {best_score}\n"
                f"**Params**: Temp={best_combo[0]}, Top-p={best_combo[1]}, Freq={best_combo[2]}\n\n"
                f"**Output**:\n{best_output}\n\n"
                f"**Stats**: "
                f"TotalRequests={stats['total_requests']}, AvgTime={stats['avg_time']:.2f}s, MaxTime={stats['max_time']:.2f}s"
            )
        else:
            # Build big output listing
            lines = []
            for combo, sc in sorted_results:
                out = outputs[combo]
                lines.append(
                    f"────────────────────────────────────────\n"
                    f"Temp={combo[0]}, Top-p={combo[1]}, Freq={combo[2]}, Score={sc}\n"
                    f"Output:\n{out}\n"
                )
            summary = (
                f"\n\n***Summary***\n"
                f"• Prompt Type: {prompt_type}\n"
                f"• Combos Tested: {len(scores)}\n"
                f"• Best Score: {best_score} @ {best_combo}\n"
                f"• Total API Requests: {stats['total_requests']}\n"
                f"• Avg Response Time: {stats['avg_time']:.2f}s\n"
                f"• Max Response Time: {stats['max_time']:.2f}s\n"
            )
            return "".join(lines) + summary


def run_autotemp(
    prompt,
    temperature_string,
    top_p_string,
    frequency_penalty_string,
    auto_select,
    max_generations,
):
    """
    This is the Gradio callback. We instantiate AutoTemp each time,
    but you could also do it once globally if you prefer.
    """
    try:
        agent = AutoTemp(auto_select=auto_select, max_generations=int(max_generations))

        # We can show recommended combos by analyzing once
        prompt_type, recommended_combos = agent.get_recommended_params(
            prompt, agent.max_generations
        )
        # Distill the recommended combos for quick display
        recommended_temps = sorted(list({c[0] for c in recommended_combos}))
        recommended_ps = sorted(list({c[1] for c in recommended_combos}))
        recommended_fs = sorted(list({c[2] for c in recommended_combos}))

        param_info = (
            f"Detected prompt type: {prompt_type}\n\n"
            f"Recommended combos:\n"
            f"  Temperatures: {recommended_temps}\n"
            f"  Top-p:        {recommended_ps}\n"
            f"  FreqPenalty:  {recommended_fs}\n\n"
            "You can override any of these by typing comma-separated values."
        )

        output = agent.run(
            prompt, temperature_string, top_p_string, frequency_penalty_string
        )
        # Return the final text plus the model version
        final = f"{param_info}\n\n{output}"
        return final, agent.model_version

    except Exception as e:
        msg = f"Error processing request: {str(e)}"
        logging.error(msg, exc_info=True)
        return msg, "error"


def main():
    iface = gr.Interface(
        fn=run_autotemp,
        inputs=[
            gr.Textbox(
                label="Prompt", placeholder="Enter your prompt here...", lines=3
            ),
            gr.Textbox(
                label="Temperature Values",
                placeholder="Comma-separated or leave blank for recommended",
                value="",
                lines=1,
            ),
            gr.Textbox(
                label="Top-p Values",
                placeholder="Comma-separated or leave blank for recommended",
                value="",
                lines=1,
            ),
            gr.Textbox(
                label="Frequency Penalties",
                placeholder="Comma-separated or leave blank for recommended",
                value="",
                lines=1,
            ),
            gr.Checkbox(
                label="Auto-select Best Output",
                value=False,
            ),
            gr.Number(
                label="Maximum Generations",
                value=9,
                minimum=1,
                maximum=50,
                step=1,
                info="Max # of parameter combos to test. Each combo does 2 calls (gen+eval).",
            ),
        ],
        outputs=[
            gr.Textbox(label="Generated Output", lines=20),
            gr.Textbox(label="Model Version", lines=1),
        ],
        title="AutoTemp: Smarter LLM Param Tuning",
        description=(
            "Automatically detect your prompt type, propose parameter combos, "
            "and evaluate the best results. This version includes short-circuiting, "
            "exponential weighting of historical data, and more robust search."
        ),
        examples=[],
        flagging_callback=CustomFlaggingCallback(),
        flagging_options=["for review"],
    )
    iface.launch()


if __name__ == "__main__":
    main()
