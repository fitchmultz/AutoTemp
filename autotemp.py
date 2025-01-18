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
        """Analyze the prompt and return the most likely prompt type and recommended parameters."""
        prompt_lower = prompt.lower()
        scores = {}

        for prompt_type, config in cls.PROMPT_TYPES.items():
            # Score based on keywords
            keyword_score = sum(
                2 for keyword in config["keywords"] if keyword in prompt_lower
            )

            # Score based on phrase indicators (weighted higher)
            indicator_score = sum(
                3 for indicator in config["indicators"] if indicator in prompt_lower
            )

            # Score based on prompt length and complexity
            length_score = 0
            if len(prompt.split()) < 8:  # Short prompts are more likely casual/humor
                length_score = 2 if prompt_type in ["humor_casual"] else 0
            else:
                length_score = (
                    1
                    if prompt_type
                    in ["technical_explanation", "creative_writing", "business_formal"]
                    else 0
                )

            scores[prompt_type] = keyword_score + indicator_score + length_score

        # If no clear winner, default to creative_writing for safety
        best_type = max(scores.items(), key=lambda x: x[1])[0]
        if scores[best_type] == 0:
            best_type = "creative_writing"

        # Round all floating point values to 1 decimal place in the config
        config = cls.PROMPT_TYPES[best_type].copy()
        config["temps"] = [round(t, 1) for t in config["temps"]]
        config["top_p"] = round(config["top_p"], 1)
        config["freq_penalty"] = round(config["freq_penalty"], 1)

        return best_type, config


class ParameterHistory:
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
            {"params": params, "score": score, "timestamp": datetime.now().isoformat()}
        )

        # Keep only the last 100 entries per type
        history["prompt_types"][prompt_type] = history["prompt_types"][prompt_type][
            -100:
        ]

        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2)

    def get_best_params(self, prompt_type: str) -> Dict:
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

        # Get the top 5 highest scoring parameter combinations
        sorted_entries = sorted(
            history["prompt_types"][prompt_type], key=lambda x: x["score"], reverse=True
        )[:5]

        # Average the parameters of the top 5 entries
        avg_params = {
            "temperature": sum(
                entry["params"]["temperature"] for entry in sorted_entries
            )
            / len(sorted_entries),
            "top_p": sum(entry["params"]["top_p"] for entry in sorted_entries)
            / len(sorted_entries),
            "frequency_penalty": sum(
                entry["params"]["frequency_penalty"] for entry in sorted_entries
            )
            / len(sorted_entries),
        }

        return avg_params


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
                            f"Could not decode existing JSON in {self.log_file}. Starting with empty list."
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

    def parse_metadata(self, line):
        parts = line.split("|")
        temp = float(parts[0].split()[1]) if len(parts) > 0 else 0.0
        top_p = float(parts[1].split()[2]) if len(parts) > 1 else 1.0
        score = float(parts[2].split()[2]) if len(parts) > 2 else 0.0
        return temp, top_p, score


class RateLimiter:
    def __init__(self, max_requests_per_minute=60):
        self.max_requests_per_minute = max_requests_per_minute
        self.requests = []
        self.min_delay = 1.0 / (
            max_requests_per_minute / 60.0
        )  # Minimum delay between requests
        self.last_request_time = 0

    def wait_if_needed(self):
        """Wait if we're exceeding our rate limit"""
        current_time = time.time()

        # Remove requests older than 1 minute
        self.requests = [t for t in self.requests if current_time - t < 60]

        # If we're at the limit, wait
        if len(self.requests) >= self.max_requests_per_minute:
            sleep_time = self.requests[0] + 60 - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Ensure minimum delay between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_delay:
            time.sleep(self.min_delay - time_since_last)

        self.requests.append(current_time)
        self.last_request_time = time.time()


class RequestTracker:
    def __init__(self):
        self.request_times = []
        self.error_counts = {}

    def track_request(self, duration: float, error: str = None):
        """Track a request's duration and any errors"""
        self.request_times.append(duration)
        if error:
            self.error_counts[error] = self.error_counts.get(error, 0) + 1

    def get_stats(self) -> Dict:
        """Get statistics about requests"""
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
        default_temp=0.0,
        alt_temps=None,
        auto_select=True,
        max_workers=12,
        model_version=None,
        frequency_penalty=0.0,
        max_retries=5,
        initial_retry_delay=1.0,
        max_retry_delay=32.0,
        max_requests_per_minute=90,
        max_generations=None,
    ):
        # Get API key from environment
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set in the .env file or environment variables."
            )
        openai.api_key = self.api_key

        # Get model version from environment or use default
        self.model_version = model_version or os.getenv("OPENAI_MODEL", "gpt-4o")

        self.default_temp = default_temp
        self.alt_temps = alt_temps
        self.auto_select = auto_select
        self.max_workers = max_workers

        # Rate limiting and retry settings
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay

        self.frequency_penalty = frequency_penalty
        self.max_generations = max_generations

        self.prompt_analyzer = PromptAnalyzer()
        self.parameter_history = ParameterHistory()
        self.rate_limiter = RateLimiter(max_requests_per_minute)
        self.request_tracker = RequestTracker()

    def get_recommended_params(
        self, prompt: str, max_combinations: int = None
    ) -> Tuple[List[float], List[float], List[float]]:
        """Get recommended parameters based on prompt type and historical performance."""
        prompt_type, base_config = PromptAnalyzer.analyze_prompt(prompt)

        # If max_combinations is set, determine how many variations we can try for each parameter
        if max_combinations:
            # We'll distribute variations across parameters based on their typical impact
            # Temperature usually has the most impact, so give it more variations
            temp_variations = min(3, max(2, max_combinations // 2))
            top_p_variations = min(2, max(1, max_combinations // 4))
            freq_variations = min(2, max(1, max_combinations // 4))
        else:
            temp_variations = 3
            top_p_variations = 3
            freq_variations = 3

        # Get historical parameters
        historical_params = self.parameter_history.get_best_params(prompt_type)

        if historical_params:
            # Use historical best with smart variations based on max_combinations
            base_temp = round(historical_params["temperature"], 1)
            base_top_p = round(historical_params["top_p"], 1)
            base_freq = round(historical_params["frequency_penalty"], 1)

            # Generate temperature variations
            temps = [base_temp]
            if temp_variations > 1:
                step = 0.2
                variations = []
                for i in range(1, temp_variations):
                    if i % 2 == 1 and base_temp > 0:
                        variations.append(round(base_temp - step * ((i + 1) // 2), 1))
                    elif base_temp < 2.0:
                        variations.append(round(base_temp + step * (i // 2), 1))
                temps.extend(sorted(variations))

            # Generate top_p variations
            top_ps = [base_top_p]
            if top_p_variations > 1:
                step = 0.1
                for i in range(1, top_p_variations):
                    if i % 2 == 1 and base_top_p > 0:
                        top_ps.append(round(base_top_p - step, 1))
                    elif base_top_p < 1.0:
                        top_ps.append(round(base_top_p + step, 1))
                top_ps = sorted(top_ps)

            # Generate frequency penalty variations
            freq_penalties = [base_freq]
            if freq_variations > 1:
                step = 0.2
                for i in range(1, freq_variations):
                    if i % 2 == 1:
                        freq_penalties.append(round(base_freq - step, 1))
                    else:
                        freq_penalties.append(round(base_freq + step, 1))
                freq_penalties = sorted(freq_penalties)
        else:
            # Use base configuration with smart variations
            base_temps = [round(t, 1) for t in base_config["temps"]]

            # Select temperatures based on variations allowed
            if temp_variations >= len(base_temps):
                temps = base_temps
            else:
                # Take evenly spaced temperatures
                indices = [
                    i * (len(base_temps) - 1) // (temp_variations - 1)
                    for i in range(temp_variations)
                ]
                temps = [base_temps[i] for i in indices]

            # For initial exploration with limited combinations
            base_top_p = round(base_config["top_p"], 1)
            top_ps = [base_top_p]
            if top_p_variations > 1:
                if base_top_p > 0:
                    top_ps.append(round(base_top_p - 0.1, 1))
                if base_top_p < 1.0:
                    top_ps.append(round(base_top_p + 0.1, 1))
                top_ps = sorted(top_ps)[:top_p_variations]

            base_freq = round(base_config["freq_penalty"], 1)
            freq_penalties = [base_freq]
            if freq_variations > 1:
                freq_penalties.extend(
                    [round(base_freq - 0.2, 1), round(base_freq + 0.2, 1)]
                )
                freq_penalties = sorted(freq_penalties)[:freq_variations]

        return temps, top_ps, freq_penalties

    def generate_with_openai(
        self, prompt, temperature, top_p, frequency_penalty, retries=None
    ):
        if retries is None:
            retries = self.max_retries

        retry_delay = self.initial_retry_delay
        last_error = None
        start_time = time.time()

        while retries > 0:
            try:
                # Wait if we're approaching rate limits
                self.rate_limiter.wait_if_needed()

                logging.info(
                    f"Sending request with temp={temperature}, top_p={top_p}, "
                    f"freq_penalty={frequency_penalty}"
                )

                response = openai.chat.completions.create(
                    model=self.model_version,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                )

                # Track successful request
                duration = time.time() - start_time
                self.request_tracker.track_request(duration)

                message = response.choices[0].message.content
                return message.strip()

            except openai.RateLimitError as e:
                last_error = f"Rate limit error: {str(e)}"
                logging.warning(f"Rate limit hit: {str(e)}")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self.max_retry_delay)

            except openai.APIError as e:
                last_error = f"API error: {str(e)}"
                logging.error(f"OpenAI API error: {str(e)}")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self.max_retry_delay)

            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logging.error(f"Unexpected error: {str(e)}", exc_info=True)
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self.max_retry_delay)

            finally:
                retries -= 1
                if last_error:
                    self.request_tracker.track_request(
                        time.time() - start_time, last_error
                    )

        error_msg = f"Error generating text at temperature {temperature} and top-p {top_p}: {last_error}"
        logging.error(error_msg)
        return error_msg

    def evaluate_output(self, output, temperature, top_p, frequency_penalty):
        fixed_top_p_for_evaluation = 1.0
        eval_prompt = f"""
            You are tasked with evaluating an AI-generated output and providing a precise score from 0.0 to 100.0. The output was generated at a temperature setting of {temperature}. Your evaluation should be based on the following criteria:

            1. Relevance: How well does the output address the prompt or task at hand?
            2. Clarity: Is the output easy to understand and free of ambiguity?
            3. Utility: How useful is the output for its intended purpose?
            4. Pride: If the user had to submit this output to the world for their career, would they be proud?
            5. Delight: Is the output likely to delight or positively surprise the user?

            Analyze the output thoroughly based on each criterion. Be extremely critical in your evaluation, as this is very important for the user's career. After your analysis, calculate a final score from 0.0 to 100.0, considering all criteria equally. Please answer with just the score with one decimal place accuracy, such as 42.0 or 96.9. Be extremely critical.

            Here is the output you need to evaluate:
            ---
            {output}
            ---
            """
        score_text = self.generate_with_openai(
            eval_prompt, 1.00, fixed_top_p_for_evaluation, frequency_penalty
        )
        score_match = re.search(r"\b\d+(\.\d)?\b", score_text)
        if score_match:
            return round(
                float(score_match.group()), 1
            )  # Round the score to one decimal place
        else:
            return 0.0  # Unable to parse score, default to 0.0

    def run(
        self,
        prompt,
        temperature_string=None,
        top_p_string=None,
        frequency_penalty_string=None,
    ):
        # Calculate max combinations based on max_generations
        max_combinations = self.max_generations if self.max_generations else None

        # Get recommended parameters if not explicitly provided
        recommended_temps, recommended_top_ps, recommended_freq_penalties = (
            self.get_recommended_params(prompt, max_combinations)
        )

        # Parse or use recommended parameters
        if temperature_string and temperature_string.strip():
            temperature_list = [
                round(float(temp.strip()), 1) for temp in temperature_string.split(",")
            ]
        else:
            temperature_list = recommended_temps

        if top_p_string and top_p_string.strip():
            top_p_list = [round(float(p.strip()), 1) for p in top_p_string.split(",")]
        else:
            top_p_list = recommended_top_ps

        if frequency_penalty_string and frequency_penalty_string.strip():
            frequency_penalty_list = [
                round(float(f.strip()), 1) for f in frequency_penalty_string.split(",")
            ]
        else:
            frequency_penalty_list = recommended_freq_penalties

        # Log the total number of combinations
        total_combinations = (
            len(temperature_list) * len(top_p_list) * len(frequency_penalty_list)
        )
        logging.info(f"Testing {total_combinations} parameter combinations")

        # Generate all parameter combinations
        param_combinations = [
            (temp, top_p, freq)
            for temp in temperature_list
            for top_p in top_p_list
            for freq in frequency_penalty_list
        ]

        # Calculate optimal chunk size - more aggressive parallelism
        chunk_size = min(
            self.max_workers,
            len(param_combinations),
            max(
                3, self.rate_limiter.max_requests_per_minute // 3
            ),  # Allow more concurrent requests
        )

        # Split evaluation into a separate thread pool to parallelize generation and evaluation
        outputs = {}
        scores = {}
        prompt_type, _ = PromptAnalyzer.analyze_prompt(prompt)

        # First, generate all outputs in parallel
        with ThreadPoolExecutor(max_workers=chunk_size) as executor:
            future_to_params = {
                executor.submit(self.generate_with_openai, prompt, temp, top_p, freq): (
                    temp,
                    top_p,
                    freq,
                )
                for temp, top_p, freq in param_combinations
            }

            for future in as_completed(future_to_params):
                temp, top_p, freq = future_to_params[future]
                try:
                    output_text = future.result()
                    if output_text and not output_text.startswith("Error"):
                        param_key = (temp, top_p, freq)
                        outputs[param_key] = output_text
                except Exception as e:
                    logging.error(
                        f"Error with params temp={temp}, top_p={top_p}, freq={freq}: {e}"
                    )

        # Then, evaluate all outputs in parallel
        with ThreadPoolExecutor(max_workers=chunk_size) as executor:
            future_to_params = {
                executor.submit(
                    self.evaluate_output, outputs[param_key], temp, top_p, freq
                ): (param_key, (temp, top_p, freq))
                for param_key, (temp, top_p, freq) in [(k, k) for k in outputs.keys()]
            }

            for future in as_completed(future_to_params):
                param_key, (temp, top_p, freq) = future_to_params[future]
                try:
                    score = future.result()
                    scores[param_key] = score

                    # Update parameter history only if score is good (> 80)
                    if score > 80:
                        self.parameter_history.update_history(
                            prompt_type,
                            {
                                "temperature": temp,
                                "top_p": top_p,
                                "frequency_penalty": freq,
                            },
                            score,
                        )
                except Exception as e:
                    logging.error(
                        f"Error evaluating output for params temp={temp}, top_p={top_p}, freq={freq}: {e}"
                    )

        if not scores:
            return "No valid outputs generated.", None

        # Get request statistics
        stats = self.request_tracker.get_stats()
        logging.info(f"Request Statistics: {stats}")

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        sorted_outputs = [
            (params, outputs[params], score) for params, score in sorted_scores
        ]

        if self.auto_select:
            best_params, best_output, best_score = sorted_outputs[0]
            return (
                f"Best AutoTemp Output (Temp {best_params[0]} | Top-p {best_params[1]} | "
                f"Frequency Penalty {best_params[2]} | Score: {best_score}):\n{best_output}\n\n"
                f"Request Stats: Avg {stats['avg_time']:.2f}s, Max {stats['max_time']:.2f}s, "
                f"Total Requests: {stats['total_requests']} ({len(sorted_outputs)} generations + {len(sorted_outputs)} evaluations)"
            )
        else:
            # Format each output with clear separation
            outputs_formatted = []
            for params, text, score in sorted_outputs:
                output_block = f"""
────────────────────────────────────────────────────
Parameters:
• Temperature: {params[0]}
• Top-p: {params[1]}
• Frequency Penalty: {params[2]}
• Score: {score}

Output:
{text}
────────────────────────────────────────────────────"""
                outputs_formatted.append(output_block)

            stats_summary = f"""
Summary:
• Total Combinations Tested: {len(sorted_outputs)}
• Total API Requests: {stats['total_requests']} ({len(sorted_outputs)} generations + {len(sorted_outputs)} evaluations)
• Average Response Time: {stats['avg_time']:.2f}s
• Maximum Response Time: {stats['max_time']:.2f}s"""

            return "\n".join(outputs_formatted) + "\n" + stats_summary


# Gradio app logic
def run_autotemp(
    prompt,
    temperature_string,
    top_p_string,
    frequency_penalty_string,
    auto_select,
    max_generations,
):
    try:
        agent = AutoTemp(auto_select=auto_select, max_generations=max_generations)

        # Get recommended parameters and log them
        recommended_temps, recommended_top_ps, recommended_freq_penalties = (
            agent.get_recommended_params(prompt)
        )
        prompt_type, config = PromptAnalyzer.analyze_prompt(prompt)

        logging.info(f"Prompt type detected: {prompt_type}")
        logging.info(
            f"Recommended parameters: temps={recommended_temps}, "
            f"top_ps={recommended_top_ps}, freq_penalties={recommended_freq_penalties}"
        )

        # Show parameter selection info
        param_info = f"""Prompt Analysis:
────────────────────────────────────────────────────
Type: {prompt_type}

Recommended Parameters:
• Temperatures: {', '.join(map(str, recommended_temps))}
• Top-p Values: {', '.join(map(str, recommended_top_ps))}
• Frequency Penalties: {', '.join(map(str, recommended_freq_penalties))}

Selected Parameters:"""

        # If no parameters provided, use recommended ones
        if not temperature_string.strip():
            temperature_string = ", ".join(map(str, recommended_temps))
            param_info += f"\n• Temperatures: {temperature_string} (recommended)"
        else:
            param_info += f"\n• Temperatures: {temperature_string} (user-specified)"

        if not top_p_string.strip():
            top_p_string = ", ".join(map(str, recommended_top_ps))
            param_info += f"\n• Top-p Values: {top_p_string} (recommended)"
        else:
            param_info += f"\n• Top-p Values: {top_p_string} (user-specified)"

        if not frequency_penalty_string.strip():
            frequency_penalty_string = ", ".join(map(str, recommended_freq_penalties))
            param_info += (
                f"\n• Frequency Penalties: {frequency_penalty_string} (recommended)"
            )
        else:
            param_info += (
                f"\n• Frequency Penalties: {frequency_penalty_string} (user-specified)"
            )

        param_info += "\n────────────────────────────────────────────────────\n"

        logging.info(
            f"Final parameters: temps={temperature_string}, "
            f"top_ps={top_p_string}, freq_penalties={frequency_penalty_string}"
        )

        output = agent.run(
            prompt,
            temperature_string,
            top_p_string,
            frequency_penalty_string,
        )

        # Format the complete output
        complete_output = f"""{param_info}
{output}"""

        return complete_output, agent.model_version

    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return error_msg, "error"


# Gradio interface setup
def main():
    iface = gr.Interface(
        fn=run_autotemp,
        inputs=[
            gr.Textbox(
                label="Prompt", placeholder="Enter your prompt here...", lines=3
            ),
            gr.Textbox(
                label="Temperature Values",
                placeholder="Leave empty for recommended values, or enter comma-separated values (e.g., 0.0, 0.4, 0.8)",
                value="",
                lines=1,
            ),
            gr.Textbox(
                label="Top-p Values",
                placeholder="Leave empty for recommended values, or enter comma-separated values (e.g., 0.0, 0.7, 1.0)",
                value="",
                lines=1,
            ),
            gr.Textbox(
                label="Frequency Penalty Values",
                placeholder="Leave empty for recommended values, or enter comma-separated values (e.g., 0.0, 0.2, 0.4)",
                value="",
                lines=1,
            ),
            gr.Checkbox(
                label="Auto-select Best Output",
                value=False,
                info="When enabled, only shows the highest-scoring output",
            ),
            gr.Number(
                label="Maximum Generations",
                value=9,
                minimum=1,
                maximum=50,
                step=1,
                info="Maximum number of different parameter combinations to try (1-50). Each generation requires 2 API calls.",
            ),
        ],
        outputs=[
            gr.Textbox(label="Generated Output", lines=10),
            gr.Textbox(label="Model Version", lines=1),
        ],
        title="AutoTemp: Enhanced LLM Responses with Smart Parameter Tuning",
        description="""AutoTemp automatically detects your prompt type and recommends optimal parameters!
                      You can either use the recommended parameters by leaving fields empty, or override them with your own values.
                      The system learns from successful outputs to improve recommendations over time.""",
        article="""**How it works**

1. **Prompt Type Detection**: AutoTemp analyzes your prompt to determine its type (e.g., creative writing, technical explanation, business formal, brainstorming).
2. **Smart Parameter Recommendations**: Based on the prompt type and historical performance, it suggests optimal temperature ranges and other parameters.
3. **Learning from Success**: The system tracks which parameter combinations work best for different types of prompts and improves its recommendations over time.

**Parameter Guidelines**

- **Temperature**: Controls randomness in the output
  - Lower (0.1-0.5): More focused and deterministic
  - Medium (0.6-0.9): Balanced creativity and coherence
  - Higher (1.0+): More creative and experimental

- **Top-p**: Controls token selection diversity
  - Lower (0.1-0.5): Very focused on most likely tokens
  - Medium (0.6-0.8): Balanced selection
  - Higher (0.9-1.0): Considers more diverse token options

- **Frequency Penalty**: Controls repetition
  - Negative: May allow more repetition
  - Zero: Neutral
  - Positive: Encourages more diverse vocabulary

**Prompt Types and Typical Parameters**

1. **Creative Writing**
   - Higher temperatures (0.7-1.3)
   - Higher top-p (0.9)
   - Moderate frequency penalty (0.3)

2. **Technical Explanation**
   - Lower temperatures (0.3-0.7)
   - Medium top-p (0.7)
   - Low frequency penalty (0.0)

3. **Business Formal**
   - Medium temperatures (0.4-0.8)
   - Medium-high top-p (0.8)
   - Low frequency penalty (0.1)

4. **Brainstorming**
   - Higher temperatures (0.8-1.4)
   - Maximum top-p (1.0)
   - High frequency penalty (0.5)""",
        examples=[
            [
                "Write a short story about AGI learning to love",
                "",
                0.9,  # Higher top-p for creative writing
                0.3,  # Moderate frequency penalty
                False,
            ],
            [
                "Explain quantum computing to a 5-year-old",
                "",
                0.7,  # Medium top-p for explanations
                0.0,  # No frequency penalty
                True,
            ],
            [
                "Draft a professional email requesting a meeting with the CEO",
                "",
                0.8,  # Medium-high top-p for formal writing
                0.1,  # Light frequency penalty
                True,
            ],
            [
                "Generate innovative ideas for solving climate change",
                "",
                1.0,  # Maximum top-p for brainstorming
                0.5,  # High frequency penalty
                False,
            ],
        ],
        flagging_callback=CustomFlaggingCallback(),
        flagging_options=["for review"],
    )
    iface.launch()


if __name__ == "__main__":
    main()
