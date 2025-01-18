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
            "temps": [0.7, 0.9, 1.1, 1.3],
            "top_p": 0.9,
            "freq_penalty": 0.3,
        },
        "technical_explanation": {
            "keywords": ["explain", "how", "what", "why", "technical", "concept"],
            "temps": [0.3, 0.5, 0.7],
            "top_p": 0.7,
            "freq_penalty": 0.0,
        },
        "business_formal": {
            "keywords": ["email", "draft", "business", "formal", "professional"],
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
            "temps": [0.8, 1.0, 1.2, 1.4],
            "top_p": 1.0,
            "freq_penalty": 0.5,
        },
    }

    @classmethod
    def analyze_prompt(cls, prompt: str) -> Tuple[str, Dict]:
        """Analyze the prompt and return the most likely prompt type and recommended parameters."""
        prompt_lower = prompt.lower()
        scores = {}

        for prompt_type, config in cls.PROMPT_TYPES.items():
            score = sum(1 for keyword in config["keywords"] if keyword in prompt_lower)
            scores[prompt_type] = score

        best_type = max(scores.items(), key=lambda x: x[1])[0]
        return best_type, cls.PROMPT_TYPES[best_type]


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
        max_workers=6,
        model_version="gpt-4",
        frequency_penalty=0.0,
        max_retries=5,
        initial_retry_delay=1.0,
        max_retry_delay=32.0,
        max_requests_per_minute=60,
    ):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
        openai.api_key = self.api_key

        self.default_temp = default_temp
        self.alt_temps = alt_temps
        self.auto_select = auto_select
        self.max_workers = max_workers
        self.model_version = model_version
        self.frequency_penalty = frequency_penalty

        # Rate limiting and retry settings
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay

        self.prompt_analyzer = PromptAnalyzer()
        self.parameter_history = ParameterHistory()
        self.rate_limiter = RateLimiter(max_requests_per_minute)
        self.request_tracker = RequestTracker()

    def get_recommended_params(self, prompt: str) -> Tuple[List[float], float, float]:
        """Get recommended parameters based on prompt type and historical performance."""
        prompt_type, base_config = PromptAnalyzer.analyze_prompt(prompt)

        # Check historical performance
        historical_params = self.parameter_history.get_best_params(prompt_type)

        if historical_params:
            # Blend historical parameters with base configuration
            temps = [
                historical_params["temperature"] - 0.2,
                historical_params["temperature"],
                historical_params["temperature"] + 0.2,
            ]
            top_p = historical_params["top_p"]
            freq_penalty = historical_params["frequency_penalty"]
        else:
            temps = base_config["temps"]
            top_p = base_config["top_p"]
            freq_penalty = base_config["freq_penalty"]

        return temps, top_p, freq_penalty

    def generate_with_openai(
        self, prompt, temperature, top_p, frequency_penalty, retries=None
    ):
        if retries is None:
            retries = self.max_retries

        retry_delay = self.initial_retry_delay
        last_error = None

        while retries > 0:
            try:
                # Wait if we're approaching rate limits
                self.rate_limiter.wait_if_needed()

                # Track request timing
                start_time = time.time()

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
                last_error = "rate_limit"
                # Use exponential backoff for rate limits
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self.max_retry_delay)

            except Exception as e:
                last_error = str(e)
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

    def run(self, prompt, temperature_string=None, top_p=None, frequency_penalty=None):
        # Get recommended parameters if not explicitly provided
        recommended_temps, recommended_top_p, recommended_freq_penalty = (
            self.get_recommended_params(prompt)
        )

        # Use provided parameters or recommended ones
        if temperature_string is None or temperature_string.strip() == "":
            temperature_list = recommended_temps
        else:
            temperature_list = [
                float(temp.strip()) for temp in temperature_string.split(",")
            ]

        top_p = float(top_p) if top_p is not None else recommended_top_p
        frequency_penalty = (
            float(frequency_penalty)
            if frequency_penalty is not None
            else recommended_freq_penalty
        )

        outputs = {}
        scores = {}
        prompt_type, _ = PromptAnalyzer.analyze_prompt(prompt)

        # Calculate optimal chunk size based on rate limits
        chunk_size = min(
            self.max_workers,
            len(temperature_list),
            self.rate_limiter.max_requests_per_minute
            // 2,  # Leave room for evaluation requests
        )

        with ThreadPoolExecutor(max_workers=chunk_size) as executor:
            future_to_temp = {
                executor.submit(
                    self.generate_with_openai, prompt, temp, top_p, frequency_penalty
                ): temp
                for temp in temperature_list
            }

            for future in as_completed(future_to_temp):
                temp = future_to_temp[future]
                try:
                    output_text = future.result()
                    if output_text and not output_text.startswith("Error"):
                        outputs[temp] = output_text
                        score = self.evaluate_output(
                            output_text, temp, top_p, frequency_penalty
                        )
                        scores[temp] = score

                        # Update parameter history
                        self.parameter_history.update_history(
                            prompt_type,
                            {
                                "temperature": temp,
                                "top_p": top_p,
                                "frequency_penalty": frequency_penalty,
                            },
                            score,
                        )

                except Exception as e:
                    logging.error(
                        f"Error while generating or evaluating output for temp {temp}: {e}"
                    )

        if not scores:
            return "No valid outputs generated.", None

        # Get request statistics
        stats = self.request_tracker.get_stats()
        logging.info(f"Request Statistics: {stats}")

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        sorted_outputs = [(temp, outputs[temp], score) for temp, score in sorted_scores]

        if self.auto_select:
            best_temp, best_output, best_score = sorted_outputs[0]
            return (
                f"Best AutoTemp Output (Temp {best_temp} | Top-p {top_p} | "
                f"Frequency Penalty {frequency_penalty} | Score: {best_score}):\n{best_output}\n\n"
                f"Request Stats: Avg {stats['avg_time']:.2f}s, Max {stats['max_time']:.2f}s, "
                f"Total Requests: {stats['total_requests']}"
            )
        else:
            return (
                "\n".join(
                    f"Temp {temp} | Top-p {top_p} | Frequency Penalty {frequency_penalty} | "
                    f"Score: {score}:\n{text}"
                    for temp, text, score in sorted_outputs
                )
                + f"\n\nRequest Stats: Avg {stats['avg_time']:.2f}s, "
                f"Max {stats['max_time']:.2f}s, Total Requests: {stats['total_requests']}"
            )


# Gradio app logic
def run_autotemp(prompt, temperature_string, top_p, frequency_penalty, auto_select):
    agent = AutoTemp(auto_select=auto_select)

    # Get recommended parameters
    recommended_temps, recommended_top_p, recommended_freq_penalty = (
        agent.get_recommended_params(prompt)
    )
    prompt_type, _ = PromptAnalyzer.analyze_prompt(prompt)

    # If no parameters provided, use recommended ones
    if not temperature_string.strip():
        temperature_string = ", ".join(map(str, recommended_temps))
    if top_p is None:
        top_p = recommended_top_p
    if frequency_penalty is None:
        frequency_penalty = recommended_freq_penalty

    output = agent.run(
        prompt,
        temperature_string,
        top_p=float(top_p),
        frequency_penalty=float(frequency_penalty),
    )

    # Add prompt type info to output
    output_with_type = f"Detected Prompt Type: {prompt_type}\n\n{output}"
    return output_with_type, agent.model_version


# Gradio interface setup
def main():
    iface = gr.Interface(
        fn=run_autotemp,
        inputs=[
            gr.Textbox(label="Prompt"),
            gr.Textbox(
                label="Temperature String (leave empty for auto-recommended temperatures)",
                placeholder="e.g., 0.4, 0.6, 0.8 or leave empty for automatic selection",
            ),
            gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=None,
                label="Top-p value (leave at 0 for auto-recommended value)",
            ),
            gr.Slider(
                minimum=-2.0,
                maximum=2.0,
                step=0.1,
                value=None,
                label="Frequency Penalty (leave at 0 for auto-recommended value)",
            ),
            gr.Checkbox(label="Auto Select Best Output"),
        ],
        outputs=[
            "text",
            "text",
        ],
        title="AutoTemp: Enhanced LLM Responses with Smart Parameter Tuning",
        description="""AutoTemp now automatically detects your prompt type and recommends optimal parameters!
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
                "",  # Empty for auto temperature
                None,  # None for auto top-p
                None,  # None for auto frequency penalty
                False,
            ],
            [
                "Explain quantum computing to a 5-year-old",
                "",
                None,
                None,
                True,
            ],
            [
                "Draft a professional email requesting a meeting with the CEO",
                "",
                None,
                None,
                True,
            ],
            [
                "Generate innovative ideas for solving climate change",
                "",
                None,
                None,
                False,
            ],
        ],
        flagging_callback=CustomFlaggingCallback(),
        flagging_options=["for review"],
    )
    iface.launch()


if __name__ == "__main__":
    main()
