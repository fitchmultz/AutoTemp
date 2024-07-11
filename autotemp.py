import json
import logging
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import gradio as gr
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


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


class AutoTemp:
    def __init__(
        self,
        default_temp=0.0,
        alt_temps=None,
        auto_select=True,
        max_workers=6,
        model_version="gpt-4o",
        frequency_penalty=0.0,
    ):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
        openai.api_key = self.api_key

        self.default_temp = default_temp
        self.alt_temps = alt_temps if alt_temps else [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
        self.auto_select = auto_select
        self.max_workers = max_workers
        self.model_version = model_version
        self.frequency_penalty = frequency_penalty

    def generate_with_openai(
        self, prompt, temperature, top_p, frequency_penalty, retries=3
    ):
        while retries > 0:
            try:
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
                # Adjusted to use attribute access instead of dictionary access
                message = response.choices[0].message.content
                return message.strip()
            except Exception as e:
                retries -= 1
                print(
                    f"Attempt failed with error: {e}"
                )  # Print the error for debugging
                if retries <= 0:
                    print(
                        f"Final error generating text at temperature {temperature} and top-p {top_p}: {e}"
                    )
                    return f"Error generating text at temperature {temperature} and top-p {top_p}: {e}"

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

    def run(self, prompt, temperature_string, top_p, frequency_penalty):
        temperature_list = [
            float(temp.strip()) for temp in temperature_string.split(",")
        ]
        outputs = {}
        scores = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
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
                    print(
                        f"Output for temp {temp}: {output_text}"
                    )  # Print the output for debugging
                    if output_text and not output_text.startswith("Error"):
                        outputs[temp] = output_text
                        scores[temp] = self.evaluate_output(
                            output_text, temp, top_p, frequency_penalty
                        )  # Pass top_p and frequency_penalty here

                except Exception as e:
                    print(
                        f"Error while generating or evaluating output for temp {temp}: {e}"
                    )

        if not scores:
            return "No valid outputs generated.", None

        # Sort the scores by value in descending order and return the sorted outputs
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        sorted_outputs = [(temp, outputs[temp], score) for temp, score in sorted_scores]

        # If auto_select is enabled, return only the best result
        if self.auto_select:
            best_temp, best_output, best_score = sorted_outputs[0]
            return f"Best AutoTemp Output (Temp {best_temp} | Top-p {top_p} | Frequency Penalty {frequency_penalty} | Score: {best_score}):\n{best_output}"
        else:
            return "\n".join(
                f"Temp {temp} | Top-p {top_p} | Frequency Penalty {frequency_penalty} | Score: {score}:\n{text}"
                for temp, text, score in sorted_outputs
            )


# Gradio app logic
def run_autotemp(prompt, temperature_string, top_p, frequency_penalty, auto_select):
    agent = AutoTemp(auto_select=auto_select)
    output = agent.run(
        prompt,
        temperature_string,
        top_p=float(top_p),
        frequency_penalty=float(frequency_penalty),
    )
    return output, agent.model_version


# Gradio interface setup
def main():
    iface = gr.Interface(
        fn=run_autotemp,
        inputs=[
            gr.Textbox(label="Prompt"),
            gr.Textbox(label="Temperature String"),
            gr.Slider(
                minimum=0.0, maximum=1.0, step=0.1, value=1.0, label="Top-p value"
            ),
            gr.Slider(
                minimum=-2.0,
                maximum=2.0,
                step=0.1,
                value=0.0,
                label="Frequency Penalty",
            ),
            gr.Checkbox(label="Auto Select"),
        ],
        outputs=[
            "text",
            "text",
        ],  # Two outputs: one for the result, one for model_version
        title="AutoTemp: Enhanced LLM Responses with Temperature and Top-p Tuning",
        description="""AutoTemp generates responses at different temperatures, evaluates them, and ranks them based on quality. 
                       Enter temperatures separated by commas for evaluation.
                       Adjust 'Top-p' to control output diversity: lower for precision, higher for creativity.
                       Toggle 'Auto Select' to either see the top-rated output or all evaluated outputs.
                       Check the FAQs at the bottom of the page for more info.""",
        article="""**FAQs**

**What's Top-p?** 'Top-p' controls the diversity of AI responses: a low 'top-p' makes output more focused and predictable, while a high 'top-p' encourages variety and surprise. Pair with temperature to fine-tune AI creativity: higher temperatures with high 'top-p' for bold ideas, or lower temperatures with low 'top-p' for precise answers. 
Using top_p=0 essentially disables the "nucleus sampling" feature, where only the most probable tokens are considered. This is equivalent to using full softmax probability distribution to sample the next word.

**How Does Temperature Affect AI Outputs?** Temperature controls the randomness of word selection. Lower temperatures lead to more predictable text, while higher temperatures allow for more novel text generation.

**How Does Top-p Influence Temperature Settings in AI Language Models?**
Top-p and temperature are both parameters that control the randomness of AI-generated text, but they influence outcomes in subtly different ways:

- **Low Temperatures (0.0 - 0.5):**
  - *Effect of Top-p:* A high `top_p` value will have minimal impact, as the model's output is already quite deterministic. A low `top_p` will further constrain the model, leading to very predictable outputs.
  - *Use Cases:* Ideal for tasks requiring precise, factual responses like technical explanations or legal advice. For example, explaining a scientific concept or drafting a formal business email.

- **Medium Temperatures (0.5 - 0.7):**
  - *Effect of Top-p:* `top_p` starts to influence the variety of the output. A higher `top_p` will introduce more diversity without sacrificing coherence.
  - *Use Cases:* Suitable for creative yet controlled content, such as writing an article on a current event or generating a business report that balances creativity with professionalism.

- **High Temperatures (0.8 - 1.0):**
  - *Effect of Top-p:* A high `top_p` is crucial for introducing creativity and surprise, but may result in less coherent outputs. A lower `top_p` can help maintain some coherence.
  - *Use Cases:* Good for brainstorming sessions, generating creative writing prompts, or coming up with out-of-the-box ideas where a mix of novelty and relevance is appreciated.

- **Extra-High Temperatures (1.1 - 2.0):**
  - *Effect of Top-p:* The output becomes more experimental and unpredictable, and `top_p`'s influence can vary widely. It's a balance between randomness and diversity.
  - *Use Cases:* Best for when you're seeking highly creative or abstract ideas, such as imagining a sci-fi scenario or coming up with a plot for a fantasy story, where coherence is less of a priority compared to novelty and uniqueness.

Adjusting both temperature and top-p helps tailor the AI's output to your specific needs.""",
        examples=[
            [
                "Write a short story about AGI learning to love",
                "0.5, 0.7, 0.9, 1.1",
                1.0,
                0.0,
                False,
            ],
            [
                "Create a dialogue between a chef and an alien creating an innovative new recipe",
                "0.3, 0.6, 0.9, 1.2",
                0.9,
                0.0,
                True,
            ],
            [
                "Explain quantum computing to a 5-year-old",
                "0.4, 0.8, 1.2, 1.5",
                0.8,
                0.0,
                False,
            ],
            [
                "Draft an email to a hotel asking for a special arrangement for a marriage proposal",
                "0.4, 0.7, 1.0, 1.3",
                0.7,
                0.0,
                True,
            ],
            [
                "Describe a futuristic city powered by renewable energy",
                "0.5, 0.75, 1.0, 1.25",
                0.6,
                0.0,
                False,
            ],
            [
                "Generate a poem about the ocean's depths in the style of Edgar Allan Poe",
                "0.6, 0.8, 1.0, 1.2",
                0.5,
                0.0,
                True,
            ],
            [
                "What are some innovative startup ideas for improving urban transportation?",
                "0.45, 0.65, 0.85, 1.05",
                0.4,
                0.0,
                False,
            ],
            [
                "Explain cybersecurity to a 5-year-old who has never used a computer before",
                "0.00, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5",
                1.0,
                0.0,
                False,
            ],
        ],
        flagging_callback=CustomFlaggingCallback(),
        flagging_options=["for review"],
    )
    iface.launch()


if __name__ == "__main__":
    main()
