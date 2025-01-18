# AutoTemp

AutoTemp is an intelligent parameter tuning system for Large Language Models that automatically optimizes generation parameters based on prompt type and historical performance.

## Features

- **Smart Prompt Analysis**: Automatically detects prompt types (creative writing, technical explanation, business formal, etc.)
- **Dynamic Parameter Optimization**: Intelligently selects and tunes temperature, top-p, and frequency penalty values
- **Historical Learning**: Learns from successful generations to improve future parameter recommendations
- **Parallel Processing**: Efficiently generates and evaluates multiple outputs simultaneously
- **Customizable**: Supports both automatic and manual parameter configuration

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/AutoTemp.git
   cd AutoTemp
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root:

   ```bash
   OPENAI_API_KEY=your_api_key_here
   OPENAI_MODEL=gpt-4  # Optional, defaults to gpt-4o if not specified
   ```

## Usage

1. Start the Gradio interface:

   ```bash
   python autotemp.py
   ```

2. Access the web interface at `http://localhost:7860`

3. Enter your prompt and optionally configure:
   - Temperature values
   - Top-p values
   - Frequency penalty values
   - Auto-select best output
   - Maximum number of generations

## Parameter Guidelines

### Temperature (0.0 - 2.0)

- **Low (0.1-0.5)**: More focused, deterministic outputs
- **Medium (0.6-0.9)**: Balanced creativity and coherence
- **High (1.0+)**: More creative and experimental

### Top-p (0.0 - 1.0)

- **Low (0.1-0.5)**: Very focused token selection
- **Medium (0.6-0.8)**: Balanced selection
- **High (0.9-1.0)**: More diverse token options

### Frequency Penalty (-2.0 - 2.0)

- **Negative**: May allow more repetition
- **Zero**: Neutral
- **Positive**: Encourages more diverse vocabulary

## Prompt Types

AutoTemp automatically detects and optimizes for different prompt types:

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
   - High frequency penalty (0.5)

5. **Humor/Casual**
   - Balanced temperatures (0.6-1.0)
   - High top-p (0.9)
   - Moderate frequency penalty (0.3)

## Performance Optimization

- Utilizes parallel processing for both generation and evaluation
- Configurable maximum workers (default: 12)
- Adjustable rate limiting (default: 90 requests per minute)
- Smart retry logic with exponential backoff

## Data Storage

- Parameter history is stored in `flagged/parameter_history.json`
- Flagged outputs are stored in `flagged/flagged_data.json`
- Historical data is used to improve parameter recommendations

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with OpenAI's GPT-4o API
- Interface powered by Gradio
- Inspired by the need for better parameter tuning in LLM applications
