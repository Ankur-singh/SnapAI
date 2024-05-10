## SnapAI

Repository for all LLM features powering the [snapjobs.me](https://snapjobs.me) site. Hermes2 LLM has been used for resume bullet point generation, job description parsing, and entity extraction. While resume bullet point generation is a user-facing feature, parsing and entity extraction are used to populate our database, which powers the advanced search on the website.

### Setup

**Note**: Make sure you have [ollama](https://ollama.com/) installed on your system.

```bash
git clone https://github.com/snapjobs-me/SnapAI
cd SnapAI
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ollama create snapai -f hermes2_modelfile
```

To run the gradio app

```bash
gardio main.py
```

