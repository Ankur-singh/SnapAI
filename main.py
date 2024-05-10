import gradio as gr
from langchain_openai import ChatOpenAI

import config
from chains import create_chains


llm = ChatOpenAI(
    openai_api_base=config.OLLAMA_HOST + "/v1",
    openai_api_key="ollama",
    model="mario-llama:latest",
    temperature=0,
)

parse_jb_chain = create_chains(llm, config.JobDescriptionPrompt, config.JobDescription)
extract_entities_chain = create_chains(llm, config.NERPrompt, config.Entities)
generation_chain = create_chains(llm, config.GenerationPrompt)


def generate_response(model, job_description, resume_bullet_point):
    return generation_chain.invoke(
        {
            "job_description": job_description,
            "resume_bullet_point": resume_bullet_point,
        }
    )


def parse_job_description(job_description):
    return dict(parse_jb_chain.invoke(job_description))


def extract_entities(document):
    return dict(extract_entities_chain.invoke(document))


with gr.Blocks() as demo:
    with gr.Tab("Generation"):
        gr.Interface(
            fn=generate_response,
            inputs=[
                gr.Dropdown(
                    choices=config.MODELS.keys(),
                    label="Choose Model",
                    value=next(iter(config.MODELS.keys())),
                ),
                gr.Textbox(
                    label="Job Description",
                    lines=8,
                    placeholder="Enter Job Description here...",
                ),
                gr.Textbox(
                    label="Your Resume Points",
                    lines=7,
                    placeholder="Enter your Resume bullet point here...",
                ),
            ],
            outputs=gr.Textbox(
                lines=29,
                label="Tailored Output",
                placeholder="Generated Resume bullet points",
            ),
            title="SnapAI",
        )

    with gr.Tab("Parsing"):
        gr.Markdown(
            """
            ## Extraction
            Paste the job description below and we will extract details like employment type, seniority, job category, location, salary, minimum experience, skills required and job requirements.
            """
        )

        gr.Interface(
            fn=parse_job_description,
            inputs=gr.Textbox(
                label="Job Description",
                placeholder="Enter Job Description here...",
            ),
            outputs=gr.Json(label="Parsed Details"),
            submit_btn="Parse",
        )

    with gr.Tab("Entity Extraction"):
        gr.Markdown(
            """
            ## Entity Extraction
            Paste the document below and we will extract the entities from the document.
            """
        )

        gr.Interface(
            fn=extract_entities,
            inputs=gr.Textbox(
                label="Job Description",
                placeholder="Enter Job Description here...",
            ),
            outputs=gr.Json(label="Extracted Entities"),
            submit_btn="Extract",
        )


demo.launch(debug=True, share=True)
