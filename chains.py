from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser


def create_chains(llm, prompts, model=None):
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", prompts.SYSTEM_PROMPT),
            ("human", prompts.USER_PROMPT),
        ],
    )

    if model:
        pydantic_schema = model.model_json_schema()
        parser = PydanticOutputParser(pydantic_object=model)
        chat_template = chat_template.partial(schema=pydantic_schema)
        return chat_template | llm | parser
    return chat_template | llm | StrOutputParser()
