import datetime

import uvicorn
from fastapi import FastAPI
from langchain.agents import initialize_agent, AgentType, ZeroShotAgent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.chains import APIChain, LLMMathChain, LLMChain
from langchain.chains.api import open_meteo_docs
from langchain.llms import OpenAI
from langchain.tools import tool
from langchain.tools import Tool
from langchain.document_loaders import WebBaseLoader
from langchain_community.agent_toolkits.openapi.planner import RequestsPostToolWithParsing, RequestsGetToolWithParsing
from langchain_community.callbacks import OpenAICallbackHandler
from langchain_community.utilities import RequestsWrapper
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool, BaseTool
from pydantic.v1 import BaseModel, Field
from langchain.agents.agent_toolkits.openapi import planner
from typing import Any, Callable, Dict, List, Optional

from starlette.middleware.cors import CORSMiddleware

from PromptKernel.planner_prompt import API_ORCHESTRATOR_PROMPT, API_CONTROLLER_PROMPT, PARSING_GET_PROMPT, \
    PARSING_POST_PROMPT, API_CONTROLLER_TOOL_NAME, API_CONTROLLER_TOOL_DESCRIPTION, API_PLANNER_TOOL_NAME, \
    API_PLANNER_TOOL_DESCRIPTION, API_PLANNER_PROMPT

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# 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
# llm = OpenAI(model="gpt-4-1106-preview", temperature=0)
llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
total_cb = OpenAICallbackHandler()


# @app.get("/")
# def root():
#     a = 2
#     chat_model = ChatOpenAI()
#     text = "What would be a good company name for a company that makes colorful socks?"
#     messages = [HumanMessage(content=text)]
#     # response = llm.invoke(text)
#     # chat_model.invoke(messages)
#
#     return {"hello world": messages}


@app.get("/weather")
def root():
    # apiMovie = 'https://developer.themoviedb.org/reference/search-movie'
    # apiPerson = 'https://developer.themoviedb.org/reference/search-person'
    # apiMovieTrending = 'https://developer.themoviedb.org/reference/trending-movies'

    class SearchForAMovie(BaseModel):
        query: str = Field()
        apiDocumentation: str = Field()

    tools = [
        StructuredTool.from_function(get_search_movie_api_doc, name="get_search_movie_api_doc",
                                     description="Get the API "
                                                 "documentation for "
                                                 "searching for "
                                                 "movies by their "
                                                 "original, "
                                                 "translated and "
                                                 "alternative "
                                                 "titles."),
        StructuredTool.from_function(search_movie, name="search_movie",
                                     description="Use the api documentation and retrieve movies informations",
                                     args_schema=SearchForAMovie)
        # Tool.from_function(search_movie),
        # get_weekday
    ]
    agent_kwargs = {
        "system_message": "Act as an action planned using the provided tools. Think about a detailed plan to answer "
                          "the question below using the tools provided.",
    }
    agent = initialize_agent(
        tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, agent_kwargs=agent_kwargs, verbose=True
    )

    response = agent.run("Give me all avatar movies")
    # chain = APIChain.from_llm_and_api_docs(
    #     llm,
    #     open_meteo_docs.OPEN_METEO_DOCS,
    #     verbose=True,
    #     limit_to_domains=["https://api.open-meteo.com/"],
    # )
    # response = chain.run(
    #     "What is the weather like right now in Munich, Germany in degrees Fahrenheit?"
    # )

    return {'response': response, 'cb': total_cb}


@app.get("/weather2")
def root():
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)

    class CalculatorInput(BaseModel):
        question: str = Field()

    tools = [
        Tool.from_function(
            func=llm_math_chain.run,
            name="Calculator",
            description="useful for when you need to answer questions about math",
            args_schema=CalculatorInput,
            # coroutine= ... <- you can specify an async method if desired as well
        )
    ]

    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    response = agent.run(
        "what is 1256 power 0.43?",
        callbacks=[total_cb],
    )

    return {"calculation": response, "total": total_cb}


@app.get("/weather3")
def root():
    loader = WebBaseLoader('https://developer.themoviedb.org/reference/search-movie')
    document = loader.load()
    header = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyMWM2YjJmMzIyNjk4N2I1ZDc0OGFhODA0ZTFlZjE3ZSIsInN1YiI6IjY1N2U0YjcxOGYyNmJjMWNmNTc1MWEwZCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.QMZ6t5UJ3ooSa7U9zXY569qvZxmVTVOQpBlzhes7a-I",
        "accept": "application/json",
    }
    requests_wrapper = RequestsWrapper(headers=header)

    chain = APIChain.from_llm_and_api_docs(
        llm,
        document[0].page_content,
        verbose=True,
        limit_to_domains=["https://api.themoviedb.org/3/search/movie"],
        headers=header
    )
    response = chain.run(
        "Give me all avatar movies",
        callbacks=[total_cb],
    )
    return {'response': response, 'cb': total_cb}

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None


@app.post("/")
def root():
    loader = WebBaseLoader('https://developer.themoviedb.org/reference/search-movie')
    document = loader.load()
    header = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyMWM2YjJmMzIyNjk4N2I1ZDc0OGFhODA0ZTFlZjE3ZSIsInN1YiI6IjY1N2U0YjcxOGYyNmJjMWNmNTc1MWEwZCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.QMZ6t5UJ3ooSa7U9zXY569qvZxmVTVOQpBlzhes7a-I",
        "accept": "application/json",
    }
    requests_wrapper = RequestsWrapper(headers=header)

    tools = [
        create_api_planner_tool(document[0].page_content, llm),
        create_api_controller_tool(document[0].page_content, requests_wrapper, llm),
    ]
    prompt = PromptTemplate(
        template=API_ORCHESTRATOR_PROMPT,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "tool_names": ", ".join([tool.name for tool in tools]),
            "tool_descriptions": "\n".join(
                [f"{tool.name}: {tool.description}" for tool in tools]
            ),
        },
    )
    agent = ZeroShotAgent(
        llm_chain=LLMChain(llm=llm, prompt=prompt),
        allowed_tools=[tool.name for tool in tools],
    )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
    )

    response = agent_executor.run(
        "Give me all avatar movies",
        callbacks=[total_cb],
    )

    return {'response': response, 'cb': total_cb}


def create_api_planner_tool(
        api_spec: str, llm_inside: BaseLanguageModel
) -> Tool:
    from langchain.chains.llm import LLMChain

    prompt = PromptTemplate(
        template=API_PLANNER_PROMPT,
        input_variables=["query"],
        partial_variables={"endpoints": api_spec},
    )
    chain = LLMChain(llm=llm_inside, prompt=prompt)
    tool_inside = Tool(
        name=API_PLANNER_TOOL_NAME,
        description=API_PLANNER_TOOL_DESCRIPTION,
        func=chain.run,
    )
    return tool_inside


def create_api_controller_tool(
        api_spec: str,
        requests_wrapper: RequestsWrapper,
        llm_inside: BaseLanguageModel,
) -> Tool:
    """Expose controller as a tool.

    The tool is invoked with a plan from the planner, and dynamically
    creates a controller agent with relevant documentation only to
    constrain the context.
    """

    def _create_and_run_api_controller_agent(plan_str: str) -> str:
        agent = create_api_controller_agent(api_spec, requests_wrapper, llm_inside)
        return agent.run(plan_str)

    return Tool(
        name=API_CONTROLLER_TOOL_NAME,
        func=_create_and_run_api_controller_agent,
        description=API_CONTROLLER_TOOL_DESCRIPTION,
    )


def create_api_controller_agent(
        api_docs: str,
        requests_wrapper: RequestsWrapper,
        llm_inside: BaseLanguageModel,
) -> Any:
    from langchain.agents.agent import AgentExecutor
    from langchain.agents.mrkl.base import ZeroShotAgent
    from langchain.chains.llm import LLMChain

    get_llm_chain = LLMChain(llm=llm_inside, prompt=PARSING_GET_PROMPT)
    post_llm_chain = LLMChain(llm=llm_inside, prompt=PARSING_POST_PROMPT)
    tools: List[BaseTool] = [
        RequestsGetToolWithParsing(
            requests_wrapper=requests_wrapper, llm_chain=get_llm_chain
        ),
        RequestsPostToolWithParsing(
            requests_wrapper=requests_wrapper, llm_chain=post_llm_chain
        ),
    ]
    prompt = PromptTemplate(
        template=API_CONTROLLER_PROMPT,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "api_docs": api_docs,
            "tool_names": ", ".join([tool.name for tool in tools]),
            "tool_descriptions": "\n".join(
                [f"{tool.name}: {tool.description}" for tool in tools]
            ),
        },
    )
    agent = ZeroShotAgent(
        llm_chain=LLMChain(llm=llm_inside, prompt=prompt),
        allowed_tools=[tool_inside.name for tool_inside in tools],
    )
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)


def get_weekday() -> str:
    """Get the current weekday."""
    return datetime.datetime.now().strftime("%A")


def get_search_movie_api_doc(query: str) -> str:
    """Get the API documentation for searching for movies by their original, translated and alternative titles."""
    loader = WebBaseLoader('https://developer.themoviedb.org/reference/search-movie')
    document = loader.load()
    return document[0].page_content


def search_movie(query: str, api_documentation: str) -> str:
    """Connect to the api and search for movies by their original, translated and alternative titles."""
    chain = APIChain.from_llm_and_api_docs(
        llm,
        api_documentation,
        verbose=True,
    )
    response = chain.run(
        query
    )
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
