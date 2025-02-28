# https://www.youtube.com/watch?v=yDHqhEYbAvw

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

load_dotenv()

web_search_agent = Agent(
    model = Groq(id="llama-3.3-70b-versatile"),
    tools = [DuckDuckGo()],
    instructions = "Always include the sources",
    show_tool_calls = True,
    markdown = True,
    debug_mode = True
)


web_search_agent.print_response("Guess my name. I'm female, starts with letter D and ends with A anh having total 9 letters in my Indian name, my name meaning is direction of the Dhruv star.", stream=True)


# agent = Agent(
#     model=Groq(id="llama-3.3-70b-versatile"),
#     markdown=True
# )
#
# # Get the response in a variable
# # run: RunResponse = agent.run("Share a 2 sentence horror story.")
# # print(run.content)
#
# # Print the response in the terminal
# agent.print_response("What is the capital of India?")

