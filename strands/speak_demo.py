from strands import Agent
from strands_tools import speak

agent = Agent(tools=[speak])
agent("Say hello and introduce yourself using the speak tool with mode='polly' to generate audio")
