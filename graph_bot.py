from graph_agents import *
from dotenv import load_dotenv

load_dotenv()

graph = create_graph()
config = {"configurable" : {"thread_id" : "2"}}

start_chat()