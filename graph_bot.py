from graph_agents import *
from dotenv import load_dotenv

load_dotenv()

graph = create_graph()
config = {"configurable" : {"thread_id" : "1"}}

start_chat()