import json
import random
import openai
import time
import os
import sys

os.chdir('C:/Users/finsc/generative_agents_-exp/reverie/backend_server')
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

backend_server_path = os.path.join(parent_dir, 'reverie', 'backend_server')
sys.path.append(backend_server_path)

print("Current Working Directory:", os.getcwd())

from persona.persona import *
from persona.cognitive_modules import converse
from global_methods import *

from maze import *

from utils import *

openai.api_key = openai_api_key

from persona.cognitive_modules.plan import _choose_retrieved
storage = "../../environment/frontend_server/storage"
# Function to load simulation and history if provided
def load_simulation(sim_code, load_history=False, history_file=None):
    sim_folder = f"{storage}/{sim_code}"
    with open(f"{sim_folder}/reverie/meta.json") as json_file:
        reverie_meta = json.load(json_file)

    maze = Maze(reverie_meta['maze_name'])

    personas = {}
    for persona_name in reverie_meta['persona_names']:
        persona_folder = f"{sim_folder}/personas/{persona_name}"
        curr_persona = Persona(persona_name, persona_folder)
        personas[persona_name] = curr_persona
    
    
    personas["Klaus Mueller"].scratch.known_personas.append("Maria Lopez")
    personas["Maria Lopez"].scratch.known_personas.append("Klaus Mueller")

    if load_history and history_file:
        curr_file = history_file
        rows = read_file_to_list(curr_file, header=True, strip_trail=True)[1]
        clean_whispers = []
        for row in rows:
            agent_name = row[0].strip()
            whispers = row[1].split(";")
            whispers = [whisper.strip() for whisper in whispers]
            for whisper in whispers:
                clean_whispers.append([agent_name, whisper])

        load_history_via_whisper(personas, clean_whispers)

    return maze, personas

# Function to generate chats
def generate_chats(maze, personas, num_chats=20):
    all_chats = []
    for i in range(num_chats):
        chat = agent_chat_v2(maze, personas["Maria Lopez"], personas["Klaus Mueller"])
        all_chats.append(chat)
    return all_chats

# Function to save chats to a file
def save_chats_to_file(file_path, all_chats):
    with open(file_path, 'w') as file:
        for idx, chat in enumerate(all_chats):
            file.write(f"Chat {idx + 1}:\n")
            for speaker, message in chat:
                file.write(f'{speaker}: {message}\n')
            file.write("\n")

# Define simulation codes and history files
simulations = {
    #"sentimentAnalysis_1": "agent_history_init_n3.csv",
    # "sentimentAnalysisEvil_1": "evilHistory.csv",
    #"sentimentAnalysis_2": "agent_history_init_n3.csv",
    #"sentimentAnalysisEvil_2": "evilHistory.csv",
    #"sentimentAnalysisWithHistEvil_1": None
    # "sentimentAnalysisWithHistEvil_2": None
    #"2sentimentAnalysis_1" : "agent_history_init_n3.csv",
   # "2sentimentAnalysis_2" : "agent_history_init_n3.csv",
    #"2sentimentAnalysisEvil_1": "evilHistory.csv",
    #"2sentimentAnalysisEvil_2": "evilHistory.csv",
    "2sentimentAnalysisWithHistEvil_1": None,
    "2sentimentAnalysisWithHistEvil_2": None,
}

# Generate and save chats for each simulation with and without history
for sim_code, history_file in simulations.items():
    # Load simulation without history
    maze, personas = load_simulation(sim_code, load_history=False)
    chats_without_history = generate_chats(maze, personas, 50)
    save_chats_to_file(f"../../analysis/{sim_code}_characterRelationshipPrompt_without_history.txt", chats_without_history)

    # Load simulation with history only if a history file is provided
    if history_file is not None:
        maze, personas = load_simulation(sim_code, load_history=True, history_file=f"../../environment/frontend_server/static_dirs/assets/the_ville/{history_file}")
        chats_with_history = generate_chats(maze, personas, 50)
        save_chats_to_file(f"../../analysis/{sim_code}_with_history.txt", chats_with_history)


print("Chat generation complete.")
