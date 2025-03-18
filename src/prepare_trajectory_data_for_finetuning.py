import pandas as pd
import datasets
import os
from datasets import Dataset, DatasetDict
from google.colab import drive
from huggingface_hub import notebook_login
drive.mount('/content/drive')

# !pip install datasets
# !pip install trl
# !pip install --upgrade sympy
# !pip install --upgrade trl

class LLMAgent:
    def __init__(self, player_id, layout_name, model_name):
        self.player_id = player_id
        self.layout_name = layout_name
        self.player_names = ['Alice', 'Bob']

        self.player_actions = []

        # Set other player ID
        if int(self.player_id) == 0:
            self.other_player_id = 1
        else:
            self.other_player_id = 0

        EnvDescriptions= {
            'cramped_room': 'Environment Details: The environment is rectangular with 2 onions dispensers (o0, o1), cooker (c0), plate dispenser (p0) and delivery area (d0). Additionally there are kitchen counters (k0 to k8) which can be used to temporarily store onions and plates while you do something else. Objects on counters can be picked up later and should be considered as they may be closer than items in dispensers.',

            'no_counter_door': 'Environment Details: The environment is cramped with little space to move aroundrectangular with 2 onions dispensers (o0, o1), cooker (c0) and plate dispenser (p0). The delivery area (d0) is inaccessible behind closed gates and can be accessed by opening one of the gates (g0, g1). A gate can only be opened by a player if they are not carrying an object. Once the door is opened it will only stay open for a brief time and then close on its own',

            'soup_passing':"Environment Details: The environment is divided into 2 partitions. Alice is in the left partition with access to onion dispenser (o0), plate dispenser (p1), cooker (c0), delivery area (d0) and kitchen counters (k0, k1, k8, k11, k12). Bob is in the right partition with access to onion dispenser (o1), plate dispenser (p0), cooker (c1) and kitchen counters (k3, k4, k6, k10, k14, k15). Only Alice has access to the delivery area (d0). There are shared counters (s0, s1, s2, s3) which can be used to pass onions, plates or cooked soup from one player to the other. Note that the objects on the shared counters can be accessed by both players. Objects on counters can be picked up later and should be considered as they may be closer than items in dispensers. ",

            'soup_passing_door':"Environment Details: The environment is divided into 2 partitions. Alice is in the left partition with access to onion dispenser o0, plate dispenser p1, cooker c0, and delivery area d0. Bob is in the right partition with access to onion dispenser o1, plate dispenser p0, and cooker c1.  The two partitions are connected by gate g0 which can be opened if a player is holding nothing. Opening the gate will allow players to move freely between partitions.  The gate will close after enough time automatically. ",

            'asymmetric_advantages' : 'Environment Details: There are two partitions in the current environment. Bob is in the left partition with access to onion dispenser o0, delivery area d0, plate dispenser p0 and kitchen counters k0, k1, k2, k3, k4, k11, k12, k16, k18, k20, k21, k22, k23. Alice is in the right partition and has access to onion dispenser o1, delivery area d1, plate dispenser p1 and kitchen counters k6, k7, k8, k9, k10, k14, k15, k17, k19, k25, k26, k27, k28. Both have access to both cookers c0 and c1 which are on the partition line. Kitchen counters (k0 to k28) which can be used to temporarily store onions and plates while you do something else. Objects on counters can be picked up later and should be considered as they may be closer than items in dispensers.',

            'forced_coordination' : 'Environment Details: The environment is split into two partitions, one with each player. In the right partition, Alice has access to cookers (c0, c1),  delivery area (d0) and kitchen counters (k6, k8, k12). In the left partition, Bob has access to onion dispensers (o0, o1), plate dispenser (p0) and kitchen counters (k1, k10). Kitchen counters can be used to temporarily store onions and plates while you do something else. Both players have access to shared counters (s0, s1, s2) which can be used to transfer onions and plates to the other player depending on the situation. Note that the objects on the shared counters can be accessed by both players. ',


            'coordination_ring': 'Environment Details: The environment is narrow and circular with room for only one player to walk along a path or access places. It features onion dispensers (o0, o1), plate dispenser (p0), cookers (c0, c1), and a delivery area (d0). Additionally there are kitchen counters (k0 to k10) which should only be used to temporarily store onions and plates while you do something else. Objects on counters can be picked up later and should be considered as they may be closer than items in dispensers',

            'counter_circuit_o_1order': 'Environment Details: The environment is circular with two onion dispensers (o0, o1), plate dispenser (p0), cookers (c0, c1) and delivery area d0. There are also the shared counters (s0, s1, s2, s3) which can be used to pass objects from one player to the other. Additionally there are kitchen counters (k0 to k15) which can be used to temporarily store onions and plates while you do something else. ',

            'bottleneck': 'Environment Details: There is 1 onion dispenser (o0), plate dispenser (p0), cookers (c0, c1) and delivery area (d0). Additionally there are kitchen counters (k0 to k17) which can be used to temporarily store onions and plates while you do something else. ',

            'large_room': 'Environment Details: There are 2 onions dispensers (o0, o1), cooker (c0), plate dispenser (p0) and delivery area (d0). Additionally there are kitchen counters (k0 to k19) which can be used to temporarily store onions and plates while you do something else. ',

            'centre_objects': 'Environment Details: There is 1 onion dispenser (o0), cooker (c0), plate dispenser (p0) and delivery area (d0). Additionally there are kitchen counters (k0 to k24) which can be used to temporarily store onions and plates while you do something else. ',

            'centre_pots': 'Environment Details: There are 2 onion dispenser (o0, 1), 2 cookers (c0, c1), 2 plate dispensers (p0, 1) and 2 delivery areas (d0, d1). Additionally there are kitchen counters (k0 to k14) which can be used to temporarily store onions and plates while you do something else. ',
        }

        self.llm_system_prompt = "You are a friendly chat assistant who is correct and brief at all times."
        self.partner_action_inference_string = ''

        self.rules = f'''Players must coordinate to make onion soups with 3 onions each. Once a soup is cooked it needs to be placed on a plate and delivered. Players can only carry one item at a time. A soup can only be loaded onto plate by a player if they are holding a plate. The goal is to maximize the number of deliveries.'''

        self.conventions_explicit_help = f'''
        1. We will try to maximize the number of deliveries.
        2. We will try to be efficient and prepare for the next soup while the current soup is cooking.
        3. If we are in the same section, we will minimize our movement to avoid getting in each other's way.
        4. We will prefer helping the other player with their cooking and delivery if the situation arises.'''



        self.conventions = f'''1. We want to be efficient and prepare for the next soup while the current soup is cooking.
        '''

        self.pi_prompt = f'''I am {self.player_names[self.player_id]}. I am playing the game Overcooked with my partner {self.player_names[self.other_player_id]}. {EnvDescriptions[self.layout_name]}
        Overcooked has the following rules: {self.rules}. We have agreed to follow the following conventions: {self.conventions}. I'll provide my action history, current state, teammate's status, and my possible actions. Help me understand my partner's intentions and needs. describe what my partner intends to do or needs in one sentence only. Do not say anything else.'''

        self.verifier_prompt = f'''I am {self.player_names[self.player_id]}. I am playing the game Overcooked with my partner {self.player_names[self.other_player_id]}. {EnvDescriptions[self.layout_name]}
        Overcooked has the following rules: {self.rules}. We have agreed to follow the following conventions: {self.conventions}.'''

        self.verifier_system_prompt = 'You are an action verification agent for Overcooked. I will provide you with my inventory, location information, and state information for me and my partner and my selected action. You need to check whether the action satisfies the criteria: 1. Rule Following: It follows to the rules of the game. 2. Convention Following: It adheres to the mentioned conventions 3. Safety: The selected action does not lead to the game being stuck. Your response should be Reasoning:<Brief Reasoning for Verification> followed by "Verification: Okay" if selected action follows **all three** criteria and "Verification: Not Okay" otherwise. Do not say anything else.'


        # With COT
        # self.base_prompt = f'''I am {self.player_names[self.player_id]}. I am playing the game Overcooked with my partner {self.player_names[self.other_player_id]}. {EnvDescriptions[self.layout_name]}
        # Overcooked has the following rules: {self.rules}. We have agreed to follow the following conventions: {self.conventions}. I'll provide my action history, current state, teammate's status, and my possible actions. Help me select the best action from the list. Format your response as: Explanation:<Brief explanation for my next action>. Action: <action>. Only select one action. Do not say anything else. Got it?'''

        # Without COT
        self.base_prompt = f'''I am {self.player_names[self.player_id]}. I am playing the game Overcooked with my partner {self.player_names[self.other_player_id]}. {EnvDescriptions[self.layout_name]}
        Overcooked has the following rules: {self.rules}. We have agreed to follow the following conventions: {self.conventions}. I'll provide my action history, current state, teammate's status, and my possible actions. Help me select the best action from the list. Format your response as: Action: <action>. Only select one action. Do not say anything else.'''

        # self.base_prompt = f'''I am {self.player_names[self.player_id]}. I am playing the game Overcooked with my partner {self.player_names[self.other_player_id]}. Overcooked has the following rules: {self.rules}. We have agreed to follow the following conventions: {self.conventions}. I'll provide my action history, current state, teammate's status, and my possible actions. Help me select the best action from the list. Format your response as: Action: <action>. Only select one action. Do not say anything else. Got it?'''
        self.assistant_response_initial = f'''Got it!'''


llm_agent = LLMAgent(layout_name='cramped_room', player_id=0, model_name='gpt2')
directory_path = "/content/drive/MyDrive/Overcooked RL Agent Trajectories/agentic_evals/game_logs/ai/cramped_room/"
combined_data = combine_csv_files(directory_path)

trajectories = combined_data
messages = [[{"role":"user", "content": llm_agent.base_prompt+"\n\n"+trajectories['full_state_description'].iloc[i]}, {"role":"assistant", "content": trajectories['llm_response'].iloc[i]}] for i in range(len(trajectories))]

prompts = [llm_agent.base_prompt+"\n\n"+trajectories['full_state_description'].iloc[i] for i in range(len(trajectories))]
df = pd.DataFrame({'prompt': prompts, 'messages': messages})

# Calculate split sizes
train_size = int(0.8 * len(df))
val_size = int(0.1 * len(df))
test_size = len(df) - train_size - val_size

# Create Dataset objects
train_dataset = Dataset.from_pandas(df[:train_size])
val_dataset = Dataset.from_pandas(df[train_size:train_size + val_size])
test_dataset = Dataset.from_pandas(df[train_size + val_size:])

# Combine datasets into a DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

dataset.push_to_hub("emilbiju/overcooked-dataset-ppo")