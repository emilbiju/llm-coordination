import pickle
import json
import numpy as np
from tqdm import trange
from typing import Any

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        # Handle tuples (for positions and orientations)
        if isinstance(obj, tuple):
            return list(obj)
        # Try to convert object to dict if it has __dict__
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        try:
            # Try to convert to string representation if possible
            return str(obj)
        except:
            return f"<Object of type {type(obj).__name__}>"

def convert_state_to_dict(state):
    """Convert Overcooked state object to dictionary"""
    state_dict = {
        "player_positions": state.player_positions,
        "player_orientations": state.player_orientations,
        "curr_order": str(state.curr_order),
    }
    
    # Handle objects dictionary - convert tuple keys to strings
    objects_dict = {}
    for obj_pos, obj in state.objects.items():
        # Convert tuple key to string representation
        pos_key = str(list(obj_pos))
        objects_dict[pos_key] = obj
    state_dict["objects"] = objects_dict
    
    return state_dict

def get_transitions_attributes(transitions):
    """Get all numpy array attributes from transitions"""
    attrs = {}
    for attr in dir(transitions):
        if not attr.startswith('_'):
            value = getattr(transitions, attr)
            if isinstance(value, np.ndarray):
                attrs[attr] = value.shape
    return attrs

def convert_trajectory_to_jsonl(pickle_file, jsonl_file):
    """Convert pickle trajectory file to JSONL format"""
    # Load pickle file
    with open(pickle_file, 'rb') as f:
        trajectory = pickle.load(f)
    
    # Get transitions data
    transitions = trajectory['encoded_transitions']
    
    # Print available attributes
    print("Available transition attributes:")
    attrs = get_transitions_attributes(transitions)
    for attr, shape in attrs.items():
        print(f"{attr}: shape {shape}")
    
    # Convert each timestep to a JSON line
    with open(jsonl_file, 'w') as f:
        for i in trange(len(trajectory['states'])):
            timestep_data = {
                "timestep": i,
                "state": convert_state_to_dict(trajectory['states'][i]),
                "reward": trajectory['rewards'][i] if i < len(trajectory['rewards']) else None,
                "info": trajectory['infos'][i] if i < len(trajectory['infos']) else None,
            }
            
            # Add all transition arrays for this timestep
            for attr, shape in attrs.items():
                value = getattr(transitions, attr)
                if i < len(value):
                    timestep_data[attr] = value[i].tolist()
            
            json.dump(timestep_data, f, cls=NumpyEncoder)
            f.write('\n')

if __name__ == "__main__":
    pickle_file = '/future/u/herumbshandilya/home/PantheonRL/unident_s_trajectory.pkl'
    jsonl_file = '/future/u/herumbshandilya/home/PantheonRL/unident_s_trajectory.jsonl'
    convert_trajectory_to_jsonl(pickle_file, jsonl_file)
    print(f"\nConverted {pickle_file} to {jsonl_file}")
    
    # Print first few lines of the JSONL file
    print("\nFirst few lines of the JSONL file:")
    with open(jsonl_file, 'r') as f:
        for i, line in enumerate(f):
            if i < 3:  # Print first 3 lines
                print(f"\nLine {i+1}:")
                print(json.dumps(json.loads(line), indent=2)) 