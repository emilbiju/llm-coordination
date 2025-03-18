import pickle
import numpy as np

def print_state_info(state, step_num):
    """Print detailed information about a state"""
    print(f"\nState at step {step_num}:")
    print(f"Player positions: {state.player_positions}")
    print(f"Player orientations: {state.player_orientations}")
    print(f"Objects: {state.objects}")
    print(f"Current order: {state.curr_order}")

def print_trajectory_info(trajectory, num_steps_to_show=5):
    """Print information about the trajectory in a readable format"""
    print("\n=== Trajectory Information ===")
    
    # Print basic statistics
    print("\nBasic Statistics:")
    print(f"Number of states: {len(trajectory['states'])}")
    print(f"Number of rewards: {len(trajectory['rewards'])}")
    print(f"Number of info dictionaries: {len(trajectory['infos'])}")
    
    # Print first few states
    print(f"\n=== First {num_steps_to_show} Steps Information ===")
    for i in range(min(num_steps_to_show, len(trajectory['states']))):
        print_state_info(trajectory['states'][i], i)
        if i < len(trajectory['rewards']):
            print(f"Reward: {trajectory['rewards'][i]}")
            print(f"Info: {trajectory['infos'][i]}")
    
    # Print encoded transitions info
    print("\n=== Encoded Transitions Info ===")
    transitions = trajectory['encoded_transitions']
    print(f"Type of transitions: {type(transitions)}")
    
    # Get the attributes that contain the actual data
    data_attrs = [attr for attr in dir(transitions) if not attr.startswith('_')]
    print("\nTransitions data:")
    for attr in data_attrs:
        data = getattr(transitions, attr)
        if isinstance(data, (np.ndarray, list)):
            if isinstance(data, np.ndarray):
                print(f"\n{attr}:")
                print(f"Shape: {data.shape}")
                print(f"First {num_steps_to_show} entries:")
                if len(data) > 0:
                    print(data[:num_steps_to_show])
            else:
                print(f"\n{attr}:")
                print(f"Length: {len(data)}")
                print(f"First {num_steps_to_show} entries:")
                print(data[:num_steps_to_show])

# Load and analyze the trajectory
with open('unident_s_trajectory.pkl', 'rb') as f:
    trajectory = pickle.load(f)

print_trajectory_info(trajectory) 