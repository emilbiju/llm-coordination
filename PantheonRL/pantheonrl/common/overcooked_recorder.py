import numpy as np
from typing import List, Optional, Dict, Tuple
import pickle

from .multiagentenv import SimultaneousEnv
from .trajsaver import SimultaneousTransitions, MultiTransitions

class OvercookedRecorder(SimultaneousEnv):
    """
    Recorder for Overcooked environments that preserves full state information

    :param env: The environment to record
    """

    def __init__(self, env):
        super(OvercookedRecorder, self).__init__(partners=env.partners[0])
        self.env = env

        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.allegoobs = []  # Encoded observations for the ego agent
        self.allegoacts = []  # Actions for the ego agent
        self.allaltobs = []  # Encoded observations for the partner agent
        self.allaltacts = []  # Actions for the partner agent
        self.allflags = []   # Done flags
        self.incomplete = False

        # Store full state information
        self.full_states = []
        self.rewards = []
        self.infos = []

    def multi_step(
                    self,
                    ego_action: np.ndarray,
                    alt_action: np.ndarray
                ) -> Tuple[Tuple[Optional[np.ndarray], Optional[np.ndarray]],
                           Tuple[float, float], bool, Dict]:
        """
        This function calls the embedded environment's multi_step and records
        the new actions and observations.
        """
        obs, rews, done, info = self.env.multi_step(ego_action, alt_action)
        
        # Record encoded observations and actions
        self.allegoacts.append(ego_action)
        self.allaltacts.append(alt_action)
        
        if not done:
            self.allegoobs.append(obs[0])
            self.allaltobs.append(obs[1])
            self.allflags.append(0)  # NOT_DONE
        else:
            self.allflags.append(1)  # DONE
            self.incomplete = False

        # Record full state information
        self.full_states.append(self.env.env.base_env.state)
        self.rewards.append(rews)
        self.infos.append(info)
        
        return obs, rews, done, info

    def multi_reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function calls the embedded environment's multi_reset and records
        the new observations.
        """
        obs = self.env.multi_reset()
        self.allegoobs.append(obs[0])
        self.allaltobs.append(obs[1])
        self.incomplete = True
        
        # Record initial state
        self.full_states.append(self.env.env.base_env.state)
        return obs

    def get_transitions(self) -> SimultaneousTransitions:
        """ Return the recorded transitions """
        egoobsarr = np.array(self.allegoobs)
        altobsarr = np.array(self.allaltobs)
        if self.incomplete:
            egoobsarr = egoobsarr[:-1]
            altobsarr = altobsarr[:-1]
        
        transitions = SimultaneousTransitions(
                    egoobsarr,
                    np.array(self.allegoacts),
                    altobsarr,
                    np.array(self.allaltacts),
                    np.array(self.allflags)
                )
        
        # Save full trajectory information
        trajectory = {
            "states": self.full_states[:-1] if self.incomplete else self.full_states,
            "rewards": self.rewards,
            "infos": self.infos,
            "encoded_transitions": transitions
        }
        
        return trajectory

    def write_trajectory(self, file):
        """Write the full trajectory information to a file"""
        trajectory = self.get_transitions()
        with open(file, 'wb') as f:
            pickle.dump(trajectory, f) 