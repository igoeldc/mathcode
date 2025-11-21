from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class Environment(ABC):
    """
    Abstract base class for RL environments

    Follows the standard RL interface with states, actions, and rewards.
    """

    @abstractmethod
    def reset(self) -> Any:
        """
        Reset environment to initial state

        Returns:
        --------
        state : Any
            Initial state
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        """
        Take action in environment

        Parameters:
        -----------
        action : Any
            Action to take

        Returns:
        --------
        next_state : Any
            Resulting state after action
        reward : float
            Immediate reward received
        done : bool
            Whether episode has terminated
        info : dict
            Additional information (optional)
        """
        pass

    @abstractmethod
    def get_states(self) -> List[Any]:
        """
        Get all possible states

        Returns:
        --------
        List of all states in the environment
        """
        pass

    @abstractmethod
    def get_actions(self, state: Any) -> List[Any]:
        """
        Get valid actions for a given state

        Parameters:
        -----------
        state : Any
            State to query

        Returns:
        --------
        List of valid actions in this state
        """
        pass

    def render(self) -> None:  # noqa: B027
        """Render the environment (optional)"""
        pass


class GridWorld(Environment):
    """
    Simple grid world environment

    Agent navigates a grid to reach goal while avoiding obstacles.
    """

    def __init__(self, height: int = 4, width: int = 4,
                 start: Tuple[int, int] = (0, 0),
                 goal: Tuple[int, int] = (3, 3),
                 obstacles: List[Tuple[int, int]] = None):
        """
        Initialize grid world

        Parameters:
        -----------
        height : int
            Grid height
        width : int
            Grid width
        start : Tuple[int, int]
            Starting position (row, col)
        goal : Tuple[int, int]
            Goal position (row, col)
        obstacles : List[Tuple[int, int]]
            List of obstacle positions
        """
        self.height = height
        self.width = width
        self.start_pos = start
        self.goal_pos = goal
        self.obstacles = obstacles if obstacles else []
        self.current_pos = start

        # Actions: up, down, left, right
        self.actions = ['up', 'down', 'left', 'right']
        self.action_effects = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

    def reset(self) -> Tuple[int, int]:
        """Reset to starting position"""
        self.current_pos = self.start_pos
        return self.current_pos

    def step(self, action: str) -> Tuple[Tuple[int, int], float, bool, dict]:
        """
        Take action in grid world

        Parameters:
        -----------
        action : str
            One of 'up', 'down', 'left', 'right'

        Returns:
        --------
        next_state : Tuple[int, int]
            New position
        reward : float
            -1 for each step, +10 for reaching goal, -10 for obstacle
        done : bool
            True if reached goal or obstacle
        info : dict
            Empty dict
        """
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}")

        # Calculate new position
        delta = self.action_effects[action]
        new_row = self.current_pos[0] + delta[0]
        new_col = self.current_pos[1] + delta[1]
        new_pos = (new_row, new_col)

        # Check boundaries
        if not (0 <= new_row < self.height and 0 <= new_col < self.width):
            # Hit wall - stay in place
            new_pos = self.current_pos
            reward = -1.0
            done = False
        elif new_pos in self.obstacles:
            # Hit obstacle - episode ends
            self.current_pos = new_pos
            reward = -10.0
            done = True
        elif new_pos == self.goal_pos:
            # Reached goal
            self.current_pos = new_pos
            reward = 10.0
            done = True
        else:
            # Normal move
            self.current_pos = new_pos
            reward = -1.0
            done = False

        return self.current_pos, reward, done, {}

    def get_states(self) -> List[Tuple[int, int]]:
        """Get all grid positions"""
        states = []
        for row in range(self.height):
            for col in range(self.width):
                states.append((row, col))
        return states

    def get_actions(self, state: Tuple[int, int]) -> List[str]:
        """All actions available from any state"""
        return self.actions.copy()

    def render(self) -> None:
        """Print grid with agent position"""
        for row in range(self.height):
            line = ""
            for col in range(self.width):
                pos = (row, col)
                if pos == self.current_pos:
                    line += "A "  # Agent
                elif pos == self.goal_pos:
                    line += "G "  # Goal
                elif pos in self.obstacles:
                    line += "X "  # Obstacle
                else:
                    line += ". "  # Empty
            print(line)
        print()


## Example Code

# # Create grid world
# env = GridWorld(height=4, width=4, start=(0, 0), goal=(3, 3),
#                 obstacles=[(1, 1), (2, 2)])
#
# # Reset environment
# state = env.reset()
# env.render()
#
# # Take some actions
# state, reward, done, info = env.step('right')
# print(f"Reward: {reward}, Done: {done}")
# env.render()
