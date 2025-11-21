"""Tests for reinforcement learning environments"""

import pytest
from mathcode.reinforcement_learning import GridWorld


class TestGridWorld:
    """Test GridWorld environment"""

    def test_initialization(self):
        """Test environment initialization"""
        env = GridWorld(height=4, width=4)
        assert env.height == 4
        assert env.width == 4
        assert env.current_pos == (0, 0)
        assert env.goal_pos == (3, 3)

    def test_reset(self):
        """Test environment reset"""
        env = GridWorld(height=4, width=4, start=(0, 0))
        env.current_pos = (2, 2)
        state = env.reset()
        assert state == (0, 0)
        assert env.current_pos == (0, 0)

    def test_step_valid_move(self):
        """Test valid movement"""
        env = GridWorld(height=4, width=4, start=(1, 1))
        env.reset()

        # Move right
        next_state, reward, done, info = env.step('right')
        assert next_state == (1, 2)
        assert reward == -1.0
        assert not done

        # Move up
        next_state, reward, done, info = env.step('up')
        assert next_state == (0, 2)

    def test_step_boundary(self):
        """Test hitting boundary"""
        env = GridWorld(height=4, width=4, start=(0, 0))
        env.reset()

        # Try to move up from top-left corner
        next_state, reward, done, info = env.step('up')
        assert next_state == (0, 0)  # Stay in place
        assert reward == -1.0
        assert not done

    def test_step_goal(self):
        """Test reaching goal"""
        env = GridWorld(height=4, width=4, start=(3, 2), goal=(3, 3))
        env.reset()

        next_state, reward, done, info = env.step('right')
        assert next_state == (3, 3)
        assert reward == 10.0
        assert done

    def test_step_obstacle(self):
        """Test hitting obstacle"""
        env = GridWorld(height=4, width=4, start=(0, 0), obstacles=[(0, 1)])
        env.reset()

        next_state, reward, done, info = env.step('right')
        assert next_state == (0, 1)
        assert reward == -10.0
        assert done

    def test_get_states(self):
        """Test getting all states"""
        env = GridWorld(height=3, width=3)
        states = env.get_states()
        assert len(states) == 9
        assert (0, 0) in states
        assert (2, 2) in states

    def test_get_actions(self):
        """Test getting available actions"""
        env = GridWorld(height=4, width=4)
        actions = env.get_actions((0, 0))
        assert len(actions) == 4
        assert 'up' in actions
        assert 'down' in actions
        assert 'left' in actions
        assert 'right' in actions

    def test_invalid_action(self):
        """Test invalid action raises error"""
        env = GridWorld(height=4, width=4)
        env.reset()

        with pytest.raises(ValueError, match="Invalid action"):
            env.step('invalid_action')


class TestGridWorldScenarios:
    """Test specific GridWorld scenarios"""

    def test_simple_path(self):
        """Test moving along a simple path"""
        env = GridWorld(height=2, width=2, start=(0, 0), goal=(1, 1))
        env.reset()

        # Path: right, down
        state, reward, done, _ = env.step('right')
        assert state == (0, 1)
        assert not done

        state, reward, done, _ = env.step('down')
        assert state == (1, 1)
        assert reward == 10.0
        assert done

    def test_obstacle_avoidance(self):
        """Test navigating around obstacles"""
        env = GridWorld(height=3, width=3, start=(0, 0), goal=(0, 2),
                       obstacles=[(0, 1)])
        env.reset()

        # Can't go directly right, must go around
        # Path: down, right, right, up
        env.step('down')  # (1, 0)
        env.step('right')  # (1, 1)
        env.step('right')  # (1, 2)
        state, reward, done, _ = env.step('up')  # (0, 2)

        assert state == (0, 2)
        assert done
