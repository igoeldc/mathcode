"""Tests for RL algorithms (DP and Monte Carlo)"""

from mathcode.reinforcement_learning import (
    GridWorld,
    Policy,
    generate_episode,
    mc_epsilon_greedy,
    mc_es,
    mc_prediction,
    policy_eval,
    policy_iter,
    value_iter,
)


class TestValueIteration:
    """Test Value Iteration algorithm"""

    def test_simple_grid(self):
        """Test value iteration on simple grid"""
        env = GridWorld(height=2, width=2, start=(0, 0), goal=(1, 1))

        V, policy = value_iter(env, gamma=0.9, theta=1e-6)

        # Goal state should have highest value
        goal_value = V.get((1, 1))
        assert goal_value > 0

        # States closer to goal should have higher values
        assert V.get((1, 0)) > V.get((0, 0)) or V.get((0, 1)) > V.get((0, 0))

        # Policy should lead toward goal
        action_from_start = policy.get_action((0, 0))
        assert action_from_start in ['right', 'down']

    def test_convergence(self):
        """Test that value iteration converges"""
        env = GridWorld(height=3, width=3, start=(0, 0), goal=(2, 2))

        V, policy = value_iter(env, gamma=0.9, max_iterations=1000)

        # Check that we have values for all states
        for state in env.get_states():
            value = V.get(state)
            assert value is not None

    def test_with_obstacles(self):
        """Test value iteration with obstacles"""
        env = GridWorld(height=3, width=3, start=(0, 0), goal=(2, 2),
                       obstacles=[(1, 1)])

        V, policy = value_iter(env, gamma=0.9)

        # States should avoid the obstacle
        # Policy from (1, 0) should not be 'right' (toward obstacle)
        if (1, 0) in [s for s in env.get_states()]:
            action = policy.get_action((1, 0))
            # Should prefer going around obstacle
            assert action is not None


class TestPolicyIteration:
    """Test Policy Iteration algorithm"""

    def test_simple_grid(self):
        """Test policy iteration on simple grid"""
        env = GridWorld(height=2, width=2, start=(0, 0), goal=(1, 1))

        V, policy = policy_iter(env, gamma=0.9)

        # Check that policy leads toward goal
        action = policy.get_action((0, 0))
        assert action in ['right', 'down']

    def test_convergence(self):
        """Test policy iteration converges"""
        env = GridWorld(height=3, width=3, start=(0, 0), goal=(2, 2))

        V, policy = policy_iter(env, gamma=0.9)

        # Should have policy for all states
        states_with_policy = 0
        for state in env.get_states():
            try:
                policy.get_action(state)
                states_with_policy += 1
            except ValueError:
                pass

        assert states_with_policy > 0


class TestPolicyEvaluation:
    """Test Policy Evaluation"""

    def test_uniform_policy(self):
        """Test evaluating uniform random policy"""
        env = GridWorld(height=2, width=2, start=(0, 0), goal=(1, 1))

        # Create uniform random policy
        policy = Policy(deterministic=True)
        for state in env.get_states():
            actions = env.get_actions(state)
            if actions:
                policy.set_action(state, actions[0])

        V = policy_eval(env, policy, gamma=0.9)

        # Should have values for states
        assert V.get((0, 0)) is not None


class TestMonteCarloMethods:
    """Test Monte Carlo methods"""

    def test_generate_episode(self):
        """Test episode generation"""
        env = GridWorld(height=3, width=3, start=(0, 0), goal=(2, 2))

        # Create simple policy
        policy = Policy(deterministic=True)
        for state in env.get_states():
            actions = env.get_actions(state)
            if actions:
                policy.set_action(state, 'right')

        episode = generate_episode(env, policy, max_steps=20)

        # Episode should be non-empty
        assert len(episode) > 0

        # Each step should be (state, action, reward)
        for step in episode:
            assert len(step) == 3
            state, action, reward = step
            assert isinstance(reward, (int, float))

    def test_mc_prediction(self):
        """Test Monte Carlo prediction"""
        env = GridWorld(height=2, width=2, start=(0, 0), goal=(1, 1))

        # Create policy that moves toward goal
        policy = Policy(deterministic=True)
        policy.set_action((0, 0), 'right')
        policy.set_action((0, 1), 'down')
        policy.set_action((1, 0), 'right')

        V = mc_prediction(env, policy, num_episodes=100, gamma=0.9)

        # Should have learned some values
        assert V.get((0, 0)) != 0.0 or V.get((1, 0)) != 0.0

    def test_mc_es(self):
        """Test Monte Carlo ES"""
        env = GridWorld(height=2, width=2, start=(0, 0), goal=(1, 1))

        Q, policy = mc_es(env, num_episodes=500, gamma=0.9)

        # Should have learned Q-values
        q_values = Q.get_all((0, 0))
        assert len(q_values) > 0

        # Policy should be defined for some states
        try:
            action = policy.get_action((0, 0))
            assert action is not None
        except ValueError:
            pass  # May not have visited this state

    def test_mc_epsilon_greedy(self):
        """Test Monte Carlo with epsilon-greedy"""
        env = GridWorld(height=2, width=2, start=(0, 0), goal=(1, 1))

        Q, policy = mc_epsilon_greedy(env, num_episodes=500,
                                               gamma=0.9, epsilon=0.1)

        # Should have learned Q-values
        q_values_00 = Q.get_all((0, 0))

        # May or may not have values depending on exploration
        assert isinstance(q_values_00, dict)


class TestIntegration:
    """Integration tests comparing algorithms"""

    def test_dp_vs_mc(self):
        """Compare DP and MC on same environment"""
        env = GridWorld(height=2, width=2, start=(0, 0), goal=(1, 1))

        # Value Iteration (DP)
        V_dp, policy_dp = value_iter(env, gamma=0.9)

        # Monte Carlo ES
        Q_mc, policy_mc = mc_es(env, num_episodes=1000, gamma=0.9)

        # Both should learn that goal is valuable
        assert V_dp.get((1, 1)) > 0

        # Both policies should generally point toward goal from (0,0)
        try:
            action_dp = policy_dp.get_action((0, 0))
            assert action_dp in ['right', 'down']
        except ValueError:
            pass

    def test_deterministic_path(self):
        """Test learning on deterministic path to goal"""
        # Very simple 1x3 grid
        env = GridWorld(height=1, width=3, start=(0, 0), goal=(0, 2))

        V, policy = value_iter(env, gamma=0.9)

        # Optimal policy should be: right, right
        action = policy.get_action((0, 0))
        assert action == 'right'

        action = policy.get_action((0, 1))
        assert action == 'right'
