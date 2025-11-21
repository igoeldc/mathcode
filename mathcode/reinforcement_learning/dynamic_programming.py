from typing import Tuple

from .environment import Environment
from .policy import Policy, QFunc, VFunc


def value_iter(env: Environment,
                    gamma: float = 0.9,
                    theta: float = 1e-6,
                    max_iterations: int = 1000) -> Tuple[VFunc, Policy]:
    """
    Value Iteration algorithm

    Solves Bellman optimality equation:
    V*(s) = max_a Σ_s' P(s'|s,a)[R(s,a,s') + γV*(s')]

    Parameters:
    -----------
    env : Environment
        Environment with known dynamics
    gamma : float
        Discount factor (0 <= gamma <= 1)
    theta : float
        Convergence threshold
    max_iterations : int
        Maximum number of iterations

    Returns:
    --------
    V : VFunc
        Optimal value function
    policy : Policy
        Optimal policy derived from V
    """
    V = VFunc(default_value=0.0)
    states = env.get_states()

    # Initialize V(s) = 0 for all states
    for state in states:
        V.set(state, 0.0)

    # Value iteration loop
    for iteration in range(max_iterations):
        delta = 0.0

        # Update each state
        for state in states:
            old_value = V.get(state)

            # Compute max_a Q(s, a)
            actions = env.get_actions(state)
            if not actions:
                continue

            action_values = []
            for action in actions:
                # For deterministic environments, simulate one step
                # Set environment to current state before taking action
                if hasattr(env, 'current_pos'):
                    env.current_pos = state

                # Take action
                next_state, reward, done, _ = env.step(action)

                # Compute Q(s, a) = R + γV(s')
                q_value = reward
                if not done:
                    q_value += gamma * V.get(next_state)

                action_values.append(q_value)

            # V(s) = max_a Q(s, a)
            new_value = max(action_values)
            V.set(state, new_value)

            # Track maximum change
            delta = max(delta, abs(old_value - new_value))

        # Check convergence
        if delta < theta:
            print(f"Value iteration converged in {iteration + 1} iterations")
            break
    else:
        print(f"Value iteration stopped after {max_iterations} iterations")

    # Extract optimal policy
    policy = Policy(deterministic=True)
    for state in states:
        actions = env.get_actions(state)
        if not actions:
            continue

        # Find action that maximizes Q(s, a)
        best_action = None
        best_value = float('-inf')

        for action in actions:
            # Set environment to current state before taking action
            if hasattr(env, 'current_pos'):
                env.current_pos = state

            # Simulate action
            next_state, reward, done, _ = env.step(action)

            q_value = reward
            if not done:
                q_value += gamma * V.get(next_state)

            if q_value > best_value:
                best_value = q_value
                best_action = action

        if best_action is not None:
            policy.set_action(state, best_action)

    return V, policy


def policy_eval(env: Environment,
                      policy: Policy,
                      gamma: float = 0.9,
                      theta: float = 1e-6,
                      max_iterations: int = 1000) -> VFunc:
    """
    Policy Evaluation using Bellman expectation equation

    Computes value function for given policy:
    V^π(s) = Σ_a π(a|s) Σ_s' P(s'|s,a)[R(s,a,s') + γV^π(s')]

    Parameters:
    -----------
    env : Environment
        Environment
    policy : Policy
        Policy to evaluate
    gamma : float
        Discount factor
    theta : float
        Convergence threshold
    max_iterations : int
        Maximum iterations

    Returns:
    --------
    V : VFunc
        Value function for the policy
    """
    V = VFunc(default_value=0.0)
    states = env.get_states()

    # Initialize V(s) = 0 for all states
    for state in states:
        V.set(state, 0.0)

    # Iterative policy evaluation
    for _ in range(max_iterations):
        delta = 0.0

        for state in states:
            old_value = V.get(state)

            # Get action from policy
            try:
                action = policy.get_action(state)
            except ValueError:
                continue

            # Set environment to current state before taking action
            if hasattr(env, 'current_pos'):
                env.current_pos = state

            # Simulate action
            next_state, reward, done, _ = env.step(action)

            # V^π(s) = R + γV^π(s')
            new_value = reward
            if not done:
                new_value += gamma * V.get(next_state)

            V.set(state, new_value)

            delta = max(delta, abs(old_value - new_value))

        if delta < theta:
            break

    return V


def policy_iter(env: Environment,
                     gamma: float = 0.9,
                     theta: float = 1e-6,
                     max_iterations: int = 100) -> Tuple[VFunc, Policy]:
    """
    Policy Iteration algorithm

    Alternates between:
    1. Policy Evaluation: Compute V^π
    2. Policy Improvement: Make policy greedy w.r.t. V^π

    Parameters:
    -----------
    env : Environment
        Environment
    gamma : float
        Discount factor
    theta : float
        Convergence threshold for policy evaluation
    max_iterations : int
        Maximum policy improvement iterations

    Returns:
    --------
    V : VFunc
        Optimal value function
    policy : Policy
        Optimal policy
    """
    states = env.get_states()

    # Initialize random policy
    policy = Policy(deterministic=True)
    for state in states:
        actions = env.get_actions(state)
        if actions:
            policy.set_action(state, actions[0])  # Arbitrary initial policy

    # Policy iteration loop
    for iteration in range(max_iterations):
        # 1. Policy Evaluation
        V = policy_eval(env, policy, gamma, theta)

        # 2. Policy Improvement
        policy_stable = True

        for state in states:
            old_action = None
            try:
                old_action = policy.get_action(state)
            except ValueError:
                pass

            # Find best action
            actions = env.get_actions(state)
            if not actions:
                continue

            best_action = None
            best_value = float('-inf')

            for action in actions:
                # Set environment to current state before taking action
                if hasattr(env, 'current_pos'):
                    env.current_pos = state

                next_state, reward, done, _ = env.step(action)

                q_value = reward
                if not done:
                    q_value += gamma * V.get(next_state)

                if q_value > best_value:
                    best_value = q_value
                    best_action = action

            # Update policy
            if best_action is not None:
                policy.set_action(state, best_action)

                if best_action != old_action:
                    policy_stable = False

        # Check if policy is stable
        if policy_stable:
            print(f"Policy iteration converged in {iteration + 1} iterations")
            break
    else:
        print(f"Policy iteration stopped after {max_iterations} iterations")

    # Final policy evaluation
    V = policy_eval(env, policy, gamma, theta)

    return V, policy


def q_value_iter(env: Environment,
                      gamma: float = 0.9,
                      theta: float = 1e-6,
                      max_iterations: int = 1000) -> QFunc:
    """
    Q-Value Iteration

    Directly computes Q* using Bellman optimality equation:
    Q*(s,a) = Σ_s' P(s'|s,a)[R(s,a,s') + γ max_a' Q*(s',a')]

    Parameters:
    -----------
    env : Environment
        Environment
    gamma : float
        Discount factor
    theta : float
        Convergence threshold
    max_iterations : int
        Maximum iterations

    Returns:
    --------
    Q : QFunc
        Optimal Q-function
    """
    Q = QFunc(default_value=0.0)
    states = env.get_states()

    # Initialize Q(s, a) = 0 for all state-action pairs
    for state in states:
        for action in env.get_actions(state):
            Q.set(state, action, 0.0)

    # Q-value iteration loop
    for iteration in range(max_iterations):
        delta = 0.0

        for state in states:
            actions = env.get_actions(state)

            for action in actions:
                old_q = Q.get(state, action)

                # Set environment to current state before taking action
                if hasattr(env, 'current_pos'):
                    env.current_pos = state

                # Simulate action
                next_state, reward, done, _ = env.step(action)

                # Q(s,a) = R + γ max_a' Q(s', a')
                new_q = reward
                if not done:
                    new_q += gamma * Q.get_max_value(next_state)

                Q.set(state, action, new_q)

                delta = max(delta, abs(old_q - new_q))

        if delta < theta:
            print(f"Q-value iteration converged in {iteration + 1} iterations")
            break
    else:
        print(f"Q-value iteration stopped after {max_iterations} iterations")

    return Q


## Example Code

# from .environment import GridWorld
#
# # Create environment
# env = GridWorld(height=4, width=4, start=(0, 0), goal=(3, 3))
#
# # Value Iteration
# V, policy = value_iter(env, gamma=0.9)
# print("Optimal Value Function:")
# for state in env.get_states():
#     print(f"V({state}) = {V.get(state):.2f}")
#
# print("\nOptimal Policy:")
# for state in env.get_states():
#     try:
#         action = policy.get_action(state)
#         print(f"π({state}) = {action}")
#     except ValueError:
#         pass
#
# # Policy Iteration
# V, policy = policy_iter(env, gamma=0.9)
#
# # Q-Value Iteration
# Q = q_ter(env, gamma=0.9)
# best_action = Q.get_best_action((0, 0))