from collections import defaultdict
from typing import Any, List, Tuple

from .environment import Environment
from .policy import Policy, QFunc, VFunc


def generate_episode(env: Environment, policy: Policy,
                     max_steps: int = 1000) -> List[Tuple[Any, Any, float]]:
    """
    Generate episode following policy

    Parameters:
    -----------
    env : Environment
        Environment to interact with
    policy : Policy
        Policy to follow
    max_steps : int
        Maximum steps per episode

    Returns:
    --------
    episode : List[Tuple[state, action, reward]]
        List of (state, action, reward) tuples
    """
    episode = []
    state = env.reset()

    for _ in range(max_steps):
        # Get action from policy (or random if policy doesn't cover this state)
        try:
            action = policy.get_action(state)
        except ValueError:
            # Random action if policy not defined
            actions = env.get_actions(state)
            if not actions:
                break
            import random
            action = random.choice(actions)

        # Take action
        next_state, reward, done, _ = env.step(action)

        # Store transition
        episode.append((state, action, reward))

        state = next_state

        if done:
            break

    return episode


def mc_prediction(env: Environment,
                           policy: Policy,
                           num_episodes: int = 1000,
                           gamma: float = 0.9,
                           first_visit: bool = True) -> VFunc:
    """
    Monte Carlo Prediction (Policy Evaluation)

    Estimates V^π(s) by averaging returns following policy π.

    Parameters:
    -----------
    env : Environment
        Environment
    policy : Policy
        Policy to evaluate
    num_episodes : int
        Number of episodes to generate
    gamma : float
        Discount factor
    first_visit : bool
        If True, use first-visit MC. If False, use every-visit MC.

    Returns:
    --------
    V : VFunc
        Estimated value function
    """
    V = VFunc(default_value=0.0)
    returns = defaultdict(list)  # state -> list of returns

    for _ in range(num_episodes):
        # Generate episode
        episode = generate_episode(env, policy)

        # Compute returns for each state
        G = 0.0  # Return
        visited_states = set()

        # Iterate backwards through episode
        for t in range(len(episode) - 1, -1, -1):
            state, _, reward = episode[t]

            # Update return
            G = reward + gamma * G

            # First-visit or every-visit
            if first_visit:
                if state not in visited_states:
                    visited_states.add(state)
                    returns[state].append(G)
            else:
                returns[state].append(G)

    # Average returns for each state
    for state, return_list in returns.items():
        V.set(state, sum(return_list) / len(return_list))

    return V


def mc_es(env: Environment,
                   num_episodes: int = 5000,
                   gamma: float = 0.9) -> Tuple[QFunc, Policy]:
    """
    Monte Carlo Exploring Starts (ES)

    Finds optimal policy using exploring starts assumption.
    Each episode starts from a random state-action pair.

    Parameters:
    -----------
    env : Environment
        Environment
    num_episodes : int
        Number of episodes
    gamma : float
        Discount factor

    Returns:
    --------
    Q : QFunc
        Optimal Q-function
    policy : Policy
        Optimal policy
    """
    import random

    Q = QFunc(default_value=0.0)
    returns = defaultdict(list)  # (state, action) -> list of returns
    policy = Policy(deterministic=True)

    # Initialize policy and Q arbitrarily
    states = env.get_states()
    for state in states:
        actions = env.get_actions(state)
        if actions:
            policy.set_action(state, random.choice(actions))
            for action in actions:
                Q.set(state, action, 0.0)

    for _ in range(num_episodes):
        # Exploring start: random initial state and action
        start_state = random.choice(states)
        actions = env.get_actions(start_state)
        if not actions:
            continue

        start_action = random.choice(actions)

        # Generate episode
        episode = []

        # First step: exploring start
        env.current_pos = start_state
        next_state, reward, done, _ = env.step(start_action)
        episode.append((start_state, start_action, reward))

        # Continue episode following policy
        state = next_state
        while not done and len(episode) < 1000:
            try:
                action = policy.get_action(state)
            except ValueError:
                actions = env.get_actions(state)
                if not actions:
                    break
                action = random.choice(actions)

            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        # Update Q and policy
        G = 0.0
        visited_pairs = set()

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = reward + gamma * G

            # First-visit MC
            if (state, action) not in visited_pairs:
                visited_pairs.add((state, action))
                returns[(state, action)].append(G)

                # Update Q(s, a) as average of returns
                Q.set(state, action, sum(returns[(state, action)]) / len(returns[(state, action)]))

                # Update policy to be greedy w.r.t. Q
                policy.make_greedy(state, Q.get_all(state))

    return Q, policy


def mc_epsilon_greedy(env: Environment,
                               num_episodes: int = 5000,
                               gamma: float = 0.9,
                               epsilon: float = 0.1,
                               epsilon_decay: float = 0.9999) -> Tuple[QFunc, Policy]:
    """
    Monte Carlo Control with Epsilon-Greedy Policy

    On-policy Monte Carlo control using epsilon-greedy exploration.

    Parameters:
    -----------
    env : Environment
        Environment
    num_episodes : int
        Number of episodes
    gamma : float
        Discount factor
    epsilon : float
        Initial exploration rate
    epsilon_decay : float
        Epsilon decay rate per episode

    Returns:
    --------
    Q : QFunc
        Learned Q-function
    policy : Policy
        Learned epsilon-greedy policy
    """

    Q = QFunc(default_value=0.0)
    returns = defaultdict(list)
    policy = Policy(deterministic=False)

    # Initialize Q and policy
    states = env.get_states()
    for state in states:
        actions = env.get_actions(state)
        if actions:
            for action in actions:
                Q.set(state, action, 0.0)

            # Initialize epsilon-greedy policy
            uniform_prob = {a: 1.0 / len(actions) for a in actions}
            policy.set_probabilities(state, uniform_prob)

    current_epsilon = epsilon

    for _ in range(num_episodes):
        # Generate episode using current policy
        episode = generate_episode(env, policy)

        if not episode:
            continue

        # Update Q using returns
        G = 0.0
        visited_pairs = set()

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = reward + gamma * G

            if (state, action) not in visited_pairs:
                visited_pairs.add((state, action))
                returns[(state, action)].append(G)

                # Update Q
                Q.set(state, action, sum(returns[(state, action)]) / len(returns[(state, action)]))

                # Update policy to be epsilon-greedy w.r.t. Q
                q_values = Q.get_all(state)
                if q_values:
                    policy.make_epsilon_greedy(state, q_values, current_epsilon)

        # Decay epsilon
        current_epsilon *= epsilon_decay
        current_epsilon = max(current_epsilon, 0.01)  # Minimum epsilon

    return Q, policy


def mc_off_policy(env: Environment,
                           target_policy: Policy,
                           num_episodes: int = 5000,
                           gamma: float = 0.9) -> QFunc:
    """
    Off-Policy Monte Carlo Control with Importance Sampling

    Learns optimal Q-function while following a different behavior policy.

    Parameters:
    -----------
    env : Environment
        Environment
    target_policy : Policy
        Target policy to learn about (typically greedy)
    num_episodes : int
        Number of episodes
    gamma : float
        Discount factor

    Returns:
    --------
    Q : QFunc
        Learned Q-function for target policy
    """

    Q = QFunc(default_value=0.0)
    C = defaultdict(float)  # Cumulative sum of importance sampling weights

    # Behavior policy: epsilon-greedy or uniform random
    behavior_policy = Policy(deterministic=False)
    states = env.get_states()

    for state in states:
        actions = env.get_actions(state)
        if actions:
            # Uniform behavior policy for exploration
            uniform_prob = {a: 1.0 / len(actions) for a in actions}
            behavior_policy.set_probabilities(state, uniform_prob)

            for action in actions:
                Q.set(state, action, 0.0)

    for _ in range(num_episodes):
        # Generate episode using behavior policy
        episode = generate_episode(env, behavior_policy)

        if not episode:
            continue

        # Process episode
        G = 0.0  # Return
        W = 1.0  # Importance sampling weight

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = reward + gamma * G

            # Update cumulative weight
            C[(state, action)] += W

            # Update Q using weighted importance sampling
            old_q = Q.get(state, action)
            Q.set(state, action, old_q + (W / C[(state, action)]) * (G - old_q))

            # Update target policy to be greedy
            target_policy.make_greedy(state, Q.get_all(state))

            # Update importance sampling weight
            # W = W * π(a|s) / b(a|s)
            # Since behavior is uniform and target is deterministic greedy:
            try:
                target_action = target_policy.get_action(state)
                if action != target_action:
                    break  # Weight becomes 0, stop processing this episode
                # If deterministic greedy matches, π(a|s) = 1
                # W *= 1 / (1/num_actions) = num_actions
                W *= len(env.get_actions(state))
            except ValueError:
                break

    return Q


## Example Code

# from .environment import GridWorld
# import random
#
# # Create environment
# env = GridWorld(height=4, width=4, start=(0, 0), goal=(3, 3))
#
# # Create random policy
# policy = Policy(deterministic=False)
# for state in env.get_states():
#     actions = env.get_actions(state)
#     if actions:
#         uniform_probs = {a: 1.0 / len(actions) for a in actions}
#         policy.set_probabilities(state, uniform_probs)
#
# # Monte Carlo Prediction
# V = monte_carlo_prediction(env, policy, num_episodes=1000, gamma=0.9)
# print("Value Function (Monte Carlo):")
# print(f"V((0,0)) = {V.get((0, 0)):.2f}")
#
# # Monte Carlo ES (Control)
# Q, optimal_policy = monte_carlo_es(env, num_episodes=5000, gamma=0.9)
# print("\nOptimal Policy (MC ES):")
# try:
#     action = optimal_policy.get_action((0, 0))
#     print(f"π((0,0)) = {action}")
# except ValueError:
#     pass
#
# # Epsilon-Greedy Monte Carlo
# Q, policy = monte_carlo_epsilon_greedy(env, num_episodes=5000, epsilon=0.1)