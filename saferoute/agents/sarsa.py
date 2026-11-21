"""SARSA: on-policy temporal-difference learning.

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * Q(s',a') - Q(s,a) ]

The target uses the action a' the agent ACTUALLY takes next (chosen with
epsilon-greedy, exploration included). That is what "on-policy" means:
it learns the value of the policy it is really following, random moves and all.

The name comes from the five values used in one update:
    S (state), A (action), R (reward), S' (next state), A' (next action).
"""

from saferoute.agents.base_agent import BaseAgent


class SarsaAgent(BaseAgent):

    def update(self, state, action, reward, next_state, next_action, terminated):
        if terminated:
            target = reward                                                  # no future after goal or crash
        else:
            # =========== This line is the only difference from Q-learning ==============
            #  it uses the action actually taken next, not the best one.
            target = reward + self.gamma * self.Q[next_state, next_action]   # the action really taken

        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error