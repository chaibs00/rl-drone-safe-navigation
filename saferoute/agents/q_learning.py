"""Q-Learning: off-policy temporal-difference learning.

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

The target uses the BEST next action (max), whatever action the agent
actually takes next. That is what "off-policy" means: it learns the value
of the greedy policy while behaving with an epsilon-greedy one.
"""

from saferoute.agents.base_agent import BaseAgent


class QLearningAgent(BaseAgent):

    def update(self, state, action, reward, next_state, next_action, terminated):
        if terminated:
            target = reward                                          # no future after goal or crash
        else:
            target = reward + self.gamma * self.Q[next_state].max()  # best next action

        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

        