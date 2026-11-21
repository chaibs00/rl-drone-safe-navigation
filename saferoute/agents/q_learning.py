"""Q-Learning: off-policy temporal-difference learning.

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

The target uses the BEST next action (max), whatever action the agent
actually takes next. That is what "off-policy" means: it learns the value
of the greedy policy while behaving with an epsilon-greedy one.

With a shield, the max is taken over the allowed (safe) actions only.
"""

from saferoute.agents.base_agent import BaseAgent


class QLearningAgent(BaseAgent):

    def update(self, state, action, reward, next_state, next_action, terminated):
        if terminated:
            target = reward                                          # no future after goal or crash
        else:
            best_next = self.Q[next_state, self.allowed_actions(next_state)].max()
            target = reward + self.gamma * best_next                 # best ALLOWED next action

        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error