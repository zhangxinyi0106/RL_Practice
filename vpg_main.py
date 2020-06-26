import gym
import numpy as np
from vpg import VPGAgent
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    agent = VPGAgent(policy_lr=0.01, value_lr=0.01, input_dims=[3],
                     mem_size=2000, n_actions=1, batch_size=10)

    train_score_history = []
    avg_train_score_history = []

    for i in range(5000):
        for j in range(10):
            obs = env.reset()
            done = False
            train_score = 0
            while not done:
                act = agent.act(obs)
                new_state, reward, done, _ = env.step(act)
                agent.record(obs, act, reward, new_state, done)
                train_score += reward
                obs = new_state
            train_score_history.append(train_score)

        # learn every 10 trajectory
        agent.learn()
        agent.clear_store()

        avg_train_score_history.append(np.mean(train_score_history[-100:]))
        print('episode %s score %d last 100 games avg reward %.2f' %
              (i, train_score, float(avg_train_score_history[-1])))

    # since there is no explore,test is the same as train
    plt.plot(avg_train_score_history)
    plt.show()
