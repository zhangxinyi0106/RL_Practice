import gym
import numpy as np
from ddpg import DDPGAgent
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    agent = DDPGAgent(actor_lr=0.001, critic_lr=0.001, input_dims=[3], tau=0.01,
                      gamma=0.99, mem_size=50000, n_actions=1, batch_size=64)

    train_score_history = []
    avg_train_score_history = []
    test_score_history = []

    for i in range(1000):
        obs = env.reset()
        done = False
        train_score = 0

        while not done:
            act = agent.act(obs)
            new_state, reward, done, _ = env.step(act)
            agent.record(obs, act, reward, new_state, done)
            agent.learn()
            train_score += reward
            obs = new_state

        train_score_history.append(train_score)
        avg_train_score_history.append(np.mean(train_score_history[-100:]))
        print('episode %s score %d last 100 games avg reward %.2f' %
              (i, train_score, float(avg_train_score_history[-1])))

        # testing
        if i % 10 == 0:
            test_sore_list = []
            for j in range(3):
                obs = env.reset()
                done = False
                test_score = 0
                while not done:
                    act = agent.act(obs, test=True)
                    new_state, reward, done, _ = env.step(act)
                    test_score += reward
                    obs = new_state
                test_sore_list.append(test_score)
            test_score_history.append(sum(test_sore_list) / len(test_sore_list))
        # testing end

    plt.plot(test_score_history)
    plt.show()
    #
    # plt.plot(train_score_history)
    # plt.show()
