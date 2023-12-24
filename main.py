from dqn import DQN
import argparse
from collections import namedtuple
import gymnasium as gym 
import matplotlib.pylab as plt
import numpy as np
import random
import torch

def main(args):
    
    if args.render:
        env = gym.make("ALE/DonkeyKong-v5", render_mode="human", frameskip=10, obs_type="grayscale")
    else:
        env = gym.make("ALE/DonkeyKong-v5", frameskip=10, obs_type="grayscale")
        
    observation, info = env.reset()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    model = DQN(observation.shape, args.max_exp, env.action_space.n)
    model.to(device)
    Experience = namedtuple("Experience", ["state", "reward", "action", "next_state", "terminal"])
    
    # plt.imshow(observation)
    # plt.show()
    
    fig, ax = plt.subplots(2, 2)
    plt.tight_layout()
    
    end_episode_loss = []
    actions_taken = [0 for i in range(env.action_space.n)]
    returns = []
        
    for e in range(args.episodes):
        
        avg_return = 0
        in_epsisode_avg_returns = []
        in_epsisode_avg_Q = []
        in_episode_loss = []
        in_episode_steps = []
        i = 0
        
        terminated = truncated = False
        while not terminated or truncated:
            
            ''' 
            With probability eps select a random action at
            otherwise select at = maxa Q*(φ(st), a; θ)
            '''
            P = random.random()
            if P < 1 - args.eps:
                with torch.no_grad(): # guess we need this
                    actions = model(model.convert(observation)) # use the network to find the set of actions 
                    action = torch.argmax(actions)
            else:
                action = env.action_space.sample()  # agent policy that uses the observation and info
            
            # Execute action at in emulator and observe reward rt and image xt+1
            prev_observation = observation
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Set st+1 = st, at, xt+1 and preprocess φt+1 = φ(st+1)        
            # Store transition (φt, at, rt, φt+1) in D
            new_experience = Experience(prev_observation, reward, action, observation, terminated or truncated)
            model.experiences.append(new_experience)
            
            if terminated or truncated:
                continue
            
            # accure enough experiences first
            if len(model.experiences) < model.experiences.maxlen:
                continue
            
            ''' 
            Sample random minibatch of transitions (φj , aj , rj , φj+1) from D
            
            Set yj =    rj for terminal φj+1
                        rj + gamma * maxa0 Q(φj+1, a0; θ) for non-terminal φj+1
            '''
            yj, Q_sa, rewards = model.find_targets(args.batch_size, args.gamma)
            
            yj.to(device)
            Q_sa.to(device)
            rewards.to(device)
            
            output = model.loss(yj, Q_sa) 

            # save for plotting
            actions_taken[action] += 1
            in_epsisode_avg_returns.append(torch.mean(rewards).detach().numpy())
            in_epsisode_avg_Q.append(torch.mean(Q_sa).detach().numpy())
            
            i += 1
            if i % 10 == 0:
                in_episode_loss.append((output).detach().numpy())
                in_episode_steps.append(i)
                if i % 20 == 0:
                    print(f"Loss at Episode {e+1} Step {i}: {round(float(in_episode_loss[-1]), 4)}")
                ax[0, 0].plot(in_episode_steps, in_episode_loss)
                ax[0, 0].set_title("In Episode Loss")
                ax[0, 0].grid()
                ax[0, 1].plot(range(1, len(in_epsisode_avg_Q)+1), in_epsisode_avg_Q)
                ax[0, 1].set_title("Average Q")
                ax[0, 1].grid()
                ax[1, 1].plot(range(1, len(in_epsisode_avg_returns) + 1), in_epsisode_avg_returns)
                ax[1, 1].set_title("Returns")
                ax[1, 1].grid()
                fig.savefig(args.out)
            
            # backprop
            model.optimizer.zero_grad()
            output.backward()
            model.optimizer.step()

            
        # reset before next episode
        observation, info = env.reset()
        returns.append(sum(in_epsisode_avg_returns) / i)
        end_episode_loss.append(sum(in_episode_loss) / i)
        
        print("===========================")
        print(f"End of Episode {e+1}:") 
        print(f"Average Return | {round(returns[-1], 4)}") 
        print(f"Average Loss   | {round(end_episode_loss[-1], 4)}") 
        print("===========================")
        
        ax[1, 0].plot(range(1, len(end_episode_loss) + 1), end_episode_loss)
        ax[1, 0].set_title("End of Episode Loss")
        ax[1, 0].grid()
        fig.savefig(args.out)
    
    env.close()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--out", default="plot.png", type=str)
    parser.add_argument("--episodes", default=100, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_exp", default=256, type=int)
    parser.add_argument("--eps", default=0.15, type=float)
    parser.add_argument("--gamma", default=0.9, type=float)
    args = parser.parse_args()
    
    main(args)