"""
Main script to play Gin Rummy: PPO Agent vs Random Agent

Usage:
python main.py --player1 ppo --player2 random --model ./models/ppo_gin_rummy/ppo_gin_rummy_final
"""

from src.gin_rummy_api import GinRummyEnvAPI
from agents.random_agent import RandomAgent
from agents.ppo_agent import PPOAgent
import argparse
import random


def play_game(env, agents_dic, agent_names, max_steps=1000, interactive=False, verbose=True):
    """
    Play a single game between two agents.
    
    Args:
        env: GinRummyEnvAPI instance
        agents_dic: Dictionary mapping player names to agents
        agent_names: Dictionary mapping player names to agent names
        max_steps: Maximum steps per game
        interactive: Whether to wait for user input each step
        verbose: Whether to print game progress
        
    Returns:
        Dictionary with game statistics
    """
    env.reset(seed=None)  # Random seed for variety
    
    step_count = 0
    rewards = {'player_0': 0, 'player_1': 0}
    
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.get_current_state()
        
        # Update rewards
        rewards[agent] += reward
        
        if term or trunc:
            action = None
            if verbose:
                print(f"\n{'='*50}")
                print(f"Game ended!")
                print(f"Player 0 ({agent_names['player_0']}) total reward: {rewards['player_0']:.2f}")
                print(f"Player 1 ({agent_names['player_1']}) total reward: {rewards['player_1']:.2f}")
                
                if rewards['player_0'] > rewards['player_1']:
                    print(f"Winner: {agent_names['player_0']} (player_0)")
                elif rewards['player_1'] > rewards['player_0']:
                    print(f"Winner: {agent_names['player_1']} (player_1)")
                else:
                    print("Draw!")
                print(f"{'='*50}\n")
        else:
            action = agents_dic[agent].do_action()
            
            if verbose:
                print(f"Step {step_count}: {agent} takes action {action}")
        
        env.step(action)
        
        if interactive and not (term or trunc):
            input('Press Enter to continue...')
        
        step_count += 1
        if step_count >= max_steps:
            if verbose:
                print(f"\nMax steps ({max_steps}) reached. Ending game.")
            break
    
    # Determine winner
    winner = None
    if rewards['player_0'] > rewards['player_1']:
        winner = 'player_0'
    elif rewards['player_1'] > rewards['player_0']:
        winner = 'player_1'
    
    return {
        'steps': step_count,
        'rewards': rewards,
        'winner': winner,
        'winner_name': agent_names.get(winner, 'Draw')
    }


def main():
    parser = argparse.ArgumentParser(description='Play Gin Rummy with different agents')
    parser.add_argument('--player1', type=str, default='random', 
                       choices=['random', 'ppo'], help='Type of player 1')
    parser.add_argument('--player2', type=str, default='random',
                       choices=['random', 'ppo'], help='Type of player 2')
    parser.add_argument('--model', type=str, default='./models/ppo_gin_rummy/ppo_gin_rummy_final',
                       help='Path to PPO model (if using PPO agent)')
    parser.add_argument('--render', type=str, default='ansi', 
                       choices=['ansi', 'human'], help='Render mode')
    parser.add_argument('--interactive', action='store_true',
                       help='Wait for Enter key after each step')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per game')
    parser.add_argument('--no-randomize', action='store_true',
                       help='Do not randomize player positions (player1 always player_0)')
    
    args = parser.parse_args()
    
    # Create environment
    env = GinRummyEnvAPI(render_mode=args.render)
    
    # Create agents
    if args.player1 == 'random':
        agent1 = RandomAgent(env)
        agent1_name = 'Random Agent 1'
    else:  # ppo
        agent1 = PPOAgent(model_path=args.model, env=env)
        agent1_name = 'PPO Agent 1'
    
    if args.player2 == 'random':
        agent2 = RandomAgent(env)
        agent2_name = 'Random Agent 2'
    else:  # ppo
        agent2 = PPOAgent(model_path=args.model, env=env)
        agent2_name = 'PPO Agent 2'
    
    randomize_positions = not args.no_randomize
    
    # Randomly assign positions
    if randomize_positions and random.random() < 0.5:
        print(f"Positions: player_0={agent2_name}, player_1={agent1_name}\n")
        agent1.set_player('player_1')
        agent2.set_player('player_0')
        agents_dic = {'player_0': agent2, 'player_1': agent1}
        agent_names = {'player_0': agent2_name, 'player_1': agent1_name}
    else:
        print(f"Positions: player_0={agent1_name}, player_1={agent2_name}\n")
        agent1.set_player('player_0')
        agent2.set_player('player_1')
        agents_dic = {'player_0': agent1, 'player_1': agent2}
        agent_names = {'player_0': agent1_name, 'player_1': agent2_name}
    
    # Play single game
    result = play_game(env, agents_dic, agent_names,
                      max_steps=args.max_steps, 
                      interactive=args.interactive,
                      verbose=True)
    
    env.close()


if __name__ == '__main__':
    main()