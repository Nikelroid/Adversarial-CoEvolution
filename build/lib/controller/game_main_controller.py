# In your other class:
from src.gin_rummy_api import GinRummyEnvAPI
from agents.random_agent import RandomAgent

if __name__=='__main__':
    # Advanced usage with custom actions
    env = GinRummyEnvAPI(render_mode="ansi")
    agent1 = RandomAgent(env)
    env.reset(seed=42)
    
    step_count = 0
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.get_current_state()
        print(agent, obs['observation'][0],info)
        if term or trunc:
            action = None
        else:
            action = agent1.do_action(agent)
        
        env.step(action)
        print(agent, action)
        input('Press Enter to continue...')
        
        step_count += 1
        if step_count >= 30:  # Stop after 10 steps for demo
            break
    
    env.close()
    print(f"Completed {step_count} steps")