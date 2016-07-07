from environment import Environment
from simulator import Simulator
import pandas as pd

def run_sim(n_trials, agent):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(agent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    return sim.run(n_trials=n_trials)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

def run_sims(n_sims, n_trials, agent):
    """
    Run n_sims with n_trials each with the agent.
    Returns the results as a dataframe.
    """
    results = []
    for i in range(n_sims):
        sim_results = run_sim(n_trials, agent)
        results.append(sim_results)
    df_results = pd.DataFrame(results)
    df_results.columns = ['n_dest_reached', 'last_dest_fail', 'sum_time_left', 
                          'n_penalties', 'last_penalty', 'len_qvals']
    return df_results