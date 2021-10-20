import numpy as np

from settings.SettingsAccess import settings

def calculate_run_performance(archives, target_pos):
    min_dist = np.inf
    for time_step in archives:
        agent = time_step.particles[settings.agent_index]
        dist_to_target = np.sqrt((agent.x - target_pos[0])**2 + (agent.y - target_pos[1])**2)
        if (dist_to_target < min_dist):
            min_dist = dist_to_target

    return min_dist