import json

def get_archive_as_json_str(archive, agent, target_pos):
    inputs = [] # each data point corresponds to the data that exists in one timestep
    outputs = []

    for a in archive:
        data_point = [ target_pos[0],  target_pos[1] ]
        for p in a.particles:
            data_point = data_point + [ p.x, p.y, p.vx, p.vy, p.m ] 
        inputs.append( [data_point] )

        agent_acc = agent.get_thrust(a)
        outputs.append( [ float(agent_acc[0]), float(agent_acc[1]) ] )

    data_points = { 'input': inputs, 'output': outputs }

    return json.dumps( data_points, indent=4 )



