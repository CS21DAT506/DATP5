from settings.settings_access import settings

def get_info_str(iteration_num, status):
    sep = f"{settings.info_str_separator} "
    info_str = f"Iteration: {iteration_num}{sep}Succeeded: {status['run_succeeded']}"
    if (status['run_succeeded']):
        info_str += f"{sep}{status['time_spent']}"
    else:
        info_str += f"{sep}Error message: '{status['error_message']}'"
    info_str += f"{sep}Total runs: {settings.num_of_iterations}"

    return info_str
