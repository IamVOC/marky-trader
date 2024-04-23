def print_metrics(total_reward):
    sample_mean = sum(total_reward)/len(total_reward)
    sample_dispersion = sum([(x - sample_mean)**2 for x in total_reward])/(len(total_reward)-1)
    print(f'Point estimate J = {sample_mean}')
    print(f'Dispersion = {sample_dispersion}')

