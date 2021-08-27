# random utility functions that are useful within ist

import torch
from collections import defaultdict

def get_demon_momentum(curr_iter, total_iter, init_mom, delay_ratio):
    """given the current and total number of iterations, this method
    will use the DEMON momentum decay policy to calculate the current
    momentum value within the decay schedule

    Parameters
    curr_iter: the index of the current iteration
    total_iter: the total number of iterations to occur during training
    init_mom: the initial momentum value during training
    delay_ratio: the ratio of iterations at which the decay schedule should begin
    """

    base_mom = 0.9 # used to scale the momentum value properly

    # calculate the iteration at which decay will begin
    first_iter = int(delay_ratio*total_iter)
    final_iter = total_iter - first_iter

    # calculate current momentum value in the schedule
    if curr_iter > first_iter:
        curr_iter = curr_iter - first_iter
        z = (final_iter - curr_iter) / (final_iter)
        next_mom = (base_mom) * (z / ((1 - base_mom) + (base_mom * z)))
        if init_mom != base_mom:
            next_mom = next_mom * (init_mom / base_mom)
        return next_mom
    else:
        return init_mom

def aggregate_resnet_optimizer_statistics(opt, site_opts):
    """opt is expected to be a new optimizer that is being passed in without
    any state (i.e., it has been freshly initialized)"""

    new_states = defaultdict(list)
    for site_opt in site_opts:
        # loop through the actual optimizer and other optimizer concurrently
        # and store the buffers available for param groups on different sites
        for new_pg, site_pg in zip(opt.param_groups, site_opt.param_groups):
            for new_p, site_p in zip(new_pg['params'], site_pg['params']):
                # at this point, the new optimizer keys all momentum buffers with new_p
                # while the state of the old optimizer can be accessed with the site_p
                if site_p.grad is not None:
                    site_state = site_opt.state[site_p]
                    if 'momentum_buffer' in site_state:
                        curr_buff = site_state['momentum_buffer']
                        new_states[new_p].append(curr_buff)

    # update momentum buffers on the global optimizer with the site optimizer statistics
    for p, buff_list in new_states.items():
        assert len(buff_list)
        if len(buff_list) > 1:
            all_buff = torch.cat([x[None, :] for x in buff_list], dim=0)
            avg_buff = torch.sum(all_buff, dim=0) / len(buff_list)
        else:
            avg_buff = buff_list[0]
        opt.state[p] = {'momentum_buffer': avg_buff}
