# This consumes the results output by mufasa.py and plots them.
# There is a bit of processing of the sparse canonical attack matrix (the one which expresses the attacks in terms of
#   the respective basis vectors) which enables this.

import configargparse
from pathlib import Path
import pprint
from os.path import splitext
import numpy as np
import torch
import matplotlib.pyplot as plt
import json

# Using a reserved value to flag a failed query (exceeded the max number of iterations)
_MAX_QUERIES = 10000
_SUCCS = 0
_FAIL = 1


def stats_summary(succs, queries, canonical_adv, mufasa_input_args):
    summary = {}
    num_attacks = succs.numel()
    # write the query count to a list
    all_blows = [queries[i].item() for i in range(num_attacks)]
    # to compute the median, set the failed queries to local_max_queries+1
    all_blows_for_median = [all_blows[i] if succs[i] else float('inf') for i in range(num_attacks)]
    summary['queries_med'] = np.median(all_blows_for_median)
    # flag as failures queries already flagged as failed or that require more than MAX_QUERIES
    failure_count = [_SUCCS if(succs[i] and all_blows[i] <= _MAX_QUERIES) else _FAIL for i in range(num_attacks)]
    summary['failed_queries_percentage'] = failure_count.count(_FAIL) / num_attacks
    # compute the successful blows (under both succs[i] and <= _MAX_QUERIES conditions) and return to main script to
    # plot the CDF
    succs_blows = [all_blows[i] for i in range(num_attacks) if(succs[i] and all_blows[i] <= _MAX_QUERIES)]
    # compute norm stats for all of the attacks
    if mufasa_input_args.attack_mode == 'standard_simba':
        attacks_l2_norm = [mufasa_input_args.epsilon * torch.norm(canonical_adv[i], p=2).item()
                           for i in range(num_attacks)]
        summary['norm_l2_avg'] = np.mean(attacks_l2_norm)
        summary['norm_l2_min'] = np.min(attacks_l2_norm)
        summary['norm_l2_max'] = np.max(attacks_l2_norm)
        summary['norm_l2_med'] = np.median(attacks_l2_norm)
    else:  # Then none of those measures matter.
        pass

    pprint.pprint(summary)

    return succs_blows, summary


def plot_results():

    parser = configargparse.ArgumentParser(
        description="Method that reads in an output structure from mufasa.py, processes it, and draws plots.",
        add_help=False
    )

    required_named_arguments = parser.add_argument_group("required named arguments")
    required_named_arguments.add_argument(
        '--input_file', type=str, required=True,
        help="The name of the mufasa result file to be processed and plotted, including any required path info."
    )

    optional_arguments = parser.add_argument_group("optional arguments")

    # We switched help off in order to get our required arguments before our optional ones. Now we just add it back:
    optional_arguments.add_argument(
        '-h', '--help', action='help', default=configargparse.SUPPRESS,
        help="show this help message and exit"
    )
    optional_arguments.add_argument(
        '--config_file', is_config_file=True,
        help="Optional file from which to read parameter values. In the case of multiple specifications, the override "
             "order is (command line) > (environment vars) > (config file) > (defaults), as in the ConfigArgParse "
             "docs. See the docs for the valid config file format options."
    )
    optional_arguments.add_argument(
        '--display_plots', action='store_true',
        help="If specified, display the resulting PDF and CDF. It requires the user to close the window to continue "
             "running. (default: '%(default)s')"
    )
    optional_arguments.add_argument(
        '--bins', type=int, default=250,
        help="The number of bins in the histogram. (default: %(default)d)"
    )
    optional_arguments.add_argument(
        '--image_save_ext', type=str, default='.png',
        help="The file format in which you want your plot images saved. (default: '%(default)s')"
    )
    optional_arguments.add_argument(
        '--result_dir', type=str, default='results',
        help="Directory for saving results. (default: '%(default)s')"
    )

    args = parser.parse_args()

    input_file = Path(args.input_file)
    input_data = torch.load(input_file)

    output_dir_hist = Path(args.result_dir + '/histograms')
    output_dir_summ = Path(args.result_dir + '/summaries')
    output_dir_hist.mkdir(parents=True, exist_ok=True)
    output_dir_summ.mkdir(parents=True, exist_ok=True)

    canonical_adv = input_data['canonical_adv']
    succs = input_data['succs']
    queries = input_data['queries']
    mufasa_input_args = input_data['input_args']

    # Collate successful attacks and prints+creates summary
    succs_blows, summary = stats_summary(succs, queries, canonical_adv, mufasa_input_args)
    # Save summary as json
    with open(output_dir_summ / ('summary_' + splitext(input_file.name)[0] + '.json'), 'w') as summary_out:
        json.dump(summary, summary_out)

    # Empirical PDF
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n_hist, bins_hist, patches_hist = ax.hist(succs_blows, range=[0, _MAX_QUERIES], bins=args.bins,
                                              histtype='stepfilled')
    plt.xlabel("queries per image")
    plt.ylabel("samples fooled at each query")
    fig.savefig(output_dir_hist / ('histogram_' + splitext(input_file.name)[0] + args.image_save_ext))

    if args.display_plots:
        plt.show()
        plt.close()

    # Empirical CDF
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n_cumhist, bins_cumhist, patches_cumhist = ax.hist(succs_blows, range=[0, _MAX_QUERIES], bins=args.bins,
                                                       cumulative=True, histtype='stepfilled')
    plt.xlabel("queries per image")
    plt.ylabel("cumulative sum of samples fooled")
    fig.savefig(output_dir_hist / ('cum_hist_' + splitext(input_file.name)[0] + args.image_save_ext))

    if args.display_plots:
        plt.show()
        plt.close()

    # Save empirical distributions to file for later consumption
    hist_data = {'n_hist': n_hist, 'bins_hist': bins_hist, 'patches_hist': patches_hist}
    cum_hist_data = {'n_cumhist': n_cumhist, 'bins_cumhist': bins_cumhist, 'patches_cumhist': patches_cumhist}
    torch.save({'hist_data': hist_data, 'cum_hist_data': cum_hist_data},
               output_dir_hist / ('hist_data_' + splitext(input_file.name)[0] + '.pt'))


if __name__ == "__main__":
    plot_results()
