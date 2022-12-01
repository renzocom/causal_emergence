import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import models
import compute
from pathlib import Path
import pickle

def run_and_plot_analysis_01(n_nodes,
                             threshold,
                             det,
                             macro_method,
                             measures,
                             all_input_dist,
                             avg_input_combs,
                             bar_width=0.6,
                             bar_plot_width=2.8,
                             bar_colors=['#b8def5', '#ff7575', '#72BC74'],
                             save_fig=False,
                             save_dir='',
                             no_labels=False):
    '''
    Calculates causation measures for state transitions (all and average)
    in the negative majority model at micro and macro scale.

    Parameters
    ----------
    n_nodes : int
    threshold : int
    det : 0 < float < 1
    macro_method : 'maxent' or 'stationary'

    Returns
    -------
    all_transition_results : dict[scale[measure]] == 2d-array
    avg_transition_results : dict[input_average[scale[measure]]] == float
    model : dict

    '''
    print('Creating model...')
    model = models.create_neg_majority_model(n_nodes, threshold, det, macro_method)

    print('Running analysis...')

    all_transition_results, avg_transition_results = run_analysis_01(model, all_input_dist, avg_input_combs)

    print('Plotting results...')
    plot_all_transition_analysis_01(all_transition_results, measures, model, no_labels=no_labels)
    if save_fig:
        plt.savefig(Path(save_dir) / f'analysis_01_all_transition_{all_input_dist}_neg-maj-system_nodes_{n_nodes}_thr_{threshold}_det_{det:.2f}.eps')
        plt.savefig(Path(save_dir) / f'analysis_01_all_transition_{all_input_dist}_neg-maj-system_nodes_{n_nodes}_thr_{threshold}_det_{det:.2f}.png',dpi=200)

    plot_avg_transition_analysis_01(avg_transition_results, measures, bar_width=bar_width, plot_width=bar_plot_width,
                                             colors=bar_colors, no_labels=no_labels)
    if save_fig:
        plt.savefig(Path(
            save_dir) / f'analysis_01_avg_transition_{all_input_dist}_neg-maj-system_nodes_{n_nodes}_thr_{threshold}_det_{det:.2f}.eps')
        plt.savefig(Path(
            save_dir) / f'analysis_01_avg_transition_{all_input_dist}_neg-maj-system_nodes_{n_nodes}_thr_{threshold}_det_{det:.2f}.png',
                    dpi=200)
    print('Done.')


def run_analysis_01(model, all_input_dist, avg_input_combs):
    '''
    Calculates causation measures for state transitions (all and average)
    in the negative majority model at micro and macro scale.

    Parameters
    ----------
    model : output of models.create_neg_majority_model()

    Returns
    -------
    all_transition_results : dict[scale[measure]] == 2d-array
    avg_transition_results : dict[input_average[scale[measure]]] == float

    '''
    n_states = model['n_states']
    micro_tpm, macro_tpm = model['micro_tpm'], model['macro_tpm']
    states, macrostates, macro2micro_ixs = model['microstates'], model['macrostates'], model['macro2micro_ixs']

    # ALL TRANSITIONS
    micro_all_transition = compute.calc_measures_all_transitions(micro_tpm, states, which_input_dist=all_input_dist)
    macro_all_transition = compute.calc_measures_all_transitions(macro_tpm, ['0', '1'], which_input_dist=all_input_dist)
    emergence_all_transition = {}

    for measure in micro_all_transition.keys():
        y_micro = micro_all_transition[measure]
        y_macro = macro_all_transition[measure]
        y_macro_expanded = expand_macro_array(n_states, y_macro, macro2micro_ixs, macrostates)
        y_emergence = y_macro_expanded - y_micro
        emergence_all_transition[measure] = y_emergence

    all_transition_results = {'micro': micro_all_transition, 'macro': macro_all_transition, 'emergence':emergence_all_transition}

    # AVERAGE TRANSITION
    avg_transition_results = {}
    for input_dist, averaging_dist in avg_input_combs:
        micro_avg_transition = compute.calc_measures_average_transition(micro_tpm, states, averaging_dist, input_dist)
        macro_avg_transition = compute.calc_measures_average_transition(macro_tpm, macrostates, averaging_dist, input_dist)

        emergence_avg_transition = {measure : macro_avg_transition[measure] - micro_avg_transition[measure]
                                                                for measure in macro_avg_transition.keys()}
        avg_transition_results['_'.join([input_dist, averaging_dist])] = {'micro': micro_avg_transition,
                                                                          'macro': macro_avg_transition,
                                                                          'emergence': emergence_avg_transition}


    return all_transition_results, avg_transition_results

def plot_all_transition_analysis_01(all_transition_results, measures, model, no_labels=False):
    '''
    Plots results on a tpm-like array for each state transition.

    Parameters
    ----------
    all_transition_results : output of run_analysis_01()

    '''
    microstates, macrostates, macro2micro_ixs = model['microstates'], model['macrostates'], model['macro2micro_ixs']

    ncols = 3
    fig, axs = plt.subplots(figsize=(9, len(measures) * ncols), nrows=len(measures), ncols=ncols)

    for i, measure in enumerate(measures):

        y_micro = all_transition_results['micro'][measure]
        y_macro = all_transition_results['macro'][measure]
        y_emergence = all_transition_results['emergence'][measure]

        vmin = np.min([np.min(y_micro), np.min(y_macro)])
        vmax = np.max([np.max(y_micro), np.max(y_macro)])

        if vmin > -1e-08: vmin = -1
        if vmax < 1e-08: vmax = 1
        if vmin < -15: vmin = -15
        if vmax > 15: vmax = 15

        labels = ['micro', 'macro', 'emergence']

        colormap = 'RdBu_r'
        divnorm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)

        for j, y in enumerate([y_micro, y_macro, y_emergence]):

            ax = axs[i, j]
            states = [microstates, macrostates, microstates][j]

            if j == 2:
                im = ax.imshow(y, norm=divnorm, cmap='PRGn')
            else:
                im = ax.imshow(y, norm=divnorm, cmap=colormap)

            if j in [1, 2]:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[vmin, 0, vmax])
                cbar.ax.set_yticklabels([f"{vmin:.1f}", "0", f"{vmax:.1f}"])

            if not no_labels:
                if j == 1:
                    ax.set_title(f"{measure}\n{labels[j]}")
                else:
                    ax.set_title(labels[j])

            ax.set_xticks(range(len(states)), states)
            ax.set_yticks(range(len(states)), states)

    plt.tight_layout()

def plot_avg_transition_analysis_01(avg_transition_results, measures, bar_width=0.5, plot_width=3, no_labels=False, colors=None):
    input_average_combs = avg_transition_results.keys()

    if colors is None:
        colors = ['tab:blue', 'tab:red', 'tab:green']

    ncols = len(input_average_combs)
    fig, axs = plt.subplots(figsize=(ncols * plot_width, 3 * len(measures)), nrows=len(measures), ncols=ncols)

    for i, measure in enumerate(measures):

        ylim_max = 1.1 * max([avg_transition_results[run][scale][measure] for run in avg_transition_results.keys()
                                                                        for scale in ['micro', 'macro', 'emergence']])
        ylim_min = 1.1 * min([avg_transition_results[run][scale][measure] for run in avg_transition_results.keys()
                              for scale in ['micro', 'macro', 'emergence']])
        print(measure, ylim_max, ylim_min)
        if ylim_min > 0 : ylim_min = 0


        for j, input_average in enumerate(input_average_combs):

            ax = axs[i, j]
            result = avg_transition_results[input_average]

            for k, scale in enumerate(result.keys()):
                ax.bar([k], [result[scale][measure]], label=scale, color=colors[k], width=bar_width)

            ax.set_ylim(ylim_min, ylim_max)

            if not no_labels:
                ax.set_title(input_average)
                ax.set_xticks([0, 1, 2], ['micro', 'macro', 'emergence'])
            else:
                ax.set_xticks([0, 1, 2], [])

            if j == 0:
                ax.set_ylabel(measure)

            if i == 0 and j == 3:
                ax.legend()
    plt.tight_layout()
    plt.suptitle("average transition - local + stationary", y=1.02)

    # plt.savefig(FIGURES_DIR/f'measures-averages_neg-maj-system_nodes_{n_nodes}_thr_{threshold}_det_{det:.2f}.eps')
    # plt.savefig(FIGURES_DIR/f'measures-averages_neg-maj-system_nodes_{n_nodes}_thr_{threshold}_det_{det:.2f}.png', dpi=200)

    # plt.savefig("majority_network_measures_averages.eps")


def run_all_analysis_02(n_states=16,
                        main_micro_transition=['0000', '1111'],
                        nonmain_micro_transition=['0000', '1110'],
                        main_macro_transition=['0', '1'],
                        nonmain_macro_transition=['0', '1'],
                        state_labels='mirror',
                        save_dir='',
                        fname='causal_emergence_analysis_02_results'):
    '''
    Runs full causal emergence on bipartite model for different intervention distributions.

    micro_transition_main = ['0000', '1111']
    micro_transition_nonmain = ['0000', '1100']
    micro_transition_nonmain2 = ['0010', '1111']
    micro_transition_nonmain3 = ['0000', '1110'] (*)
    '''

    all_results = {}
    # SINGLE TRANSITIONS
    print('SYMMETRIC MODEL - SINGLE TRANSITION ANALYSIS')
    for input_dist in ['maxent', 'perturb', 'stationary']:
        for transition_type, micro_transition, macro_transition in zip(['main', 'nonmain'],
                                                                       [main_micro_transition, nonmain_micro_transition],
                                                                       [main_macro_transition, nonmain_macro_transition]):
            print(input_dist, micro_transition)
            results = run_single_analysis_02(n_states, 'single', which_input_dist=input_dist,
                                               micro_transition=micro_transition, macro_transition=macro_transition,
                                               state_labels=state_labels, asymmetry=0)
            all_results[input_dist + '_single_' + transition_type] = results

    # AVERAGE OVER TRANSITIONS
    print('SYMMETRIC MODEL - AVERAGE OVER TRANSITIONS ANALYSIS')

    for input_dist in ['maxent', 'stationary', 'perturb']:
        for averaging_dist in ['maxent', 'stationary']:
            print(input_dist, averaging_dist)
            results = run_single_analysis_02(n_states, 'average', which_input_dist=input_dist,
                                               which_averaging_dist=averaging_dist,
                                               state_labels=state_labels, asymmetry=0)

            all_results[input_dist + '_' + averaging_dist + '_average'] = results

    # ASYMMETRIC MODEL (AVERAGE OVER TRANSITIONS)
    print('ASSYMETRIC MODEL - AVERAGE OVER TRANSITIONS ANALYSIS')
    asymmetry = 0.7
    for input_dist, averaging_dist in zip(['maxent', 'stationary'], ['maxent', 'stationary']):
        print(input_dist, averaging_dist)
        results = run_single_analysis_02(n_states, 'average', which_input_dist=input_dist,
                                           which_averaging_dist=averaging_dist,
                                           state_labels=state_labels, asymmetry=asymmetry)

        all_results[f"{input_dist}_{averaging_dist}_average_asy_{asymmetry:.1f}"] = results

    print("Done. Saving results...")
    fpath = Path(save_dir) / (fname + '.pkl')
    with open(fpath, "wb") as f:
        pickle.dump(all_results, f)
    print(f"Results saved at {fpath}")

def run_single_analysis_02(n_states,
                            single_or_average_transition,
                            which_input_dist,
                            micro_transition=None,
                            macro_transition=None,
                            which_averaging_dist=None,
                            asymmetry=0,
                            state_labels='default',
                            print_info=False,
                            inf_ceil=10):
    '''
    Calculates causation measures for state transitions (all or single)
    in the bipartite model for all degeneracy and determinism model
    parameters at micro and macro scale.

    Parameters
    ----------
    n_states : int

    single_or_average_transition : str, 'single' or 'all'
        whether calculate measures for a single transition (specified by 'micro_transition' and 'macro_transition')
        or the average across all transitions (specified by 'which_averaging_dist')

    which_input_dist : str, 'maxent', 'stationary', 'perturb
        Interventional distribution P(causes) to calculate contrafactuals.

    micro_transition : 2-list (e.g. ['0000', '1000'])

    macro_transition : 2-list (e.g. ['0', '1'])

    which_averaging_dist : str, 'maxent', 'stationary', 'perturb
        Distribution to average across all transitions.

    asymmetry : float (-1 < x < 1)
        Assymetry in between the macro groups in the bipartite model

    state_labels : 'default' or 'mirror'

    print_info : bool

    Returns
    -------
    results : dict[measure][level][det_i, deg_j]
        measure : str ('eells', 'cheng', etc)
        level : str ('micro', 'macro', 'emergence', 'abs_emergence')
    '''

    model_degs = np.linspace(0.999, 0.001, 10)  # decreases across rows; degeneracy is between 0 and 1
    model_dets = np.linspace(0.001, 0.999, 10)  # increases across columns

    measures = compute.get_measure_labels()

    results = {}  # create result data structure: results[stat][scale][i, j]
    for m in measures + ['det', 'deg', 'EI', 'det_norm', 'deg_norm', 'EI_norm']:
        results[m] = {'micro': np.zeros((model_degs.shape[0], model_dets.shape[0])),
                         'macro': np.zeros((model_degs.shape[0], model_dets.shape[0]))}

    for i, model_deg in enumerate(model_degs):
        for j, model_det in enumerate(model_dets):
            if print_info:
                print(f"deg={model_deg:.2f}, det={model_det:.2f}")

            micro_tpm, micro_states, _ = models.create_bipartite_micro_tpm(model_det, model_deg, n_states,
                                                                             asymmetry=asymmetry,
                                                                             state_labels=state_labels)
            macro_tpm, macro_states = models.create_bipartite_macro_tpm()

            # SYSTEM LEVEL
            # compute degeneracy, determinism, EI and eff
            for tpm, level in zip([micro_tpm, macro_tpm], ['micro', 'macro']):
                for bool, s in zip([False, True], ['', '_norm']):
                    det, deg, EI = compute.calc_system_det_deg_EI(tpm, normalize=bool)
                    results['det' + s][level][i, j] = det
                    results['deg' + s][level][i, j] = deg
                    results['EI' + s][level][i, j] = EI

            # TRANSITION LEVEL
            # compute causal measures
            if single_or_average_transition == 'single':
                micro_out = compute.calc_measures(micro_tpm, micro_states,
                                                  micro_transition, which_input_dist)
                macro_out = compute.calc_measures(macro_tpm, macro_states,
                                                  macro_transition, which_input_dist)

            elif single_or_average_transition == 'average':
                micro_out = compute.calc_measures_average_transition(micro_tpm, micro_states,
                                                             which_averaging_dist, which_input_dist)
                macro_out = compute.calc_measures_average_transition(macro_tpm, macro_states,
                                                             which_averaging_dist, which_input_dist)
            else:
                raise ValueError("Either 'single' or 'average' transition.")

            for m in measures:
                results[m]['micro'][i, j] = micro_out[m]
                results[m]['macro'][i, j] = macro_out[m]

    for m in measures:
        if np.any(np.isposinf(results[m]['macro'])):
            inf_ceil = 10
            n_inf = np.sum(np.isposinf(results[m]['macro']))
            print(f"Substituting {n_inf} infinite values for {inf_ceil} (for plotting) in {m} measure.")
            results[m]['macro'][np.isposinf(results[m]['macro'])] = inf_ceil

        if np.any(np.isneginf(results[m]['macro'])):
            n_inf = np.sum(np.isneginf(results[m]['macro']))
            print(f"{n_inf} infinite values in {m} measure.")

    # add causal emergence level
    for m in measures:
        results[m]['emergence'] = results[m]['macro'] - results[m]['micro']
        results[m]['abs_emergence'] = np.abs(results[m]['macro']) - np.abs(results[m]['micro'])

    return results

def plot_analysis_02(all_results, figure_dir='', no_labels=True, save_fig=False):
    figure_dir = Path(figure_dir)
    no_titles = no_labels

    causal_primitives = ['sufficiency', 'necessity', 'point_det_coef', 'point_deg_coef']
    causal_measures = ['galton', 'suppes', 'eells', 'cheng', 'good', 'lewis_II_cpw', 'bit_flip_transition', 'effect_ratio']

    #############
    # FIGURE S2 #
    #############
    title = 'figS2_causal_primitives'
    scale = 'micro'
    which_measures = causal_primitives
    which_results = ['perturb_single_main', 'perturb_single_nonmain', 'perturb_stationary_average',
                     'maxent_maxent_average', 'stationary_stationary_average']

    plot_analysis_by_result_02(all_results, which_results=which_results, which_measures=which_measures,
                               scale=scale, no_titles=no_titles, no_labels=no_labels, colormap='RdBu_r')
    if not no_titles: plt.suptitle(title, y=1.02)
    if save_fig: plt.savefig(figure_dir / f"{title}.eps")

    ############
    # FIGURE 5 #
    ############
    title = 'fig5_causal_measures_RdBu_perturb'
    scale = 'micro'
    which_measures = causal_measures
    which_results = ['perturb_single_main', 'perturb_single_nonmain', 'perturb_stationary_average',
                     'maxent_maxent_average', 'stationary_stationary_average']

    plot_analysis_by_result_02(all_results, which_results=which_results, which_measures=which_measures,
                                scale=scale, no_titles=no_titles, no_labels=no_labels, colormap='RdBu_r')
    if not no_titles: plt.suptitle(title, y=1.02)
    if save_fig: plt.savefig(figure_dir / f"{title}.eps")

    title = 'fig5_good_RdBu_perturb'
    which_measures = ['good']
    plot_analysis_by_result_02(all_results, which_results=which_results, which_measures=which_measures,
                               scale=scale, no_titles=no_titles, no_labels=no_labels, colormap='RdBu_r',
                               colorbar_ceil=10, colsize=3.5, rowsize=3)
    if not no_titles: plt.suptitle(title, y=1.02)
    if save_fig: plt.savefig(figure_dir / f"{title}.eps")

    ############
    # FIGURE 6 #
    ############
    title = 'fig6_causal_emergence_PRGn_perturb'
    scale = 'emergence'
    which_measures = causal_primitives + causal_measures
    which_results = ['perturb_single_main', 'perturb_single_nonmain', 'perturb_stationary_average']

    plot_analysis_by_result_02(all_results, which_results=which_results, which_measures=which_measures,
                               scale=scale, no_titles=no_titles, no_labels=no_labels, colormap='PRGn')

    if not no_titles: plt.suptitle(title, y=1.02)
    if save_fig: plt.savefig(figure_dir / f"{title}.eps")

    title = 'fig6_good_PRGn_perturb'
    which_measures = ['good']

    plot_analysis_by_result_02(all_results, which_results=which_results, which_measures=which_measures,
                               scale=scale, no_titles=no_titles, no_labels=no_labels, colormap='PRGn',
                               colorbar_ceil=10, colsize=3.5, rowsize=3)
    if not no_titles: plt.suptitle(title, y=1.02)
    if save_fig: plt.savefig(figure_dir / f"{title}.eps")

    #############
    # FIGURE S3 #
    #############
    title = 'figS3_causal_emergence_PRGn_1'
    scale = 'emergence'
    which_measures = causal_primitives + causal_measures
    which_results = ['maxent_single_main', 'stationary_single_main', 'perturb_single_main', 'maxent_single_nonmain',
                     'stationary_single_nonmain', 'perturb_single_nonmain']

    plot_analysis_by_result_02(all_results, which_results=which_results, which_measures=which_measures,
                               scale=scale, no_titles=no_titles, no_labels=no_labels, colormap='PRGn')
    if not no_titles: plt.suptitle(title, y=1.02)
    if save_fig: plt.savefig(figure_dir / f"{title}.eps")

    title = 'figS3_good_PRGn_1'
    which_measures = ['good']
    plot_analysis_by_result_02(all_results, which_results=which_results, which_measures=which_measures,
                               scale=scale, no_titles=no_titles, no_labels=no_labels, colormap='PRGn',
                               colorbar_ceil=10, colsize=3.5, rowsize=3)
    if not no_titles: plt.suptitle(title, y=1.02)
    if save_fig: plt.savefig(figure_dir / f"{title}.eps")

    title = 'figS3_causal_emergence_PRGn_2'
    which_measures = causal_primitives + causal_measures
    which_results = ['maxent_maxent_average', 'stationary_stationary_average', 'perturb_stationary_average',
                     'maxent_maxent_average_asy_0.7']
    plot_analysis_by_result_02(all_results, which_results=which_results, which_measures=which_measures,
                               scale=scale, no_titles=no_titles, no_labels=no_labels, colormap='PRGn')
    if not no_titles: plt.suptitle(title, y=1.02)
    if save_fig: plt.savefig(figure_dir / f"{title}.eps")

    title = 'figS3_good_PRGn_2'
    which_measures = ['good']
    plot_analysis_by_result_02(all_results, which_results=which_results, which_measures=which_measures,
                               scale=scale, no_titles=no_titles, no_labels=no_labels, colormap='PRGn',
                               colorbar_ceil=10, colsize=3.5, rowsize=3)
    if not no_titles: plt.suptitle(title, y=1.02)
    if save_fig: plt.savefig(figure_dir / f"{title}.eps")

def plot_analysis_by_scales_02(results,
                            which_scales=['micro', 'macro', 'emergence'],
                            which_measures=None,
                            colormap='RdBu_r'):
    if which_measures is None:
        which_measures = compute.get_measure_labels()

    # titles = {'galton': 'Galton', 'eells': 'Eells (PNS)', 'lewis_II': 'Lewis (PN)', 'cheng': 'Cheng (PS)',
    #           'suppes': 'Suppes', 'effect_ratio': 'Effect Ratio', 'det': 'Determinism', 'deg': 'Degeneracy',
    #           'EI': 'EI', 'eff': 'Effectiveness', 'det_norm': 'Determinism Coef.', 'deg_norm': 'Degeneracy Coef.',
    #           'lewis_II_cpw': 'Lewis II (CPW)', 'lewis_ratio': 'Lewis Ratio', 'lewis_ratio_cpw': 'Lewis Ratio (CPW)',
    #           'point_deg': 'Degeneracy', 'point_det': 'Determinism', 'sufficiency': 'Sufficiency',
    #           'necessity': 'Necessity', 'suppes_ratio': 'Effect ratio (no log)',
    #           'point_det_coef': 'Point Determinism Coef', 'point_deg_coef': 'Point Degeneracy Coef',
    #           'perturb_diff': 'Perturbational Sensitivity (transition)',
    #           'perturb_diff2': 'Perturb. Sensitivity (original)',
    #           'perturb_diff3': 'Perturb. Sensitivity (EMD)',
    #           'perturb_power': 'Perturb. Eells Power'}

    nrows = len(which_scales)
    ncols = len(which_measures)
    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    for n, measure in enumerate(which_measures):
        for m, scale in enumerate(which_scales):
            ax = axs[m, n] if axs.ndim == 2 else axs[n]

            X = results[measure][scale]

            X_noinf_nonan = X[(~np.isinf(X)) & (~np.isnan(X))]
            if len(X_noinf_nonan) == 0:
                vmin = -1
                vmax = 1
            else:
                vmin = np.min(X_noinf_nonan) if np.min(X_noinf_nonan) < -1e-08 else -1
                vmax = np.max(X_noinf_nonan) if np.max(X_noinf_nonan) > 1e-08 else 1

            ceil = 15
            if vmax > ceil:
                vmax = ceil
            if vmin < -ceil:
                vmin = -ceil

            divnorm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)

            im = ax.imshow(X, extent=(0, 1, 0, 1), norm=divnorm, cmap=colormap)
            ax.set_xlabel('det')
            ax.set_ylabel('deg')

            if n != len(which_measures) // 2:
                ax.set_title(measure)
            else:
                ax.set_title(f"{scale.upper()}\n{measure}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return fig, axs

def plot_analysis_by_result_02(all_results,
                            which_results=None,
                            which_measures=None,
                            scale='emergence',
                            transpose=True,
                            separate_colorbar=False,
                            no_titles=False,
                            no_labels=False,
                            colormap='RdBu_r',
                            colsize=3.5,
                            rowsize=3,
                            colorbar_ceil=5):
    '''

    Parameters
    ----------
    all_results : dict
        R = dict[result]
        R : dict[measure][scale][det_i, deg_j]

    which_results
    which_measures
    scale
    transpose
    separate_colorbar
    no_titles

    Returns
    -------

    '''
    if which_measures is None:
        which_measures = compute.get_measure_labels()
    if which_results is None:
        which_results = all_results.keys()

    def get_min_max(measure):
        Xs = [all_results[result][measure][scale] for result in which_results]
        Xs = [X[(~np.isinf(X)) & (~np.isnan(X))] for X in Xs]
        Xs = [X if len(X) > 0 else [1, -1] for X in Xs]
        vmins = [np.min(X) for X in Xs]
        vmaxs = [np.max(X) for X in Xs]
        vmin = np.min(vmins) if np.min(vmins) < -1e-08 else -1
        vmax = np.max(vmaxs) if np.max(vmaxs) > 1e-08 else 1

        if vmin > -1e-08: vmin = -1
        if vmax < 1e-08: vmax = 1

        if vmin < -colorbar_ceil: vmin = -colorbar_ceil
        if vmax > colorbar_ceil: vmax = colorbar_ceil

        return vmin, vmax

    if not transpose:
        nrows = len(which_results)
        ncols = len(which_measures)
    else:
        nrows = len(which_measures)
        ncols = len(which_results)

    fig, axs = plt.subplots(nrows, ncols, figsize=(colsize * ncols, rowsize * nrows))

    for n, measure in enumerate(which_measures):
        vmin, vmax = get_min_max(measure)
        for m, result in enumerate(which_results):

            if not transpose:
                if nrows == 1 and ncols == 1:
                    ax = axs
                elif nrows == 1:
                    ax = axs[n]
                elif ncols == 1:
                    ax = axs[m]
                else:
                    ax = axs[m, n]
            else:
                if nrows==1 and ncols==1:
                    ax = axs
                elif nrows==1:
                    ax = axs[m]
                elif ncols==1:
                    ax = axs[n]
                else:
                    ax = axs[n, m]

            X = all_results[result][measure][scale]

            if separate_colorbar:
                X_noinf_nonan = X[(~np.isinf(X)) & (~np.isnan(X))]
                if len(X_noinf_nonan) == 0:
                    vmin = -1
                    vmax = 1
                else:
                    vmin = np.min(X_noinf_nonan) if np.min(X_noinf_nonan) < -1e-08 else -1
                    vmax = np.max(X_noinf_nonan) if np.max(X_noinf_nonan) > 1e-08 else 1

                if vmax > 5:
                    vmax = 5
                if vmin < -5:
                    vmin = -5

            divnorm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)

            im = ax.imshow(X, extent=(0, 1, 0, 1), norm=divnorm, cmap=colormap)

            # Labels
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            if not no_labels:
                ax.set_xlabel('det')
                ax.set_ylabel('deg')
            else:
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            if not no_titles:
                if not transpose:
                    if n == len(which_measures) // 2:
                        ax.set_title(f"{result.upper()}\n{measure}")
                    else:
                        ax.set_title(measure)
                else:
                    if m == len(which_results) // 2:
                        ax.set_title(f"{measure.upper()}\n{result}")
                    else:
                        ax.set_title(f"{result}")

            if (not transpose) or (transpose and m == len(which_results) - 1):
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                # ticks = cbar.get_ticks()
                # print([ticks[0], 0, ticks[-1]])
                cbar.set_ticks([vmin, 0, vmax], labels=[f"{vmin:.2f}", "0", f"{vmax:.2f}"])
                # cbar.set_ticks([vmin, ticks[1], 0, ticks[-2], vmax])

    plt.subplots_adjust(top=0.9, wspace=0.12, hspace=0.15)
    plt.tight_layout()
    return fig, axs

def run_and_plot_analysis_03(save_fig=False, save_dir=''):
    n = 2

    bins = 20

    xs = np.linspace(0.001, 0.999, bins)

    prob_e_c = xs
    prob_e_not_c = xs  # decreases up
    prob_e = xs  # decreases up

    necessity = xs[::-1]  # grows up
    sufficiency = xs  # grows right

    degeneracy = xs  # decreases up
    determinism = xs  # grows right

    mesh_e_c, mesh_e_not_c = np.meshgrid(prob_e_c, prob_e_not_c)
    mesh_e_c, mesh_e = np.meshgrid(prob_e_c, prob_e)

    mesh_det, mesh_deg = np.meshgrid(determinism, degeneracy)

    results = {}
    results['eells'] = compute.calc_eells(mesh_e_c, mesh_e_not_c)
    results['lewis'] = compute.calc_lewis(mesh_e_c, mesh_e_not_c)
    results['cheng'] = compute.calc_cheng(mesh_e_c, mesh_e_not_c)
    results['good'] = compute.calc_good(mesh_e_c, mesh_e_not_c)

    results['suppes'] = compute.calc_suppes(mesh_e_c, mesh_e)

    results['effect_ratio'] = compute.calc_effect_ratio(mesh_e_c, mesh_e)
    results['effect_ratio2'] = np.log2(n) * (mesh_det - mesh_deg)
    results['effectiveness'] = compute.calc_effectiveness(mesh_e_c, mesh_e, n)
    results['effectiveness2'] = mesh_det - mesh_deg

    measures = ['eells', 'suppes', 'cheng', 'lewis', 'good', 'effect_ratio', 'effect_ratio2']

    plt.figure(figsize=(len(measures) * 4.2, 4))

    for n, key in enumerate(measures):
        plt.subplot(1, len(measures), n + 1)

        im = plt.imshow(results[key], extent=(0, 1, 0, 1), vmin=-1, vmax=1)

        plt.plot([0, 1], [1, 0], color='red', linestyle='-')

        plt.xticks([0, 0.5, 1], [0, 0.5, 1])
        if n in [0, 1, 2, 3, 4, 5]:
            plt.xlabel('sufficiency');
            plt.ylabel("necessity")
            plt.yticks([0, 0.5, 1], [0, 0.5, 1])
        else:
            plt.xlabel('determinism');
            plt.ylabel("degeneracy")
            plt.yticks([0, 0.5, 1], [1, 0.5, 0])

        if n == len(measures) - 1:
            plt.colorbar(im, fraction=0.046, pad=0.04)

        plt.title(measures[n].capitalize(), pad=20)

    plt.tight_layout()

    if save_fig:
        fname = "fig1_measures_functional_behavior"
        plt.savefig(Path(save_dir) / (fname + ".eps"))
        plt.savefig(Path(save_dir) / (fname + ".png"), dpi=200)

def plot_analysis_04(all_results, which_result='perturb_stationary_average',  save_fig=False, save_dir='',):
    colormap = 'PRGn'
    scale = 'emergence'
    result = which_result

    print(result.upper())
    plt.figure(figsize=(20, 2))

    measures = ['galton', 'suppes', 'eells', 'cheng', 'good', 'effect_ratio', 'bit_flip_transition', 'lewis_II_cpw']

    Y = []
    for n, measure in enumerate(measures):
        A = all_results[result][measure][scale]
        print(f"{measure.upper()} [min, max]")
        print(f"[{np.min(A):.2f}, {np.max(A):.2f}] (before norm)")
        norm = np.max(np.abs(A))
        A = A / norm
        print(f"[{np.min(A):.2f}, {np.max(A):.2f}] (after norm)")

        ax = plt.subplot(1, 12, n + 1)

        divnorm = mpl.colors.TwoSlopeNorm(vmin=-1, vcenter=0., vmax=1)

        plt.imshow(A, extent=(0, 1, 0, 1), norm=divnorm, cmap=colormap)
        plt.title(measure)

        Y.append(A)

    plt.tight_layout()

    Y = np.array(Y)
    Y_mean = np.mean(Y, axis=0)

    print("super average".upper())
    print(f"[{np.min(Y_mean):.2f}, {np.max(Y_mean):.2f}] (before norm)")

    plt.figure()
    divnorm = mpl.colors.TwoSlopeNorm(vmin=-1, vcenter=0., vmax=1)
    im = plt.imshow(Y_mean, extent=(0, 1, 0, 1), norm=divnorm, cmap=colormap)

    plt.colorbar()
    if save_fig:
        plt.savefig(Path(save_dir) / 'fig7_causal_emergence_super_average_normalized.eps')


def print_measures_output(micro_out, macro_out):
    print('Parameters\n==========')
    print(f"Micro transition: ({micro_out['c']}) --> ({micro_out['e']})")
    print(f"Macro transition: ({macro_out['c']}) --> ({macro_out['e']})")
    print()
    print('Micro')
    print(f"Input distribution: '{micro_out['which_input_dist']}'")
    print('Macro')
    print(f"Input distribution: '{macro_out['which_input_dist']}'")

    print()

    print()
    print('Probabilities\n=============')
    prob_input_string = [f"{x:.2f}" for x in micro_out['prob_input']]
    print(f"{f'P(s)':<20} = {prob_input_string} => {macro_out['prob_input']}")
    print(f"{f'States(s)':<20} = {micro_out['cause_purview']} => {macro_out['cause_purview']}")
    print()
    prob_not_cause_purview_string = [f"{x:.2f}" for x in micro_out['prob_not_cause_purview']]
    print(f"{'P(not-c)':<20} = {prob_not_cause_purview_string} => {macro_out['prob_not_cause_purview']}")
    print(f"{f'Not-c States(s)':<20} = {micro_out['not_cause_purview']} => {macro_out['not_cause_purview']}")

    print()
    print(f"{f'P(e | c)':<20} = {micro_out['prob_e_c']:.4f} => {macro_out['prob_e_c']:.2f}")

    print(f"{f'P(e | {{c}})':<20} = {micro_out['prob_e']:.4f} => {macro_out['prob_e']:.2f}")
    print(f"{f'P(e | {{not-c}})':<20} = {micro_out['prob_e_not_c']:.4f} => {macro_out['prob_e_not_c']:.2f}")

    print()
    print(f"{'CS_Eells (PNS)':<20} = {micro_out['eells']:.3f} => {macro_out['eells']:.2f}")
    print(f"{'CS_Lewis (PN)':<20} = {micro_out['lewis_II']:.3f} => {macro_out['lewis_II']:.2f}")
    print(f"{'CS_Cheng (PS)':<20} = {micro_out['cheng']:.3f} => {macro_out['cheng']:.2f}")
    print(f"{'CS_Suppes':<20} = {micro_out['suppes']:.3f} => {macro_out['suppes']:.2f}")
    print(f"{'CS_Good':<20} = {micro_out['good']:.3f} => {macro_out['good']:.2f}")
    print(f"{'CS_Effect_Info':<20} = {micro_out['effect_ratio']:.3f} => {macro_out['effect_ratio']:.2f}")
    print()
    print(f"{'Lewis_CPW state':<20} = {micro_out['not_c_lewis']} => {macro_out['not_c_lewis']}")
    print(
        f"{'P(e | not-c CPW)':<20} = {micro_out['prob_e_not_c_lewis']:.3f} => {macro_out['prob_e_not_c_lewis']:.2f}")
    print(f"{'CS_Lewis_CPW':<20} = {micro_out['lewis_II_cpw']:.3f} => {macro_out['lewis_II_cpw']:.2f}")
    print()
    print(f"{'CS_Bit-flip':<20} = {micro_out['bit_flip_cause']:.3f} => {macro_out['bit_flip_cause']:.2f}")

def micro_macro_model_distance(micro_tpm, macro_tpm, A_ixs, B_ixs, initial_state='uniform'):
    '''
    Calculates and plots (cumulated) distance across time between micro and macro models.

    Parameters
    ----------
    micro_tpm
    macro_tpm
    '''
    def KL(a, b):
        return np.sum(a * np.log(a / b))

    n_nodes = int(np.sqrt(micro_tpm.shape[0]))

    if initial_state=='uniform':
        n_states = 2**n_nodes
        state0 = np.ones(n_states) / n_states
    else:
        state0 = np.random.randn(n_states) + 1
        state0 = state0 / np.sum(state0)

    macro_state0 = [np.sum(state0[A_ixs]), np.sum(state0[B_ixs])]

    N = 40
    micro_effect_distribution = state0[:]
    macro_effect_distribution = macro_state0[:]

    dists = []
    for i in range(N):
        micro_effect_distribution = np.dot(micro_effect_distribution, micro_tpm)
        macro_effect_distribution = np.dot(macro_effect_distribution, macro_tpm)

        micro_effect_distribution = micro_effect_distribution / np.sum(micro_effect_distribution)
        macro_effect_distribution = macro_effect_distribution / np.sum(macro_effect_distribution)

        coarse_micro_effect_distribution = [np.sum(micro_effect_distribution[A_ixs]), np.sum(micro_effect_distribution[B_ixs])]

        dist = KL(macro_effect_distribution, coarse_micro_effect_distribution)
        dists.append(dist)
        plt.plot(micro_effect_distribution)
    plt.title('Micro effect distribution (iterations)')

    plt.figure()
    plt.plot(dists, label='KL')
    plt.plot(np.cumsum(dists), label="Cumulated sum")
    plt.xlabel('Iteration')
    plt.legend()
    plt.title('Distance between micro and macro model')


def expand_macro_array(n_microstates, y_macro, macro2micro_ixs, macrostates=['0', '1']):
    '''
    Expand matrix of macro results (for each macro transition) to
    a matrix with the respective micro transitions.

    Parameters
    ----------
    n_microstates : number of microstates

    y_macro : 2d array with macro values (n_macrostates, n_macrostates)

    macro2micro_ixs : dict = {macrostate : micro_ixs} (e.g. {'0' : [0, 1, 2, 4], '1' : [3, 5, 6, 7]})

    macrostates : list of macrostates (e.g. ['0', '1'])
    '''
    y_macro_expanded = np.zeros((n_microstates, n_microstates))
    for s1 in macrostates:
        for s2 in macrostates:
            y_macro_expanded[np.ix_(macro2micro_ixs[s1], macro2micro_ixs[s2])] = y_macro[
                macrostates.index(s1), macrostates.index(s2)]

    return y_macro_expanded

if __name__ == '__main__':
    PROJECT_DIR = Path('/Users/atopos/Drive/science/causal_emergence/')
    FIGURES_DIR = PROJECT_DIR / 'figures'

    n_nodes = 2
    threshold = 1
    det = 0.9

    macro_method = 'maxent'

    all_input_dist = 'perturb'
    avg_input_combs = ('perturb', 'stationary'), ('maxent', 'maxent'), ('stationary', 'stationary')

    measures = ['sufficiency', 'necessity', 'point_det_coef', 'point_deg_coef'] + \
               ['galton', 'suppes', 'eells', 'cheng', 'good', 'lewis_II_cpw', 'bit_flip_transition', 'effect_ratio']

    bar_colors = ['#6cb0d4', '#ed594a', '#72BC74']

    run_and_plot_analysis_01(n_nodes,
                             threshold,
                             det,
                             macro_method,
                             measures,
                             all_input_dist,
                             avg_input_combs,
                             bar_colors=bar_colors,
                             save_fig=True,
                             save_dir=FIGURES_DIR,
                             no_labels=True)


