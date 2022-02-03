import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import models
import compute

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
        print(f"{'CS_Effect_Info':<20} = {micro_out['effect_ratio']:.3f} => {macro_out['effect_ratio']:.2f}")
        print()
        print(f"{'Lewis_CPW state':<20} = {micro_out['not_c_lewis']} => {macro_out['not_c_lewis']}")
        print(f"{'P(e | not-c CPW)':<20} = {micro_out['prob_e_not_c_lewis']:.3f} => {macro_out['prob_e_not_c_lewis']:.2f}")
        print(f"{'CS_Lewis_CPW':<20} = {micro_out['lewis_II_cpw']:.3f} => {macro_out['lewis_II_cpw']:.2f}")
        print()
        print(f"{'CS_Bit-flip':<20} = {micro_out['bit_flip_cause']:.3f} => {macro_out['bit_flip_cause']:.2f}")

x = ['states', 'transition', 'c', 'e', 'which_input_dist', 'cause_purview', 'prob_e_c', 'prob_e',\
'prob_e_not_c', 'prob_e_not_c_lewis', 'prob_not_e_not_c', 'prob_input', \
'prob_not_cause_purview', 'galton', 'eells', 'suppes', 'cheng', 'lewis_II', 'lewis_ratio',\
'lewis_II_cpw', 'lewis_ratio_cpw', 'effect_ratio', 'suppes_ratio', 'point_det', 'point_deg',\
'point_det_coef', 'point_deg_coef', 'point_eff', 'sufficiency', 'necessity', 'bit_flip_transition',\
'bit_flip_cause', 'bit_flip_cause_emd', 'perturb_power']

def run_analysis(n_states,
                single_or_average_transition,
                which_input_dist,
                micro_transition=None,
                macro_transition=None,
                which_averaging_dist=None,
                asymmetry=0,
                state_labels='default',
                print_info=False):
    '''
    Single transition for all degeneracy and determinism.

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

            micro_tpm, micro_states, _ = models.create_micro_tpm(model_det, model_deg, n_states,
                                                                             asymmetry=asymmetry,
                                                                             state_labels=state_labels)
            macro_tpm, macro_states = models.create_macro_tpm()

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

    # add causal emergence level
    for m in measures:
        results[m]['emergence'] = results[m]['macro'] - results[m]['micro']
        results[m]['abs_emergence'] = np.abs(results[m]['macro']) - np.abs(results[m]['micro'])

    return results


def plot_analysis_by_scales(results,
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

            if vmax > 15:
                vmax = 15
            if vmin < -15:
                vmin = -15

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

def plot_analysis_by_result(all_results,
                            which_results=None,
                            which_measures=None,
                            scale='emergence',
                            transpose=False,
                            separate_colorbar=False,
                            no_titles=False,
                            no_labels=False,
                            colormap='RdBu_r',
                            colsize=3,
                            rowsize=3):
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
        if vmin < -5: vmin = -5
        if vmax > 5: vmax = 5

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
                ax = axs[m, n] if axs.ndim == 2 else axs[n]
            else:
                ax = axs[n, m] if axs.ndim == 2 else axs[n]


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

            #             ax.set_xticklabels()

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
                ticks = cbar.get_ticks()
                cbar.set_ticks([ticks[0], 0, ticks[-1]])

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return fig, axs