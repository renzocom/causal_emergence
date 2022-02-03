import numpy as np
import scipy as sp
import itertools

from pyemd import emd

def calc_measures_average_transition(tpm, states, which_averaging_dist, which_input_dist):
    '''
    Calculates measure averaged over all state transitions.

    Parameters
    ----------
    tpm : 2d-array
    states : list
    which_averaging_dist : 'maxent', 'stationary', 'perturb'
        distribution for averaging across all transitions
    which_input_dist : 'maxent', 'stationary', 'perturb'
        input distribution to calculate P(cause)

    Returns
    -------
    results : dict[measure]

    '''

    measures_xy = calc_measures_all_transitions(tpm, states, which_input_dist)

    results = {}
    for m in measures_xy.keys():
        results[m] = joint_expectation(measures_xy[m], tpm, which_averaging_dist)
    return results

def calc_measures_all_transitions(tpm, states, which_input_dist):
    '''
    Calculates causal measures for all transitions.

    Returns
    -------
    measures_xy : dict[measure][cause, effect]
    '''

    measure_labels = get_measure_labels()

    measures_xy = {}  # results[measure][scale][i, j]
    for label in measure_labels:
        measures_xy[label] = np.zeros(tpm.shape)

    transitions = list(itertools.product(states, states))

    for n, (cause, effect) in enumerate(transitions):
        measures = calc_measures(tpm, states, (cause, effect), which_input_dist)

        for measure in measure_labels:
            measures_xy[measure][states.index(cause), states.index(effect)] = measures[measure]

    return measures_xy


def prob_joint_c_e(tpm, which_averaging_dist):
    '''
    Returns the joint distribution of c and e given by

        p(e , c) = p(e | c) * p(c)

    where p(e|c) is the tpm and p(c) is either the uniform maxent distribution or the stationary distribution of p(e|c).
    '''

    if which_averaging_dist == 'stationary': # ATTENTION: for macro tpm this should be obtained by summing the microtpm,
                                             #            but for the bipartite model it doesn't matter
        prob_input = equilibrium_dist(tpm)
        if prob_input.ndim > 1:
            print(f"Warning: Multiple solutions for equilibrium dist: {prob_input.shape[1]} stationary distributions found. Choosing first one.")
            prob_input = prob_input[:, 0]

    elif which_averaging_dist == 'maxent':
        n_states = tpm.shape[0]
        prob_input = np.ones(n_states) / n_states

    else:
        raise ValueError("which_averaging_dist must be one of 'stationary' or 'maxent'.")

    return prob_input[:, np.newaxis] * tpm

def joint_expectation(f_xy, tpm, which_averaging_dist):
    '''
    Computes:

        \sum_{x,y} p(x,y)*f(x,y)

    Parameters
    ----------
    f_xy : 2d-array (n_states, n_states)
    tpm : 2d-array (n_states, n_states)
    which_input_dist : 'maxent' or 'stationary'

    '''

    return np.nansum(prob_joint_c_e(tpm, which_averaging_dist) * f_xy)

def state_str2int(state):
    '''
    Converts string of ints into int.

    Examples
    --------
    >>> state_str2int('0010')
    0010
    '''
    return [int(s) for s in state]

def calc_bit_flip_transition(tpm, transition, states):
    '''
    Bit-flip measure for specific transition.

    Parameters
    ----------
    tpm : 2d-array
    transition : 2-list
    states : list
    '''
    cause, effect = transition

    perturb_states = [flip_bit_in_state(cause, n) for n in range(len(cause))]
    perturb_ixs = [states.index(state) for state in perturb_states]

    dists = [sp.spatial.distance.hamming(state_str2int(effect), state_str2int(state)) for state in states]
    perturb_diffs = []
    for perturb_ix in perturb_ixs:
        probs = tpm[perturb_ix, :]
        diff = np.inner(probs, dists)
        perturb_diffs.append(diff)

    return np.mean(perturb_diffs)

def calc_bit_flip_cause(tpm, cause, states):
    '''
    Extension of Sarah Walker bit-flip measure for non-deterministic systems.

    Parameters
    ----------
    tpm : 2d-array
    transition : 2-list
    states : list
    '''
    n_states = len(states)

    cause_ix = states.index(cause)

    perturb_causes = [flip_bit_in_state(cause, n) for n in range(len(cause))]
    perturb_ixs = [states.index(state) for state in perturb_causes]

    states_mat = [[int(s) for s in state] for state in states]
    hamming_dists = sp.spatial.distance.cdist(states_mat, states_mat, 'hamming')

    metrics = []
    for perturb_ix in perturb_ixs:
        prob_effects = np.repeat(tpm[cause_ix][np.newaxis, :], n_states, axis=0)
        prob_effects_perturb = np.repeat(tpm[perturb_ix][:, np.newaxis], n_states, axis=1)

        joint = prob_effects * prob_effects_perturb

        out = joint * hamming_dists

        metric = np.sum(out)
        metrics.append(metric)

    return np.mean(metrics)

def calc_bit_flip_cause_emd(tpm, cause, states):
    '''
    Extension of Sarah Walker bit-flip measure for non-deterministic systems using EMD.

    Parameters
    ----------
    tpm : 2d-array
    transition : 2-list
    states : list
    '''

    cause_ix = states.index(cause)

    perturb_causes = [flip_bit_in_state(cause, n) for n in range(len(cause))]
    perturb_ixs = [states.index(state) for state in perturb_causes]

    states_mat = [[int(s) for s in state] for state in states]
    hamming_dists = sp.spatial.distance.cdist(states_mat, states_mat, 'hamming')

    effect_dist = tpm[cause_ix]

    metrics = []

    for perturb_ix in perturb_ixs:
        perturb_dist = tpm[perturb_ix]
        metric = emd(effect_dist.astype(float), perturb_dist.astype(float), hamming_dists.astype(float))
        metrics.append(metric)

    return np.mean(metrics)

def calc_perturb_power(tpm, transition, states):
    cause, effect = transition
    cause_ix, effect_ix = states.index(cause), states.index(effect)

    perturb_states = [flip_bit_in_state(cause, n) for n in range(len(cause))]
    perturb_ixs = [states.index(state) for state in perturb_states]

    prob_e_c = tpm[cause_ix, effect_ix]
    probs_e_perturb_c = [tpm[perturb_ix, effect_ix] for perturb_ix in perturb_ixs]
    perturb_powers = [prob_e_c - prob_e_perturb_c for prob_e_perturb_c in probs_e_perturb_c]
    return np.mean(perturb_powers)


def flip_bit_in_state(state, i):
    '''
    Flips a bit of a binary string state.

    Example
    -------
    >>> flip_bit_in_state('0010', 2)
    '0000'
    '''
    assert type(state) == str, "State must be a string."
    assert np.all([x in ['0', '1'] for x in set(state)]), "State must be a binary string."

    state_list = [s for s in state]
    state_list[i] = '0' if state_list[i] == '1' else '1'
    return ''.join(state_list)

def calc_galton(prob_c, prob_e_c, prob_e_not_c):
    return prob_c * (1 - prob_c) * (prob_e_c - prob_e_not_c)

def calc_eells(prob_e_c, prob_e_not_c):
    return prob_e_c - prob_e_not_c

def calc_suppes(prob_e_c, prob_e):
    return prob_e_c - prob_e

def calc_lewis(prob_e_c, prob_e_not_c):
    return (prob_e_c - prob_e_not_c) / prob_e_c

def calc_cheng(prob_e_c, prob_e_not_c):
    return (prob_e_c - prob_e_not_c) / (1 - prob_e_not_c)

def calc_effect_ratio(prob_e_c, prob_e):
    return np.log2(prob_e_c / prob_e)

# def calc_determinism(prob_e_c, n):
#     return (1 - np.log2(1 / prob_e_c) / np.log2(n))
#
# def calc_degeneracy(prob_e, n):
#     return (1 - np.log2(1 / prob_e) / np.log2(n))

def calc_determinism(prob_e_c, n_states, normalize):
    det = np.log2(n_states) - np.log2(1 / prob_e_c)
    if normalize:
        det = det / np.log2(n_states)
    return det

def calc_degeneracy(prob_e, n_states, normalize):
    deg = np.log2(n_states) - np.log2(1 / prob_e)
    if normalize:
        deg = deg / np.log2(n_states)
    return deg

def calc_effectiveness(prob_e_c, prob_e, n_states, normalize=True):
    return calc_determinism(prob_e_c, n_states, normalize=normalize) - calc_degeneracy(prob_e, n_states, normalize=normalize)

def calc_system_det_deg_EI(tpm, normalize):
    '''
    Calculates (normalized) determinism and degeneracy from a transition probability matrix.

    Parameters
    ----------
    tpm : 2d-array (n_states, n_states)
    normalize : bool

    Returns
    -------
    det, deg, EI

    '''

    n_states = tpm.shape[0]
    det = np.log2(n_states) - np.mean([sp.stats.entropy(dist, base=2) for dist in tpm])
    deg = np.log2(n_states) - sp.stats.entropy(np.mean(tpm, axis=0), base=2)

    if normalize:
        det = det / np.log2(n_states)
        deg = deg / np.log2(n_states)

    EI = det - deg

    return det, deg, EI

def calc_measures(tpm, states, transition, which_input_dist='maxent'):
    '''
    Calculates measures of causation.

    Parameters
    ----------
    tpm : 2d-array

    states : list

    transition : 2-list
        e.g. ['000', '100']

    which_input_dis : 'maxent', 'stationary' or 'perturb'

    Returns
    -------
    results : dict

    '''

    cause, effect = transition
    cause_ix = states.index(cause)
    effect_ix = states.index(effect)

    # NOTE: cause purview is taken to be the full state space. No need to restrict P(states)
    # to a P(causes) by normalize input distribution to cause purview (as was initially done).
    # cause_purview_ixs = [states.index(s) for s in cause_purview]
    # prob_cause_purview = prob_input[cause_purview_ixs] / np.sum(prob_input[cause_purview_ixs])

    cause_purview = states
    n_states = len(states)

    if which_input_dist =='stationary':
        prob_input = equilibrium_dist(tpm)
        if prob_input.ndim > 1:
            # TODO add average on stationary distributions?
            print(f"Warning: degenerate stationary distribution with {prob_input.shape[1]} solutions.")
            prob_input = prob_input[:, 0]

    elif which_input_dist =='maxent':
        prob_input = np.ones(n_states) / n_states

    elif which_input_dist =='perturb':
        n_bits = len(cause)
        perturb_causes = [flip_bit_in_state(cause, n) for n in range(n_bits)] + [cause] # check
        prob_input = np.array([state in perturb_causes for state in states])
        prob_input = prob_input / np.sum(prob_input)

    else:
        raise ValueError("which_input_dist must be one of 'stationary', 'perturb' or 'maxent'.")

    # P(c)
    prob_c = prob_input[cause_ix]

    # P(~c)
    prob_not_c = 1 - prob_input[cause_ix]

    # P(e | c)
    prob_e_c = tpm[cause_ix, effect_ix]

    # P(e | {c})
    prob_e = np.inner(prob_input, tpm[:, effect_ix])

    # P(e | {not-c}) considers all other cause states but c, uses the input distribution on not-c states and renormalizes
    not_cause_purview = list(set(cause_purview) - set([cause]))
    if not_cause_purview != []:
        not_cause_purview_ixs = sorted([states.index(state) for state in not_cause_purview])

        prob_not_cause_purview = prob_input[not_cause_purview_ixs] / np.sum(prob_input[not_cause_purview_ixs])
        prob_e_not_c = np.inner(prob_not_cause_purview, tpm[not_cause_purview_ixs, effect_ix])
        prob_not_e_not_c = 1 - prob_e_not_c
    else:
        print('Empty {not-c} purview.')
        prob_not_cause_purview = []
        prob_e_not_c = 0
        prob_not_e_not_c = 1

    ## P(e | not_c) using Lewis closest posssible world (CPW) based on Hamming distance
    if not_cause_purview == []:
        not_c_lewis = []
        prob_e_not_c_lewis = 0
    else:
        # hamming distance between the cause and all other states (counterfactuals)
        dists = [sp.spatial.distance.hamming(state_str2int(cause), state_str2int(state)) for state in not_cause_purview]

        # closest states (degenerate)
        lewis_ixs = np.where(dists == np.min(dists))[0]
        not_c_lewis = [not_cause_purview[ix] for ix in lewis_ixs] # closest states
        not_c_lewis_ixs = [states.index(s) for s in not_c_lewis]
        prob_e_not_c_lewis = [tpm[ix, effect_ix] for ix in not_c_lewis_ixs] # p(e|c) of closest states

        # average across all closest states
        prob_e_not_c_lewis = np.mean(prob_e_not_c_lewis)

        # within the closest states selecting that which the highest probability
        # ix = np.argmax(prob_e_not_c_lewis)
        # not_c_lewis_ix = not_c_lewis_ixs[ix]
        # not_c_lewis = states[not_c_lewis_ix]
        # prob_e_not_c_lewis = np.max(prob_e_not_c_lewis)



    # Perturbational measures
    perturb_power = calc_perturb_power(tpm, transition, states)
    bit_flip_transition = calc_bit_flip_transition(tpm, transition, states)
    bit_flip_cause = calc_bit_flip_cause(tpm, cause, states)
    bit_flip_cause_emd = calc_bit_flip_cause_emd(tpm, cause, states)

    # Causation measures
    galton = calc_galton(prob_c, prob_e_c, prob_e_not_c)
    eells = calc_eells(prob_e_c, prob_e_not_c)
    suppes = calc_suppes(prob_e_c, prob_e)
    cheng = calc_cheng(prob_e_c, prob_e_not_c)
    lewis_II = calc_lewis(prob_e_c, prob_e_not_c)

    lewis_ratio = prob_e_c / prob_e_not_c
    lewis_II_cpw = (prob_e_c - prob_e_not_c_lewis) / prob_e_c # rescaled lewis with closes possible world
    lewis_ratio_cpw = prob_e_c / prob_e_not_c_lewis # original lewis with closest possible world
    suppes_ratio = prob_e_c / prob_e

    effect_ratio = calc_effect_ratio(prob_e_c, prob_e)

    point_det = calc_determinism(prob_e_c, n_states, normalize=False)
    point_deg = calc_degeneracy(prob_e, n_states, normalize=False)
    point_det_coef = calc_determinism(prob_e_c, n_states, normalize=True)
    point_deg_coef = calc_degeneracy(prob_e, n_states, normalize=True)
    point_eff = calc_effectiveness(prob_e_c, prob_e, n_states, normalize=True)

    suf = prob_e_c
    nec = prob_not_e_not_c

    return {'states' : states,
            'transition' : transition,
            'c' : cause, 'e' : effect,
            'which_input_dist' : which_input_dist,
            'cause_purview': states,
            'not_cause_purview' : not_cause_purview,
            'prob_e_c' : prob_e_c,
            'prob_e' : prob_e,
            'prob_e_not_c' : prob_e_not_c,
            'prob_e_not_c_lewis' : prob_e_not_c_lewis,
            'prob_not_e_not_c' : prob_not_e_not_c,
            'prob_input' : prob_input,
            'prob_not_cause_purview' : prob_not_cause_purview,
            'galton' : galton,
            'eells' : eells,
            'suppes' : suppes,
            'cheng' : cheng,
            'lewis_II' : lewis_II,
            'lewis_ratio' : lewis_ratio,
            'lewis_II_cpw' : lewis_II_cpw,
            'lewis_ratio_cpw' : lewis_ratio_cpw,
            'prob_e_not_c_lewis' : prob_e_not_c_lewis,
            'not_c_lewis' : not_c_lewis,
            'effect_ratio' : effect_ratio,
            'suppes_ratio' : suppes_ratio,
            'point_det' : point_det,
            'point_deg' : point_deg,
            'point_det_coef' : point_det_coef,
            'point_deg_coef' : point_deg_coef,
            'point_eff' : point_eff,
            'sufficiency' : suf,
            'necessity' : nec,
            'bit_flip_transition' : bit_flip_transition,
            'bit_flip_cause' : bit_flip_cause,
            'bit_flip_cause_emd' : bit_flip_cause_emd,
            'perturb_power' : perturb_power}


def get_measure_labels(groups=['primitives', 'prob', 'lewis', 'info', 'perturb']):
    '''
    Returns list with measures.
    '''
    primitives = ['sufficiency', 'necessity', 'point_det_coef', 'point_deg_coef']
    prob_terms = ['prob_e']
    prob = ['galton', 'eells', 'suppes', 'lewis_II', 'cheng']
    lewis = ['lewis_ratio', 'lewis_II_cpw', 'lewis_ratio_cpw']
    info = ['effect_ratio', 'point_det', 'point_deg']
    perturb = ['bit_flip_transition', 'bit_flip_cause', 'bit_flip_cause_emd', 'perturb_power']

    EI = ['det_norm', 'deg_norm', 'det', 'deg', 'EI', 'eff']

    measure_map = {'primitives':primitives, 'prob_terms':prob_terms, 'prob':prob, 'lewis':lewis,
                   'info':info, 'perturb':perturb,  'EI': EI}

    measures = []
    for group in groups:
        measures += measure_map[group]
    return measures

def equilibrium_dist(M, combine_dist=False):
    """
    function from @author: thosvarley

    Calculates the equilibrium distribution of a given Markov chain TPM
    as the normalized eigenvector of the largest left-eigenvalue of the TPM.

    Since all TPM rows sum to 1, the largest eigenvalue should always be 1 and have real-part = 0.

    This assumes that there is only 1 valid steady-state for the Markov Chain (i.e. only one eigenvalue == 1)
    If you have a system whith more than 1 eigenvalue == 1, then the system is non-ergodic.
    We deal with this by normalizing all valid eigenvectors and then averaging them together.
    This assumes that all initial conditions are equiprobable and captures the distributions of states over all initial conditions.
    """
    w, vl, vr = sp.linalg.eig(M, left=True)

    if 1 not in np.real(w):
        where = np.argmax(np.real(w))
        pi = np.real(np.squeeze(vl[:, where] / np.sum(vl[:, where])))
        pi = np.divide(pi, np.sum(pi))

    else:
        where = np.array(np.isclose(np.real(w), 1.0).nonzero())[0]

        if where.shape[0] == 1:
            pi = np.real(np.squeeze(vl[:, where] / np.sum(vl[:, where])))
            pi = np.divide(pi, np.sum(pi))
        else:
            if combine_dist == True:
                valid_vectors = vl[:, where]
                for i in range(where.shape[0]):
                    valid_vectors[:, i] = valid_vectors[:, i] / np.sum(valid_vectors[:, i])

                pi = np.mean(valid_vectors, axis=1)
            else:
                valid_vectors = vl[:, where]
                for i in range(where.shape[0]):
                    valid_vectors[:, i] = valid_vectors[:, i] / np.sum(valid_vectors[:, i])

                pi = valid_vectors

    return pi