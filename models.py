
import numpy as np
import networkx as nx
# import dit
from itertools import product
import matplotlib.pyplot as plt


import matplotlib as mpl
import plotly.graph_objects as go

import pyphi

import compute

def create_neg_majority_model(n_nodes, threshold, det, macro_method):
    '''

    Parameters
    ----------
    n_nodes : int
    threshold : int
    det : 0 < float < 1
    macro_method : 'maxent' or 'stationary'

    Returns
    -------
    model : dict
    '''

    micro_tpm, states, (A_ixs, B_ixs) = create_neg_majority_tpm(n_nodes, threshold, det)
    micro_tpm_sbn = pyphi.convert.state_by_state2state_by_node(micro_tpm)
    micro_tpm_sbn = pyphi.convert.to_2dimensional(micro_tpm_sbn)

    macro_tpm, macrostates = create_neg_majority_macro_tpm(micro_tpm, A_ixs, B_ixs, method=macro_method)

    A_states = [states[ix] for ix in A_ixs]
    B_states = [states[ix] for ix in B_ixs]
    micro2macro_states = {**{s: '0' for s in A_states}, **{s: '1' for s in B_states}}
    macro2micro_states = {'0': A_states, '1': B_states}
    macro2micro_ixs = {'0': A_ixs, '1': B_ixs}

    model = {'n_states': len(states),
             'n_nodes': n_nodes,
             'micro_tpm':micro_tpm,
             'micro_tpm_sbn':micro_tpm_sbn,
             'macro_tpm':macro_tpm,
             'microstates':states,
             'macrostates':macrostates,
             'A_ixs':A_ixs,
             'B_ixs':B_ixs,
             'macro2micro_ixs':macro2micro_ixs,
             'macro2micro_states':macro2micro_states,
             'micro2macro_states':micro2macro_states}
    return model

def create_neg_majority_tpm(n_nodes, threshold, det):
    '''
    Create all-to-all system where nodes are negative majorities gates.

    Parameters
    ----------
    n_nodes : int
    threshold : int, threshold for majority
    det : float between 0 and 1

    Returns
    -------
    tpm : 2d matrix (2**n_nodes, 2**n_nodes)
    states : list[str]
    (A_ixs, B_ixs) : indices of A and B groups

    '''

    states = list(pyphi.utils.all_states(n_nodes))
    states = ["".join([str(s) for s in state]) for state in states]
    A_ixs, B_ixs = [], []

    micro_tpm_sbn = np.zeros((2 ** n_nodes, n_nodes))
    for i, state in enumerate(states):
        count = np.sum([int(x) for x in state])
        if not count > threshold:
            #             micro_tpm_sbn[i] = [det] * n_nodes + np.random.randn(n_nodes)/10
            micro_tpm_sbn[i] = det
            A_ixs.append(i)
        else:
            micro_tpm_sbn[i] = 1 - det
            B_ixs.append(i)

    micro_tpm = pyphi.convert.state_by_node2state_by_state(micro_tpm_sbn)

    A_states = [states[ix] for ix in A_ixs]
    B_states = [states[ix] for ix in B_ixs]
    print("Microstates: ", states)
    print(f"Macrostates: OFF = {A_states}, ON = {B_states}")



    return micro_tpm, states, (A_ixs, B_ixs)

def create_neg_majority_macro_tpm(micro_tpm, A_ixs, B_ixs, method=None):
    '''
    Calculates macro TPM from micro TPM that groups microstates into two groups A and B.

    Parameters
    ----------
    micro_tpm : 2d-array
        state-by-state TPM
    A_ixs : list of indices
    B_ixs : list of indices
    method : 'stationary' or 'maxent'

    Returns
    -------
    macro_tpm : 2d-array
    states : list[str]
    '''
    macro_states = ['0', '1']

    if method == 'stationary':
        micro_stationary = compute.equilibrium_dist(micro_tpm)

        macro_tpm = np.zeros((2, 2))
        pi = micro_stationary[A_ixs] / np.sum(micro_stationary[A_ixs])
        pi = pi[:, np.newaxis]

        tmp_A = pi * micro_tpm[A_ixs, :]
        tmp_A = np.mean(tmp_A, axis=0)
        tmp_A = tmp_A / np.sum(tmp_A)

        pi = micro_stationary[B_ixs] / np.sum(micro_stationary[B_ixs])
        pi = pi[:, np.newaxis]

        tmp_B = pi * micro_tpm[B_ixs, :]
        tmp_B = np.mean(tmp_B, axis=0)
        tmp_B = tmp_B / np.sum(tmp_B)

    elif method == 'maxent':
        macro_tpm = np.zeros((2, 2))
        tmp_A = np.mean(micro_tpm[A_ixs, :], axis=0)
        tmp_B = np.mean(micro_tpm[B_ixs, :], axis=0)
    else:
        raise ValueError("Invalid 'method'.")

    macro_tpm[0, 0] = np.mean(tmp_A[A_ixs])
    macro_tpm[0, 1] = np.mean(tmp_A[B_ixs])
    macro_tpm[1, 0] = np.mean(tmp_B[A_ixs])
    macro_tpm[1, 1] = np.mean(tmp_B[B_ixs])

    macro_tpm = macro_tpm / np.sum(macro_tpm, axis=1)[:, np.newaxis]
    return macro_tpm, macro_states

def create_bipartite_micro_tpm(det, deg, n_states, asymmetry, state_labels='default'):
    '''
    Create tpm for bipartite model where states transition between to macro groups A and B.

    Parameters
    ----------
    det : float (0 < x < 1)
    deg : float (0 < x < 1)
    n_states : n_states = 2**n
    asymmetry : float (-1 < x < 1)
    state_labels : 'default' (binary ordering, e.g. ['00', '01'] and ['10', '11'])
                    or 'mirror' (states main causal link is paired with its flipped version,
                    e.g. ['00', '01'] and ['11', '10']).

    Returns
    -------
    tpm : np.array
    states : list
    (A_states, B_states) : states split in groups A and B
    '''

    AB = list(range(n_states))
    # divide (a)symmetrically states into A and B groups
    cut_AB = n_states // 2 + np.round((n_states // 2 - 1) * asymmetry).astype(int)
    A, B = AB[:cut_AB], AB[cut_AB:]
    n_A, n_B = len(A), len(B)

    # cm and degeneracy algorithm
    max_iter = min(n_A, n_B) - 1
    n_iter = np.round(deg * max_iter).astype(int)

    # A to B
    A2B, _ = degeneracy_algorithm(n_iter, A, B)

    # B to A
    B2A, _ = degeneracy_algorithm(n_iter, B, A)

    tpm = np.zeros((n_states, n_states))
    for a, b in product(A, B):
        p = (1 + det * (n_B - 1)) / n_B
        if A2B[a] == b:
            tpm[a, b] = p
        else:
            tpm[a, b] = (1 - p) / (n_B - 1)

    for b, a in product(B, A):
        p = (1 + det * (n_A - 1)) / n_A
        if B2A[b] == a:
            tpm[b, a] = p
        else:
            tpm[b, a] = (1 - p) / (n_A - 1)

    n_bits = int(np.ceil(np.log2(n_states)))
    bits = ["".join(x) for x in
            product(["0", "1"], repeat=n_bits)]  # ['000', '001', '010', '011', '100', '101', '110', '111']

    states = bits[:n_states]
    if state_labels == 'default':
        A_states, B_states = [states[a] for a in A], [states[b] for b in B]

    elif state_labels == 'mirror':
        def flip_state(state):
            '''
            Example
            -------
            >>> flip_state('010')
            '101'
            '''
            new_state = ['1' if s == '0' else '0' for s in state]
            return ''.join(new_state)

        if n_A == n_B: # symmetric
            A_states = [states[a] for a in A]
            B_states = [flip_state(state) for state in A_states]

        elif n_A < n_B:
            A_states = [states[a] for a in A] # e.g. [000, 001]
            B_states = [flip_state(state) for state in A_states] # incomplete e.g. [111, 110]
            B_states = B_states + list(set(states) - set(A_states + B_states)) # completes e.g. [111, 110, 010, 011, ...]

        else: # n_A > n_B
            A_states = [states[a] for a in A[:n_B]] # incomplete, only first n_B
            B_states = [flip_state(state) for state in A_states]
            A_states = A_states + list(set(states) - set(A_states + B_states))

        states = A_states + B_states
    else:
        raise ValueError("state_label must be 'default' or 'mirror'.")

    return tpm, states, (A_states, B_states)

def create_bipartite_macro_tpm():
    '''
    Returns the macro tpm and macro states of the bipartite model.
    '''
    macro_states = ['0', '1']
    macro_tpm = np.array([[0, 1], [1, 0]])

    return macro_tpm, macro_states

def get_A2B_map(cm, A, B):
    '''
    Gets main causal links mapping from states in A to B.

    Parameters
    ----------
    cm : 2d-array
        connectivity matrix
    A : list
    B : list

    Returns
    -------
    dict : map from states in A to B
    '''

    check_cm_consistency(cm)
    assert cm.shape == (len(A), len(B)), "cm is inconsistent with A and B."

    B_ixs = [list(row).index(1) for row in cm]
    A2B = {a: B[B_ixs[n]] for n, a in enumerate(A)}
    return A2B

def check_cm_consistency(cm):
    '''
    Checks if connectivity matrix of main causes to effects makes sense

    Parameters
    ----------
    cm : 2d-array
    '''
    assert np.all(np.sum(cm,
                         axis=1)), "CM is inconsistent (causes with multiple main effects)."  # check cause only lead to one main effect
    zeros = np.where(np.sum(cm, axis=0) == 0)[0]
    if len(zeros) > 0:
        assert zeros[-1] == cm.shape[
            1] - 1, "CM is inconsistent (sparse zeroed columns)."  # check if zeroed columns are all in the right side
        assert np.all(np.diff(
            zeros)), "CM is inconsistent (non-contiguous zeroed columns)."  # check if zeroed columns are contiguous


def get_initial_cm(n_A, n_B):
    '''
    Returns initial connectivity matrix between main causal links in bipartite model (A and B groups of states).
    '''
    if n_A <= n_B:
        cm = np.eye(n_A, n_B)

    else:  # n_A > n_B
        m = n_A // n_B
        r = n_A % n_B
        cm1 = np.tile(np.eye(n_B, n_B), (m, 1))
        cm2 = np.eye(r, n_B)
        cm = np.concatenate((cm1, cm2))

    return cm


def get_model_layout(A_states, B_states, column_gap=2):
    '''
    Returns
    -------
    pos :dict
        {state : (x,y)} with layout of two macro states model.
    '''

    pos = {}
    for i, state in enumerate(A_states):
        pos[state] = np.array([0, 3 - i])

    for i, state in enumerate(B_states):
        pos[state] = np.array([column_gap, 3 - i])
    return pos

def degeneracy_algorithm(n_iter, A, B):
    '''
    Algorithm for increasing degeneracy in the bipartite model (two groups of macro states, A and B).

    Parameters
    ----------
    n_iter : number of iterations
    A : list
        set of states in group A
    B : list
        set of states in group B

    Returns
    -------
    A2B : dict
        main effect map from states in A to states in B
    cm : 2d-array (n_A, nB)
        connectivity matrix
    '''
    n_A, n_B = len(A), len(B)
    cm = get_initial_cm(n_A, n_B)
    check_cm_consistency(cm)

    for n in range(n_iter):
        cm = degenerate(cm)
    A2B = get_A2B_map(cm, A, B)
    return A2B, cm


def degenerate(cm):
    '''
    Step increase in the algorithm for increasing degeneracy of the bipartite model

    cm : 2d-array
    '''
    # get source column
    all_in_degrees = np.sum(cm, axis=0)
    if np.min(all_in_degrees) != 0:  # surjective cm
        source_col = cm.shape[1] - 1  # source column is the last column
    else:
        source_col = np.argmin(
            all_in_degrees) - 1  # source column is the column before the first column with zero in-degree

    # get target column
    in_degrees = np.sum(cm[:, :source_col],
                        axis=0)  # number incoming causes to effects excluding the source col and beyond
    target_col = np.argmin(in_degrees)  # next poorest column

    # move causal links from source column to target column
    source_rows = np.where(cm[:, source_col] == 1)[0]
    cm[source_rows, source_col] = 0
    cm[source_rows, target_col] = 1

    return cm

##############
## PLOTTING ##
##############

def shorten_line(x_ini, y_ini, x_end, y_end, rho):
    '''
    Shortens a line from (x_ini, y_ini) to (x_end, y_end) by a factor of rho.

    Parameters
    ---------
    first point (x_ini, y_ini)
    second point (x_end, y_end)
    rho : shorting factor (float)
    '''
    x_center = (x_ini + x_end) / 2
    y_center = (y_ini + y_end) / 2
    d = np.sqrt((x_ini - x_end) ** 2 + (y_ini - y_end) ** 2)
    d_new = d / 2 * rho

    cos_a = (x_end - x_ini) / d
    sin_a = (y_end - y_ini) / d

    x_correction = d_new * cos_a
    y_correction = d_new * sin_a

    x_short_ini = x_center - x_correction
    y_short_ini = y_center - y_correction
    x_short_end = x_center + x_correction
    y_short_end = y_center + y_correction

    return x_short_ini, y_short_ini, x_short_end, y_short_end


def plot_state_space(tpm, states, A_ixs, B_ixs):
    '''
    Plots state space as a Markov chain with two columns of microstates in A (left) and B (right).

    Parameters
    ----------
    tpm : 2d array
        State-by-state
    states : list of state labels
    A_ixs : list
        Indices in group A
    B_ixs : list
        Indices in group B
    '''
    xys = [(0, i) for i, ix in enumerate(A_ixs)] + [(1, i) for i, ix in enumerate(B_ixs)]
    pos = {label: xy for label, xy in zip(states, xys)}

    G = nx.MultiDiGraph()
    edge_labels = {}

    for i, origin_state in enumerate(states):
        for j, destination_state in enumerate(states):
            prob = tpm[i, j]
            if prob > 0:
                G.add_edge(origin_state, destination_state, weight=prob, label="{:.02f}".format(prob))
                edge_labels[(origin_state, destination_state)] = label = "{:.02f}".format(prob)

    plt.figure(figsize=(8, 10))
    node_size = 200
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_weight='bold', font_size=20, font_color='black')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, label_pos=0.7, font_color='tab:blue')
    plt.axis('off')
    plt.title('State Space')

def plot_tpm_graph(tpm, states, A_states, B_states, flip_direction=False,
                   width=600,
                   height=500,
                   renderer=None,
                   show_nodelabels=True,
                   colormap='plasma',
                   node_size=30,
                   node_color='#a8d9ff',
                   rho_arrow=0.8,
                   arrowwidth=1.5,
                   arrowhead=3):
    '''
    Plots TPM using Plotly.

    Parameters
    ----------
    tpm : 2d-array
    states : list of states
    in_states : list (same as A_states or B_states)
    out_states : list (same as B_states or A_states)
    flip_direction : bool
        Whether states transitions shown are from A to B, or from B to A (flipped)

    Returns
    -------
    fig
    '''
    # Generates nx.Graph() with tpm

    pos = get_model_layout(A_states, B_states)

    G = nx.DiGraph()

    for state in states:
        G.add_node(state, pos=pos[state])

    if flip_direction:
        out_states, in_states = B_states, A_states
    else:
        out_states, in_states = A_states, B_states

    for in_state in in_states:
        for out_state in out_states:

            out_state_ix = states.index(out_state)
            in_state_ix = states.index(in_state)

            prob = tpm[out_state_ix, in_state_ix]

            if prob != 0:
                G.add_edge(out_state, in_state, prob=prob)

    layout = go.Layout(width=width,
                       height=height,
                       paper_bgcolor='#ffffff',
                       plot_bgcolor='#ffffff',
                       margin=dict(l=20, r=20, t=20, b=20))
    fig = go.Figure(layout=layout)

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(showlegend=False)

    xs = [pos[0] for pos in pos.values()]
    ys = [pos[1] for pos in pos.values()]

    if show_nodelabels:
        node_labels = list(pos.keys())
    else:
        node_labels = None

    nodes_trace = go.Scatter(x=xs,
                             y=ys,
                             mode='markers+text',
                             marker=dict(symbol='circle',
                                         size=node_size,
                                         color=node_color,
                                         line=dict(color='black', width=2),
                                         opacity=1),
                             text=node_labels,
                             textfont=dict(size=12),
                             name='State',
                             hoverinfo='text', )
    # colorbar hack
    colorbar_trace = go.Scatter(x=[0, 0],
                                y=[0, 0],
                                mode='markers',
                                marker=dict(
                                    color=[0, 1],
                                    opacity=0,
                                    colorbar=dict(title="P(e|c)"),
                                    colorscale=colormap))

    arrows = []
    for edge in G.edges:
        node_ini, node_end = edge

        probability = G.edges[edge]['prob']
        x_ini, y_ini = pos[node_ini]
        x_end, y_end = pos[node_end]

        x_ini, y_ini, x_end, y_end = shorten_line(x_ini, y_ini, x_end, y_end, rho=rho_arrow)

        cm = plt.get_cmap(colormap)
        color = mpl.colors.rgb2hex(cm(probability))
        arrows.append(go.layout.Annotation(dict(
            x=x_end,
            y=y_end,
            xref="x", yref="y",
            text="",
            showarrow=True,
            axref="x", ayref='y',
            ax=x_ini,
            ay=y_ini,
            arrowhead=arrowhead,
            arrowwidth=arrowwidth,
            arrowcolor=color, )))

    fig.update_layout(annotations=arrows)
    #     fig.add_traces(node_traces)

    fig.add_trace(nodes_trace)
    fig.add_trace(colorbar_trace)

    fig.show(renderer=renderer)

    return fig

def plot_all_tpm_matrices(micro_tpm_sbs, micro_tpm_sbn, macro_tpm_maxent, macro_tpm_stat, microstates, macrostates):
    '''
    Plots TPM for micro and macro system:
        - micro TPM state-by-node
        - micro TPM state-by-state
        - micro stationary distribution
        - macro TPM (maxent)
        - macro TPM (stationary)

    '''
    cmap = 'viridis'
    cmap = 'inferno_r'

    n_nodes = micro_tpm_sbn.shape[1]
    micro_stationary = compute.equilibrium_dist(micro_tpm_sbs)

    fig, axs = plt.subplots(ncols=5, figsize=(20, 4))
    ax = axs[0]
    im = ax.imshow(micro_tpm_sbn, vmin=0, vmax=1, cmap=cmap)
    ax.set_xticks(range(n_nodes))
    ax.set_xticklabels(np.arange(1, n_nodes + 1))
    ax.set_xlabel("node")
    ax.set_yticks(range(2 ** n_nodes))
    ax.set_yticklabels(microstates)
    ax.set_ylabel("state")
    ax.set_title('Micro TPM - state-by-node')

    ax = axs[1]
    im = ax.imshow(micro_tpm_sbs, vmin=0, vmax=1, cmap=cmap)
    ax.set_xticks(range(2 ** n_nodes))
    ax.set_xticklabels(microstates, rotation=-90)
    ax.set_xlabel("state")
    ax.set_yticks(range(2 ** n_nodes))
    ax.set_yticklabels(microstates)
    ax.set_ylabel("state")
    ax.set_title('Micro TPM - state-by-state')

    ax = axs[2]
    ax.imshow(micro_stationary[np.newaxis, :], vmin=0, vmax=1, cmap=cmap)
    ax.set_title('Micro Stationary Distribution')

    for ix, macro_tpm, title in zip([3, 4], [macro_tpm_stat, macro_tpm_maxent],
                                    ['Macro TPM (Stationary)', 'Macro TPM (Maxent)']):
        ax = axs[ix]
        im = ax.imshow(macro_tpm, vmin=0, vmax=1, cmap=cmap)
        for (j, i), label in np.ndenumerate(macro_tpm):
            ax.text(i, j, f"{label:.3f}", ha='center', va='center')
        ax.set_xticks(range(2))
        ax.set_xticklabels(macrostates, rotation=-90)
        ax.set_xlabel("state")
        ax.set_yticks(range(2))
        ax.set_yticklabels(macrostates)
        ax.set_ylabel("state")
        ax.set_title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)

def plot_tpm_matrices(micro_tpm, micro_states, macro_tpm, macro_states, colormap='inferno_r'):
    '''
    Plot micro and macro TPM matrices.

    micro_tpm : 2d-array (n_states, n_states)
    micro_states : list of length n_states
    macro_tpm : 2d-array (2, 2)
    macro_states : list of length 2
    colormap : colormap name

    '''
    n_states = micro_tpm.shape[0]
    fig, axes = plt.subplots(2, 3, figsize=(8, 5), gridspec_kw={'height_ratios': [5, 1], 'width_ratios': [10, 10, 1]})
    ax = axes[0, 0]
    ax.imshow(micro_tpm, vmin=0, vmax=1, cmap=colormap)
    ax.set_xticks(range(n_states))
    ax.set_xticklabels(micro_states)
    ax.set_yticks(range(n_states))
    ax.set_yticklabels(micro_states)

    ax = axes[0, 1]
    im = ax.imshow(macro_tpm, vmin=0, vmax=1, cmap=colormap)
    ax.set_xticks(range(2))
    ax.set_xticklabels(macro_states)
    ax.set_yticks(range(2))
    ax.set_yticklabels(macro_states)
    ax = axes[0, 2]
    plt.colorbar(im, cax=ax, shrink=0.1)

    ax = axes[1, 0]
    ax.imshow(np.mean(micro_tpm, axis=0)[np.newaxis, :], vmin=0, vmax=1, cmap=colormap)
    ax.set_xticks(range(n_states))
    ax.set_xticklabels(micro_states)
    ax.set_yticks([])

    axes[1, 1].axis('off')
    axes[1, 2].axis('off')

# PLOT DEGENERACY ALGORITHM

def plot_main_effects(cause2main_effect, states, A_states, B_states, ax=None):
    '''
    Plots networkx graph with the main effects between A_states and B_states.

    Parameters
    ----------
    cause2main_effect : dict
        Map from A states to B states.
    states : list
        All states
    A_states, B_states : list
    ax : ax for plotting
    '''
    G = nx.DiGraph()
    pos = get_model_layout(A_states, B_states)

    for state in states:
        G.add_node(state, pos=pos[state])

    for cause, main_effect in cause2main_effect.items():
        G.add_edge(cause, main_effect)

    # plt.subplot(1,2,1)
    nodes = nx.draw_networkx(G, pos=pos, node_size=500, ax=ax, node_color='#9cdbff', edge_color='k');

def plot_degeneracy_algorithm(n_states, asymmetry):
    '''
    Plots graph with main causal links of the model for different degeneracy values.

    Parameters
    ----------
    n_states : 2*n = n_states
    asymmetry : -1 < x < 1
    '''

    AB = list(range(n_states))
    # divide (a)symmetrically states into A and B groups
    cut_AB = n_states // 2 + np.round((n_states // 2 - 1) * asymmetry).astype(int)
    A, B = AB[:cut_AB], AB[cut_AB:]
    n_A, n_B = len(A), len(B)
    max_iter = min(n_A, n_B)

    n_cols = 4
    n_rows = int(np.ceil(max_iter / n_cols))
    plt.figure(figsize=(n_cols * 4, n_rows * 4))
    for n_iter in range(max_iter):
        deg = n_iter / (max_iter - 1)

        A2B, cm = degeneracy_algorithm(n_iter, A, B)

        ax = plt.subplot(n_rows, n_cols, n_iter + 1)
        plot_main_effects(A2B, A + B, A, B, ax=ax)
        plt.title(f"iter = {n_iter} (model deg = {deg:.2f})")
    plt.suptitle(f"n_states = {n_states}, asymmetry = {asymmetry:.1f}", fontsize=16)
    plt.subplots_adjust(top=0.8)


