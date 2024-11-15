import numpy as np
import pandas as pd
from graphviz import Digraph


class Markov:
    def __init__(self, transitions, rewards=None, node_names=None, prob_dist=None, values=None, ):
        # data의 타입에 따라 클래스 변수 초기화
        if isinstance(transitions, pd.DataFrame):
            self.transitions = transitions
            self._node_names = list(
                self.transitions.columns if (node_names is None) else node_names)

        elif isinstance(transitions, dict):
            self.transitions = self._dict_to_dataframe(transitions)
            self._node_names = list(
                self.transitions.columns if (node_names is None) else node_names)

        else:
            if isinstance(transitions, np.ndarray):
                if node_names is None:
                    self._node_names = [
                        f'Node_{i}' for i in range(transitions.shape[0])]
                else:
                    self._node_names = list(node_names)
                self.transitions = pd.DataFrame(
                    transitions, index=self._node_names, columns=self._node_names)
            elif isinstance(transitions, list):
                if node_names is None:
                    self._node_names = [
                        f'Node_{i}' for i in range(len(transitions[0]))]
                else:
                    self._node_names = list(node_names)
                self.transitions = pd.DataFrame(
                    transitions, index=self._node_names, columns=self._node_names)

            else:
                raise ValueError(
                    "Transitions must be a Pandas DataFrame, NumPy ndarray, list, or dict.")

    def transitions_as_array(self):
        return self.transitions.values

    @property
    def node_names(self):
        return self._node_names

    def save_to_csv(self):
        pass

    def _dict_to_dataframe(self, _dict):
        df = pd.DataFrame(0, index=_dict.keys(), columns=_dict.keys())
        for from_state, destinations in _dict.items():
            for to_state, probability in destinations.items():
                df.loc[from_state, to_state] = probability

        return df


class PlotMarkov:
    def __init__(self, markov):
        self.markov = markov
        self.transitions = self._dataframe_to_dict(markov.transitions)
        self._probabilities = {}
        self._values = {}

    def _draw_graph(self, show_probabilities=False, show_values=False):
        graph = Digraph()
        graph.attr(rankdir='LR')
        graph.attr('node', shape='circle',
                   style='filled', fillcolor='lightblue')

        # Add nodes
        for state in self.transitions.keys():
            label = f'<B>{state}</B>'

            if show_probabilities:
                p = self._probabilities.get(state, 0)
                label += f'<br/><FONT COLOR="Red" POINT-SIZE="10">p = {p:1.2f}</FONT>'
            if show_values:
                v = self._values.get(state, 0)
                label += f'<br/><FONT COLOR="Green" POINT-SIZE="10">v = {v:.2f}</FONT>'

            graph.node(state, label=f'<{label}>',
                       width='0.8', height='0.8', fixedsize='true')

        # Add edges with probabilities
        for from_state, destinations in self.transitions.items():
            for to_state, probability in destinations.items():
                graph.edge(from_state, to_state,
                           label=f' {probability:.2f} ')
        return graph

    def _dataframe_to_dict(self, frame):
        transitions = frame.T.to_dict()
        transitions = {
            from_state: {
                to_state: probability
                for to_state, probability in destinations.items() if probability != 0
            }
            for from_state, destinations in transitions.items()
        }
        return transitions
