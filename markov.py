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

    def draw(self):
        pass

    def save_to_csv(self):
        pass

    def _dict_to_dataframe(self, _dict):
        df = pd.DataFrame(0, index=_dict.keys(), columns=_dict.keys())
        for from_state, destinations in _dict.items():
            for to_state, probability in destinations.items():
                df.loc[from_state, to_state] = probability

        return df
