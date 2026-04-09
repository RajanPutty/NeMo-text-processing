import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)


class PercentageFst(GraphFst):
    def __init__(self):
        super().__init__(name="percentage", kind="verbalize")

        # extract number part (remove labels and quotes)
        # example: integer: "२०" → २०
        integer_part = (
            pynutil.delete("integer:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # extract percent symbol
        # example: percent: "%" → %
        percent_part = (
            pynutil.delete("percent:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # combine both → २०%
        graph = integer_part + delete_space + percent_part

        # remove outer wrapper: percentage { ... }
        delete_tokens = self.delete_tokens(graph)

        self.fst = delete_tokens.optimize()