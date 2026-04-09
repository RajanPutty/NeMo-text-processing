import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    INPUT_LOWER_CASED,
    GraphFst,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path


class PercentageFst(GraphFst):
    def __init__(self, cardinal, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="percentage", kind="classify")

        # load percent words and flip mapping: प्रतिशत → %
        percent_graph = pynini.string_file(
            get_abs_path("data/percentage/percent_symbol.tsv")
        ).invert()

        # reuse number logic (बीस → २०, पाँच सौ → ५००)
        integer_graph = cardinal.graph_no_exception

        # match: <number> + <percent word>
        # and convert into structured format
        final_graph = (
            pynutil.insert('integer: "')
            + integer_graph
            + pynutil.insert('"')
            + delete_space
            + pynutil.insert(' percent: "')
            + percent_graph
            + pynutil.insert('"')
        )

        # wrap as: percentage { ... }
        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()