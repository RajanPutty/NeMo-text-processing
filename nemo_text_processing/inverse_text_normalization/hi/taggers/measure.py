# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2024 and onwards Google, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    NEMO_CHAR,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import apply_fst, get_abs_path


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure
        e.g. ऋण बारह किलोग्राम -> measure { decimal { negative: "true"  integer_part: "१२"  fractional_part: "५०"} units: "kg" }
        e.g. ऋण बारह किलोग्राम -> measure { cardinal { negative: "true"  integer_part: "१२"} units: "kg" }
        e.g. सात शून्य शून्य ओक स्ट्रीट -> measure { units: "address" cardinal { integer: "७०० ओक स्ट्रीट" } preserve_order: true }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        measure: MeasureFst
    """

    def get_structured_address_graph(self):
        """
        ITN structured address tagger for state/city + pincode patterns.
        Converts spoken digit words to Devanagari digits for pincodes.

        Examples:
            "अमरावती छह पाँच पाँच नौ तीन शून्य" -> units: "address" cardinal { integer: "अमरावती ६५५९३०" } preserve_order: true
            "मुंबई, महाराष्ट्र आठ तीन नौ चार आठ आठ" -> units: "address" cardinal { integer: "मुंबई, महाराष्ट्र ८३९४८८" } preserve_order: true
        """
        # State/city keywords
        states = pynini.string_file(get_abs_path("data/address/states.tsv"))
        cities = pynini.string_file(get_abs_path("data/address/cities.tsv"))
        state_city_names = pynini.union(states, cities).optimize()

        # Digit word -> Devanagari digit (inverted mapping)
        num_word = (
            pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
            | pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        ).invert().optimize()

        # Pincode: exactly 6 spoken digit words -> 6-digit Devanagari number
        delete_one_space = pynutil.delete(" ")
        pincode = num_word + pynini.closure(delete_one_space + num_word, 5, 5)

        # Pattern: state/city [, state] space pincode
        pattern = (
            state_city_names
            + pynini.closure(
                pynini.accep(",") + pynini.accep(" ") + state_city_names, 0, 1
            )
            + pynini.accep(" ") + pincode
        ).optimize()

        graph = (
            pynutil.insert('units: "address" cardinal { integer: "')
            + convert_space(pattern)
            + pynutil.insert('" } preserve_order: true')
        )
        return pynutil.add_weight(graph, 1.0).optimize()

    def get_address_graph(self):
        """
        ITN address tagger that converts spoken digit words, special character words,
        and ordinal words back to their written forms when address context keywords
        are present.

        Special characters (हाइफ़न -> -, बटा -> /) act as "connectors" that bridge
        adjacent words without spaces, handling patterns like "डी-५", "२९८९/बी",
        "ट्रॉय-शेंक्टाडी" naturally.

        Examples:
            "सात शून्य शून्य ओक स्ट्रीट" -> "७०० ओक स्ट्रीट"
            "छह छह हाइफ़न चार, पार्कहर्स्ट रोड" -> "६६-४, पार्कहर्स्ट रोड"
            "डी हाइफ़न पाँच शॉप तीन" -> "डी-५ शॉप ३"
            "सेकंड फ्लोर नंबर आठ शून्य आठ" -> "२nd फ्लोर नंबर ८०८"
        """
        # Digit word -> Devanagari digit (inverted mapping)
        num_word = (
            pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
            | pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        ).invert().optimize()

        # Special char word -> actual char (हाइफ़न -> -, बटा -> /)
        # These are NOT part of the digit block; instead they serve as "connectors"
        # that bridge adjacent words without spaces.
        special_word = pynini.string_file(get_abs_path("data/address/special_characters.tsv"))

        # Ordinal word -> Devanagari digit + suffix (फ़र्स्ट -> १st, सेकंड -> २nd, etc.)
        ordinal_word = pynini.string_file(get_abs_path("data/address/ordinals.tsv"))

        # Address context keywords (FSA)
        context_keywords_fsa = pynini.string_file(get_abs_path("data/address/context.tsv"))

        # Passthrough for literal digit characters that may appear in input
        # (e.g. Extended Arabic-Indic digit zero ۰ = U+06F0)
        digit_passthrough = pynini.string_map([
            ("۰", "۰"), ("۱", "۱"), ("۲", "۲"), ("۳", "۳"), ("۴", "۴"),
            ("۵", "۵"), ("۶", "۶"), ("۷", "۷"), ("۸", "۸"), ("۹", "۹"),
        ]).optimize()
        digit_unit = pynini.union(num_word, digit_passthrough).optimize()

        # All digit inputs (for text_word exclusion)
        all_digit_inputs = pynini.project(digit_unit, "input").optimize()

        # All ordinal inputs (for text_word exclusion)
        all_ordinal_inputs = pynini.project(ordinal_word, "input").optimize()

        # Non-space, non-comma character
        non_space_non_comma = pynini.difference(
            NEMO_CHAR, pynini.union(NEMO_WHITE_SPACE, pynini.accep(","))
        ).optimize()

        # Any word: sequence of non-space, non-comma characters
        any_word = pynini.closure(non_space_non_comma, 1).optimize()

        # Text word: any word NOT a digit word and NOT an ordinal
        # Note: special word inputs (हाइफ़न, बटा) ARE text words when standalone;
        # they only convert to symbols when used as connectors in a chain.
        text_word = pynini.difference(
            any_word, pynini.union(all_digit_inputs, all_ordinal_inputs)
        ).optimize()

        # Digit block: one or more digit words/characters concatenated (spaces deleted)
        delete_one_space = pynutil.delete(" ")
        digit_block = digit_unit + pynini.closure(
            pynutil.add_weight(delete_one_space + digit_unit, -1.0)
        )

        # Special word connector: bridges two words without spaces
        # Consumes the space before and after the special word
        # Examples: " हाइफ़न " -> "-", " बटा " -> "/"
        connector = delete_one_space + special_word + delete_one_space

        # Matchable unit: digit block, ordinal, or text word
        matchable = pynini.union(
            pynutil.add_weight(digit_block, -0.1),
            pynutil.add_weight(ordinal_word, -0.2),
            pynutil.add_weight(text_word, 0.1),
        ).optimize()

        # Chain: one or more matchable units optionally connected by special chars
        # The connector's negative weight ensures bridging is preferred over
        # separate elements when a special word appears between words.
        # Examples:
        #   "डी हाइफ़न पाँच" -> chain: text("डी") + conn("-") + digit("५") = "डी-५"
        #   "दो नौ आठ नौ बटा बी" -> chain: digit("२९८९") + conn("/") + text("बी") = "२९८९/बी"
        #   "ट्रॉय हाइफ़न शेंक्टाडी" -> chain: text + conn("-") + text = "ट्रॉय-शेंक्टाडी"
        chain = matchable + pynini.closure(
            pynutil.add_weight(connector + matchable, -0.5)
        )

        # Optional trailing comma (attached directly to preceding word in spoken form)
        opt_comma = pynini.closure(pynini.accep(","), 0, 1)

        # Element: chain with optional trailing comma
        element = chain + opt_comma

        # Full address: elements separated by spaces
        address_content = element + pynini.closure(pynini.accep(" ") + element)

        # Context detection using sigma_star-based matching for robust keyword detection
        # This allows the keyword to appear anywhere in the input without window size limits
        any_char = pynini.union(
            pynini.difference(NEMO_CHAR, NEMO_WHITE_SPACE),
            NEMO_WHITE_SPACE,
        ).optimize()
        sigma_star = pynini.closure(any_char).optimize()

        # Word separator: space or comma (for keyword boundary detection)
        word_sep = pynini.union(pynini.accep(" "), pynini.accep(",")).optimize()

        # Input must contain at least one context keyword as a complete word
        # (bounded by space, comma, or string boundary)
        input_pattern = pynini.union(
            # keyword at start, followed by separator and more content
            context_keywords_fsa + word_sep + sigma_star,
            # keyword at end, preceded by content and space
            sigma_star + pynini.accep(" ") + context_keywords_fsa,
            # keyword in middle, bounded by separators
            sigma_star + pynini.accep(" ") + context_keywords_fsa + word_sep + sigma_star,
            # just the keyword alone
            context_keywords_fsa,
        ).optimize()

        # Compose: only process inputs matching the context pattern
        address_graph = pynini.compose(input_pattern, address_content).optimize()

        graph = (
            pynutil.insert('units: "address" cardinal { integer: "')
            + convert_space(address_graph)
            + pynutil.insert('" } preserve_order: true')
        )
        return pynutil.add_weight(graph, 1.05).optimize()

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="measure", kind="classify")

        cardinal_graph = cardinal.graph_no_exception
        decimal_graph = decimal.final_graph_wo_negative

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("ऋण", "\"true\"") + delete_extra_space,
            0,
            1,
        )

        measurements_graph = pynini.string_file(get_abs_path("data/measure/measurements.tsv")).invert()
        paune_graph = pynini.string_file(get_abs_path("data/numbers/paune.tsv")).invert()

        self.measurements = pynutil.insert("units: \"") + measurements_graph + pynutil.insert("\" ")
        graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        graph_integer_paune = pynutil.insert("integer_part: \"") + paune_graph + pynutil.insert("\"")

        graph_saade_single_digit = pynutil.add_weight(
            pynutil.delete("साढ़े")
            + delete_space
            + graph_integer
            + delete_space
            + pynutil.insert(" fractional_part: \"५\""),
            0.1,
        )
        graph_sava_single_digit = pynutil.add_weight(
            pynutil.delete("सवा")
            + delete_space
            + graph_integer
            + delete_space
            + pynutil.insert(" fractional_part: \"२५\""),
            0.1,
        )
        graph_paune_single_digit = pynutil.add_weight(
            pynutil.delete("पौने")
            + delete_space
            + graph_integer_paune
            + delete_space
            + pynutil.insert(" fractional_part: \"७५\""),
            1,
        )
        graph_dedh_single_digit = pynutil.add_weight(
            pynini.union(pynutil.delete("डेढ़") | pynutil.delete("डेढ़"))
            + delete_space
            + pynutil.insert("integer_part: \"१\"")
            + delete_space
            + pynutil.insert(" fractional_part: \"५\""),
            0.1,
        )
        graph_dhaai_single_digit = pynutil.add_weight(
            pynutil.delete("ढाई")
            + delete_space
            + pynutil.insert("integer_part: \"२\"")
            + delete_space
            + pynutil.insert(" fractional_part: \"५\""),
            1,
        )

        graph_exceptions = (
            graph_saade_single_digit
            | graph_sava_single_digit
            | graph_paune_single_digit
            | graph_dedh_single_digit
            | graph_dhaai_single_digit
        )

        graph_measurements = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal_graph
            + pynutil.insert(" }")
            + delete_extra_space
            + self.measurements
        )
        graph_measurements |= (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + delete_extra_space
            + self.measurements
        )
        graph_quarterly_measurements = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + graph_exceptions
            + pynutil.insert(" }")
            + delete_extra_space
            + self.measurements
        )
        graph_exception_bai = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + delete_space
            + pynini.cross("बाई", "x")
            + delete_space
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + pynini.closure(delete_extra_space + self.measurements)
        )

        # Address graphs
        address_graph = self.get_address_graph()
        structured_address_graph = self.get_structured_address_graph()

        graph = (
            graph_measurements
            | graph_quarterly_measurements
            | graph_exception_bai
            | address_graph
            | structured_address_graph
        )
        self.graph = graph.optimize()

        final_graph = self.add_tokens(graph)
        self.fst = final_graph
