"""
Resample a set of single base pair sequence variants
=========================================
Copyright 2023 GlaxoSmithKline Research & Development Limited. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=========================================
..Input::
    --tsv_file :
    A .tsv file (tab separated) containing the following columns in that order:
    variant_id, variant_chromosome,variant_position,
    restrict_window_chromosome, restrict_window_start, restrict_window_end

    .tsv file example
    chr10_102115065_A_C_b38	chr10	102115065	chr10	102022001	102218609	-
    chr10_103471557_G_A_b38	chr10	103471557	chr10	103354003	103550611	-
    chr10_104033611_G_A_b38	chr10	104033611	chr10	103987513	104184121	-
    chr10_112389547_T_C_b38	chr10	112389547	chr10	112275749	112472357	+

    Alternatively the columns can be provided as arguments for a single
    variant:
    --variant_id,
    --variant_chromosome,
    --variant_position,
    --restrict_window_chromosome,
    --restrict_window_start,
    --restrict_window_end

..Arguments::
    --variant_id,
    --variant_chromosome,
    --variant_position,
    --restrict_window_chromosome,
    --restrict_window_start,
    --restrict_window_end
    --position_base

..Usage::
    python ./variant_resampling.py \
        --tsv_file example_variant.tsv \
        --out sampled_variants_min1_max1000 \
        --ref_genome hg38.fa \
        --min_distance 1 \
        --max_distance 1000 \
        --position_base 1

..Output::
    A tsv file with the following columns:
    chr pos ref_base var_base tag
    tag may be used to link a resample variant to an input (anchor) variant.
"""
import argparse
import random

import pandas as pd
from pysam import FastaFile


# Functions =================
def sample_variant_position(
    pos: int,
    min_dist: int,
    max_dist: int,
    restrict_start: int = None,
    restrict_end: int = None,
) -> int:
    """Sample a position relative to a given position

    ..Arguments::
        pos: int
            Position to sample relative from.
        min_dist: int
            Minimum distance from the start position,
        max_dist: int
            Maximum distance from the start position.
        restrict_start: int
            Sampled position not permitted to be smaller/before.
        restrict_end: int
            Sampled position not permitted to be larger/after.

    ..Arguments::
        Samples between min and max distance allowed before or after the
        anchor position. Further restricted by an allowed window
    """
    # handle no  provided restriction window
    if restrict_end is None:
        restrict_end = pos + max_dist
    if restrict_start is None:
        restrict_start = pos - max_dist

    # set range from where to sample
    range_start_lower = max(pos - max_dist, restrict_start)
    range_end_lower = min(pos - min_dist, restrict_end)
    range_start_higher = max(pos + min_dist, restrict_start)
    range_end_higher = min(pos + max_dist, restrict_end)

    # estimate range sizes
    range_size_lower = max(0, range_end_lower - range_start_lower + 1)
    range_size_higher = max(0, range_end_higher - range_start_higher + 1)

    assert (range_size_lower > 0) | (range_size_higher > 0), (
        "At least one " "range must be " "larger than 0!"
    )

    # choose between higher or lower with probability proportional to the
    # range sizes
    direction = random.choices(
        ["lower", "higher"], weights=[range_size_lower, range_size_higher]
    )[0]

    if direction == "higher":
        sampled_pos = random.randrange(range_start_higher, range_end_higher + 1)
    else:
        sampled_pos = random.randrange(range_start_lower, range_end_lower + 1)

    return sampled_pos


def sample_variant_base(
    ref_base: str,
) -> str:
    """Sample a variant base given a reference base"""
    bases = ["A", "C", "G", "T", "N"]

    # ensure captial letter encoding
    ref_base = ref_base.upper()
    # ensure valid base
    assert (
        ref_base in bases
    ), f"reference base {ref_base} not in allowed list of characters"

    # select a  sample one of the three
    bases = bases[:4]
    # remove ref base form list of choices
    bases = [b for b in bases if b != ref_base]

    var_base = random.choice(bases)

    return var_base


def sample_variant(
    variant_position: int,
    chromosome: str,
    reference_genome: FastaFile,
    min_distance: int,
    max_distance: int,
    restrict_window_start: int = None,
    restrict_window_end: int = None,
    variant_id: str = "var_1",
    tag: str = ".",
) -> pd.DataFrame:
    """Sample a sequence variant relative to a given position

    ..Arguments::
        variant_position: int
            Position to sample relative from.
        chromosome: str
            Chromosome to operate on. Must match naming in provided fasta file.
        reference_genome: FastaFile
            Pysam loaded fasta file.
        min_distance: int
            Minimum distance from the start position,
        max_distance: int
            Maximum distance from the start position.
        restrict_window_start: int
            Optional: Sampled position not permitted to be smaller/before.
        restrict_window_end: int
            Optional: Sampled position not permitted to be larger/after.
        variant_id: str
            ID for the anchor variant.
        tag: str
            Optional: a tag to associate with the sampled variant e.g.
            the variant sampled relative to.

    ..Outputs::
        sampled_variant: pd.DataFrame
            Dataframe holding chromosome position ref_base var_base
            variant_anchor tag
    """
    # sample a position
    sampled_position = sample_variant_position(
        pos=variant_position,
        min_dist=min_distance,
        max_dist=max_distance,
        restrict_start=restrict_window_start,
        restrict_end=restrict_window_end,
    )

    # fetch reference base at sampled position
    ref_base = reference_genome.fetch(
        chromosome, sampled_position, sampled_position + 1
    ).upper()

    # sample a possible variant
    var_base = sample_variant_base(ref_base)

    # assemble
    var_df = pd.DataFrame(
        {
            "chr": [chromosome],
            "pos": [sampled_position],
            "ref_base": [ref_base],
            "var_base": [var_base],
            "var_anchor": [variant_id],
            "tag": [tag],
        }
    )

    return var_df


if __name__ == "__main__":
    # define command line arguments
    parser = argparse.ArgumentParser(
        description="(Sample variants around a given sequence position and within "
        "provided constraints"
    )
    parser.add_argument(
        "--tsv_file",
        dest="tsv_file",
        type=str,
        required=False,
        default=None,
        help="Input file in tab separated format.\n"
        "Requires the following columns:\n"
        "\tvariant_id\n"
        "\tvariant_chromosome\n"
        "\trestrict_window_chromosome\n"
        "\trestrict_window_start\n"
        "\trestrict_window_end\n",
    )
    parser.add_argument(
        "--ref_genome",
        dest="ref_genome",
        type=str,
        required=True,
        help="Path to reference genome fasta file, needs to be indexed and "
        "match the coordinate system in the query file.",
    )
    parser.add_argument(
        "--out",
        dest="out_file",
        default="./out_resampled_variants.tsv",
        type=str,
        required=False,
        help="Path and name for output file default=./tss_query_file_seq_model.tsv",
    )
    parser.add_argument(
        "--position_base",
        dest="position_base",
        default=1,
        type=int,
        required=False,
        help="Position base. Indicate if the variant and sequence position "
        "are in 1 based or 0 based coordinates. Default=1. "
        "If the position_base is 1 than 1 based coordinates are reported. "
        "or zero based otherwise.",
    )
    parser.add_argument(
        "--num_samples",
        dest="num_samples",
        default=1,
        type=int,
        required=False,
        help="Number of variants to sample per variant anchor. Default=1",
    )
    parser.add_argument(
        "--min_distance",
        dest="min_distance",
        default=1,
        type=int,
        required=False,
        help="Maximum distance of a sampled variant from the prog. variant "
        "position. Default = 1000."
        " Min of 1 ensures no variant is sampled from the exact same postion.",
    )
    parser.add_argument(
        "--max_distance",
        dest="max_distance",
        default=1000,
        type=int,
        required=False,
        help="Maximum distance of a sampled variant from the prog. variant "
        "position. Default = 1000.",
    )
    parser.add_argument(
        "--variant_id",
        dest="variant_id",
        default="example_variant",
        type=str,
        required=False,
        help="Single variant id." "Default = 'example_variant'.",
    )
    parser.add_argument(
        "--variant_chromosome",
        dest="variant_chromosome",
        default=None,
        type=str,
        required=False,
        help="Single chromosome of the variant.",
    )
    parser.add_argument(
        "--variant_position",
        dest="variant_position",
        default=None,
        type=int,
        required=False,
        help="Single variant chromosome. Must match chromosome names in the "
        "reference genome.",
    )
    parser.add_argument(
        "--restrict_window_chromosome",
        dest="restrict_window_chromosome",
        default=None,
        type=str,
        required=False,
        help="Single chromosome of the restriction window.",
    )
    parser.add_argument(
        "--restrict_window_start",
        dest="restrict_window_start",
        default=None,
        type=int,
        required=False,
        help="Single start position of the restriction window.",
    )
    parser.add_argument(
        "--restrict_window_end",
        dest="restrict_window_end",
        default=None,
        type=int,
        required=False,
        help="Single end position of the restriction window.",
    )
    parser.add_argument(
        "--tag",
        dest="tag",
        default=".",
        type=str,
        required=False,
        help="Single tag to add to.",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        required=False,
        type=int,
        default=1234,
        help="Random seed. Default = 1234.",
    )

    # fetch arguments
    args = parser.parse_args()

    # set seed
    random.seed(args.seed)

    # check that sensible position_base was supplied
    assert args.position_base in [
        0,
        1,
    ], f"position_base not in [0, 1] got {args.position_base}"

    # read tsv file if provided
    if args.tsv_file:
        var_df = pd.read_csv(
            args.tsv_file,
            sep="\t",
            header=None,
            names=[
                "variant_id",
                "variant_chromosome",
                "variant_position",
                "restrict_window_chromosome",
                "restrict_window_start",
                "restrict_window_end",
                "tag",
            ],
            index_col=False,
        )
    else:
        # else make a single entry dataframe
        var_df = pd.DataFrame(
            {
                "variant_id": [args.variant_id],
                "variant_chromosome": [args.variant_chromosome],
                "variant_position": [args.variant_position],
                "restrict_window_chromosome": [args.restrict_window_chromosome],
                "restrict_window_start": [args.restrict_window_start],
                "restrict_window_end": [args.restrict_window_end],
                "tag": [args.tag],
            }
        )

    # switch positions to 0-based index for python indexing
    # if position_base == 1
    if args.position_base == 1:
        var_df["variant_position"] -= 1
        var_df["restrict_window_start"] -= 1

    # open fasta file
    fa = FastaFile(args.ref_genome)

    # for each entry in variant data frame
    # sample num_samples new variants
    per_input_entry = 0
    collect_sampled_variants = []
    for per_input_entry in range(len(var_df)):
        # ensure variant and restriction window chromosomes match!
        assert (
            var_df["variant_chromosome"][per_input_entry]
            == var_df["restrict_window_chromosome"][per_input_entry]
        )

        # run sample variant <num_samples> times
        for per_num_sample in range(args.num_samples):
            sampled_variant = sample_variant(
                variant_position=var_df["variant_position"][per_input_entry],
                chromosome=var_df["variant_chromosome"][per_input_entry],
                reference_genome=fa,
                min_distance=args.min_distance,
                max_distance=args.max_distance,
                restrict_window_start=var_df["restrict_window_start"][per_input_entry],
                restrict_window_end=var_df["restrict_window_end"][per_input_entry],
                variant_id=var_df["variant_id"][per_input_entry],
                tag=var_df["tag"][per_input_entry],
            )

            collect_sampled_variants.append(sampled_variant)

    # concatinate dataframe
    sampled_df = pd.concat(collect_sampled_variants)
    sampled_df.reset_index(inplace=True, drop=True)

    # if position_base == 1 --> modify the positions of the variant to
    #  be one based again
    if args.position_base == 1:
        sampled_df["pos"] += 1

    # save variants to tsv
    sampled_df.to_csv(args.out_file, sep="\t", header=False, index=False)
