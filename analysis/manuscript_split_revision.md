# Manuscript wording for the annotation-level split

## Methods: replace the current split sentence

Replace:

> Splits were drawn at the level of raw actions before multi-label flattening,
> so that no action text appears in more than one split.

with:

> Source annotations were partitioned by their unique action/annotation IDs
> before multi-foundation flattening. All rows derived from the same source ID
> were retained in a single split, preventing the dimension-specific copies of
> one annotation from being shared between training and evaluation. This
> annotation-level split treats separately recorded annotations as distinct
> labeled observations: Social Chemistry may contain annotations with the same
> surface action string but different contexts, rules of thumb, moral-foundation
> assignments, or judgments. The resulting internal validation and test sets
> therefore measure generalization to held-out annotations rather than strict
> generalization to lexically unseen action strings. Flattening produced 201,023
> training, 25,170 validation, and 25,141 test rows.

Optional quantitative disclosure for Methods or Supplementary Information:

> An exact-string audit found that 3,796 of 20,343 validation annotations
> (18.7%) and 3,861 of 20,344 test annotations (19.0%) used an action string that
> also occurred in a separately recorded training annotation. These are shared
> surface forms rather than shared source IDs; source IDs and all rows derived
> from each source annotation remain disjoint across splits.

The percentages above use annotation IDs as the denominator. The corresponding
expanded-row counts are 4,714/25,170 (18.7%) for validation and 4,803/25,141
(19.1%) for test.

## Results: change the internal-monitor description

Replace “for held-out actions” with:

> for held-out source annotations

Full revised sentence:

> Judgement accuracy measured whether the model assigned the highest likelihood
> to the correct virtue, vice, or neutral completion for held-out source
> annotations. Together with validation RSA, this metric tracks internal
> generalization across annotations; it is not a strict test of lexically unseen
> action strings.

## Methods or Results: preserve the external-generalization claim

> The cross-dataset behavioral evaluation is separate from the internal Social
> Chemistry split. The nine out-of-distribution benchmarks are externally
> sourced evaluation corpora and were not constructed by repartitioning or
> reusing Social Chemistry records. No benchmark example entered the ReSO, DPO,
> or shuffled-control training loss or checkpoint-selection rule. HarmBench was
> additionally logged in the training-dynamics analysis as a passive monitor,
> but it did not affect optimization, early stopping, or checkpoint selection.
> Consequently, shared surface forms among separately recorded Social Chemistry
> annotations do not create record-level train--test duplication with the nine
> external benchmark evaluations.

This paragraph establishes data-provenance independence. It should not be
expanded into an unverified claim that no short phrase happens to occur in both
independently collected corpora.

## Limitations

> The Social Chemistry split was defined at the source-annotation level rather
> than by grouping identical action strings. This preserves annotation IDs and
> keeps all multi-foundation rows derived from one annotation in a single split,
> but separately recorded annotations with the same surface action may occur in
> different splits. Internal RSA and judgment metrics should therefore be
> interpreted as held-out-annotation evaluation. Evidence for transfer beyond
> the Social Chemistry corpus comes from the nine independently sourced
> out-of-distribution benchmark suites, none of which supplied training examples
> or a checkpoint-selection signal.

## Response to an editor or reviewer

> Thank you for prompting us to clarify the split unit. The data were split by
> unique source action/annotation ID before multi-foundation flattening, and all
> descendants of a source annotation were kept in the same partition. Our
> previous wording that no action text appeared in more than one split was
> imprecise: separately recorded annotations can share a surface action string
> while differing in their associated context or moral labels. We have corrected
> the Methods and now describe the internal metrics as held-out-annotation
> evaluation. The paper's cross-dataset behavioral results use nine external
> benchmark corpora that are not derived from Social Chemistry and that supplied
> neither training examples nor checkpoint-selection signals. The annotation-
> level overlap therefore does not constitute shared records between the
> training corpus and those external evaluations. We also added this distinction
> to the limitations.

## Claims to avoid without additional experiments

- “No action text appears in more than one split.”
- “The internal validation set contains only unseen behaviors.”
- “Text overlap had no effect on the internal metrics.”
- “There is no lexical overlap whatsoever between Social Chemistry and every
  external benchmark.”

