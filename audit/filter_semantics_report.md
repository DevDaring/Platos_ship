# audit/filter_semantics_report.md

## What the code does

`Code_Phase_2/CPU_Only/src/confidence_weighted_protocol.py::filter_peers_by_confidence`

```python
if confidence is None or confidence < confidence_threshold:
    filtered_out += 1        # dropped
else:
    included.append(peer)    # retained
```

**Behaviour A: a peer is RETAINED when confidence >= 60 and dropped otherwise.**
Missing/unparsed confidence counts as below threshold. The focal agent is never
treated as its own peer. This matches what the manuscript describes.

## The defect this exposed

The pre-flight gate was written as

    Delta = P(loud | wrong) - P(loud | correct)

described as "positive only when high confidence actually marks a wrong
answer". For a **retain-high** filter that is the wrong sign: if confidence
were higher when a peer is wrong, the filter would preferentially keep WRONG
peers, which is harmful rather than useful.

The correctly signed quantity is

    Delta_ret = P(retained | correct) - P(retained | wrong)

positive when the filter preferentially keeps correct peers. `corrected_gate.py`
now computes this, and the AUROC positive class is stated as **correct**.

The empirical conclusion is unchanged — the filter is useless either way,
because the discrimination is under one percentage point — but the stated
criterion was reversed and would not have survived a careful reviewer.

## Unit tests (`Code_Phase_2/CPU_Only/tests/test_filter_semantics.py`, 4 passing)

| Test | Asserts |
|---|---|
| boundary values | conf 0, 59, missing -> dropped; 60, 61, 100 -> retained; focal excluded |
| useful direction | correct peers confident -> gap +1.0, gate passes |
| harmful direction | wrong peers confident -> gap -1.0, gate fails |
| single class | no correct peers -> gap `None`, `undefined_single_class`, never 0 |

## Measured retention / removal

| Condition | Peers offered | Removed | Retained |
|---|---:|---:|---:|
| legacy C5 (no confidence line in persona prompt) | 600 | 600 (**100%**) | 0% |
| C5R (repaired, wrong-anchored) | 600 | 6 (**1.0%**) | 99.0% |
| C5H (repaired, honest peers) | 600 | 24 (**4.0%**) | 96.0% |

Legacy C5 therefore never tested filtering at all; it is labelled a broken
diagnostic in Table 1 and is excluded from the filter conclusion.

## Gate values (correctly signed)

| Substrate | n | P(ret\|correct) | P(ret\|wrong) | Delta_ret | AUROC | Decision |
|---|---:|---:|---:|---:|---:|---|
| Phase-1 C3/C4 dumb R1 | 12,332 | 0.9975 | 0.9940 | **+0.0036** | 0.62 | failed |
| C5R anchored R0 | 600 | undefined (0 correct) | 0.9900 | undefined | – | undefined_single_class |
| C5H honest R0 | 578 | 1.0000 | 0.9935 | **+0.0065** | 0.58 | failed |

Threshold 0.10. The sign is the useful one; the magnitude is not. Undefined
stays undefined where the correct class is empty.

## Matched C5R vs unfiltered C4

Both on the same 100-question mitigation subset: C5R Round-1 accuracy 82.7%
against C4 84.1% on the full pool (Table 2). Since the filter removes only 1%
of peers, C5R is close to unfiltered C4 by construction — which is the point.
