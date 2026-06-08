# Controller and Shielding Protocol

The evaluation follows a paired raw-vs-shielded protocol:

1. A raw controller proposes an action from the finite candidate group.
2. CertiTwin-RSM checks whether the raw action is certified safe.
3. If certified safe, the raw action is passed through.
4. If not certified safe, the shield projects to a certified safe candidate while minimizing deviation from the raw action.
5. If no certified safe candidate exists, fallback is counted and the minimum-risk candidate is selected.

The raw controller is not retrained by the shield. Raw and shielded versions use the same test candidate groups and utility definition.
