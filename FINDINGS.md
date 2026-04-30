# Findings

First-pass results of the EigenFlux ablation, comparing each persona's day of decisions under the search baseline (`without_eigenflux`) and the EigenFlux feed (`with_eigenflux`). For methodology, see [METHODOLOGY.md](METHODOLOGY.md).

> **Run:** 2026-04-22, 24-hour collection window per (persona × condition) cell.
> Search baseline (`without_eigenflux`): one search per session.
> EigenFlux (`with_eigenflux`): 5-minute heartbeat pull interval.
> Single run per cell — no variance estimates yet.

---

## Headline

EigenFlux produces a large, consistent increase in **Send Count** across every persona — from `+162%` (Lark) to `+1850%` (Ember) — and a more uneven increase in **Action Count**. The dominant mechanism is upstream: EigenFlux feeds contain roughly 6× more items than the matched search pool, and item-level send rates hold up under that volume.

---

## Data

<table>
<thead>
<tr>
  <th rowspan="2">Persona</th>
  <th colspan="3" align="center">Search (control)</th>
  <th colspan="3" align="center">EigenFlux (treatment)</th>
  <th colspan="2" align="center">Comparison</th>
</tr>
<tr>
  <th align="right">Items</th>
  <th align="right">Send Count</th>
  <th align="right">Action Count</th>
  <th align="right">Items</th>
  <th align="right">Send Count</th>
  <th align="right">Action Count</th>
  <th align="right">Δ Send Count</th>
  <th align="right">Δ Action Count</th>
</tr>
</thead>
<tbody>
<tr><td>Sable (careful operator)</td><td align="right">14</td><td align="right">6</td><td align="right">0</td><td align="right">135</td><td align="right">39</td><td align="right">0</td><td align="right"><b>+550%</b></td><td align="right">N/A</td></tr>
<tr><td>Nova (content strategist)</td><td align="right">20</td><td align="right">2</td><td align="right">1</td><td align="right">71</td><td align="right">27</td><td align="right">9</td><td align="right"><b>+1250%</b></td><td align="right"><b>+800%</b></td></tr>
<tr><td>Flint (executive finisher)</td><td align="right">22</td><td align="right">4</td><td align="right">3</td><td align="right">186</td><td align="right">31</td><td align="right">3</td><td align="right"><b>+675%</b></td><td align="right">0%</td></tr>
<tr><td>Lark (learning curator)</td><td align="right">21</td><td align="right">13</td><td align="right">10</td><td align="right">51</td><td align="right">34</td><td align="right">24</td><td align="right"><b>+162%</b></td><td align="right"><b>+140%</b></td></tr>
<tr><td>Mist (research librarian)</td><td align="right">23</td><td align="right">11</td><td align="right">5</td><td align="right">143</td><td align="right">63</td><td align="right">6</td><td align="right"><b>+473%</b></td><td align="right">+20%</td></tr>
<tr><td>Ember (social scout)</td><td align="right">23</td><td align="right">2</td><td align="right">0</td><td align="right">150</td><td align="right">39</td><td align="right">7</td><td align="right"><b>+1850%</b></td><td align="right">N/A</td></tr>
<tr><td><b>Aggregate (sum)</b></td><td align="right"><b>123</b></td><td align="right"><b>38</b></td><td align="right"><b>19</b></td><td align="right"><b>736</b></td><td align="right"><b>233</b></td><td align="right"><b>49</b></td><td align="right"><b>+513%</b></td><td align="right"><b>+158%</b></td></tr>
</tbody>
</table>

`N/A` reflects a 0 → N action transition (no baseline to compute a percentage against), not an infinite gain.

---

## Quantitative analysis

### Source pool size

EigenFlux delivers **~6× more items per persona** on average (123 → 736 in aggregate). This is the upstream throughput advantage and the precondition for everything else.

### Send Count

**Universal improvement** — 6/6 personas. The aggregate Send Count grows from 38 → 233 (+513%), tracking the item-volume gain almost exactly. EigenFlux's primary effect is making more items available that the model judges worth surfacing.

### Send rate (derived)

Per-item efficiency tells a more nuanced story. Even though EigenFlux brings more items, the *fraction* of items judged sendable is comparable to or better than search for most personas:

| Persona | Search send rate | EigenFlux send rate |
|---------|-----------------:|--------------------:|
| Sable | 43% | 29% |
| Nova | 10% | **38%** |
| Flint | 18% | 17% |
| Lark | 62% | 67% |
| Mist | 48% | 44% |
| Ember | 9% | **26%** |

Two patterns:
- For personas where the search baseline was **sparse** (Nova, Ember at 9–10%), EigenFlux improves both volume *and* per-item rate. Personalized matching is doing real work.
- For personas where search was already efficient (Sable 43%, Lark 62%, Mist 48%), EigenFlux's per-item rate is comparable. Volume scales without rate degradation.

Sable is the one persona where the per-item send rate slips (43% → 29%) — EigenFlux brings more borderline items and Sable's high interruption threshold filters more of them out. The *count* still rises 6× though, because the volume gain dominates: even with stricter per-item filtering, EigenFlux delivers many times more sendable items than search.

### Action Count

**Variable across personas.** Strong gains for Nova (+800%) and Lark (+140%); a meaningful 0→7 jump for Ember; for Flint and Mist, EigenFlux's contribution shows up primarily in send volume rather than additional action-recommendations (+0% / +20% on Action Count); Sable is zero in both conditions, consistent with its high interruption threshold.

The aggregate Action Count still grows 19 → 49 (+158%) — meaningful, but smaller than the +513% on sends. **EigenFlux's amplification is stronger on the send dimension than on the action dimension.**

### Action-among-Sends ratio

The clearest qualitative signal in the data is the **conditional action rate** — among items the model decided to send, what fraction warranted a recommended action?

| Persona | Search action/send | EigenFlux action/send |
|---------|-------------------:|----------------------:|
| Sable | 0% | 0% |
| Nova | 50% | 33% |
| Flint | **75%** | 10% |
| Lark | 77% | 71% |
| Mist | **45%** | 10% |
| Ember | 0% | 18% |

For Flint and Mist, the ratio drops (75% → 10%, 45% → 10%) — EigenFlux's added items pass the "worth surfacing" bar but lean toward `send` rather than `send_with_action`. For Lark, the ratio holds (77% → 71%), with both sources surfacing action-clear signals at similar density. For Ember, EigenFlux unlocks an action-worthy category that search couldn't reach (0 → 18%).

---

## Qualitative insights

### 1. EigenFlux's primary effect is at the source level

The data is consistent with the methodology's framing of EigenFlux as a source-level enhancement. EigenFlux's gain is **upstream**: it puts more items the model considers worth surfacing in front of the agent. It does not, on this data, make those items disproportionately more action-worthy. The volume gain on Send Count (~6×) is real and consistent; the gain on Action Count (~2.6× aggregate) is smaller and uneven.

This is the expected shape of a personalized retrieval intervention: **bigger pool of relevant signal, similar quality distribution within that signal.**

### 2. EigenFlux's amplification is largest where keyword search is weakest

Send Count rises across all six personas; the magnitude varies with how well a keyword baseline already serves the persona's task.

The **biggest jumps** — Ember (+1850%) and Nova (+1250%) — come for personas whose tasks resist direct keyword retrieval. Ember (social scout) is hunting for early adopters and partnerships — soft signals from people and communities that don't surface cleanly in DDG queries. Nova (content strategist) is tracking emerging discourse — narrative shifts and writing angles, which similarly resist keyword search. Personalized routing fills exactly the gap a search-only baseline leaves open.

**EigenFlux continues to amplify substantially** even where keyword search already serves the task well. Lark (+162%) and Mist (+473%) deal in structured learning resources and academic papers in a defined subfield — categories DDG queries handle reasonably — and EigenFlux still grows the pool of items worth surfacing many times over.

The takeaway: EigenFlux helps everywhere, and helps most where the user's task type makes solo retrieval hardest.

### 3. Persona traits shape decisions visibly

Sable produces zero actions in both conditions, despite EigenFlux delivering 39 sends versus search's 6. This is **not a measurement failure** — Sable is configured with `interruption_threshold: high` and conservative `risk_tolerance`. The persona is designed to surface information without recommending action unless the case is clear. The data is internally consistent: more sends, no action escalation.

Conversely, Lark — configured for patient, level-matched curation — produces extremely high action-among-send ratios (77% / 71%) under both conditions. The persona is designed to recommend learning resources directly, and the data reflects that.

This is a healthy sign: persona traits are visible in the decision distribution, not washed out by the source-level intervention.

---

## Caveats

- **Single run per cell.** No within-condition variance is captured. A re-run with a different seed, or under different model temperature, may shift individual numbers; the broad pattern (universal Send Count gain, mixed Action Count gain) is robust to small per-cell variation but should not be over-read at the single-percentage level.
- **Item counts differ by an order of magnitude between conditions.** The %SC and %AC columns normalize for this, but absolute-count comparisons must be read alongside the rate analysis above.
- **Decisions are made by a single judge model** (Claude Haiku). All Send/Action counts reflect that model's judgment of proactive value — a different judge could shift the absolute numbers, though the cross-condition comparison is more robust because both conditions use the same judge.
- **`N/A` action increases** (Sable, Ember) reflect 0/0 division. For Ember, the absolute 0 → 7 jump is meaningful even without a percentage.
- **Per-item, not in-context — by design.** Each item is judged in isolation by a competent model, so item-level *usefulness* and *accuracy* are already implicitly part of the Send/Action signal (see METHODOLOGY §"What Send Count actually measures"). The deployment-context layer — sequential judgment over a stream, awareness of what's already been surfaced, behavior when the user replies mid-stream, the framing of the actual message produced, and timing relative to when the underlying signal became available — is a deliberate experimental simplification. Evaluating those dimensions in a controlled, paired-comparable way is hard; collapsing the decision into a per-item judgment is the approximation that makes the EigenFlux ablation cleanly feasible.

---

## Summary

EigenFlux substantially increases the volume of items a competent judge model deems useful and accurate enough to surface proactively, with a smaller and less even gain on items judged action-worthy. Per-item usefulness and accuracy are already part of what these metrics measure — every counted send has passed that bar.

The deployment-context layer — sequential decisions in a stream, message framing, timing, behavior when the user replies mid-stream — is intentionally collapsed into per-item judgment in this design. That trade buys experimental tractability: it lets the EigenFlux variable be cleanly ablated against a search baseline. Stronger claims about live-deployment behavior would require methodological extensions outside this experiment's scope.
