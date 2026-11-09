# Reading and setting an ant's hidden randomness (eps)

Run the notebooks in order. Each one makes a single point, and its first cell gives the question, the recorded results and the conclusion.

| # | Notebook | Key point | Saves |
|---|---|---|---|
| 01 | data_and_rule_check | Logged ants follow rule + uniform noise exactly | |
| 02 | v1_plain_crossentropy_baseline | Plain cross-entropy on one state cannot recover the rule | v1_margin_agreement.csv |
| 03 | v2_brain_model | Mixture loss + rule-only inputs learn the rule exactly | brain_model.pt |
| 04 | q1_model_A_reads_eps_from_history | A transformer reads eps from history, compressed | seq_A.pt |
| 05 | q1_controls_eps_leaks_through_states | Most of that signal leaks through the states | seq_C.pt, seq_D.pt |
| 06 | q1_model_B_given_the_rule | Given the rule, eps is read accurately (caveat: near-trivial) | seq_B.pt |
| 07 | q2_eps_inside_the_model | eps is partly linearly readable in layers 2 and 3 | q2_eps_vectors.npz |
| 08 | q3_setting_eps_by_injection | Adding an eps vector at layer 2 sets eps | q3_injection_settings.json |
| 09 | q3_unseen_eps_values | Unseen eps (0.4, 0.7) can be read and set | seq_B_heldout.pt |
| 10 | q3_closed_loop_in_simulator | A read-then-inject controller steers a live colony | q3_closed_loop.csv |

**Setup.** Put this folder anywhere and check the paths at the top of `ant_common.py` (`PROJECT`, `DATA_DIR`, `OUT_DIR`). You can also set them with the `ANT_PROJECT`, `ANT_DATA_DIR` and `ANT_OUT_DIR` environment variables. Outputs go to `data/eps_models_v2`, so your earlier `eps_models` folder is not overwritten.

**Smoke test.** Run with `ANT_QUICK=1` to execute every notebook in a few minutes using tiny settings. The numbers from a smoke test mean nothing.

**Recorded numbers.** The results and analysis cells quote your earlier full GPU runs of the same code. A fresh run will land close to them but not on them exactly, because weight initialisation and data order change.
