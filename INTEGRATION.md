# Original module interfaces

The external detector implementation is deliberately not distributed here. These notes describe integration points rather than a turnkey training command.

## GPDM

Instantiate `gqdet.gpdm.P2DIAGPDM` using the GPDM parameters under `DFINE` in the selected experiment configuration. Its call is `p3_updated, diagnostics = module(p2, p3, p4)`. Retain the external detector's P4 and P5: encoder inputs are `[p3_updated, p4, p5]`. The identity branch is preserved inside GPDM. Include its learnable parameters in the optimizer.

## RGQS

Construct `gqdet.rgqs.RGQSSelection(config['BRGQACriterion'])`. Call `_build_brgqa_indices(outputs, targets, indices_o2o, indices_go)` with the external detector's frozen predictions and original primary/union assignments. Boxes are normalized cxcywh; logits are raw logits; the experiment is single-class ship detection. The result contains selected query/GT pairs, current IoUs, normalized auxiliary weights and diagnostics.

The original selection uses adjacent auxiliary decoder outputs, current-IoU eligibility and positive selection score, excludes primary/union positives, retains the per-GT candidate pool, then greedily resolves query/GT conflicts. A query is used at most once and each GT receives at most the configured auxiliary budget. Accepted candidates are re-ranked within each GT before quality/rank weights are normalized. The methods preserve the original `torch.topk` behavior, including its tie behavior.

Remove selected auxiliary query positions from the corresponding layer's negative classification mask. The original detector's primary matching and regression/local branches are otherwise external and unchanged. Use `_rgqa_lambda(epoch)` for the auxiliary multiplier. Epoch values are zero-based global epochs.

## QCC and auxiliary losses

`qcc_classification` accepts logits, a binary matched-class mask, detached matching IoUs, the current negative exponent, the existing branch's GT normalizer and an optional auxiliary-query mask. Masks must broadcast to the logits. The target is matched IoU and the weights are positive IoU or negative sigmoid-confidence raised to the negative exponent.

`rgqs_auxiliary_losses` accepts only selected query logits/boxes and their matched GTs. Pass current IoU (not refinement-aware quality) for the classification soft target and the original criterion's `gamma` as `target_gamma`. Compute matched GIoU externally without detaching the predicted boxes. The denominator is the batch GT normalizer after the detector's original distributed reduction/clamping, not the number of auxiliaries. Empty selected tensors yield zero losses connected to their inputs. Multiply each returned loss by its configured loss coefficient and the RGQS schedule once.

The standalone tensor interfaces are newly separated code, not the full experiment criterion. They do not implement the external decoder, denoising, FGL/DDF or other original detector losses.

## LJO and experiment settings

Use `lgrm_cost_class(epoch)` and `mal_gamma(epoch)` from `gqdet.ljo`. Apply any dataset-specific `matching_class_cost_overrides` through `matching_class_cost(config, epoch, scheduled_value)`. Change the existing matcher's classification cost and QCC's negative exponent only; do not change the soft-target exponent.

Preserve all other experiment settings in configs/, including optimizer parameter groups, warmup, learning-rate schedule, data augmentation, EMA, training budget and validation-based checkpoint selection. Names of external/custom classes in these configuration records do not imply those classes are included in this package. Exact end-to-end reproduction requires implementing the integration and the remaining configured training components in a separately obtained detector environment.
