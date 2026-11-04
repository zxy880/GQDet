"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import datetime
import json
import math
import time

import torch

from ..misc import dist_utils, stats
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch
from .final_schedule import apply_runtime_schedule, matching_class_cost


class DetSolver(BaseSolver):
    def fit(self):
        self.train()
        args = self.cfg
        metric_names = ["AP50:95", "AP50", "AP75", "APsmall", "APmedium", "APlarge"]
        best_ap5095_only = bool(args.yaml_cfg.get("best_ap5095_only", False))
        best_checkpoint_name = args.yaml_cfg.get(
            "best_checkpoint_name", "best_ap5095.pth"
        )
        reload_best_at_stop_epoch = bool(
            args.yaml_cfg.get("reload_best_at_stop_epoch", True)
        )
        save_last_all_stages = bool(args.yaml_cfg.get("save_last_all_stages", False))
        checkpoint_epochs = set(args.yaml_cfg.get("checkpoint_epochs", []))
        final_ema_restart = bool(args.yaml_cfg.get("final_ema_restart", False))
        final_ema_retain_history = bool(args.yaml_cfg.get("final_ema_retain_history", False))
        final_ema_start_epoch = int(args.yaml_cfg.get("final_ema_start_epoch", -1))
        final_ema_decay = float(args.yaml_cfg.get("final_ema_decay", args.yaml_cfg.get("ema", {}).get("decay", 0.9999)))
        final_ema_warmups = int(args.yaml_cfg.get("final_ema_warmups", args.yaml_cfg.get("ema", {}).get("warmups", 1000)))
        final_ema_restarted = False
        trainfix_global_lr = bool(args.yaml_cfg.get("trainfix_global_epoch_lr", False))
        # Historical experimental schedules are opt-in only.  A normal
        # baseline must retain the matcher and criterion constants from YAML.
        enable_trainfix_lgrm = bool(args.yaml_cfg.get("enable_trainfix_lgrm", False))
        validate_raw_from_epoch = int(args.yaml_cfg.get("validate_raw_from_epoch", 80))
        validate_raw_every = int(args.yaml_cfg.get("validate_raw_every", 2))
        best_raw_checkpoint_name = args.yaml_cfg.get(
            "best_raw_checkpoint_name", "best_val_ap5095_raw.pth"
        )
        best_raw_ap5095 = -1.0

        def lgrm_cost_class(global_epoch):
            """Classification matching-cost schedule in global epochs."""
            if global_epoch <= 54:
                return 2.0
            if global_epoch <= 69:
                progress = (global_epoch - 55) / 14.0
                return 1.0 + 0.5 * (1.0 + math.cos(math.pi * progress))
            return 1.0

        def mal_gamma(global_epoch):
            """Only MAL's negative modulation changes; quality targets do not."""
            if global_epoch <= 54:
                return 1.5
            if global_epoch <= 79:
                progress = (global_epoch - 55) / 24.0
                return 2.0 - 0.25 * (1.0 + math.cos(math.pi * progress))
            return 2.0

        def log_lrs(global_epoch):
            detail = ", ".join(
                f"{pg.get('group_name', 'unnamed')}={pg['lr']:.10g}"
                for pg in self.optimizer.param_groups
            )
            print(f"[LR] global_epoch={global_epoch} {detail}", flush=True)

        if self.use_wandb:
            import wandb

            wandb.init(
                project=args.yaml_cfg["project_name"],
                name=args.yaml_cfg["exp_name"],
                config=args.yaml_cfg,
            )
            wandb.watch(self.model)

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-" * 42 + "Start training" + "-" * 43)
        top1 = 0
        best_stat = {
            "epoch": -1,
        }
        if best_ap5095_only and self.output_dir:
            log_path = self.output_dir / "log.txt"
            if log_path.exists():
                with log_path.open() as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            value = record["test_coco_eval_bbox"][0]
                            if value > top1:
                                top1 = value
                                best_stat = {
                                    "epoch": record["epoch"],
                                    "coco_eval_bbox": value,
                                }
                        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                            continue

        def update_primary_best(eval_epoch, test_stats):
            nonlocal top1, best_stat
            current = float(test_stats["coco_eval_bbox"][0])
            if current > top1:
                top1 = current
                best_stat = {
                    "epoch": eval_epoch,
                    "coco_eval_bbox": current,
                }
                if self.output_dir:
                    dist_utils.save_on_master(
                        self.state_dict(), self.output_dir / best_checkpoint_name
                    )
            print(f"best_stat: {best_stat}")

        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                self.last_epoch,
                self.use_wandb
            )
            if best_ap5095_only:
                update_primary_best(self.last_epoch, test_stats)
            else:
                for k in test_stats:
                    best_stat["epoch"] = self.last_epoch
                    best_stat[k] = test_stats[k][0]
                    top1 = test_stats[k][0]
                    print(f"best_stat: {best_stat}")

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epochs):
            apply_runtime_schedule(self, epoch)
            # The policy is tied to the true global epoch, never an internal
            # scheduler offset.  Iteration warmup owns the first 500 successful
            # optimizer updates and then this policy takes over.
            if trainfix_global_lr and (
                self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished()
            ):
                self.lr_scheduler.set_epoch(epoch)
            log_lrs(epoch)
            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if (
                epoch == self.train_dataloader.collate_fn.stop_epoch
                and reload_best_at_stop_epoch
            ):
                self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                if self.ema:
                    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                    print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            if (
                final_ema_restart
                and self.ema
                and not final_ema_restarted
                and (epoch == final_ema_start_epoch or
                     (final_ema_retain_history and epoch > final_ema_start_epoch))
            ):
                if not final_ema_retain_history:
                    raw_state = dist_utils.de_parallel(self.model).state_dict()
                    self.ema.module.load_state_dict(raw_state, strict=True)
                    self.ema.updates = 0
                self.ema.decay = final_ema_decay
                self.ema.warmups = final_ema_warmups
                if final_ema_warmups == 0:
                    self.ema.decay_fn = lambda x, decay=final_ema_decay: decay
                else:
                    self.ema.decay_fn = lambda x, decay=final_ema_decay, warmups=final_ema_warmups: decay * (
                        1 - math.exp(-x / warmups)
                    )
                final_ema_restarted = True
                print(
                    f"[Final EMA] {'retain history' if final_ema_retain_history else 'restart'} at epoch {epoch}: "
                    f"decay={self.ema.decay} warmups={self.ema.warmups} updates={self.ema.updates}"
                )

            # Preserve fixed baseline matcher/loss settings unless an
            # configuration enables the LJO schedule.
            matcher = getattr(self.criterion, "matcher", None)
            if matcher is None:
                raise RuntimeError("Detection criterion requires criterion.matcher")
            if enable_trainfix_lgrm:
                matcher.cost_class = matching_class_cost(args.yaml_cfg, epoch, lgrm_cost_class(epoch))
                if not hasattr(self.criterion, "set_mal_gamma"):
                    raise RuntimeError("LGRM schedule requires criterion.set_mal_gamma")
                self.criterion.set_mal_gamma(mal_gamma(epoch))
                print(
                    f"[LGRM] global_epoch={epoch} matcher_cost_class={matcher.cost_class:.10f} "
                    f"cost_bbox={float(matcher.cost_bbox):.1f} cost_giou={float(matcher.cost_giou):.1f} "
                    f"mal_gamma={self.criterion.mal_gamma:.6f}",
                    flush=True,
                )
            else:
                stable_gamma = getattr(self.criterion, "mal_gamma", self.criterion.gamma)
                print(
                    f"[Baseline] global_epoch={epoch} matcher_cost_class={matcher.cost_class:.10f} "
                    f"cost_bbox={float(matcher.cost_bbox):.1f} cost_giou={float(matcher.cost_giou):.1f} "
                    f"gamma={float(stable_gamma):.6f}",
                    flush=True,
                )

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                epochs=args.epochs,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                use_wandb=self.use_wandb,
                output_dir=self.output_dir,
            )
            train_stats["matcher_cost_class"] = float(matcher.cost_class)
            train_stats["mal_gamma"] = float(self.criterion.mal_gamma)

            if (not trainfix_global_lr) and (
                self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished()
            ):
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and (
                epoch < self.train_dataloader.collate_fn.stop_epoch
                or save_last_all_stages
            ):
                checkpoint_paths = [self.output_dir / "last.pth"]
                completed_epochs = epoch + 1
                if completed_epochs in checkpoint_epochs:
                    checkpoint_paths.append(
                        self.output_dir / f"checkpoint_epoch{completed_epochs:04}.pth"
                    )
                elif not checkpoint_epochs and completed_epochs % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                epoch,
                self.use_wandb,
                output_dir=self.output_dir,
            )

            if best_ap5095_only:
                for k in test_stats:
                    if self.writer and dist_utils.is_main_process():
                        for i, v in enumerate(test_stats[k]):
                            self.writer.add_scalar(f"Test/{k}_{i}".format(k), v, epoch)
                update_primary_best(epoch, test_stats)

                # From epoch 80 onward, additionally inspect raw weights every
                # two epochs without replacing EMA as the default validation.
                if epoch >= validate_raw_from_epoch and (
                    (epoch - validate_raw_from_epoch) % validate_raw_every == 0
                ):
                    raw_stats, _ = evaluate(
                        self.model,
                        self.criterion,
                        self.postprocessor,
                        self.val_dataloader,
                        self.evaluator,
                        self.device,
                        epoch,
                        self.use_wandb,
                        output_dir=self.output_dir,
                    )
                    raw_ap5095 = float(raw_stats["coco_eval_bbox"][0])
                    train_stats["raw_AP50_95"] = raw_ap5095
                    train_stats["raw_AP75"] = float(raw_stats["coco_eval_bbox"][2])
                    if raw_ap5095 > best_raw_ap5095:
                        best_raw_ap5095 = raw_ap5095
                        if self.output_dir:
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / best_raw_checkpoint_name
                            )
                    print(
                        f"[TrainFix-Raw] epoch={epoch} AP50:95={raw_ap5095:.6f} "
                        f"best_raw={best_raw_ap5095:.6f}",
                        flush=True,
                    )
            else:
                # Legacy D-FINE two-stage checkpoint behavior.
                for k in test_stats:
                    if self.writer and dist_utils.is_main_process():
                        for i, v in enumerate(test_stats[k]):
                            self.writer.add_scalar(f"Test/{k}_{i}".format(k), v, epoch)

                    if k in best_stat:
                        best_stat["epoch"] = (
                            epoch if test_stats[k][0] > best_stat[k] else best_stat["epoch"]
                        )
                        best_stat[k] = max(best_stat[k], test_stats[k][0])
                    else:
                        best_stat["epoch"] = epoch
                        best_stat[k] = test_stats[k][0]

                    if best_stat[k] > top1:
                        best_stat_print["epoch"] = epoch
                        top1 = best_stat[k]
                        if self.output_dir:
                            if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                                dist_utils.save_on_master(
                                    self.state_dict(), self.output_dir / "best_stg2.pth"
                                )
                            else:
                                dist_utils.save_on_master(
                                    self.state_dict(), self.output_dir / "best_stg1.pth"
                                )

                    best_stat_print[k] = max(best_stat[k], top1)
                    print(f"best_stat: {best_stat_print}")  # global best

                    if best_stat["epoch"] == epoch and self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            if test_stats[k][0] > top1:
                                top1 = test_stats[k][0]
                                dist_utils.save_on_master(
                                    self.state_dict(), self.output_dir / "best_stg2.pth"
                                )
                        else:
                            top1 = max(test_stats[k][0], top1)
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg1.pth"
                            )

                    elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        best_stat = {
                            "epoch": -1,
                        }
                        if self.ema:
                            self.ema.decay -= 0.0001
                            self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                            print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if self.use_wandb:
                wandb_logs = {}
                for idx, metric_name in enumerate(metric_names):
                    wandb_logs[f"metrics/{metric_name}"] = test_stats["coco_eval_bbox"][idx]
                wandb_logs["epoch"] = epoch
                wandb.log(wandb_logs)

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def val(self):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            epoch=-1,
            use_wandb=False,
        )

        if self.output_dir:
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )

        return
