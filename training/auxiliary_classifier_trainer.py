import time
from tqdm.auto import tqdm

import torch
from torch import nn

import utils
from training.classification_trainer import ClassificationTrainer


class AuxiliaryClassifierTrainer(ClassificationTrainer):
    """
    Trainer pour modèles avec classifiers auxiliaires (ex: GoogleLeNet/Inception).

    Hérite de ClassificationTrainer et override train_one_epoch() et evaluate()
    pour gérer les sorties multiples pendant l'entraînement.

    Features:
    - Détection automatique du nombre de classifiers auxiliaires
    - Poids configurables via YAML (aux_classifier_weights)
    - Métriques séparées pour chaque classifier (loss, acc1, acc5)
    - Désactivation automatique des auxiliaires en eval
    """

    def __init__(self, args):
        super().__init__(args)

        # Validation et lecture des poids des classifiers auxiliaires
        if not hasattr(args, 'aux_classifier_weights'):
            raise ValueError(
                "Configuration must contain 'aux_classifier_weights' list. "
                "Example: aux_classifier_weights: [0.3, 0.3]"
            )

        self.aux_weights = args.aux_classifier_weights

        if not isinstance(self.aux_weights, list) or len(self.aux_weights) == 0:
            raise ValueError(
                f"aux_classifier_weights must be a non-empty list, got: {self.aux_weights}"
            )

        print(f"[AuxiliaryClassifierTrainer] Initialized with {len(self.aux_weights)} "
              f"auxiliary classifiers, weights: {self.aux_weights}")

    def train_one_epoch(self, epoch):
        """
        Training loop avec gestion des classifiers auxiliaires.

        Calcule la loss totale comme:
        total_loss = main_loss + sum(weight_i * aux_loss_i)

        Logger les métriques pour chaque classifier séparément.
        """
        self.model.train()

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch}/{self.args.epochs} [Train]"
        )

        # Initialize metrics pour le main classifier et chaque auxiliaire
        total_loss = 0.0
        total_loss_main = 0.0
        total_losses_aux = [0.0] * len(self.aux_weights)

        total_correct_1_main = 0.0
        total_correct_5_main = 0.0
        total_correct_1_aux = [0.0] * len(self.aux_weights)
        total_correct_5_aux = [0.0] * len(self.aux_weights)

        total_samples = 0

        start_time = time.time()

        for i, (image, target) in pbar:
            # Data preparation
            image, target = image.to(self.device), target.to(self.device)
            prepare_time = time.time() - start_time

            # Forward pass - retourne (main_output, aux1_output, aux2_output, ...)
            outputs = self.model(image)

            # Validation du nombre de sorties
            if not isinstance(outputs, tuple):
                raise RuntimeError(
                    f"Model must return tuple in training mode, got {type(outputs)}"
                )

            num_aux_classifiers = len(outputs) - 1
            if num_aux_classifiers != len(self.aux_weights):
                raise ValueError(
                    f"Model returned {num_aux_classifiers} auxiliary classifiers, "
                    f"but config specifies {len(self.aux_weights)} weights. "
                    f"Please update aux_classifier_weights in config."
                )

            # Séparer main et auxiliaires
            main_output = outputs[0]
            aux_outputs = outputs[1:]

            # Calculer loss pour le main classifier
            loss_main = self.criterion(main_output, target)

            # Calculer losses pour les classifiers auxiliaires
            losses_aux = []
            for aux_output in aux_outputs:
                loss_aux = self.criterion(aux_output, target)
                losses_aux.append(loss_aux)

            # Loss totale = main + weighted sum des auxiliaires
            loss_total = loss_main
            for weight, loss_aux in zip(self.aux_weights, losses_aux):
                loss_total += weight * loss_aux

            # Backward pass
            self.optimizer.zero_grad()
            loss_total.backward()

            if self.args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

            self.optimizer.step()

            # Compute metrics pour le main classifier
            acc1_main, acc5_main = utils.metrics.accuracy(main_output, target, topk=(1, 5))

            # Compute metrics pour chaque auxiliaire
            accs_1_aux = []
            accs_5_aux = []
            for aux_output in aux_outputs:
                acc1_aux, acc5_aux = utils.metrics.accuracy(aux_output, target, topk=(1, 5))
                accs_1_aux.append(acc1_aux)
                accs_5_aux.append(acc5_aux)

            # Update metrics
            batch_size = image.shape[0]
            total_loss += loss_total.item() * batch_size
            total_loss_main += loss_main.item() * batch_size

            for idx, loss_aux in enumerate(losses_aux):
                total_losses_aux[idx] += loss_aux.item() * batch_size

            total_correct_1_main += acc1_main.item() * batch_size / 100.0
            total_correct_5_main += acc5_main.item() * batch_size / 100.0

            for idx in range(len(self.aux_weights)):
                total_correct_1_aux[idx] += accs_1_aux[idx].item() * batch_size / 100.0
                total_correct_5_aux[idx] += accs_5_aux[idx].item() * batch_size / 100.0

            total_samples += batch_size

            # Update tensorboard - Main classifier
            self.writer.add_scalar('train/loss_total', loss_total.item(), self.n_iter)
            self.writer.add_scalar('train/loss_main', loss_main.item(), self.n_iter)
            self.writer.add_scalar('train/acc1_main', acc1_main.item(), self.n_iter)
            self.writer.add_scalar('train/acc5_main', acc5_main.item(), self.n_iter)

            # Update tensorboard - Auxiliary classifiers
            for idx in range(len(self.aux_weights)):
                self.writer.add_scalar(f'train/loss_aux{idx+1}', losses_aux[idx].item(), self.n_iter)
                self.writer.add_scalar(f'train/acc1_aux{idx+1}', accs_1_aux[idx].item(), self.n_iter)
                self.writer.add_scalar(f'train/acc5_aux{idx+1}', accs_5_aux[idx].item(), self.n_iter)

            self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]["lr"], self.n_iter)

            # Compute computation time / efficiency
            process_time = time.time() - start_time - prepare_time
            compute_efficiency = process_time / (process_time + prepare_time)

            # Update progress bar - afficher les métriques principales
            pbar.set_postfix({
                'loss': f'{total_loss/total_samples:.4f}',
                'main_acc@1': f'{total_correct_1_main/total_samples*100:.2f}%',
                'aux1_acc@1': f'{total_correct_1_aux[0]/total_samples*100:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'eff': f'{compute_efficiency:.2%}'
            })

            start_time = time.time()
            self.n_iter += 1

    def evaluate(self, epoch):
        """
        Evaluation utilisant SEULEMENT le classifier principal.

        Le modèle en mode eval retourne un seul tensor (pas de tuple).
        Métriques identiques au ClassificationTrainer standard.
        """
        self.model.eval()

        total_loss = 0.0
        correct_1 = 0
        correct_5 = 0
        total_samples = 0

        header = f"Epoch {epoch}/{self.args.epochs} [Val]"
        pbar = tqdm(self.test_loader, desc=header)

        with torch.no_grad():
            for image, target in pbar:
                # Data preparation
                image = image.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                # Forward - en mode eval, retourne seulement main_output (pas de tuple)
                output = self.model(image)

                # Validation que c'est bien un tensor simple
                if isinstance(output, tuple):
                    raise RuntimeError(
                        "Model returned tuple in eval mode. "
                        "Ensure model.forward() returns single tensor when self.training=False"
                    )

                loss = self.criterion(output, target)

                # Compute metrics
                acc1, acc5 = utils.metrics.accuracy(output, target, topk=(1, 5))

                batch_size = image.shape[0]
                total_loss += loss.item() * batch_size
                correct_1 += acc1.item() * batch_size / 100.0
                correct_5 += acc5.item() * batch_size / 100.0
                total_samples += batch_size

                # Update progress
                pbar.set_postfix({
                    'loss': f'{total_loss/total_samples:.4f}',
                    'acc@1': f'{correct_1/total_samples*100:.2f}%',
                    'acc@5': f'{correct_5/total_samples*100:.2f}%'
                })

        avg_loss = total_loss / total_samples
        avg_acc1 = correct_1 / total_samples * 100.0
        avg_acc5 = correct_5 / total_samples * 100.0

        # Update tensorboard
        if self.writer:
            self.writer.add_scalar('eval/loss', avg_loss, epoch)
            self.writer.add_scalar('eval/acc1', avg_acc1, epoch)
            self.writer.add_scalar('eval/acc5', avg_acc5, epoch)

        # is best model ?
        is_best = avg_acc1 > self.best_acc1

        # Save checkpoint
        metrics = {
            'loss': avg_loss,
            'acc1': avg_acc1,
            'acc5': avg_acc5,
        }
        self.save_checkpoint_with_metrics(epoch, metrics, is_best=is_best)

        # Update best accuracy
        if is_best:
            self.best_acc1 = avg_acc1

        return avg_acc1
