from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.utils.tensorboard.writer import SummaryWriter


class TrainingEngine:
    def __init__(self, model, device, train_loader, val_loader, hyperparams, directory):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Core logic
        self.model = model.to(device)
        self.directory = directory + "/" + timestamp
        self.writer = SummaryWriter(self.directory + "/verbose")

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Hyperparameters
        self.lr = hyperparams["learning_rate"]
        self.batch_size = hyperparams["batch_size"]
        self.max_epochs = hyperparams["max_epochs"]
        self.patience = hyperparams["early_stopping_patience"]
        self.gradient_clip_norm = hyperparams["gradient_clip_norm"]
        self.energy_weight = hyperparams["loss_weights"]["energy"]
        self.force_weight = hyperparams["loss_weights"]["force"]
        self.nac_weight = hyperparams["loss_weights"]["nac"]
        self.dipole_weight = hyperparams["loss_weights"]["dipole"]

        # Optimizer
        self.optimizer = Adam(model.parameters(), lr=self.lr)

    def loss_function(self, batch, energies, mu, nac):
        # Reshape batch.energies to match predicted energies shape
        batch_size = energies.shape[0]
        num_states = energies.shape[1]
        target_energies = batch.energies.view(batch_size, num_states)

        loss_E = F.mse_loss(energies, target_energies)

        loss_mu = 0.0
        if hasattr(batch, "transition_dipoles"):
            loss_mu = F.mse_loss(mu, batch.transition_dipoles)

        loss_nac = 0.0
        if hasattr(batch, "nac"):
            # Reshape batch.nac to match predicted nac shape
            # batch.nac shape is (batch_size * num_coupling_pairs, num_atoms, 3)
            # nac shape is (batch_size, num_coupling_pairs, 3) - molecule-level NAC
            # We need to aggregate over atoms for target NAC
            num_coupling_pairs = nac.shape[1]
            num_atoms = batch.nac.shape[1]
            target_nac_per_atom = batch.nac.view(
                batch_size, num_coupling_pairs, num_atoms, 3
            )
            # Sum over atoms to get molecule-level NAC
            target_nac = target_nac_per_atom.sum(dim=2)
            loss_nac = F.mse_loss(nac, target_nac)

        # Forces from ground state only
        loss_F = torch.tensor(0.0, device=energies.device)

        # Only compute forces if pos requires gradients (i.e., during training)
        if batch.pos.requires_grad:
            E0 = energies[:, 0].sum()
            grad_outputs = torch.autograd.grad(
                E0, batch.pos, create_graph=True, allow_unused=True
            )[0]

            if grad_outputs is not None:
                forces = -grad_outputs
                # batch.forces shape is (batch_size * num_states, num_atoms, 3)
                # We only compute forces for ground state, so take ground state forces
                target_forces = batch.forces.view(batch_size, num_states, -1, 3)[
                    :, 0
                ]  # Ground state forces
                loss_F = F.mse_loss(forces, target_forces)

        loss = (
            self.energy_weight * loss_E
            + self.dipole_weight * loss_mu
            + self.force_weight * loss_F
            + self.nac_weight * loss_nac
        )

        return loss

    def training_step(self, batch):
        """
        Single training step for the MANA model.
        batch: A batch of molecular graphs with attributes:
            - pos: Atomic positions (num_atoms, 3)
            - energies: Reference energies (num_molecules, num_singlet_states + 1)
            - forces: Reference forces (num_atoms, 3)
            - transition_dipoles: (optional) Reference transition dipole moments (num_molecules, 3)
            - nac: (optional) Reference non-adiabatic couplings (num_molecules, num_coupling_pairs, 3)
        Returns:
            - loss: Computed loss value.
            - loss_E: Computed loss value for energy.
            - loss_F: Computed loss value for force.
            - loss_nac: Computed loss value for non-adiabatic couplings.
            - loss_mu: Computed loss values for dipoles.
        """
        batch.pos.requires_grad_(True)
        self.optimizer.zero_grad()

        energies, mu, nac = self.model(batch)

        loss = self.loss_function(batch, energies, mu, nac)
        loss.backward()

        self.optimizer.step()

        return loss

    def train_epoch(self, index):
        """
        Trains one full epoch
        """
        running_loss = 0.0
        last_loss = 0.0

        for i, data in enumerate(self.train_loader):
            loss = self.training_step(data)
            running_loss += loss.item()

            if i % 100 == 99:
                last_loss = running_loss / 100
                print(f"     batch {i + 1} loss: {last_loss}")
                tb_x = index * len(self.train_loader) + i + 1
                self.writer.add_scalar("Loss/train", last_loss, tb_x)
                running_loss = 0.0

        return last_loss

    def train(self):
        epoch_number = 0
        best_vloss = 1_000_000.0

        for epoch in range(self.max_epochs):
            print(f"EPOCH {epoch_number + 1}")

            self.model.train()
            avg_loss = self.train_epoch(epoch_number)

            running_vloss = 0.0
            self.model.eval()

            with torch.no_grad():
                for i, vdata in enumerate(self.val_loader):
                    venergies, vmu, vnac = self.model(vdata)
                    vloss = self.loss_function(vdata, venergies, vmu, vnac)
                    running_vloss += vloss.item()

            avg_vloss = running_vloss / len(self.val_loader)
            print(f"LOSS: train {avg_loss}, validation {avg_vloss}")

            self.writer.add_scalars(
                "Training vs. Validation Loss",
                {"Training": avg_loss, "Validation": avg_vloss},
                epoch_number + 1,
            )
            self.writer.flush()

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss

                torch.save(self.model.state_dict(), f"{self.directory}/best_model.pth")
                print(f"  Best model saved with validation loss: {best_vloss}")

            epoch_number += 1
