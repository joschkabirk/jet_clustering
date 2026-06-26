"""Module with functions related to calculating jet substructure."""

import logging

import awkward as ak
import fastjet
import numpy as np
import vector

vector.register_awkward()

pylogger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def calc_deltaR(particles, jet):
    jet = ak.unflatten(ak.flatten(jet), counts=1)
    return particles.deltaR(jet)


class JetSubstructure:
    """Class to calculate and store the jet substructure variables.

    Definitions as in slide 7 here:
    https://indico.cern.ch/event/760557/contributions/3262382/attachments/1796645/2929179/lltalk.pdf
    """

    def __init__(
        self,
        particles,
        R=0.8,
        beta=1.0,
        use_wta_pt_scheme=False,
    ):
        """Run the jet clustering and calculate the substructure variables. The clustering is
        performed with the kt algorithm and the WTA pt scheme.

        Parameters
        ----------
        particles : awkward array
            The particles that are clustered into jets. Have to be vector Momentum4D objects
        R : float, optional
            The jet radius, by default 0.8
        beta : float, optional
            The beta parameter for N-subjettiness, by default 1.0
        use_wta_pt_scheme : bool, optional
            Whether to use the WTA pt scheme for the clustering, by default False
        """

        pylogger.info(f"Calculating substructure for {len(particles)} jets")
        mask_too_few_particles = ak.num(particles) < 3
        n_jets_with_nparticles_too_small = ak.sum(mask_too_few_particles)
        if n_jets_with_nparticles_too_small > 0:
            pylogger.info(
                f"There are {n_jets_with_nparticles_too_small} jets with less than 3 particles."
            )
            raise ValueError("Jets with too few particles are not allowed.")

        self.R = R
        self.beta = beta
        self.particles = particles
        self.particles_sum = ak.sum(particles, axis=1)
        self.jet_mass = self.particles_sum.mass
        self.jet_pt = self.particles_sum.pt
        self.jet_eta = self.particles_sum.eta
        self.jet_phi = self.particles_sum.phi
        self.jet_n_constituents = ak.num(particles)

        if use_wta_pt_scheme:
            jetdef = fastjet.JetDefinition(
                fastjet.kt_algorithm, self.R, fastjet.WTA_pt_scheme
            )
        else:
            jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, self.R)
        pylogger.info("Clustering jets with fastjet")
        pylogger.info(f"Jet definition: {jetdef}")
        self.cluster = fastjet.ClusterSequence(particles, jetdef)
        self.inclusive_jets = self.cluster.inclusive_jets()
        self.exclusive_jets_1 = self.cluster.exclusive_jets(n_jets=1)
        self.exclusive_jets_2 = self.cluster.exclusive_jets(n_jets=2)
        self.exclusive_jets_3 = self.cluster.exclusive_jets(n_jets=3)

        pylogger.info("Calculating N-subjettiness")
        self._calc_d0()
        self._calc_tau1()
        self._calc_tau2()
        self._calc_tau3()
        self.tau21 = self.tau2 / self.tau1
        self.tau32 = self.tau3 / self.tau2
        pylogger.info("Calculating D2")
        # D2 as defined in https://arxiv.org/pdf/1409.6298.pdf
        self.d2 = self.cluster.exclusive_jets_energy_correlator(njets=1, func="d2")

    def _calc_d0(self):
        """Calculate the d0 values."""
        self.d0 = ak.sum(self.particles.pt * self.R**self.beta, axis=1)

    def _calc_tau1(self):
        """Calculate the tau1 values."""
        self.delta_r_1i = calc_deltaR(self.particles, self.exclusive_jets_1[:, :1])
        self.pt_i = self.particles.pt
        # calculate the tau1 values
        self.tau1 = ak.sum(self.pt_i * self.delta_r_1i**self.beta, axis=1) / self.d0

    def _calc_tau2(self):
        """Calculate the tau2 values."""
        delta_r_1i = calc_deltaR(self.particles, self.exclusive_jets_2[:, :1])
        delta_r_2i = calc_deltaR(self.particles, self.exclusive_jets_2[:, 1:2])
        self.pt_i = self.particles.pt
        # add new axis to make it broadcastable
        min_delta_r = ak.min(
            ak.concatenate(
                [
                    delta_r_1i[..., np.newaxis] ** self.beta,
                    delta_r_2i[..., np.newaxis] ** self.beta,
                ],
                axis=-1,
            ),
            axis=-1,
        )
        self.tau2 = ak.sum(self.pt_i * min_delta_r, axis=1) / self.d0

    def _calc_tau3(self):
        """Calculate the tau3 values."""
        delta_r_1i = calc_deltaR(self.particles, self.exclusive_jets_3[:, :1])
        delta_r_2i = calc_deltaR(self.particles, self.exclusive_jets_3[:, 1:2])
        delta_r_3i = calc_deltaR(self.particles, self.exclusive_jets_3[:, 2:3])
        self.pt_i = self.particles.pt
        min_delta_r = ak.min(
            ak.concatenate(
                [
                    delta_r_1i[..., np.newaxis] ** self.beta,
                    delta_r_2i[..., np.newaxis] ** self.beta,
                    delta_r_3i[..., np.newaxis] ** self.beta,
                ],
                axis=-1,
            ),
            axis=-1,
        )
        self.tau3 = ak.sum(self.pt_i * min_delta_r, axis=1) / self.d0

    def get_subjettiness_illustration_data(self, jet_idx: int, N: int = 2) -> dict:
        """Return data needed to illustrate the tau_N subjettiness calculation for a single jet.

        Parameters
        ----------
        jet_idx : int
            Index of the jet to illustrate.
        N : int, optional
            Number of subjets (1, 2, or 3), by default 2.

        Returns
        -------
        dict with keys:
            "constituents": dict with "eta", "phi", "pt" arrays (one entry per particle)
            "subjet_axes": dict with "eta", "phi" arrays (one entry per subjet axis)
            "subjet_assignment": int array — index of the nearest subjet axis per particle
            "distances": float array — ΔR to the nearest subjet axis per particle (the
                         values that enter the tau_N sum, before the beta exponent)
        """
        if N not in (1, 2, 3):
            raise ValueError("N must be 1, 2, or 3")

        exclusive_jets_map = {
            1: self.exclusive_jets_1,
            2: self.exclusive_jets_2,
            3: self.exclusive_jets_3,
        }
        axes = exclusive_jets_map[N][jet_idx]  # shape: (N,)
        particles = self.particles[jet_idx]  # shape: (n_particles,)

        # ΔR from each particle to each subjet axis using vector's deltaR
        # axes[i] is a scalar; wrap it in a length-1 array so deltaR can broadcast
        delta_r_per_axis = np.stack(
            [
                np.array(particles.deltaR(ak.unflatten(axes[i : i + 1], counts=1)[0]))
                for i in range(N)
            ],
            axis=1,
        )  # shape: (n_particles, N)

        subjet_assignment = np.argmin(
            delta_r_per_axis, axis=1
        )  # nearest axis per particle
        distances = delta_r_per_axis[np.arange(len(particles)), subjet_assignment]

        return {
            "constituents": {
                "eta": np.array(particles.eta),
                "phi": np.array(particles.phi),
                "pt": np.array(particles.pt),
            },
            "subjet_axes": {
                "eta": np.array(axes.eta),
                "phi": np.array(axes.phi),
            },
            "subjet_assignment": subjet_assignment,
            "distances": distances,
        }

    def get_substructure_as_ak_array(self):
        """Return the substructure variables as a dictionary."""
        return ak.Array(
            {
                "tau1": self.tau1,
                "tau2": self.tau2,
                "tau3": self.tau3,
                "tau21": self.tau21,
                "tau32": self.tau32,
                "d2": self.d2,
                "jet_mass": self.jet_mass,
                "jet_pt": self.jet_pt,
                "jet_eta": self.jet_eta,
                "jet_phi": self.jet_phi,
                "jet_n_constituents": self.jet_n_constituents,
            }
        )
