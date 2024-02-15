"""
Single-column time-varying-updraft framework with moisture advection handled by
[PyMPDATA](http://github.com/open-atmos/PyMPDATA/)
"""

import numpy as np

from PySDM.environments.impl.moist import Moist

from PySDM.impl import arakawa_c
from PySDM import Formulae
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.initialisation.equilibrate_wet_radii import equilibrate_wet_radii


class Kinematic1D(Moist):
    def __init__(self, *, dt, mesh, thd_of_z, rhod_of_z, z0=0):
        super().__init__(dt, mesh, [])
        self.thd0 = thd_of_z(z0 + mesh.dz * arakawa_c.z_scalar_coord(mesh.grid))
        self.rhod = rhod_of_z(z0 + mesh.dz * arakawa_c.z_scalar_coord(mesh.grid))
        self.formulae = Formulae()

    def register(self, builder):
        super().register(builder)
        self.formulae = builder.particulator.formulae
        rhod = builder.particulator.Storage.from_ndarray(self.rhod)
        self._values["current"]["rhod"] = rhod
        self._tmp["rhod"] = rhod

    def get_water_vapour_mixing_ratio(self) -> np.ndarray:
        return self.particulator.dynamics["EulerianAdvection"].solvers.advectee.get()

    def get_thd(self) -> np.ndarray:
        return self.thd0

    def init_attributes(
        self, *, spatial_discretisation, n_sd_per_mode, nz_tot, aerosol_modes_by_kappa
    ):
        super().sync()
        self.notify()

        attributes = {
            k: np.empty(0) for k in ("dry volume", "kappa times dry volume", "multiplicity")
        }
        with np.errstate(all="raise"):
            positions = spatial_discretisation.sample(
                backend=self.particulator.backend,
                grid=self.mesh.grid,
                n_sd= self.particulator.n_sd,
            )
            (
                attributes["cell id"],
                attributes["cell origin"],
                attributes["position in cell"],
            ) = self.mesh.cellular_attributes(positions)

            rhod = self["rhod"].to_ndarray()
            cell_id = attributes["cell id"]
            domain_volume = np.prod(np.array(self.mesh.size))
            
            for i, (kappa, spectrum) in enumerate(aerosol_modes_by_kappa.items()):
                sampling = ConstantMultiplicity(spectrum)
                r_dry, n_per_kg = sampling.sample(backend=self.particulator.backend, n_sd= n_sd_per_mode[i]*nz_tot)
                v_dry = self.formulae.trivia.volume(radius=r_dry)
                attributes["dry volume"] = np.append(attributes["dry volume"], v_dry)
                attributes["kappa times dry volume"] = np.append(attributes["kappa times dry volume"], v_dry * kappa)
                ind_start= i*int(len(cell_id)/2)
                ind_end= (i+1)*int(len(cell_id)/2)
                attributes["multiplicity"]= np.append(attributes["multiplicity"], n_per_kg * rhod[cell_id[ind_start:ind_end]] * domain_volume)
                #np.random.choice(rhod[cell_id], len(n_per_kg))
            r_wet = equilibrate_wet_radii(
                r_dry= self.formulae.trivia.radius(volume=attributes["dry volume"]),
                environment=self,
                cell_id=attributes["cell id"],
                kappa_times_dry_volume=attributes["kappa times dry volume"],
            )
        
        attributes["volume"]= self.formulae.trivia.volume(radius=r_wet)
        
        return attributes

    @property
    def dv(self):
        return self.mesh.dv
