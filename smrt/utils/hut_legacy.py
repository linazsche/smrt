# coding: utf-8

"""Wrapper to original HUT matlab using SMRT framework. To use this module, extra installation are needed:

* get HUT. Decompress the archive somewhere on your disk.

* in the file snowemis_nlayers change the 6 occurences of the "do" variable into "dos" because it causes a syntax error in Octave.

* install the oct2py module using :code:`pip install oct2py` or :code:`easy_install install oct2py`.

* install Octave version 3.6 or above.

* for convenience you can set the HUT_DIR environment variable to point to HUT path. This path can also be programmatically set with :py:func:`set_hut_path`.

In case of problem check the instructions given in http://blink1073.github.io/oct2py/source/installation.html

"""

import os
from collections.abc import Sequence

import numpy as np
import xarray as xr

from oct2py import octave

from smrt.core.result import PassiveResult, concat_results
from smrt.core.globalconstants import DENSITY_OF_ICE, FREEZING_POINT

# python-space path to n-layer HUT
_hut_path = None


def set_hut_path(path):
    """set the path where HUT archive has been uncompressed, i.e. where the file `snowemis_nlayer.m` is located."""
    global _hut_path

    if path != _hut_path:
        # octave.restoredefaultpath() # risk of bad interference
        octave.addpath(path)
        octave.addpath(os.path.dirname(__file__))
        _hut_path = path


try:
    # set
    set_hut_path(os.environ["HUT_DIR"])
except KeyError:
    pass


def run(sensor, snowpack, ke_option=0, grainsize_option=1, nout=1, hut_path=None):
    """call HUT for the snowpack and sensor configuration given as argument. Any microstructure model that defines the "radius" parameter
    is valid.

    :param snowpack: describe the snowpack.
    :param sensor: describe the sensor configuration.
    :param ke_option: see HUT snowemis_nlayers.m code
    :param grainsize_option: see HUT snowemis_nlayers.m code
    """

    if hut_path is not None:
        set_hut_path(hut_path)

    if isinstance(snowpack, Sequence):
        result_list = [run(sensor, sp, ke_option=ke_option, grainsize_option=grainsize_option) for sp in snowpack]
        return concat_results(result_list, ("snowpack", range(len(snowpack))))

    if snowpack.substrate is not None:
        Tg = snowpack.substrate.temperature
        # roughness_rms = getattr(snowpack.substrate, "roughness_rms", 0)
        try:
            # soil
            roughness_rms = snowpack.substrate.roughness_rms * 1000  # (mm)
            soil_eps = snowpack.substrate.permittivity(sensor.frequency)
        except AttributeError:
            # water
            roughness_rms = 1  # (mm)
            soil_eps = np.nan

    else:
        Tg = FREEZING_POINT
        roughness_rms = 0  # specular
        soil_eps = 1

    snow = []  # snow is a N Layer (snowpack+soil) row and 8 columns. Each colum has a data (see snowemis_nlayer)
    enough_warning = False

    # snow and ice layers
    for lay in snowpack.layers:
        # density = lay.frac_volume * DENSITY_OF_ICE
        # snow.append((lay.temperature - FREEZING_POINT, lay.thickness * density, 2000 * lay.microstructure.radius, density / 1000,
        #  lay.liquid_water, lay.salinity, 0, 0))
        try:
            # ice
            if lay.ice_type == "fresh":
                density = np.nan
                liquid_water = 0
                Weq = lay.thickness * 1000  # (mm)
        except AttributeError:
            # snow
            density = lay.density / 1000  # (g/cm^3)
            liquid_water = lay.liquid_water
            Weq = (lay.thickness * 1000) * density  # (mm)

        snow.append(
            (
                lay.temperature - FREEZING_POINT,
                Weq,
                2000 * lay.microstructure.radius,
                density,
                liquid_water * 100,
                lay.salinity,
                0,
                0,
            )
        )

        if lay.salinity and enough_warning:
            print("Warning: salinity in HUT is ppm")
            enough_warning = True

    # substrate
    snow.append((Tg - FREEZING_POINT, 0, 0, 0, 0, 0, roughness_rms, soil_eps))

    thetad = np.degrees(sensor.theta)

    # coords = [('theta', sensor.theta), ('polarization', sensor.polarization)]
    coords = [("theta", sensor.theta), ("polarization", ["V", "H"])]

    if nout == 1:
        outV = np.zeros((0, sensor.theta.shape[0]))
        outH = np.zeros((0, sensor.theta.shape[0]))
    else:
        outV = np.zeros((sensor.theta.shape[0], nout))
        outH = np.zeros((sensor.theta.shape[0], nout))

    # run MATLAB code separately for polarizations
    for pol in sensor.polarization:
        if pol == "V":
            outV = [
                octave.snowemis_nlayer(
                    otulo, np.array(snow), sensor.frequency / 1e9, 0, ke_option, grainsize_option, nout=nout
                )
                for otulo in thetad
            ]
        elif pol == "H":
            outH = [
                octave.snowemis_nlayer(
                    otulo, np.array(snow), sensor.frequency / 1e9, 1, ke_option, grainsize_option, nout=nout
                )
                for otulo in thetad
            ]

    if nout == 1:
        # [Tb, __] = snowemis_nlayer(...)
        TbV = outV
        TbH = outH

        Tb = xr.DataArray(
            np.vstack((TbV, TbH)).T,
            attrs=dict(
                mode=sensor.mode,
            ),
            coords=coords,
        )
        # Teff = None

        return PassiveResult(Tb)

    else:
        # [Tb, Teff] = snowemis_nlayer(...)
        TbV = np.array(outV)[:, 0]
        TbH = np.array(outH)[:, 0]
        TeffV = np.array(outV)[:, 1]
        TeffH = np.array(outH)[:, 1]

        Tb = xr.DataArray(
            np.stack((TbV, TbH), axis=-1),
            attrs=dict(
                mode=sensor.mode,
            ),
            coords=coords,
        )
        Teff = xr.DataArray(
            np.stack((TeffV, TeffH), axis=-1),
            attrs=dict(
                mode=sensor.mode,
            ),
            coords=coords,
        )

        return PassiveResult(Tb), Teff
