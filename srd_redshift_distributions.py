#  Niko Sarcevic
#  nikolina.sarcevic@gmail.com
#  github/nikosarcevic
#  ----------

import numpy as np
from numpy import exp
import pandas
from scipy.integrate import simpson
import yaml


# noinspection PyDefaultArgument
class SRDRedshiftDistributions(object):
    """
        Generate the LSST DESC type redshift distributions
        for lens and source sample for year 1 and year 10.
        See the LSST DESC Science Requirements Document (SRD)
        https://arxiv.org/abs/1809.01669. The model used here
        is the Smail type redshift distribution. This class
        reads the parameters automatically from a yaml file
        included in this repository (lsst_desc_parameters.yaml).
        ...
        Attributes
        ----------
        redshift_range: array
        galaxy_sample: string
            galaxy sample for which the redshift distribution will be
            calculated. Accepted values are "source_galaxies" and
            "lens_galaxies".
        forecast_year: string
            year that corresponds to the SRD forecast. Accepted values
            are "1" and "10"
         """

    def __init__(self,
                 redshift_range,
                 galaxy_sample={},
                 forecast_year={}):

        self.redshift_range = redshift_range

        supported_galaxy_samples = {"lens_sample", "source_sample"}
        if galaxy_sample in supported_galaxy_samples:
            self.galaxy_sample = galaxy_sample
        else:
            raise ValueError(f"galaxy_sample must be one of {supported_galaxy_samples}.")

        supported_forecast_years = {"1", "10"}
        if forecast_year in supported_forecast_years:
            self.forecast_year = forecast_year
        else:
            raise ValueError(f"forecast_year must be one of {supported_forecast_years}.")

        # Read in the LSST DESC redshift distribution parameters
        with open("parameters/lsst_desc_parameters.yaml", "r") as f:
            lsst_desc_parameters = yaml.load(f, Loader=yaml.FullLoader)

        self.srd_parameters = lsst_desc_parameters[self.galaxy_sample][self.forecast_year]

    def smail_type_distribution(self,
                                redshift_range,
                                pivot_redshift=None,
                                alpha=None,
                                beta=None):

        """
        Generate the LSST DESC SRD parametric redshift distribution (Smail-type).
        For details check LSST DESC SRD paper https://arxiv.org/abs/1809.01669, equation 5.
        The redshift distribution parametrisation is a smail type of the form
        N(z) = (z / z0) ^ beta * exp[- (z / z0) ^ alpha],
        where z is redshift, z0 is pivot redshift, and alpha and beta are power law indices.
        ----------
        Arguments:
            redshift_range: array
                redshift range
            pivot_redshift: float
                pivot redshift
            alpha: float
                power law index in the exponent
            beta: float
                power law index in the prefactor
        Returns:
            redshift_distribution: array
                A Smail-type redshift distribution over a range of redshifts.
                """

        if not pivot_redshift:
            pivot_redshift = self.srd_parameters["z_0"]
        if not alpha:
            alpha = self.srd_parameters["alpha"]
        if not beta:
            beta = self.srd_parameters["beta"]

        redshift_distribution = [z ** beta * exp(-(z / pivot_redshift) ** alpha) for z in redshift_range]

        return redshift_distribution

    def get_redshift_distribution(self,
                                  redshift_range=None,
                                  normalised=True,
                                  save_file=True,
                                  file_format="npy"):
        """
        Generate the LSST type redshift distribution
        for lens and source sample for year 1 and year 10.
        See the LSST DESC Science Requirements Document (SRD)
        https://arxiv.org/abs/1809.01669, eq. 5. The model is
        the Smail type redshift distribution of the form
        N(z) = (z / z0) ^ beta * exp[- (z / z0) ^ alpha] where
        z is the redshift, z0 is the pivot redshift, and
        beta and alpha are power law parameters. LSST DESC
        has a set of z0, beta, and alpha parameters for
        lens and source galaxy sample for year 1 and year 10
        science requirements. The value of the parameters can
        be found in the SRD paper. The parameters are automatically
        read from a yaml file included in this repository
        (lsst_desc_parameters.yaml).
        ----------
        Arguments:
            redshift_range: array
                an array of redshifts over which the redshift distribution
                will be defined. If not specified, the SRD default will
                be used (redshift interval 0.01 < z < 4.).
            normalised: bool
                normalise the redshift distribution (defaults to True)
            save_file: bool
                option to save the output as a .csv (defaults to True).
                Saves the redshift range and the corresponding redshift
                distribution to a .csv file (with the redshift and the
                redshift distribution columns).
            file_format: string
                file format of the output file (defaults to .npy).
                Accepts .npy and .csv.
        Returns:
            srd_redshift_distribution (array):
                an LSST DESC SRD redshift distribution of a galaxy sample
                for a chosen forecast year.
        """

        # If redshift range is not specified, use the default LSST DESC redshift range
        if not redshift_range:
            redshift_range = self.redshift_range

        # Generate the LSST DESC SRD redshift distribution
        redshift_distribution = self.smail_type_distribution(redshift_range)

        # Normalise the redshift distribution
        if normalised:
            normalisation = np.array(simpson(redshift_distribution, redshift_range))
            redshift_distribution = np.array(redshift_distribution / normalisation)

        combined_data = {"redshift": redshift_range, "dndz": redshift_distribution}
        # Save the redshift distribution to a .csv file

        if save_file:
            self.save_to_file(combined_data, file_format)

        return redshift_distribution

    def save_to_file(self, data, file_format="npy"):

        if file_format == "npy":
            np.save(f"./srd_{self.galaxy_sample}_dndz_year_{self.forecast_year}.npy", data)
        elif file_format == "csv":
            dndz_df = pandas.DataFrame(data)
            dndz_df.to_csv(f"./srd_{self.galaxy_sample}_dndz_year_{self.forecast_year}.csv", index=False)
