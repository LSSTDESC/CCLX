#  Niko Sarcevic
#  nikolina.sarcevic@gmail.com
#  github.com/nikosarcevic
#  ----------


import numpy as np
import pandas
from scipy.integrate import simpson, cumtrapz
from scipy.special import erf
import yaml


class Binning:

    def __init__(self,
                 redshift_range,
                 redshift_distribution,
                 forecast_year={}):
        """
        Performs the slicing of the input redshift distribution into tomographic bins.
        The binning algorithm follows the LSST DESC prescription. For more details, see
        the LSST DESC Science Requirements Document (SRD) Appendix D (link to paper:
        https://arxiv.org/abs/1809.01669).
        The methods allow for slicing of the initial redshift distribution into a source or
        lens galaxy sample for the appropriate LSST DESC forecast year (year 1 or year 10).

        ...
        Attributes
        ----------
        redshift_range: array
            An interval of redshifts for which
            the redshift distribution is generated
        redshift_distribution: array
            A redshift distribution over given
            redshift_range
        forecast_year: str
            year that corresponds to the SRD forecast. Accepted values
            are "1" and "10"
        """

        supported_forecast_years = {"1", "10"}
        if forecast_year in supported_forecast_years:
            self.forecast_year = forecast_year
        else:
            raise ValueError(f"forecast_year must be one of {supported_forecast_years}.")

        self.redshift_range = redshift_range
        self.redshift_distribution = redshift_distribution
        self.forecast_year = forecast_year

        with open("parameters/lsst_desc_parameters.yaml", "r") as f:
            self.lsst_parameters = yaml.load(f, Loader=yaml.FullLoader)

        self.lens_params = self.lsst_parameters["lens_sample"][self.forecast_year]
        self.source_params = self.lsst_parameters["source_sample"][self.forecast_year]

    def true_redshift_distribution(self, upper_edge, lower_edge, variance, bias):
        """A function that returns the true redshift distribution of a galaxy sample.
         The true distribution of galaxies is defined as a convolution of an overall galaxy redshift distribution and
         a probability distribution p(z_{ph}|z)  at a given z (z_{ph} is a photometric distribution at a given z).
         Overall galaxy redshift distribution is a Smail type distribution (n(z) = (z/z_0)^alpha exp[-(z/z_0)^beta]).
         The true distribution defined here is following Ma, Hu & Huterer 2018
          (see https://arxiv.org/abs/astro-ph/0506614 eq. 6).

           Arguments:
               upper_edge (float): upper edge of the redshift bin
               lower_edge (float): lower edge of the redshift bin
               variance (float): variance of the photometric distribution
               bias (float): bias of the photometric distribution
            Returns:
                true_redshift_distribution (array): true redshift distribution of a galaxy sample"""
        # Calculate the scatter
        scatter = variance * (1 + self.redshift_range)
        # Calculate the upper and lower limits of the integral
        lower_limit = (upper_edge - self.redshift_range + bias) / np.sqrt(2) / scatter
        upper_limit = (lower_edge - self.redshift_range + bias) / np.sqrt(2) / scatter

        # Calculate the true redshift distribution
        true_redshift_distribution = 0.5 * np.array(self.redshift_distribution) * (erf(upper_limit) - erf(lower_limit))

        return true_redshift_distribution

    def compute_equal_number_bounds(self, redshift_range, redshift_distribution, n_bins):
        """
        Determines the redshift values that divide the distribution into bins
        with an equal number of galaxies.

        Arguments:
            redshift_range (array): an array of redshift values
            redshift_distribution (array): the corresponding redshift distribution defined over redshift_range
            n_bins (int): the number of tomographic bins

        Returns:
            An array of redshift values that are the boundaries of the bins.
        """

        # Calculate the cumulative distribution
        cumulative_distribution = cumtrapz(redshift_distribution, redshift_range, initial=0)
        total_galaxies = cumulative_distribution[-1]

        # Find the bin edges
        bin_edges = []
        for i in range(1, n_bins):
            fraction = i / n_bins * total_galaxies
            # Find the redshift value where the cumulative distribution crosses this fraction
            bin_edge = np.interp(fraction, cumulative_distribution, redshift_range)
            bin_edges.append(bin_edge)

        return [redshift_range[0]] + bin_edges + [redshift_range[-1]]

    def source_bins(self, normalised=True, save_file=True, file_format='npy'):
        """split the initial redshift distribution of source galaxies into tomographic bins.
        LSST DESC case, sources are split into 5 tomographic bins (year 1 and year 10 forecast).
        Each bin has equal number of galaxies. Variance is 0.05 for both forecast years while z_bias is zero.
        For more information about the redshift distributions and binning,
        consult the LSST DESC Science Requirements Document (SRD) https://arxiv.org/abs/1809.01669,
        Appendix D.
        ----------
        Arguments:
            normalised (bool): normalise the redshift distribution (defaults to True).
            save_file (bool): option to save the output as a .csv (defaults to True).
                Saves the redshift range and the corresponding redshift
                distributions for each bin. The bin headers are the
                dictionary keys.
            file_format (str): format of the output file (defaults to 'npy').
                Accepted values are 'csv' and 'npy'.
        Returns:
            A source galaxy sample (dictionary), appropriately binned."""
        # Get the bin edges as redshift values directly
        bins = self.compute_equal_number_bounds(self.redshift_range,
                                                self.redshift_distribution,
                                                self.source_params["n_tomo_bins"])

        # Get the bias and variance values for each bin
        source_z_bias_list = np.repeat(self.source_params["z_bias"],
                                       self.source_params["n_tomo_bins"])
        source_z_variance_list = np.repeat(self.source_params["sigma_z"],
                                           self.source_params["n_tomo_bins"])

        # Create a dictionary of the redshift distributions for each bin
        source_redshift_distribution_dict = {}
        # Loop over the bins: each bin is defined by the upper and lower edge of the bin
        for index, (x1, x2) in enumerate(zip(bins[:-1], bins[1:])):
            z_bias = source_z_bias_list[index]
            z_variance = source_z_variance_list[index]
            source_redshift_distribution_dict[index] = self.true_redshift_distribution(x1, x2, z_variance, z_bias)

        # Normalise the distributions
        if normalised:
            norm_factor = []
            for key in sorted(source_redshift_distribution_dict.keys()):
                norm_factor.append(simpson(source_redshift_distribution_dict[key], self.redshift_range))
                source_redshift_distribution_dict[key] /= norm_factor[-1]

            # Create a combined dictionary
        combined_data = {'redshift_range': self.redshift_range,
                         'bins': source_redshift_distribution_dict}

        # Save the data
        if save_file:
            self.save_to_file(combined_data, "source", file_format)

        return source_redshift_distribution_dict

    def lens_bins(self,
                  normalised=True,
                  save_file=True,
                  file_format='npy'):
        """
        Split the initial redshift distribution of lens galaxies (lenses) into tomographic bins.
        In the LSST DESC case, lenses are split into 5 tomographic bins (year 1 forecast) or 10
        tomographic bins (year 10). Binning is performed in such a way that the bins are spaced
        by 0.1 in photo-z between 0.2 ≤ z ≤ 1.2 for Y10, and 5 bins spaced by 0.2 in photo-z in
        the same redshift range.
        Variance is 0.03 for both forecast years while z_bias is zero.
        For more information about the redshift distributions and binning,
        consult the LSST DESC Science Requirements Document (SRD) https://arxiv.org/abs/1809.01669,
        Appendix D.
        ----------
        Arguments:
            normalised: bool
                normalise the redshift distribution (defaults to True).
            save_file: bool
                option to save the output as a .csv (defaults to True).
                Saves the redshift range and the corresponding redshift
                distributions for each bin. The bin headers are the
                dictionary keys.
            file_format: str (defaults to 'npy')
                format of the output file. Accepted values are 'csv' and 'npy'.
        Returns: dictionary
                A lens galaxy sample, appropriately binned. Depending on the forecast year
                chosen while initialising the class, it will output a lens sample for year 1
                (5 bins) or lens galaxy sample for year 10 (10 bins).
        """
        # Define the bin edges
        bins = np.arange(self.lens_params["bin_start"],
                         self.lens_params["bin_stop"] + self.lens_params["bin_spacing"],
                         self.lens_params["bin_spacing"])

        # Get the bias and variance values for each bin
        lens_z_bias_list = np.repeat(self.lens_params["z_bias"],
                                     self.lens_params["n_tomo_bins"])
        lens_z_variance_list = np.repeat(self.lens_params["sigma_z"],
                                         self.lens_params["n_tomo_bins"])

        # Create a dictionary of the redshift distributions for each bin
        lens_redshift_distribution_dict = {}
        for index, (x1, x2) in enumerate(zip(bins[:-1], bins[1:])):
            z_bias = lens_z_bias_list[index]
            z_variance = lens_z_variance_list[index]
            lens_redshift_distribution_dict[index] = self.true_redshift_distribution(x1, x2, z_variance, z_bias)

        # Normalise the distributions
        if normalised:
            norm_factor = []
            for i, key in enumerate(list(sorted(lens_redshift_distribution_dict.keys()))):
                norm_factor.append(simpson(lens_redshift_distribution_dict[key], self.redshift_range))
                lens_redshift_distribution_dict[key] /= norm_factor[i]

        combined_data = {'redshift_range': self.redshift_range,
                         'bins': lens_redshift_distribution_dict}

        # Save the distributions to a file
        if save_file:
            self.save_to_file(combined_data, "lens", file_format)

        return lens_redshift_distribution_dict

    def get_bin_centers(self, decimal_places=2, save_file=True):
        """Method to calculate the bin centers for the source and lens galaxy samples.
        The bin centers are calculated as the redshift value where
        the redshift distribution is maximised.
        The bin centers are rounded to the specified number of decimal places.

        Arguments:
            decimal_places (int): number of decimal places to round the bin centers to (defaults to 2)
            save_file (bool): option to save the output as a .npy file (defaults to True)
        Returns: a nested dictionary of bin centers for source and lens galaxy samples
         for year 1 and year 10 forecast (keys are the forecast years).
            """
        bin_centers = {"sources": [], "lenses": []}

        # Calculate bin centers for sources
        source_bins = self.source_bins(normalised=True, save_file=False)
        for index in range(self.source_params["n_tomo_bins"]):
            bin_center = self.find_bin_center(source_bins[index], self.redshift_range, decimal_places)
            bin_centers["sources"].append(bin_center)

        # Calculate bin centers for lenses
        lens_bins = self.lens_bins(normalised=True, save_file=False)
        for index in range(self.lens_params["n_tomo_bins"]):
            bin_center = self.find_bin_center(lens_bins[index], self.redshift_range, decimal_places)
            bin_centers["lenses"].append(bin_center)

        if save_file:
            # Save to .npy file if save_file is True
            np.save(f'./srd_bin_centers_y_{self.forecast_year}.npy', bin_centers)

        return bin_centers

    def find_bin_center(self, bin_distribution, redshift_range, decimal_places=2):
        """Helper method to calculate and round the bin center."""
        max_index = np.argmax(bin_distribution)
        return round(redshift_range[max_index], decimal_places)

    def save_to_file(self, data, name, file_format="npy"):

        if file_format == "npy":
            np.save(f"./srd_{name}_bins_year_{self.forecast_year}.npy", data)
        elif file_format == "csv":
            dndz_df = pandas.DataFrame(data)
            dndz_df.to_csv(f"./srd_{name}_bins_year_{self.forecast_year}.csv", index=False)
