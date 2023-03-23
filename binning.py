#  Niko Sarcevic
#  nikolina.sarcevic@gmail.com
#  github.com/nikosarcevic
#  ----------

from math import fsum
import numpy
import pandas
from scipy.integrate import simpson
from scipy.stats import uniform, norm
import yaml


class Gaussian(object):
    """
    A class that computes a probability distribution function.

    ...
        Attributes
        ----------
            interval: The interval over which the distribution is normalised.
            spread: The width of the distribution.
            displacement: The deviation from the given mean.
        Returns:
            A class object containing a normalised gaussian probability
            distribution function, corresponding to P(x|y). The argument
            of the function is the mean.
    """

    def __init__(self, interval, spread, displacement):
        self.interval = interval
        self.spread = spread
        self.displacement = displacement

    def get_distribution(self, x):
        """
        Returns a gaussian roughly centred around x.
        The gaussian is normalised over self.interval.
        Arguments:
            x: A real number
        Returns:
            A gaussian with mean (x - displacement),
            normalised over interval.
        """

        if x < numpy.min(self.interval) or x > numpy.max(self.interval):
            raise ValueError(f"Value {x} lies outside of given interval {self.interval}.")
        rv = norm()
        standard_deviation = self.spread * (1 + x)
        mean = x - self.displacement
        exponent = (numpy.array(self.interval) - mean) / standard_deviation
        distribution = rv.pdf(exponent) / standard_deviation
        # The distribution is normalised over (-∞, ∞), but not over the given interval.
        # It is therefore renormalised in the output.
        return distribution / numpy.trapz(distribution, self.interval)


class NormaliseDistribution(object):
    """
    Constructs a normalised distribution out of a set of values
    defined over a given interval.

    ...
        Attributes
        ----------
            interval: The interval over which the
                      distribution is defined.
            distribution_values: An array that makes up the
                                 distribution.
        Returns: A normalised distribution.
    """

    def __init__(self, interval, distribution_values):
        self.interval = interval
        self.distribution_values = distribution_values

    def get_distribution(self):
        norm_fac = numpy.trapz(self.distribution_values, self.interval)
        normalised_distribution = self.distribution_values / norm_fac
        return normalised_distribution


class ConvolveDistributions(object):
    """
    Takes distributions p(x|y) and q(y) and computes
                 p(x) ∝ ∫ p(x|y) q(y) dy.
    Note that the final distribution is not normalised.
    The integration should be done over restricted ranges
    of y. This is implemented by imposing a set of filters
    F, such that q(y) = F(y) * p(y).
    Parameters:
        initial_distribution: class object
            A NormaliseDistribution class object.
        conditional_distribution:   class object
            A Gaussian class object.
        filters: array
            An array of window functions, i.e. an array of arrays
            with values with limited support.
    Returns:
        The unnormalised joint described above.
    """

    def __init__(self, initial_distribution, conditional_distribution, filters):
        self.conditional_distribution = conditional_distribution
        self.initial_distribution = initial_distribution
        self.filter_list = filters

    def get_distribution(self):
        return numpy.array([self.compute_distribution(el) for el in self.filter_list])

    def compute_distribution(self, tomo_filter):
        ys = self.initial_distribution.interval
        xs = self.conditional_distribution.interval

        initial_pdf = self.initial_distribution.get_distribution()
        conditional_pdf = lambda x: self.conditional_distribution.get_distribution(x)
        joint_pdf = numpy.zeros((len(xs), len(ys)))

        # This defines the integrand p(x|y) p(y) in the integral above
        for i, y in enumerate(ys):
            joint_pdf[:, i] = conditional_pdf(y) * initial_pdf[i] * tomo_filter[i]

        # This performs the actual integration over y
        final_distribution = numpy.zeros((len(xs),))
        for i in range(len(xs)):
            final_distribution[i] = numpy.trapz(joint_pdf[i, :], ys)

        return final_distribution


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

        with open("parameters/lsst_desc_parameters.yaml", "r") as f:
            self.lsst_parameters = yaml.load(f, Loader=yaml.FullLoader)

        self.redshift_range = redshift_range
        self.redshift_distribution = redshift_distribution
        self.forecast_year = forecast_year

        self.lens_params = self.lsst_parameters["lens_sample"][self.forecast_year]
        self.source_params = self.lsst_parameters["source_sample"][self.forecast_year]

        self.n_tomobins_lens = self.lens_params["n_tomo_bins"]
        self.n_tomobins_source = self.source_params["n_tomo_bins"]

        self.z_start_lens = self.lens_params["bin_start"]
        self.z_stop_lens = self.lens_params["bin_stop"]
        self.z_spacing_lens = self.lens_params["bin_spacing"]

        self.galaxy_bias = self.lens_params["galaxy_bias_values"]
        self.z_variance = self.source_params["z_variance"]
        self.z_bias = self.source_params["z_bias"]
        self.sigmaz_lens = self.lens_params["sigma_z"]

    def compute_equal_area_bounds(self, redshift_range, redshift_distribution, bin_number):
        """
        Determines the index values of areas of equal probability.

        Arguments:
            redshift_range : an array of floats
            redshift_distribution : an array of floats
            bin_number : number of intervals (default: 5)

        Returns:
            An array of indices that represent the boundary
            values of the areas of equal probability.
        """

        xs = redshift_range
        ys = redshift_distribution
        step_size = xs[1] - xs[0]

        # Ensure that the redshift distribution is normalised
        norm_distribution = simpson(ys, xs)
        ys /= norm_distribution

        q = 1 / bin_number
        quantile_bounds = []
        probability_sum = 0
        for y in ys:
            upper_bound = numpy.where(ys == y)[0][0]
            probability_sum += step_size * y
            if probability_sum >= q:
                quantile_bounds.append(upper_bound)
                probability_sum = 0

        return [0] + quantile_bounds + [len(xs) - 1]

    def source_bins(self,
                    normalised=True,
                    save_file=True):
        """
        Split the initial redshift distribution of source galaxies (sources) into tomographic bins.
        In LSST DESC case, sources are split into 5 tomographic bins where each bin
        has equal number of galaxies. This holds for both year 1 and year 10 forecast.
        For more information about the redshift distributions and binning, consult the LSST DESC
        Science Requirements Document (SRD) https://arxiv.org/abs/1809.01669, Appendix D.
        ----------
        Arguments:
            normalised: bool
                normalise the redshift distribution (defaults to True).
            save_file: bool
                option to save the output as a .csv (defaults to True).
                Saves the redshift range and the corresponding redshift
                distributions for each bin. The bin headers are the
                dictionary keys.
        Returns: dictionary
            A redshift distribution of source galaxies binned into five tomographic bins
        """

        bins = [self.redshift_range[i] for i in
                self.compute_equal_area_bounds(self.redshift_range,
                                               self.redshift_distribution,
                                               self.n_tomobins_source)]
        bin_centers = [0.5 * fsum([bins[i] + bins[i + 1]]) for i in range(len(bins[:-1]))]
        pdf_z = NormaliseDistribution(self.redshift_range, numpy.array(self.redshift_distribution))

        z_bias_list = numpy.repeat(self.z_bias, self.n_tomobins_source)
        z_variance_list = numpy.repeat(self.z_variance, self.n_tomobins_source)

        source_redshift_distribution_dict = {}
        for index, (x1, x2) in enumerate(zip(bins[:-1], bins[1:])):
            z_bias = z_bias_list[index]
            z_variance = z_variance_list[index]
            core = Gaussian(self.redshift_range, spread=z_variance, displacement=z_bias)
            tomofilter = uniform.pdf(self.redshift_range, loc=x1, scale=x2-x1)
            photoz_model = ConvolveDistributions(pdf_z, core, [tomofilter])
            source_redshift_distribution_dict[bin_centers[index]] = photoz_model.get_distribution()[0]
        if normalised:
            norm_factor = []
            for i, key in enumerate(list(sorted(source_redshift_distribution_dict.keys()))):
                norm_factor.append(simpson(source_redshift_distribution_dict[key], self.redshift_range))
                source_redshift_distribution_dict[key] = source_redshift_distribution_dict[key] / norm_factor[i]

        if save_file:
            z_df = pandas.DataFrame({"z": self.redshift_range})
            dndz_df = pandas.DataFrame(source_redshift_distribution_dict)
            lens_df = pandas.concat([z_df, dndz_df], axis=1)
            lens_df.to_csv(f"source_bins_srd_y{self.forecast_year}.csv", index=False)

        return source_redshift_distribution_dict

    def lens_bins(self,
                  normalised=True,
                  save_file=True):
        """
        Split the initial redshift distribution of lens galaxies (lenses) into tomographic bins.
        In the LSST DESC case, lenses are split into 5 tomographic bins (year 1 forecast) or 10
        tomographic bins (year 10). Binning is performed in such a way that the bins are spaced
        by 0.1 in photo-z between 0.2 ≤ z ≤ 1.2 for Y10, and 5 bins spaced by 0.2 in photo-z in
        the same redshift range. For more information about the redshift distributions and binning,
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
        Returns: dictionary
                A lens galaxy sample, appropriately binned. Depending on the forecast year
                chosen while initialising the class, it will output a lens sample for year 1
                (5 bins) or lens galaxy sample for year 10 (10 bins).
        """

        bins = numpy.arange(self.z_start_lens,
                            self.z_stop_lens + self.z_spacing_lens,
                            self.z_spacing_lens)

        bin_centers = [0.5 * fsum([bins[i] + bins[i + 1]]) for i in range(len(bins[:-1]))]
        pdf_z = NormaliseDistribution(self.redshift_range, numpy.array(self.redshift_distribution))
        lens_redshift_distribution_dict = {}
        for index, (x1, x2) in enumerate(zip(bins[:-1], bins[1:])):
            core = Gaussian(self.redshift_range, spread=self.sigmaz_lens, displacement=0.)
            tomofilter = uniform.pdf(self.redshift_range, loc=x1, scale=x2-x1)
            photoz_model = ConvolveDistributions(pdf_z, core, [tomofilter])
            lens_redshift_distribution_dict[bin_centers[index]] = photoz_model.get_distribution()[0]

        if normalised:
            norm_factor = []
            for i, key in enumerate(list(sorted(lens_redshift_distribution_dict.keys()))):
                norm_factor.append(simpson(lens_redshift_distribution_dict[key], self.redshift_range))
                lens_redshift_distribution_dict[key] = lens_redshift_distribution_dict[key] / norm_factor[i]

        if save_file:
            z_df = pandas.DataFrame({"z": self.redshift_range})
            dndz_df = pandas.DataFrame(lens_redshift_distribution_dict)
            lens_df = pandas.concat([z_df, dndz_df], axis=1)
            lens_df.to_csv(f"lens_bins_srd_y{self.forecast_year}.csv", index=False)

        return lens_redshift_distribution_dict
