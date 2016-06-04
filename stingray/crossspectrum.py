__all__ = ["Crossspectrum", "AveragedCrossspectrum"]

import numpy as np
import scipy
import scipy.stats
import scipy.fftpack
import scipy.optimize

import stingray.lightcurve as lightcurve
import stingray.utils as utils

class Crossspectrum(object):

    def __init__(self, lc1=None, lc2=None, norm='none'):
        """
        Make a cross spectrum from a (binned) light curve.
        You can also make an empty Crossspectrum object to populate with your
        own fourier-transformed data (this can sometimes be useful when making
        binned periodograms).

        Parameters
        ----------
        lc1: lightcurve.Lightcurve object, optional, default None
            The first light curve data for the channel/band of interest.

        lc2: lightcurve.Lightcurve object, optional, default None
            The light curve data for the reference band.

        norm: {'frac', 'abs', 'leahy', 'none'}, default 'none'
            The normalization of the (real part of the) cross spectrum.

        Attributes
        ----------
        freq: numpy.ndarray
            The array of mid-bin frequencies that the Fourier transform samples

        power: numpy.ndarray
            The array of cross spectra (complex numbers)

        df: float
            The frequency resolution

        m: int
            The number of averaged cross-spectra amplitudes in each bin.

        n: int
            The number of data points/time bins in one segment of the light
            curves.

        nphots1: float
            The total number of photons in light curve 1

        nphots2: float
            The total number of photons in light curve 2
        """

        assert isinstance(norm, str), "norm is not a string!"

        assert norm.lower() in ["frac", "abs", "leahy", "none"], \
                "norm must be 'frac', 'abs', 'leahy', or 'none'!"

        self.norm = norm.lower()

        ## check if input data is a Lightcurve object, if not make one or
        ## make an empty Crossspectrum object if lc1 == None or lc2 == None
        if lc1 is None or lc2 is None:
            if lc1 is not None or lc2 is not None:
                 raise TypeError("You can't do a cross spectrum with just one "
                         "light curve!")
            # else:
            #      print("Please specify input light curves!")
            self.freq = None
            self.power = None
            self.df = None
            self.nphots1 = None
            self.nphots2 = None
            self.m = 1
            self.n = None
            return

        self._make_crossspectrum(lc1, lc2)

    def _make_crossspectrum(self, lc1, lc2):

        ## make sure the inputs work!
        assert isinstance(lc1, lightcurve.Lightcurve), \
                        "lc1 must be a lightcurve.Lightcurve object!"
        assert isinstance(lc2, lightcurve.Lightcurve), \
                        "lc2 must be a lightcurve.Lightcurve object!"


        ## total number of photons is the sum of the
        ## counts in the light curve
        self.nphots1 = np.sum(lc1.counts)
        print("nphots1: " + str(self.nphots1))
        self.nphots2 = np.sum(lc2.counts)
        print("nphots2: " + str(self.nphots2))


        self.meancounts1 = np.mean(lc1.counts)
        print("meancounts1: " + str(self.meancounts1))
        self.meancounts2 = np.mean(lc2.counts)
        print("meancounts2: " + str(self.meancounts2))

        ## the number of data points in the light curve
        assert lc1.counts.shape[0] == lc2.counts.shape[0], \
            "Light curves do not have same number of time bins per segment."
        assert lc1.dt == lc2.dt, \
                "Light curves do not have same time binning dt."
        self.n = lc1.counts.shape[0]

        ## the frequency resolution
        self.df = 1.0/lc1.tseg

        ## the number of averaged periodograms in the final output
        ## This should *always* be 1 here
        self.m = 1

        ## make the actual Fourier transform and compute cross spectrum
        self.freq, self.unnorm_power = self._fourier_cross(lc1, lc2)

        ## If co-spectrum is desired, normalize here. Otherwise, get raw back
        ## with the imaginary part still intact.
        self.power = self._normalize_crossspectrum(self.unnorm_power, lc1.tseg)

    def _fourier_cross(self, lc1, lc2):
        """
        Fourier transform the two light curves, then compute the cross spectrum.
        computed as CS = lc1 x lc2* (where lc2 is the one that gets
        complex-conjugated)

        Parameters
        ----------
        lc1: lightcurve.Lightcurve object
            One light curve to be Fourier transformed. Ths is the band of
            interest or channel of interest.

        lc2: lightcurve.Lightcurve object
            Another light curve to be Fourier transformed. This is the reference
            band.

        Returns
        -------
        fr: numpy.ndarray
            The squared absolute value of the Fourier amplitudes

        """
        fourier_1 = scipy.fftpack.fft(lc1.counts)  # do Fourier transform 1
        fourier_2 = scipy.fftpack.fft(lc2.counts)  # do Fourier transform 2

        print("fourier_1: " + str(fourier_1))
        print("fourier_2: " + str(fourier_2))

        freqs = scipy.fftpack.fftfreq(lc1.counts.shape[0], lc1.dt)
        cross = fourier_1[freqs > 0] * np.conj(fourier_2[freqs > 0])

        print("cross: " + str(cross))

        return freqs[freqs > 0], cross


    def rebin(self, df, method="mean"):
        """
        Rebin the cross spectrum to a new frequency resolution df.

        Parameters
        ----------
        df: float
            The new frequency resolution

        Returns
        -------
        bin_cs = Crossspectrum object
            The newly binned cross spectrum
        """

        # rebin cross spectrum to new resolution
        binfreq, bincs, step_size = utils.rebin_data(self.freq[1:],
                                                     self.power[1:], df,
                                                     method=method)

        # make an empty cross spectrum object
        # note: syntax deliberate to work with subclass Powerspectrum
        bin_cs = self.__class__()

        # store the binned periodogram in the new object
        bin_cs.freq = np.hstack([binfreq[0]-self.df, binfreq])
        bin_cs.power = np.hstack([self.power[0], bincs])
        bin_cs.df = df
        bin_cs.n = self.n
        bin_cs.norm = self.norm
        bin_cs.nphots1 = self.nphots1
        bin_cs.nphots2 = self.nphots2
        bin_cs.m = int(step_size)

        return bin_cs

    def _normalize_crossspectrum(self, unnorm_power, tseg):
        """
        Normalize the real part of the cross spectrum to Leahy, absolute rms^2,
        fractional rms^2 normalization, or not at all.

        Parameters
        ----------
        unnorm_power: numpy.ndarray
            The unnormalized cross spectrum.

        tseg: int
            The length of the Fourier segment, in seconds.

        Returns
        -------
        power: numpy.nd.array
            The normalized co-spectrum (real part of the cross spectrum). For
            'none' normalization, imaginary part is returned as well.
        """

        # The "effective" counst/bin is the geometrical mean of the counts/bin
        # of the two light curves
        actual_nphots = np.sqrt(self.nphots1 * self.nphots2)
        actual_mean = np.sqrt(self.meancounts1 * self.meancounts2)

        assert actual_mean > 0.0, \
                "Mean count rate is <= 0. Something went wrong."

        if self.norm.lower() == 'leahy':
            print("unnorm_powers in _normalize_crossspectrum are: " + str(unnorm_power.real))
            c = unnorm_power.real
            power = c * 2. / actual_nphots
            print("powers in _normalize_crossspectrum are: " + str(power))

        elif self.norm.lower() == 'frac':
            c = unnorm_power.real / np.float(self.n**2.)
            power = c * 2. * tseg / (actual_mean**2.0)

        elif self.norm.lower() == 'abs':
            c = unnorm_power.real / np.float(self.n**2.)
            power = c * (2. * tseg)

        elif self.norm.lower() == 'none':
            power = unnorm_power

        else:
            raise Exception("Normalization not recognized!")

        return power

    def rebin_log(self, f=0.01):
        """
        Logarithmic rebin of the periodogram.
        The new frequency depends on the previous frequency
        modified by a factor f:

        dnu_j = dnu_{j-1}*(1+f)

        Parameters
        ----------
        f: float, optional, default 0.01
            parameter that steers the frequency resolution


        Returns
        -------
        binfreq: numpy.ndarray
            the binned frequencies

        binpower: numpy.ndarray
            the binned powers

        nsamples: numpy.ndarray
            the samples of the original periodogramincluded in each
            frequency bin
        """

        minfreq = self.freq[1] * 0.5  # frequency to start from
        maxfreq = self.freq[-1]  # maximum frequency to end
        binfreq = [minfreq, minfreq + self.df]  # first
        df = self.freq[1]  # the frequency resolution of the first bin

        # until we reach the maximum frequency, increase the width of each
        # frequency bin by f
        while binfreq[-1] <= maxfreq:
            binfreq.append(binfreq[-1] + df*(1.0+f))
            df = binfreq[-1] - binfreq[-2]

        # compute the mean of the powers that fall into each new frequency bin
        binpower, bin_edges, binno = scipy.stats.binned_statistic(
            self.freq, self.power, statistic="mean", bins=binfreq)

        # compute the number of powers in each frequency bin
        nsamples = np.array([len(binno[np.where(binno == i)[0]])
                             for i in range(np.max(binno))])

        # the frequency resolution
        df = np.diff(binfreq)

        # shift the lower bin edges to the middle of the bin and drop the
        # last right bin edge
        binfreq = binfreq[:-1] + df/2

        return binfreq, binpower, nsamples


class AveragedCrossspectrum(Crossspectrum):

    def __init__(self, lc1, lc2, segment_size=1, norm='none'):
        """
        Make an averaged cross spectrum from a light curve by segmenting two
        light curves, Fourier-transforming each segment and then averaging the
        resulting cross spectra.

        Parameters
        ----------
        lc1: lightcurve.Lightcurve object OR
            iterable of lightcurve.Lightcurve objects
            One light curve data to be Fourier-transformed. This is the band
            of interest or channel of interest.

        lc2: lightcurve.Lightcurve object OR
            iterable of lightcurve.Lightcurve objects
            Second light curve data to be Fourier-transformed. This is the
            reference band.

        segment_size: float, default 1 second
            The size of each segment to average. Note that if the total duration
            of each Lightcurve object in lc1 or lc2 is not an integer multiple
            of the segment_size, then any fraction left-over at the end of the
            time series will be lost. Otherwise you introduce artefacts.

        norm: {'frac', 'abs', 'leahy', 'none'}, default 'none'
            The normalization of the (real part of the) cross spectrum.

        Attributes
        ----------
        freq: numpy.ndarray
            The array of mid-bin frequencies that the Fourier transform samples

        power: numpy.ndarray
            The array of cross spectra

        df: float
            The frequency resolution

        m: int
            The number of averaged cross spectra

        n: int
            The number of time bins per segment of light curve?

        nphots1: float
            The total number of photons in the first (interest) light curve

        nphots2: float
            The total number of photons in the second (reference) light curve

        """
        self.type = "crossspectrum"

        print(self.type)

        assert isinstance(norm, str), "norm is not a string!"

        assert norm.lower() in ["frac", "abs", "leahy", "none"], \
                "norm must be 'frac', 'abs', 'leahy', or 'none'!"

        self.norm = norm.lower()

        assert np.isfinite(segment_size), "segment_size must be finite!"

        self.segment_size = segment_size

        Crossspectrum.__init__(self, lc1, lc2, self.norm)

        return

    def _make_segment_spectrum(self, lc1, lc2, segment_size):

        # TODO: need to update this for making cross spectra.
        assert isinstance(lc1, lightcurve.Lightcurve)
        assert isinstance(lc2, lightcurve.Lightcurve)

        assert lc1.dt == lc2.dt, \
            "Light curves do not have same time binning dt."

        assert lc1.tseg == lc2.tseg, "Lightcurves do not have same tseg."

        # number of bins per segment
        nbins = int(segment_size/lc1.dt)
        start_ind = 0
        end_ind = nbins

        cs_all = []
        nphots1_all = []
        nphots2_all = []

        while end_ind <= lc1.counts.shape[0]:
            time_1 = lc1.time[start_ind:end_ind]
            counts_1 = lc1.counts[start_ind:end_ind]
            time_2 = lc2.time[start_ind:end_ind]
            counts_2 = lc2.counts[start_ind:end_ind]
            lc1_seg = lightcurve.Lightcurve(time_1, counts_1)
            lc2_seg = lightcurve.Lightcurve(time_2, counts_2)
            cs_seg = Crossspectrum(lc1_seg, lc2_seg, norm=self.norm)
            cs_all.append(cs_seg)
            nphots1_all.append(np.sum(lc1_seg.counts))
            nphots2_all.append(np.sum(lc2_seg.counts))
            start_ind += nbins
            end_ind += nbins

        return cs_all, nphots1_all, nphots2_all

    def _make_crossspectrum(self, lc1, lc2):

        # chop light curves into segments
        if isinstance(lc1, lightcurve.Lightcurve) and \
                isinstance(lc2, lightcurve.Lightcurve):
            if self.type == "crossspectrum":
                cs_all, nphots1_all, nphots2_all = \
                    self._make_segment_spectrum(lc1, lc2, self.segment_size)

            elif self.type == "powerspectrum":
                cs_all, nphots1_all = \
                    self._make_segment_spectrum(lc1, self.segment_size)

        else:
            cs_all, nphots1_all, nphots2_all = [], [], []
            ## TODO: should be using izip from iterables if lc1 or lc2 could
            ## be long
            for lc1_seg, lc2_seg in zip(lc1, lc2):

                print("self.type: " + str(self.type))
                if self.type == "crossspectrum":
                    print("I am here!")
                    cs_sep, nphots1_sep, nphots2_sep = \
                        self._make_segment_spectrum(lc1_seg, lc2_seg,
                                                            self.segment_size)
                    nphots2_all.append(nphots2_sep)
                elif self.type == "powerspectrum":
                    cs_sep, nphots1_sep = \
                        self._make_segment_spectrum(lc1_seg, self.segment_size)

                cs_all.append(cs_sep)
                nphots1_all.append(nphots1_sep)

            cs_all = np.hstack(cs_all)
            nphots1_all = np.hstack(nphots1_all)

            if self.type == "crossspectrum":
                nphots2_all = np.hstack(nphots2_all)


        m = len(cs_all)
        nphots1 = np.mean(nphots1_all)

        power_avg = np.zeros_like(cs_all[0].power)
        for cs in cs_all:
            power_avg += cs.power

        power_avg /= np.float(m)

        self.freq = cs_all[0].freq
        self.power = power_avg
        self.m = m
        self.df = cs_all[0].df
        self.n = cs_all[0].n
        self.nphots1 = nphots1

        if self.type == "crossspectrum":
            self.nphots1 = nphots1
            nphots2 = np.mean(nphots2_all)

            self.nphots2 = nphots2

