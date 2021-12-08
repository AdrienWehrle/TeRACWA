# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, University of Zürich (UZH), Switzerland, 2019

Implementation of the Terrestrial Radar Assessment of Calving Wave 
Activity (TeRACWA)

Wehrlé, A., Lüthi, M.P., Walter, A., Jouvet, G. and Vieli, A., 2021. 
Automated detection and analysis of surface calving waves with a terrestrial 
radar interferometer at the front of Eqip Sermia, Greenland. The Cryosphere 
Discussions, pp.1-24.
    
"""

import glob
from gprifile import GPRIfile
from kneed import KneeLocator
from multiprocessing import freeze_support, cpu_count
from multiprocessing.pool import ThreadPool
import numpy as np
import os
import pandas as pd
from paramfile import ParamFile
import platform
import pickle
from scipy.fftpack import fft, fftfreq
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.signal import peak_widths
from time import strftime, localtime, ctime, time
from typing import Tuple, Union


class TeRACWA:
    """
    Run the different steps of TeRACWA and store the associated run parameters
    and results at different levels.
    The Level-2 consists of a catalog of wave generation times, wave widths,
    along-front locations (azimuth dimension), as well as associated wave
    magnitudes quantified by the power peak prominence associated with a given
    wave, called the Wave Power Index (WPI).

    Attributes:

        slc_directory : str
            Full path to the directory containing Single Look Complex (SLC)
            files acquired with a GAMMA Portable Radar Interferometer (GPRI).

        mask_filename : str | None
            Full file name of a csv file containing a 2D boolean matrix (with
            the same shape as the SLC files) that can be used to specify
            the values to include (1) or exclude (0) from the processing.
            Default is None.

        low_cutoff: int
            Low-cutoff wave length of the band-pass filter applied in the
            spectral analysis (in meters). Default is 12.3 meters (Wehrlé et
            al, 2021).

        high_cutoff: int
            High-cutoff wave length of the band-pass filter applied in the
            spectral analysis (in meters). Default is 800 meters (Wehrlé et
            al, 2021).

        coregistration: bool
            Specify if the method should be applied to raw files (.slc, False)
            or coregistrated files (.rslc, True). Default is True.

        antenna: str
            Process acquisitions from the upper (u) or lower (l) GPRI antenna.
            Default is 'u'.

        nb_cores: int
            Number of machine CPU cores to use for parallel processing.
            Default is 5. If 'auto', all available cores are used.

        verbose: bool
            Specify if processing info messages should be printed (True) or
            not (False). Default is True.

        save: bool
            Whether to save class instance in a pickle file (True) or not (False).
            Please keep in mind that pickle file transfers should be avoided
            as pickle can be unsecure. Default is True. If set to '.' the
            class instance is saved in the current directory.

        out_path : str | None
            Full path to the directory where run results should be saved. If
            not specified and save is set to True, save in the current
            directory. Default is None.

        nb_azimuth_lines: int
            Number of azimuth lines in a given SLC acquisition.

        nb_range_samples: int
            Number of range samples in a given SLC acquisition.

        range_resolution: float
            Resolution in the range dimension.

        nb_timesteps: int
            Number of time steps in the data set to process.

        acquisition_times: list
            List of Pandas datetime objects for the acquisition time of
            each file to process.

        mask: numpy.ndarray
            The boolean mask loaded from mask_filename.

        prominence_footprint: int
            Footprint around peak (in pixels) used to compute peak prominences.

        results: dict
            Dictionnary containing the following results:
                - raw power maxima (2D, nb_azimuth_lines * nb_timesteps)
                - normalized power maxima (2D, nb_azimuth_lines * nb_timesteps)
                - raw_power_peaks (2D, nb_azimuth_lines * nb_timesteps)
                - raw_peak_prominences (2D, nb_azimuth_lines * nb_timesteps)
                - raw_peak_positions (1D)
                - noise_threshold (float)
                - Level-1 peak specs (1D)
                - Level-2 peak specs (1D)
            The Level-1 peak specs include the peak azimuth line, timestep, time,
            prominence and width of the raw detected peaks after noise removal.
            The Level-2 peak specs are the Level-1 peak specs after the deletion
            of peak duplicates.
    """

    def __init__(
        self,
        slc_directory: str,
        mask_filename: Union[str, None] = None,
        low_cutoff: int = 12.3,
        high_cutoff: int = 800,
        coregistration: bool = True,
        antenna: str = "u",
        nb_cores: int = 5,
        verbose: bool = True,
        save: bool = True,
        out_path: Union[str, None] = None,
    ) -> None:

        self.slc_directory = slc_directory
        self.mask_filename = mask_filename
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.coregistration = coregistration
        self.antenna = antenna
        self.nb_cores = nb_cores
        self.verbose = verbose
        self.save = save
        self.out_path = out_path
        self.results = {}

    def list_slc_files(self) -> list:
        """
        List Single Look Complex (SLC) files from directory.

        :returns: The full paths to SLC files acquired with a specified
            antenna.
        """
        if self.verbose:
            print(f"Listing slc files in {self.slc_directory}...")

        # list files to process depending on extension
        if self.coregistration:
            slc_files = sorted(glob.glob(f"{self.slc_directory}/*{self.antenna}.rslc"))
        else:
            slc_files = sorted(glob.glob(f"{self.slc_directory}/*{self.antenna}.slc"))

        self.slc_files = slc_files

        # save the number of time steps
        self.nb_timesteps = len(slc_files)

        # store acquisition times as Pandas Datetime objects
        self.acquisition_times = [
            pd.to_datetime(
                f.split(os.sep)[-1].split(".")[0], format=f"%Y%m%d_%H%M%S{self.antenna}"
            )
            for f in slc_files
        ]

        return self.slc_files

    def remove_corrupted_files(self, minimum_filesize: float = 5e7) -> list:
        """
        Remove corrupted files (defined as files with a size lower than a
        size threshold) from the slc_files list.

        :param minimum_filesize: threshold for minimum file size.
            Default is 50 Mb.

        :returns: the updated slc_files list without corrupted files.
        """
        if self.verbose:
            print("Removing corrupted files from file list...")

        # get the initial number of slc files
        initial_length = len(self.slc_files)

        # get the size of slc files
        self.slc_files = [
            file for file in self.slc_files if os.path.getsize(file) > minimum_filesize
        ]

        # get the number of slc_files after removal
        final_length = len(self.slc_files)

        # compute the number of files removed
        nb_files_removed = initial_length - final_length

        self.nb_timesteps = final_length

        if self.verbose:
            print(f"{nb_files_removed} files removed.")

        return self.slc_files

    def get_data_specs(self) -> Tuple[int, int, float]:
        """
        Retrieve data specifications from an example file selected at the
        middle of the slc_files list.

        :returns: The number of azimuth lines, range samples and the resolution
            in the range dimension.
        """
        if self.verbose:
            print("Extracting data specs from parameter file...")

        # list parameter files associated with SLC files
        parameter_files = sorted(glob.glob(f"{self.slc_directory}/*.slc.par"))

        # load example file to extract specifications
        example_pf = ParamFile(parameter_files[int(len(parameter_files) / 2)])

        # extract specifications
        self.nb_azimuth_lines = example_pf.azimuth_lines
        self.nb_range_samples = example_pf.range_samples
        self.range_resolution = example_pf.range_pixel_spacing

        return self.nb_azimuth_lines, self.nb_range_samples, self.range_resolution

    def load_mask(self) -> np.ndarray:
        """
        Load a 2-dimensional boolean mask having the same shape as the SLC
        files to restrict the processing to the region of interest.

        :returns: The boolean mask in a numpy.ndarray.
        """

        # load mask if specified
        if self.mask_filename:
            self.mask = np.loadtxt(self.mask_filename, delimiter=",")
        else:
            self.mask = None

        return self.mask

    def spectral_analysis(self, file_number: int) -> np.ndarray:
        """
        Spectral analysis within a restricted frequency range using a Fast
        Fourier Transform (FFT) for a given acquistion.

        :param file_number: File number in the slc_files list

        :returns: The FFT maximum powers for all azimuth lines of a given
            acquistion.
        """

        def spectral_analysis_1D(signal_1D: np.ndarray) -> float:
            """
            Spectral analysis at a given azimuth line.

            :param signal_1D: Radar backscatter magnitude at a given azimuth
                line.

            :returns: The FFT maximum power at a given azimuth line.
            """

            # mask data outside of the region of interest
            masked_signal_1D = signal_1D[signal_1D != 0]

            if masked_signal_1D.shape[0] > 1:

                # build the Discrete Fourier Transform sample frequencies
                frequencies = fftfreq(n=len(masked_signal_1D), d=self.range_resolution)

                # compute the Discrete Fourier Transform
                fft_1D = np.abs(fft(masked_signal_1D))

                # apply band-pass filter (restriction on the frequency ramge)
                fft_1D_filtered = fft_1D[
                    (frequencies > 1 / self.high_cutoff)
                    & (frequencies < 1 / self.low_cutoff)
                ]

                # extract the FFT maximum power
                maximum_power_1D = np.nanmax(fft_1D_filtered)

            else:

                maximum_power_1D = np.nan

            return maximum_power_1D

        # subtract two consecutive acquisitions of backscatter amplitude
        diff_magnitude = (
            GPRIfile(self.slc_files[file_number]).getMagnitude()
            - GPRIfile(self.slc_files[file_number + 1]).getMagnitude()
        )

        # apply mask if available
        if isinstance(self.mask, np.ndarray):
            diff_magnitude_masked = diff_magnitude * self.mask
        else:
            diff_magnitude_masked = diff_magnitude

        # apply spectral analysis to all azimuth lines
        maximum_power_2D = np.apply_along_axis(
            spectral_analysis_1D, 1, diff_magnitude_masked
        )

        if self.verbose:
            print(f"{file_number}/{self.nb_timesteps}")

        return maximum_power_2D

    def run_spectral_analysis(self) -> np.ndarray:
        """
        Run spectral analysis for all files in the slc_files list.

        :returns: The raw power maxima for each azimuth line (rows) of each
            acquisition (columns).
        """
        if self.verbose:
            print("Running TeRACWA spectral analysis...")

        # if auto setting, use all available cores
        if self.nb_cores == "auto":
            self.nb_cores = cpu_count()

        # add support when Windows freezes to produce an executable
        # (Move to Linux!)
        if platform == "Windows":
            freeze_support()

        # load boolean mask
        self.load_mask()

        # save starting time
        start_time = time()
        start_local_time = ctime(start_time)

        # initialize output array
        raw_power_maxima = np.zeros((self.nb_azimuth_lines, self.nb_timesteps - 1))

        # run spectral analysis in parallel
        i = 0
        with ThreadPool(self.nb_cores) as p:

            for mx in p.map(self.spectral_analysis, range(self.nb_timesteps - 1)):
                raw_power_maxima[:, i] = mx
                i += 1

        # store results
        self.results["raw_power_maxima"] = raw_power_maxima

        # get processing time
        end_time = time()
        end_local_time = ctime(end_time)
        processing_time = (end_time - start_time) / 60

        if self.verbose:
            print(f"--- Processing time: {processing_time} minutes ---")
            print(f"--- Start time: {start_local_time} ---")
            print(f"--- End time: {end_local_time} ---")

        return self.results["raw_power_maxima"]

    def normalize_maximum_power(self) -> np.ndarray:
        """
        Normalize the time series of maximum powers per azimuth line
        individually by subtracting the mean and dividing by the standard
        deviation.

        :returns: The normalized power maxima with the same shape as the
            raw power maxima.

        """
        if self.verbose:
            print("Normalizing power maxima...")

        # azimuth line standardization
        normalized_power_maxima = (
            self.results["raw_power_maxima"]
            - np.nanmean(self.results["raw_power_maxima"], axis=1, keepdims=True)
        ) / np.nanstd(self.results["raw_power_maxima"], axis=1, keepdims=True)

        # store the normalized power spectra
        self.results["normalized_power_maxima"] = normalized_power_maxima

        return self.results["normalized_power_maxima"]

    def detect_power_peaks(self) -> np.ndarray:
        """
        Detect power peaks in 2 dimensions with an 8-connected neighborhood.

        :returns: A 2D boolean matrix of detected peaks (1) with the same
            shape as the matrices of power maxima.
        """

        if self.verbose:
            print("Detecting power peaks...")

        # define an 8-connected neighborhood
        neighborhood = generate_binary_structure(2, 2)

        # apply the local maximum filter
        local_max = (
            maximum_filter(
                self.results["normalized_power_maxima"], footprint=neighborhood
            )
            == self.results["normalized_power_maxima"]
        )

        # create the background mask
        background = self.results["normalized_power_maxima"] == 0

        # background erosion to get rid of artifact of the local maximum filter
        eroded_background = binary_erosion(
            background, structure=neighborhood, border_value=1
        )

        # remove background from the local maxima mask
        raw_power_peaks = local_max ^ eroded_background

        # store the detected power peaks
        self.results["raw_power_peaks"] = raw_power_peaks

        return self.results["raw_power_peaks"]

    def compute_power_peak_prominences(self) -> np.ndarray:
        """
        Compute the prominence of all power peaks of the data set.

        :returns: A 2D matrix of peak prominences.

        """

        if self.verbose:
            print("Computing power peak prominences...")

        def compute_peak_prominence(p: int, prominence_footprint: int = 5) -> float:
            """
            Compute the prominence of a given power peak by subtracting the
            background signal (+- prominence_footprint acquisitions) to the
            peak height.

            :param p: Peak number in raw_peak_positions.

            :returns: The prominence of a given power peak.
            """

            # save prominence footprint
            self.prominence_footprint = prominence_footprint

            # handle the matrix back boundary
            if self.results["raw_peak_positions"][p, 1] - self.prominence_footprint < 0:
                low_boundary = 0
                high_boundary = (
                    self.results["raw_peak_positions"][p, 1] + self.prominence_footprint
                )

            # handle the matrix front boundary
            elif (
                self.results["raw_peak_positions"][p, 1] + self.prominence_footprint
                > self.nb_timesteps
            ):
                low_boundary = (
                    self.results["raw_peak_positions"][p, 1] - self.prominence_footprint
                )
                high_boundary = self.nb_timesteps

            # apply default case
            else:
                low_boundary = (
                    self.results["raw_peak_positions"][p, 1] - self.prominence_footprint
                )
                high_boundary = (
                    self.results["raw_peak_positions"][p, 1] + self.prominence_footprint
                )

            # compute background signal
            minimum_background = np.nanmin(
                self.results["normalized_power_maxima"][
                    self.results["raw_peak_positions"][p, 0], low_boundary:high_boundary
                ]
            )

            # compute peak prominence
            prominence = (
                self.results["normalized_power_maxima"][
                    self.results["raw_peak_positions"][p, 0],
                    self.results["raw_peak_positions"][p, 1],
                ]
                - minimum_background
            )

            return prominence

        # extract the peak positions from the binary matrix
        self.results["raw_peak_positions"] = np.column_stack(
            np.where(self.results["raw_power_peaks"])
        )

        # vectorize prominence function
        # (numpy.vectorize, is for convenience, not performance)
        compute_prominences_vectorized = np.vectorize(compute_peak_prominence)

        # run vectorized prominence function
        peak_prominences = compute_prominences_vectorized(
            range(len(self.results["raw_peak_positions"]))
        )

        # build 2D matrix of peak prominences
        self.results["raw_peak_prominences"] = np.full(
            [
                self.results["raw_power_peaks"].shape[0],
                self.results["raw_power_peaks"].shape[1],
            ],
            np.nan,
        )

        # populate 2D matrix of peak prominences
        self.results["raw_peak_prominences"][
            self.results["raw_peak_positions"][:, 0],
            self.results["raw_peak_positions"][:, 1],
        ] = peak_prominences

        return self.results["raw_peak_prominences"]

    def remove_noise_peaks(self) -> pd.DataFrame:
        """
        Detect, remove peaks associated with noise and create the Level-1
        product.

        :returns: The Level-1 product of peak characteristics.

        """
        if self.verbose:
            print("Removing power peaks associated with noise...")

        # mask for peak prominences
        peak_mask = np.isfinite(self.results["raw_peak_prominences"])

        # extract peak prominences
        raw_peak_prominences_flat = self.results["raw_peak_prominences"][peak_mask]

        # prominence range to which the Kneedle algorithm will be applied
        prominence_range = np.arange(
            np.nanmean(raw_peak_prominences_flat),
            np.nanmax(raw_peak_prominences_flat),
            0.01,
        )

        # compute number of peaks above threshold range
        nb_peaks_above_threshold = [
            np.sum(raw_peak_prominences_flat > threshold)
            for threshold in prominence_range
        ]

        # determine curve knee/elbow
        kn = KneeLocator(
            prominence_range,
            nb_peaks_above_threshold,
            curve="convex",
            direction="decreasing",
        )

        # store threshold for noise cancelation
        self.results["noise_threshold"] = np.round(kn.knee, 2)

        # compute 2D boolean noise filter
        noise_filter = (
            self.results["raw_peak_prominences"] <= self.results["noise_threshold"]
        )

        # compute 1D boolean noise filter
        noise_peaks = noise_filter[
            self.results["raw_peak_positions"][:, 0],
            self.results["raw_peak_positions"][:, 1],
        ].flatten()

        # create a DataFrame with peak specifications (as 1D lists)
        self.results["L1_peak_specs"] = pd.DataFrame(
            self.results["raw_peak_positions"][~noise_peaks],
            columns=["azimuth_line", "timestep"],
        )

        # include peak datetime
        self.results["L1_peak_specs"]["time"] = np.array(self.acquisition_times)[
            self.results["L1_peak_specs"]["timestep"].values
        ]

        # include peak prominences
        self.results["L1_peak_specs"]["prominence"] = raw_peak_prominences_flat[
            ~noise_peaks
        ]

        return self.results["L1_peak_specs"]

    def compute_peak_boundary(self, p: int, relative_height: float) -> Tuple[int, int]:
        """
        Determine the azimuth boundaries of a given peak at a given relative_height
        (in fraction of peak height).

        :param p: Peak number in the Level-1 product.

        :returns: The left and right peak boundaries in azimuth dimension.

        """

        # extract peak time step and azimuth line
        peak_time = self.results["L1_peak_specs"]["timestep"][p]
        peak_line = self.results["L1_peak_specs"]["azimuth_line"][p]

        # extract vector of normalized power maxima at peak time step
        normalized_power_maxima_ts = self.results["normalized_power_maxima"][
            :, peak_time
        ]

        # extract peak width, width height, left and right boundaries
        width, width_height, left_ips, right_ips, = peak_widths(
            normalized_power_maxima_ts,
            np.repeat(peak_line, 2),
            rel_height=relative_height,
        )

        # convert peak boundaries to integers
        left_peak_boundary = round(left_ips[0])
        right_peak_boundary = round(right_ips[0])

        return left_peak_boundary, right_peak_boundary

    def compute_power_peak_widths(self) -> pd.Series:
        """
        Compute power peak widths for the entire data set defined as the
        peak width at half prominence.

        :returns: The column of the Level-1 product with peak widths.

        """

        if self.verbose:
            print("Computing power peak widths...")

        # vectorize peak boundary function
        # (numpy.vectorize, is for convenience, not performance)
        compute_peak_boundaries = np.vectorize(self.compute_peak_boundary)

        # run vectorized peak boundary function
        peak_boundaries = compute_peak_boundaries(
            range(len(self.results["L1_peak_specs"])), 0.5
        )

        # compute peak width as the distance between left and right boundaries
        self.results["L1_peak_specs"]["width"] = peak_boundaries[1] - peak_boundaries[0]

        return self.results["L1_peak_specs"]["width"]

    def remove_power_peak_duplicates(self) -> pd.DataFrame:
        """
        Detect, remove peak duplicates and create the Level-2 product.

        :returns: The Level-2 product of peak characteristics.

        """

        def detect_peak_duplicate(p: int) -> list:
            """
            Detect the duplicates of a given power peak. A duplicate is defined
            as a peak in the vicinity of the main peak (width at 3/4 of its
            prominence), at the same time step and with a lower prominence than
            the main peak.

            :param p: Peak number in the Level-1 product.

            :returns: The positions of the peak duplicates.

            """

            # extract peak time step and azimuth line
            peak_time = self.results["L1_peak_specs"]["timestep"][p]
            peak_prominence = self.results["L1_peak_specs"]["prominence"][p]

            left_ips, right_ips = self.compute_peak_boundary(p, 0.75)

            # apply conditions to find peak duplicates
            peak_duplicates = np.where(
                (self.results["L1_peak_specs"]["azimuth_line"] >= left_ips)
                & (self.results["L1_peak_specs"]["azimuth_line"] <= right_ips)
                & (self.results["L1_peak_specs"]["timestep"] == peak_time)
                & (self.results["L1_peak_specs"]["prominence"] < peak_prominence)
            )

            return peak_duplicates

        if self.verbose:
            print("Removing power peak duplicates...")

        # vectorize peak duplicate function
        # (numpy.vectorize, is for convenience, not performance)
        detect_peak_duplicates = np.vectorize(detect_peak_duplicate, otypes=[list])

        # run vectorized peak duplicate function
        peak_duplicates = detect_peak_duplicates(
            range(len(self.results["L1_peak_specs"]))
        )

        # stack list of peak duplicates
        peak_duplicates_stack = np.hstack(
            [pkd[0] for pkd in peak_duplicates if len(pkd[0]) > 0]
        )

        # copy the Level-1 product as a basis for the Level-2 product
        self.results["L2_peak_specs"] = self.results["L1_peak_specs"].copy()

        # remove peak duplicates to create Level-2 product
        self.results["L2_peak_specs"].drop(peak_duplicates_stack, inplace=True)

        return self.results["L2_peak_specs"]

    def save_run(self) -> None:
        """
        Save the class instance containing run parameters and method results
        in a pickle file. Please keep in mind that pickle file transfers
        should be avoided as pickle can be unsecure.

        """

        if self.verbose:
            print("Saving class instance...")

        # save results in current directory
        if self.out_path == "." or self.out_path is None:
            self.out_path = os.getcwd()

        # build time tag to produce time-dependent file names
        time_tag = strftime("%Y%m%d%H%M", localtime())

        # save class instance
        with open(f"{self.out_path}/TeRACWA_results_{time_tag}.pickle", "wb") as file:
            pickle.dump(self, file)

    def run(self) -> None:
        """
        Run the different steps of TeRACWA in sequence (which somehow a
        stupid application of a class but allows a clear structure)
        """

        # run parameters and preprocessing
        self.list_slc_files()
        self.remove_corrupted_files()
        self.get_data_specs()

        # TeRACWA core methods
        self.run_spectral_analysis()
        self.normalize_maximum_power()
        self.detect_power_peaks()

        # corrections and postprocessing
        self.compute_power_peak_prominences()
        self.remove_noise_peaks()
        self.compute_power_peak_widths()
        self.remove_power_peak_duplicates()

        # save class instance
        if self.save:
            self.save_run()
