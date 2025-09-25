import torch
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

class FFT_Hypergraph:
    def __init__(self, min_freq_bands=2, peak_height_percentile=75, min_peak_distance_hz=1.0): # Optional: minimum 1 Hz between peaks
        """
        Initializes an empty hypergraph with parameters for dynamic frequency band detection.
        
        Args:
            min_freq_bands (int): Minimum number of frequency bands to detect
            peak_height_percentile (float): Percentile threshold for peak detection
            min_peak_distance_hz (float, optional): Minimum distance between peaks in Hz.
                If None, will be set based on frequency resolution.
        """
        self.min_freq_bands = min_freq_bands
        self.peak_height_percentile = peak_height_percentile
        self.min_peak_distance_hz = min_peak_distance_hz
        
    def detect_frequency_bands(self, fft_magnitudes, sampling_rate):
        """
        Detects significant frequency bands in the signal using peak detection.
        
        Args:
            fft_magnitudes (np.ndarray): FFT magnitudes of shape [B, C, T]
            sampling_rate (float): Sampling rate of the original signal in Hz
            
        Returns:
            List[tuple]: List of (start_freq, end_freq) for each detected band
        """
        # Average across batch and channels to get overall frequency profile
        avg_spectrum = np.mean(fft_magnitudes, axis=(0, 1))
        
        # Calculate frequency resolution
        freq_resolution = sampling_rate / (2 * len(avg_spectrum))
        
        # Set minimum peak distance if not provided
        if self.min_peak_distance_hz is None:
            # Default to 2x the frequency resolution
            self.min_peak_distance_hz = 2 * freq_resolution
        
        # Convert min_peak_distance from Hz to number of samples
        min_distance = int(self.min_peak_distance_hz / freq_resolution)
        
        # Find peaks in the averaged spectrum
        height = np.percentile(avg_spectrum, self.peak_height_percentile)
        peaks, peak_props = find_peaks(avg_spectrum, 
                                     height=height,
                                     distance=min_distance)
        
        # Ensure we have at least min_freq_bands
        if len(peaks) < self.min_freq_bands:
            # If not enough peaks, divide spectrum evenly
            peaks = np.linspace(0, len(avg_spectrum)-1, self.min_freq_bands, dtype=int)
        
        # Create frequency bands around peaks
        bands = []
        frequencies = fftfreq(2 * len(avg_spectrum), 1/sampling_rate)[:len(avg_spectrum)]
        
        for i in range(len(peaks)):
            if i == 0:
                start_idx = 0
                start_freq = frequencies[start_idx]
            else:
                start_idx = (peaks[i] + peaks[i-1]) // 2
                start_freq = frequencies[start_idx]
                
            if i == len(peaks) - 1:
                end_idx = len(avg_spectrum)
                end_freq = frequencies[-1]
            else:
                end_idx = (peaks[i] + peaks[i+1]) // 2
                end_freq = frequencies[end_idx]
                
            bands.append((start_idx, end_idx, start_freq, end_freq))

        return bands

    def construct_hypergraph(self, x, sampling_rate=0.00028):
        """
        Constructs a hypergraph from the input time-series data using dynamic FFT frequencies.

        Args:
            x (np.ndarray): Time-series data of shape [B, T, C]
            sampling_rate (float): Sampling rate of the signal in Hz
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (features, edge_index)
        """
        batch_size, time_steps, channels = x.shape
        x = x.detach().cpu()
        
        # Compute FFT for each channel in the time series
        fft_magnitudes = []
        for b in range(batch_size):
            channel_ffts = []
            for c in range(channels):
                fft_result = fft(x[b, :, c])
                magnitudes = np.abs(fft_result[:time_steps])
                channel_ffts.append(magnitudes)
            fft_magnitudes.append(channel_ffts)

        fft_magnitudes = np.array(fft_magnitudes)  # [B, C, T]
        #print("FFT mag: ", fft_magnitudes.shape)
        # Detect frequency bands dynamically
        freq_bands = self.detect_frequency_bands(fft_magnitudes, sampling_rate)
        
        # Create node features
        features = torch.tensor(fft_magnitudes.reshape(batch_size * channels, -1), dtype=torch.float32)
        #print("Features size: ", features.size())
        # Construct hyperedge indices
        node_indices = []
        hyperedge_indices = []
        
        # For each detected frequency band (hyperedge)
        for edge_idx, (start_idx, end_idx, _, _) in enumerate(freq_bands):
            # Find nodes that belong to this band
            for node_idx in range(start_idx, end_idx):
                node_indices.append(node_idx)
                hyperedge_indices.append(edge_idx)
        
        # Create the edge_index tensor
        edge_index = torch.tensor([node_indices, hyperedge_indices], dtype=torch.long)

        self.features = features
        self.edge_index = edge_index
        self.freq_bands = freq_bands


    def get_hypergraph(self):
        """
        Returns the constructed hypergraph.
        """

        return self.features, self.edge_index

