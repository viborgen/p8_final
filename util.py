from sklearn.metrics import mean_squared_error
import librosa
import os
from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2
from visqol.pb2 import similarity_result_pb2
import torch
import torchaudio.transforms as T
from siren import Siren
import numpy as np
import soundFiles
import copy
from typing import Dict
from torch._C import dtype

DTYPE_BIT_SIZE: Dict[dtype, int] = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.bfloat16: 16,
    torch.complex32: 32,
    torch.complex64: 64,
    torch.complex128: 128,
    torch.cdouble: 128,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1,
    torch.quint8: 8,
    torch.qint8: 8,
    torch.qint32: 32,
    torch.quint8: 8
}

def get_coordinates_and_features(ground_truth_arr, device):
    """
    Preprocesses the ground truth array and returns coordinates and features.

    Args:
        ground_truth_arr (np.ndarray): Ground truth array.
        device (torch.device): Device to be used for computation.

    Returns:
        tuple: Tuple containing the coordinates, features, maximum value, and minimum value.

    """
    max_val = np.max(ground_truth_arr)
    min_val = np.min(ground_truth_arr)
    normalized_arr = ((ground_truth_arr - min_val) / (max_val - min_val))
    ground_truth = torch.from_numpy(normalized_arr).float().to(device, dtype=torch.float32)
    coordinates = torch.ones(ground_truth.shape[0]).nonzero(as_tuple=False).float().to(device, dtype=torch.float32)
    coordinates = coordinates / (ground_truth.shape[0] - 1) - 0.5
    coordinates *= 2
    features = ground_truth
    features = features[:, None]
    return coordinates, features, max_val, min_val

def to_coordinates_and_features(img):
    """
    Converts an image to a set of coordinates and features.

    Args:
        img (torch.Tensor): Image tensor of shape (channels, height, width).

    Returns:
        tuple: Tuple containing the coordinates and features.
            - coordinates (torch.Tensor): Tensor of coordinates of shape (num_points, 2).
            - features (torch.Tensor): Tensor of features of shape (num_points, channels).

    """
    # Coordinates are indices of all non-zero locations of a tensor of ones of the same shape as spatial dimensions of the image
    coordinates = torch.ones(img.shape[1:]).nonzero(as_tuple=False).float()
    # Normalize coordinates to lie in [-.5, .5]
    coordinates = coordinates / (img.shape[1] - 1) - 0.5
    # Convert to range [-1, 1]
    coordinates *= 2
    # Convert image to a tensor of features of shape (num_points, channels)
    features = img.reshape(img.shape[0], -1).T
    return coordinates, features

def mse(original,compressed):
    """
    Calculates the mean squared error (MSE) between the original and compressed data.

    Args:
        original (torch.Tensor or np.ndarray): Original data.
        compressed (torch.Tensor or np.ndarray): Compressed data.

    Returns:
        float: Mean squared error.

    """
    if isinstance(compressed, torch.Tensor):
        mse = (original - compressed).detach().pow(2).mean().to('cpu').item()
    else:
        mse = mean_squared_error(original, compressed)
    return mse

def visqol(original, compressed, sr, mode):
    """
    Calculates the ViSQOL/ViSQOLAudio score between the original and compressed signals.

    Args:
        original: The original signal.
        compressed: The compressed signal.
        sr: The sample rate of the signals.
        mode: The mode of operation ("audio" or "speech").

    Returns:
        The ViSQOL score between the original and compressed signals.
    """
    config = visqol_config_pb2.VisqolConfig()
    #mode = "audio"
    if mode == "audio":
        config.audio.sample_rate = 48000
        config.options.use_speech_scoring = False
        svr_model_path = "libsvm_nu_svr_model.txt"
    elif mode == "speech":
        config.audio.sample_rate = 16000
        config.options.use_speech_scoring = True
        svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
    else:
        raise ValueError(f"Unrecognized mode: {mode}")

    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)

    api = visqol_lib_py.VisqolApi()

    api.Create(config)
    if(mode=="audio"):
        if(sr != 48000):
            if isinstance(compressed, torch.Tensor):
                resampler = T.Resample(sr, 48000).to("cuda")
                original = resampler(original)
                compressed = resampler(compressed)
            else:
                original = librosa.resample(y=original, orig_sr=sr, target_sr=48000)
                compressed = librosa.resample(y=compressed, orig_sr=sr, target_sr=48000)   
    if(mode=="speech"):
        if(sr != 16000):
            if isinstance(compressed, torch.Tensor):
                resampler = T.Resample(sr, 16000).to("cuda")
                original = resampler(original)
                compressed = resampler(compressed)
            else:
                original = librosa.resample(y=original, orig_sr=sr, target_sr=16000)
                compressed = librosa.resample(y=compressed, orig_sr=sr, target_sr=16000)
    if isinstance(compressed, torch.Tensor):
        similarity_result = api.Measure(original.cpu().squeeze().detach().numpy().astype('float64'), compressed.cpu().squeeze().detach().numpy().astype('float64'))
    else:    
        similarity_result = api.Measure(original.astype('float64'), compressed.astype('float64'))

    return similarity_result.moslqo

def PSNR(origWaveform, compressedWaveform):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two waveforms.

    The PSNR is a metric used to evaluate the quality of compressed or reconstructed signals
    compared to the original signal. It measures the ratio between the maximum possible power
    of a signal and the power of the noise present in the signal.

    Args:
        origWaveform (array-like): The original waveform.
        compressedWaveform (array-like): The compressed/reconstructed waveform.

    Returns:
        float: The PSNR value in decibels (dB).
    """
    mseScore = mse(origWaveform, compressedWaveform)
    psnr = 20. * np.log10(1.) - 10. * np.log10(mseScore)

    return psnr

def clamp_waveform(waveform):
    """Clamp iwaveform values to like in [0, 1] and convert to unsigned int.

    Args:
        img (torch.Tensor):
    """
    # Values may lie outside [0, 1], so clamp input
    waveform_ = torch.clamp(waveform, 0., 1.)
    # Pixel values lie in {0, ..., 255}, so round float tensor
    return waveform_


def get_clamped_psnr(waveform_recon,waveform):
    """Get PSNR between true image and reconstructed image. As reconstructed
    image comes from output of neural net, ensure that values like in [0, 1] and
    are unsigned ints.

    Args:
        img (torch.Tensor): Ground truth image.
        img_recon (torch.Tensor): Image reconstructed by model.
    """
    return PSNR(waveform, clamp_waveform(waveform_recon))

def get_clamped_mse(waveform_recon,waveform): 
    """Get MSE between true waveform and reconstructed waveform. As reconstructed
    waveform comes from output of neural net, ensure that values like in [0, 1] and
    are unsigned ints.

    Args:
        img (torch.Tensor): Ground truth waveform.
        img_recon (torch.Tensor): Waveform reconstructed by model.
    """
    return mse(waveform, clamp_waveform(waveform_recon))

def mean(list_):
    return np.mean(list_)

def load_model_from_path(path, numLayers, layerSize, w0_param):
    """
    Loads a Siren model from a file specified by the given path.

    This function creates a Siren model with the specified number of layers, layer size, and w0 parameter.
    It then loads the model's state dictionary from the file located at the given path.

    Args:
        path (str): The path to the file containing the model's state dictionary.
        numLayers (int): The number of layers in the Siren model.
        layerSize (int): The size of each hidden layer in the Siren model.
        w0_param (float): The w0 parameter used in Siren initialization.

    Returns:
        Siren: The loaded Siren model.
    """
    func_rep = Siren(
        dim_in=1,
        dim_hidden=layerSize,
        dim_out=1,
        num_layers=numLayers,
        final_activation=torch.nn.Identity(),
        w0_initial=w0_param,
        w0=w0_param
    )
    state_dict = torch.load(path)
    func_rep.load_state_dict(state_dict)    
    return func_rep

def load_model_from_memory(state_dict, numLayers, layerSize, w0_param):
    """
    Loads a Siren model from a state dictionary stored in memory.

    This function creates a Siren model with the specified number of layers, layer size, and w0 parameter.
    It then loads the model's state dictionary from the given memory location.

    Args:
        state_dict (dict): The state dictionary containing the model's parameters.
        numLayers (int): The number of layers in the Siren model.
        layerSize (int): The size of each hidden layer in the Siren model.
        w0_param (float): The w0 parameter used in Siren initialization.

    Returns:
        Siren: The loaded Siren model.
    """
    func_rep = Siren(
        dim_in=1,
        dim_hidden=layerSize,
        dim_out=1,
        num_layers=numLayers,
        final_activation=torch.nn.Identity(),
        w0_initial=w0_param,
        w0=w0_param
    )
    func_rep.load_state_dict(state_dict)    
    return func_rep

def model_size_in_kb(model):
    """
    Calculates the total number of kilobits required to store the parameters and buffers of the given `model`.

    This function iterates through the parameters and buffers of the model and calculates the total number of bits required
    to store them. The resulting number of bits is then converted to kilobits.

    Args:
        model (nn.Module): The model for which to calculate the size.

    Returns:
        float: The size of the model in kilobits.
    """
    kb= (sum(sum(t.nelement() * DTYPE_BIT_SIZE[t.dtype] for t in tensors)
               for tensors in (model.parameters(), model.buffers())))/1024
    return kb


def model_bit_rate_kbps(model, duration = 1):
    """
    Calculates the bit rate in kilobits per second (kbps) required to transmit the model over a given duration.

    This function first calculates the total number of bits required to store the parameters and buffers of the model
    using the `model_size_in_kb` function. It then divides the number of bits by the duration to obtain the bit rate
    in kilobits per second (kbps).

    Args:
        model (nn.Module): The model for which to calculate the bit rate.
        duration (float): The duration of transmission in seconds (default: 1).

    Returns:
        float: The bit rate of the model in kilobits per second (kbps).
    """
    num_bits = model_size_in_kb(model)
    bit_rate = (num_bits/duration)
    return bit_rate

def reconstructSound(func_rep,origWaveform,sr, duration=1):
    """
    Reconstructs the sound waveform using a given functional representation.

    Args:
        func_rep (callable): Functional representation of the sound waveform.
        origWaveform (np.ndarray): Original sound waveform.
        sr (int): Sample rate of the waveform.
        duration (float, optional): Duration of the reconstructed waveform in seconds. Defaults to 1.

    Returns:
        np.ndarray: Reconstructed waveform.

    """
    max_val = np.max(origWaveform)
    min_val = np.min(origWaveform)
    coordinates = np.linspace(start=-1, stop=1, num=sr*duration)
    coordinatesTorch = torch.from_numpy(coordinates).float().unsqueeze(1)
    with torch.no_grad():
        waveform_recon = func_rep(coordinatesTorch).to('cpu').numpy()
        waveformSqueezed = waveform_recon.squeeze()
        recon_waveform = waveformSqueezed * (max_val - min_val) + min_val
    return recon_waveform

def compareModelWithWaveform(func_rep,origWaveform,duration = 1, sr = 48000,form="audio", bit_rate = 0):
    """
    Compares a model's reconstructed waveform with an original waveform.

    Args:
        func_rep (callable): Functional representation of the sound waveform.
        origWaveform (np.ndarray): Original sound waveform.
        duration (float, optional): Duration of the waveform in seconds. Defaults to 1.
        sr (int, optional): Sample rate of the waveform. Defaults to 48000.
        form (str, optional): Form of the waveform. Defaults to "audio".
        bit_rate (int, optional): Bit rate of the model. If 0, it will be calculated. Defaults to 0.

    Returns:
        tuple: Tuple containing the ViSQOL score and PSNR score.

    """
    recon_waveform = reconstructSound(func_rep,origWaveform,sr)
    visqolscore = visqol(origWaveform, recon_waveform, sr, form)
    PSNRScore = PSNR(origWaveform, recon_waveform)
    if(bit_rate == 0):
        bit_rate = model_bit_rate_kbps(func_rep, duration)
    return visqolscore, PSNRScore

def playAudioFromModel(func_rep,origWaveform,sr = 48000,duration = 1):
    """
    Generates the waveform from a model and prepares it for audio playback.

    Args:
        func_rep (callable): Functional representation of the sound waveform.
        origWaveform (np.ndarray): Original sound waveform.
        sr (int, optional): Sample rate of the waveform. Defaults to 48000.
        duration (float, optional): Duration of the waveform in seconds. Defaults to 1.

    Returns:
        np.ndarray: Waveform for audio playback.

    """
    recon_waveform = reconstructSound(func_rep,origWaveform,sr,duration)
    bit_rate_calc = model_bit_rate_kbps(func_rep, duration)
    denorm_matrix = recon_waveform.reshape(-1)
    return denorm_matrix 

def compareWaveformWithWaveform(original, compressed, sr = 48000, mode="audio"):
    """
    Compares two waveforms using the ViSQOL and PSNR scores.

    Args:
        original (np.ndarray): Original sound waveform.
        compressed (np.ndarray): Compressed sound waveform.
        sr (int, optional): Sample rate of the waveforms. Defaults to 48000.
        mode (str, optional): Determines usage of ViSQOLAudio for "audio" and ViSQOL for "speech". Defaults to "audio".

    Returns:
        tuple: Tuple containing the ViSQOL score and PSNR score.

    """
    visqolScore = visqol(original, compressed, sr, mode)
    PSNRscore = PSNR(original, compressed)
    return visqolScore,PSNRscore

def quantizeAndCompare(model, origWaveform, sr, duration, mode = "audio"):
    """
    Quantizes a model and compares it with an original waveform. 
    Calculates and prints the bitrate for 16 bit-, 14 bit- and 12 bit uniform quantization, as well as half precision
    for a given INR model.

    Args:
        model (torch.nn.Module): Model to be quantized.
        origWaveform (np.ndarray): Original sound waveform.
        sr (int): Sample rate of the waveform.
        duration (float): Duration of the waveform in seconds.
        mode (str, optional): Determines usage of ViSQOLAudio for "audio" and ViSQOL for "speech". Defaults to "audio".

    Returns:
        None

    """
    quantized_model = modelUniformQuantization(model,16)
    bit_rate_calc = ((sum(sum(t.nelement() * 16 for t in tensors)
               for tensors in (model.parameters(), model.buffers())))/1024)/duration
    ViSQOLScore, PSNRscore = compareModelWithWaveform(quantized_model,origWaveform,duration = duration, sr = sr,form=mode, bit_rate = bit_rate_calc)
    print(f'Uniform 16 bit: ViSQOLScore: {ViSQOLScore}, PSNR: {PSNRscore}, bitrate: {bit_rate_calc}')

    quantized_model = modelUniformQuantization(model,14)
    bit_rate_calc = ((sum(sum(t.nelement() * 14 for t in tensors)
               for tensors in (model.parameters(), model.buffers())))/1024)/duration
    ViSQOLScore, PSNRscore = compareModelWithWaveform(quantized_model,origWaveform,duration = duration, sr = sr,form=mode, bit_rate = bit_rate_calc)
    print(f'Uniform 14 bit: ViSQOLScore: {ViSQOLScore}, PSNR: {PSNRscore}, bitrate: {bit_rate_calc}')

    quantized_model = modelUniformQuantization(model,12)
    bit_rate_calc = ((sum(sum(t.nelement() * 12 for t in tensors)
               for tensors in (model.parameters(), model.buffers())))/1024)/duration
    ViSQOLScore, PSNRscore = compareModelWithWaveform(quantized_model,origWaveform,duration = duration, sr = sr,form=mode, bit_rate = bit_rate_calc)
    print(f'Uniform 12 bit: ViSQOLScore: {ViSQOLScore}, PSNR: {PSNRscore}, bitrate: {bit_rate_calc}')
    
    quantized16 = model
    quantized16.half()
    bit_rate_calc = model_bit_rate_kbps(quantized16, duration)
    quantized16.float()
    ViSQOLScore, PSNRscore = compareModelWithWaveform(quantized16,origWaveform,duration = duration, sr = sr,form=mode, bit_rate = bit_rate_calc)
    print(f'HALF Precision: ViSQOLScore: {ViSQOLScore}, PSNR: {PSNRscore}, bitrate: {bit_rate_calc}')

def quantizeAudioSampleAndCompare(origWaveform, mode = "audio", sample_rate = 48000, bits = 8):   
    """
    Quantizes an original waveform and compares it with the quantized waveform.

    Args:
        origWaveform (np.ndarray): Original sound waveform.
        mode (str, optional): Form of the waveform. Defaults to "audio".
        sample_rate (int, optional): Sample rate of the waveform. Defaults to 48000.
        bits (int, optional): Bit depth for quantization. Defaults to 8.

    Returns:
        None

    """
    quantization_bits = bits
    audio_data_quan = origWaveform * 2**quantization_bits/2
    audio_data_quan = audio_data_quan.astype(int)
    audio_data_quanNormalized = audio_data_quan / 2**quantization_bits
    ViSQOLScore, PSNRscore = compareWaveformWithWaveform(origWaveform, audio_data_quanNormalized, sample_rate, mode=mode)
    print(f'Uniform Quantization {bits} bit: ViSQOLScore: {ViSQOLScore}, PSNR: {PSNRscore}')

def modelUniformQuantization(modelOrig, bit_depth):
    """
    Performs uniform quantization on a model.

    Args:
        modelOrig (torch.nn.Module): Original model.
        bit_depth (int): Bit depth for quantization.

    Returns:
        torch.nn.Module: Quantized model.

    """
    # Define the quantization levels and formula
    num_levels = 2 ** bit_depth
    delta = 2.0 / (num_levels - 1)
    model = copy.deepcopy(modelOrig)
    with torch.no_grad():
        for param in model.parameters():
            weights = param.data
            nonzero_mask = weights != 0.0
            quantized_weights = torch.zeros_like(weights)
            quantized_weights[nonzero_mask] = torch.round(weights[nonzero_mask] / delta) * delta
            param.data[...] = quantized_weights

    return model

