#based on https://github.com/EmilienDupont/coin
import random
import torch
import sys
from siren import Siren
from training import Trainer
import numpy as np
from numpy import genfromtxt
import librosa
import wandb
import util
import soundFiles


def sweep(config,sweeping):
    """
    Perform a sweep or single run based on the given configuration.

    Args:
        config (dict): Configuration parameters for the sweep or single run.
        sweeping (bool): True if conducting a sweep, False for a single run.
    """
    print(config)
    # Set up torch and cuda
    dtype = torch.float32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

    # Set random seeds
    torch.manual_seed(random.randint(1, int(1e6)))
    torch.cuda.manual_seed_all(random.randint(1, int(1e6)))

    #loading sound file
    sound_file = soundFiles.soundfile("clipClassic")
    ground_truth_arr, sr = librosa.load(sound_file, sr=48000, mono=True, duration=10)
    start_time = 0
    duration = 1
    start_sample = int(start_time * sr)
    end_sample = int(start_time+duration * sr)
    ground_truth_arr = ground_truth_arr[start_sample:end_sample]

    #normalize sound file to be between 0 and 1
    max_val = np.max(ground_truth_arr)
    min_val = np.min(ground_truth_arr)
    normalized_arr = ((ground_truth_arr - min_val) / (max_val - min_val))
    ground_truth = torch.from_numpy(normalized_arr).float().to(device,dtype)

    # Setup Siren model
    func_rep = Siren(
        dim_in=1,
        dim_hidden=config['layerSize'],
        dim_out=1,
        num_layers=config['numLayers'],
        final_activation=torch.nn.Identity(),
        w0_initial=config['w0'],
        w0=config['w0']
    ).to(device)
    

    # Create coordinates
    coordinates = torch.ones(ground_truth.shape[0]).nonzero(as_tuple=False).float().to(device,dtype)
    # Normalize coordinates to [-1, 1]
    coordinates = coordinates / (ground_truth.shape[0] - 1) - 0.5
    np.set_printoptions(threshold=sys.maxsize)
    coordinates *= 2

    # Calculate model size and bit rate
    model_size = util.model_size_in_kb(func_rep)
    bit_rate_model = util.model_bit_rate_kbps(model=func_rep, duration=duration)

    # Log results to W&B
    if(sweeping):
        wandb.log({
            'model size (kb)': model_size,
            'bit rate model (kbps)': bit_rate_model,
        })
        if(model_size > ((sr*24*duration)/1024)*2): #if model size is larger than the uncompressed audio file *2 then stop the run
            wandb.run.finish(exit_code=1)
            sys.exit(0)

    # Set up training
    if(retrainMixed):
        print("Training with mixed precision")
        func_rep.load_state_dict(torch.load("state_dicts/classic/1sec/original/clipClassicPSNR.pt", map_location = device))
        func_rep = func_rep.half().to('cuda')
        func_rep = func_rep.float().to('cuda')
                
    trainer = Trainer(func_rep, lr=config['learningRate'])
    trainer.train(coordinates, ground_truth[:,None], num_iters=config['iterations'], max_val=max_val, min_val=min_val,sr=sr,sweeping=sweeping, retrainMixed=retrainMixed)
    print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')

    # Save best model
    torch.save(trainer.best_model, "best_model.pt")


def main():
    """
    Main function for running the program.

    If sweeping is enabled, it initializes Weights & Biases (W&B) project, performs a parameter sweep, and sets the run name based on the configuration.
    If sweeping is not enabled, it defines a test configuration and performs a single run of the sweep with the test configuration.

    Note:
    - The `sweeping` parameter determines whether to conduct a parameter sweep or a single run.
    - The configurations for the sweep or test run are provided within the function.
    """
    if(sweeping):
        wandb.init(project='coin')
        sweep(wandb.config,sweeping)
        wandb.run.name = "LS"+str(wandb.config.layerSize)+","+"NL"+str(wandb.config.numLayers)+","+"w0"+str(wandb.config.w0)
    else:
        testConfig = {'iterations': 10000, 'layerSize': 78, 'learningRate': 0.0002, 'numLayers': 5,'w0': 50}
        sweep(testConfig,sweeping)
   
sweep_configuration = {
    'method': 'bayes',
    'name': 'TESTING',
    'metric': {'goal': 'maximize', 'name': 'visqol'},
    'parameters': 
    {
        'layerSize': {'min': 10, 'max': 100},
        'numLayers': {'min': 5, 'max': 50},
        'iterations': {'min': 40000, 'max': 80000},
        'w0': {'min': 10, 'max': 50},
        'learningRate': {'values': [2e-4]},
     }
}

#Want to conduct a sweep? If false, it's a single run.
sweeping = False
retrainMixed = False


if(sweeping):
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='TESTING')
    agent = wandb.agent(sweep_id, function=main, count=500)

else:
    main()