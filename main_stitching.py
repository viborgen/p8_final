
import imageio.v2 as imageio
import torch
import sys
from siren import Siren
import numpy as np
import librosa
import wandb
from training import Trainer
import util
import soundFiles

#Hyperparameters for models
number_of_layers = 5
layerSize = 78
w0 = 50
#general definitinons
warmStartIterations = 1000
ScratchIterations = 70000
originalSound_dict = {}
reconSound_dict = {}
#########################
# is it wanted to warm start or train from scratch?
warmStart = True
useWandb = False
#loading the original waveform
mode = "audio"
pathAudio = soundFiles.soundfile("clipClassic")
number_of_seconds = 10
wav, sr = librosa.load(pathAudio, sr=48000, duration=number_of_seconds)
#Dividing the original waveform into smaller waveforms, where each sample is clip_duration
clip_duration = 1
number_of_clips = int(number_of_seconds/clip_duration)
#discovering which sample is the beginning of the next second
division_point = len(wav)/number_of_clips
print(f'division_point = {division_point}')
#########################
#creating a dictionary with the original waveform divided into smaller waveforms
for i in range(number_of_clips):
    key = f'originalSecond{i+1}'
    value = wav[int(division_point*i):int(division_point*(i+1))]
    originalSound_dict[key] = value

#########################
#loading the model we want to retrain (if warmstart is used), which is also used to reconstruct the first second.
func_rep = util.load_model_from_path('state_dicts/classic/1sec/original/clipClassicPSNR.pt',number_of_layers,layerSize,w0)
reconSound_dict['reconSecond1'] = util.reconstructSound(func_rep = func_rep , origWaveform = originalSound_dict['originalSecond1'], sr = sr, duration=1)
reconSoundTotal = reconSound_dict['reconSecond1']
#########################
#making sure that the model is on the GPU if it's available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')
#########################
for i in range(1, len(originalSound_dict)):
    if(warmStart):
        if(useWandb):
            wandb.init(
                # set the wandb project where this run will be logged
                project="yolo",
                name=f"SpeechwarmstartReconstruction{warmStartIterations}"
            )
        #reusing the latest instance of Siren every time
        coordinates, features, max, min = util.get_coordinates_and_features(originalSound_dict[f'originalSecond{i+1}'],device=device)
        trainer = Trainer(func_rep.to(device), lr=2e-4)
        trainer.train(coordinates, features, num_iters=warmStartIterations,max_val=max,min_val=min,sr=sr,sweeping=useWandb, mode=mode)
        func_rep.load_state_dict(trainer.best_model)
        torch.save(trainer.best_model, f'state_dicts/stitching/outputs/classicalWarm{warmStartIterations}_{i+1}.pt')
        if(useWandb):
            wandb.finish()
    if (warmStart == False):
        if(useWandb):
            wandb.init(
                # set the wandb project where this run will be logged
                project="insert_project_name",
                name=f"insert_run_name{ScratchIterations}"
            )
        #defining a new instance of Siren every time
        torch.cuda.empty_cache()
        func_rep = Siren(
            dim_in=1,
            dim_hidden=layerSize,
            dim_out=1,
            num_layers=number_of_layers,
            final_activation=torch.nn.Identity(),
            w0_initial=w0,
            w0=w0
        ).to('cuda')
        coordinates, features, max, min = util.get_coordinates_and_features(originalSound_dict[f'originalSecond{i+1}'],device=device)
        trainer = Trainer(func_rep, lr=2e-4)
        trainer.train(coordinates, features, num_iters=ScratchIterations, max_val=max,min_val=min,sr=sr,sweeping=useWandb, mode=mode)
        func_rep.load_state_dict(trainer.best_model)
        torch.save(trainer.best_model, f'state_dicts/stitching/outputs/classicalScratch{ScratchIterations}_{i+1}.pt')
        if(useWandb):
            wandb.finish()
    #reconstructing the sound and saving it to a dictionary
    reconSound_dict[f'reconSecond{i+1}'] = util.reconstructSound(func_rep = func_rep.to("cpu"), origWaveform = originalSound_dict[f'originalSecond{i+1}'], sr = sr, duration=1)
    #concatenating the reconstructed sound to the total reconstructed sound
    reconSoundTotal = np.concatenate((reconSoundTotal, reconSound_dict[f'reconSecond{i+1}']))
librosa.save(reconSoundTotal, f'/state_dicts/stitching/outputs/reconSound.wav', sr=sr)
#calculating metrics for the concatenated waveform, and logging to W&B
visqol = util.visqol(wav,reconSoundTotal,sr,mode)
psnr = util.PSNR(wav,reconSoundTotal)
if(useWandb):
    wandb.init(
        # set the wandb project where this run will be logged
        project="insert_project_name",
        name=f"insert_run_name"
    )
    wandb.log({
        'psnr': psnr,
        'visqol': visqol
    })
    wandb.log({"TotalReconstruction": wandb.Audio(reconSoundTotal, sample_rate=sr)})
    wandb.finish()