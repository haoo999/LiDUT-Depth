import os
import numpy as np
import torch
import networks

device = torch.device("cuda")

model_path = "./models/lidut-depth_640x192/"
encoder_dict = torch.load(os.path.join(model_path, "encoder.pth"))
decoder_dict = torch.load(os.path.join(model_path, "depth.pth"))

# LOADING PRETRAINED MODEL
print("   Loading pretrained encoder")
encoder = networks.LiteMono(model = 'lite-mono')

model_dict = encoder.state_dict()
encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

encoder.to(device)
encoder.eval()

print("   Loading pretrained decoder")
depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
depth_model_dict = depth_decoder.state_dict()
depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

depth_decoder.to(device)
depth_decoder.eval()

dummy_input = torch.randn(8, 3, 192, 640, dtype=torch.float).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings = np.zeros((repetitions, 1))

# GPU-WARM-UP
for _ in range(50):
    _ = depth_decoder(encoder(dummy_input))

# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = depth_decoder(encoder(dummy_input))
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
mean_fps = 1000. / mean_syn

# Calculate single inference time
single_inference_time = mean_syn / dummy_input.shape[0]

print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
print('Single Inference Time: {:.3f}ms'.format(single_inference_time))