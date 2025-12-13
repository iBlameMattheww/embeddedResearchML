from strupnet import SympNet
import torch
import json
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(CURRENT_DIR, 'params')
PARAMS_PATH = os.path.join(PARAMS_PATH, 'sympnet_params.json')

sympnet = SympNet(
    dim = 1,
    layers = 2,
    max_degree = 2,
    method = 'P',
)

""" Simple Harmonic Oscillator data generation """

def simple_harmonic_oscillator_solution(t_start, t_end, timestep):
    time_grid = torch.linspace(t_start, t_end, int((t_end-t_start)/timestep)+1)
    p_sol = torch.cos(time_grid)
    q_sol = torch.sin(time_grid)
    pq_sol = torch.stack([p_sol, q_sol], dim=-1)
    return pq_sol, time_grid.unsqueeze(dim=1)

timestep=0.05
x, t = simple_harmonic_oscillator_solution(t_start=0, t_end=1, timestep=timestep)
x_test, t_test = simple_harmonic_oscillator_solution(t_start=1, t_end=4, timestep=timestep)
x0, x1, t0, t1 = x[:-1, :], x[1:, :], t[:-1, :], t[1:, :]
x0_test, x1_test, t0_test, t1_test = x_test[:-1, :], x_test[1:, :], t_test[:-1, :], t_test[1:, :]

""" Training SympNet on SHO data """

optimizer = torch.optim.Adam(sympnet.parameters(), lr=0.01)
mse = torch.nn.MSELoss()
for epoch in range(1000):
    optimizer.zero_grad()    
    x1_pred = sympnet(x=x0, dt=t1 - t0)
    loss = mse(x1, x1_pred)
    loss.backward()
    optimizer.step()

params = []
for i, layer in enumerate(sympnet.layers_list):
    a = layer.params["a"].detach().cpu().numpy()
    w = layer.params["w"].detach().cpu().numpy()
    params.append({
        "layer": i,
        "a": a.tolist(),
        "w": w.tolist(),
    })

if not os.path.exists(os.path.dirname(PARAMS_PATH)):
    os.makedirs(os.path.dirname(PARAMS_PATH))
with open(PARAMS_PATH, "w") as f:
    json.dump(params, f, indent=2)


""" Evaluating trained SympNet on test data """

print("Final loss value: ", loss.item())

x1_test_pred = sympnet(x=x0_test, dt=t1_test - t0_test)

print("Test set error", torch.norm(x1_test_pred - x1_test).item())

print(sympnet.eval())
for name, param in sympnet.named_parameters():
    print(name, param.shape)
