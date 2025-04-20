import torch
import numpy as np
from train.TrainPlanningOperator3D import smooth_chi

def load_raw_data(data_dir, Ntotal):
    mask = np.load(f'{data_dir}/mask.npy')[:Ntotal]
    dist_in = np.load(f'{data_dir}/dist_in.npy')[:Ntotal]
    output = np.load(f'{data_dir}/output.npy')[:Ntotal]
    goals = np.load(f'{data_dir}/goals.npy')[:Ntotal]
    
    return (
        torch.tensor(mask, dtype=torch.float),
        torch.tensor(dist_in, dtype=torch.float),
        torch.tensor(output, dtype=torch.float),
        torch.tensor(goals, dtype=torch.float)
    )


def preprocess_data(mask, dist_in, output, goals, smooth_coef, sub, Sx, Sy, Sz, ntrain, ntest):
    input_chi = smooth_chi(mask, dist_in, smooth_coef)

    # Apply subsampling and reshape
    def prepare_tensor(tensor, n_samples):
        return tensor[:n_samples, ::sub, ::sub, ::sub][:, :Sx, :Sy, :Sz].reshape(n_samples, Sx, Sy, Sz, 1)

    mask_train = prepare_tensor(mask, ntrain)
    mask_test = prepare_tensor(mask, -ntest)
    chi_train = prepare_tensor(input_chi, ntrain)
    chi_test = prepare_tensor(input_chi, -ntest)
    y_train = prepare_tensor(output, ntrain)
    y_test = prepare_tensor(output, -ntest)

    goals_train = goals[:ntrain].reshape(ntrain, 3, 1)
    goals_test = goals[-ntest:].reshape(ntest, 3, 1)

    return (mask_train, chi_train, y_train, goals_train), (mask_test, chi_test, y_test, goals_test)


def get_dataloaders(train_data, test_data, batch_size):
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*train_data),
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*test_data),
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader, test_loader

def load_pno_model(
    ckpt_path,
    modes=12,
    width=32,
    nlayers=1,
    device=None
):
    # Auto-select device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize and load model
    model = PlanningOperator3D(modes, modes, modes, width, nlayers)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model