import torch
import numpy as np
import matplotlib.pyplot as plt
from train.TrainPNO3D import smooth_chi, PNO3D


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
    mask_test = prepare_tensor(mask[-ntest:], ntest)
    chi_train = prepare_tensor(input_chi, ntrain)
    chi_test = prepare_tensor(input_chi[-ntest:], ntest)
    y_train = prepare_tensor(output, ntrain)
    y_test = prepare_tensor(output[-ntest:], ntest)

    goals_train = goals[:ntrain].reshape(ntrain, 3, 1)
    goals_test = goals[-ntest:].reshape(ntest, 3, 1)

    return (mask_train, chi_train, y_train, goals_train), (mask_test, chi_test, y_test, goals_test)


def move_to_device(batch, device):
    return tuple(t.to(device) for t in batch)

def get_dataloaders(train_data, test_data, batch_size=1):
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*train_data),
        batch_size=batch_size,
        shuffle=False,
    )
    
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*test_data),
        batch_size=batch_size,
        shuffle=False,
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
    
    # Initialize the model
    model = PNO3D(modes, modes, modes, width, nlayers)

    # Load the checkpoint safely (weights only)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    
    # Load weights into model
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def plot_model_output(model, data_loader, device, n_samples=1, slice_axis=2, contour_levels=10):
    """
    Plot the model output and ground truth side-by-side using contour plots.
    Also compute and print the relative error for each sample.

    :param model: The trained model.
    :param data_loader: The DataLoader containing the test or train set.
    :param device: The device (CUDA or CPU) to run the model on.
    :param n_samples: The number of samples to plot.
    :param slice_axis: The axis along which to slice the 3D output for contour plotting (0, 1, or 2).
    :param contour_levels: The number of contour levels to plot.
    """
    
    model.eval()

    for idx, (mask, chi, output, goals) in enumerate(data_loader):
        if idx >= n_samples:
            break

        # Move tensors to device
        mask, chi, output, goals = mask.to(device), chi.to(device), output.to(device), goals.to(device)

        with torch.no_grad():
            pred = model(chi, goals) * mask

        # Convert to numpy arrays
        pred_np = pred.cpu().numpy().squeeze()
        output_np = output.cpu().numpy().squeeze()

        # Compute relative error: mean absolute error / mean ground truth value
        abs_error = np.abs(pred_np - output_np)
        rel_error = abs_error.mean() / (np.abs(output_np).mean() + 1e-8)
        print(f"Sample {idx+1} - Relative Error: {rel_error:.4f}")

        # Select slice
        if slice_axis == 0:
            pred_slice = pred_np[int(pred_np.shape[0] // 2), :, :]
            gt_slice = output_np[int(output_np.shape[0] // 2), :, :]
        elif slice_axis == 1:
            pred_slice = pred_np[:, int(pred_np.shape[1] // 2), :]
            gt_slice = output_np[:, int(output_np.shape[1] // 2), :]
        elif slice_axis == 2:
            pred_slice = pred_np[:, :, int(pred_np.shape[2] // 2)]
            gt_slice = output_np[:, :, int(output_np.shape[2] // 2)]
        else:
            raise ValueError("slice_axis must be 0, 1, or 2")

        # Side-by-side plots
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Prediction
        pred_contour = axs[0].contourf(pred_slice, levels=contour_levels, cmap='viridis')
        axs[0].contour(pred_slice, levels=contour_levels, colors='black', linewidths=0.5)
        axs[0].set_title(f"Prediction (Sample {idx+1})")
        axs[0].set_xlabel("X-axis")
        axs[0].set_ylabel("Y-axis")
        fig.colorbar(pred_contour, ax=axs[0])

        # Ground truth
        gt_contour = axs[1].contourf(gt_slice, levels=contour_levels, cmap='viridis')
        axs[1].contour(gt_slice, levels=contour_levels, colors='black', linewidths=0.5)
        axs[1].set_title("Ground Truth")
        axs[1].set_xlabel("X-axis")
        axs[1].set_ylabel("Y-axis")
        fig.colorbar(gt_contour, ax=axs[1])

        plt.tight_layout()
        plt.show()
