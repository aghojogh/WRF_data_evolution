import matplotlib.pyplot as plt


def plot_refinement_loss(generated_result):
    plt.figure(figsize=(5, 3))
    plt.plot(generated_result["refinement_loss_history"])
    plt.xlabel("PyTorch step")
    plt.ylabel("Loss")
    plt.title("PyTorch refinement loss")
    plt.tight_layout()
    plt.show()
