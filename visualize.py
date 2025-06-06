import matplotlib.pyplot as plt

def plot_histograms(image, noise):
    flat_image = image.view(-1).cpu().numpy()
    flat_noise = noise.view(-1).cpu().numpy()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(flat_image, bins=50, color='blue', alpha=0.7)
    plt.title('CT Image Pixel Distribution')

    plt.subplot(1, 2, 2)
    plt.hist(flat_noise, bins=50, color='red', alpha=0.7)
    plt.title('Gaussian Noise Distribution')

    plt.tight_layout()
    plt.show()

def plot_samples(image, noise):
    plt.figure(figsize=(10, 4))
    for i in range(4):
        plt.subplot(2, 4, i + 1)
        plt.imshow(image[i, 0].cpu(), cmap='gray')
        plt.title("CT Image")
        plt.axis('off')
        plt.subplot(2, 4, i + 5)
        plt.imshow(noise[i, 0].cpu(), cmap='gray')
        plt.title("Noise")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
