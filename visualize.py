import matplotlib.pyplot as plt

def visualize(generated):
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(generated[i, 0].cpu(), cmap='gray')
        plt.axis('off')
    plt.show()