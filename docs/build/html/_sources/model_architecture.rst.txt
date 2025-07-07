Model Architecture
==================

This section provides a detailed overview of the Conditional Normalizing Flow with UNet (CNF-UNet) architecture employed in this project for Lung CT image generation. Understanding this architecture is key to grasping how the model learns to generate high-quality synthetic images.

Overview of CNF-UNet
--------------------
The core of this project's image generation capability lies in the integration of Conditional Normalizing Flows (CNF) with a UNet-like structure. This combination allows for a powerful generative model that can learn complex data distributions while leveraging the hierarchical feature extraction benefits of the UNet.

[Elaborate here on the high-level concept: How CNF works (invertible transformations, exact likelihood), how UNet contributes (multi-scale feature extraction, skip connections), and how conditioning is applied (e.g., through UNet features).]

Normalizing Flows (NF)
---------------------
Normalizing Flows are a class of generative models that transform a simple base distribution (e.g., a Gaussian) into a complex data distribution through a sequence of invertible and differentiable transformations. This allows for exact likelihood computation and efficient sampling.

[Detail the specific types of flow layers used (e.g., coupling layers, permutations, non-linearities) and why they were chosen.]

UNet Integration
----------------
The UNet architecture, originally developed for biomedical image segmentation, is well-suited for processing images due to its encoder-decoder structure with skip connections. In our CNF-UNet, the UNet part typically acts as a powerful feature extractor that provides rich, multi-scale conditional information to the normalizing flow.

[Explain how the UNet is used: Does it provide features at different scales to different flow blocks? Is it an encoder for the conditioning? How are skip connections utilized in this context?]

Conditional Aspect
------------------
The "Conditional" aspect of CNF-UNet means the model's generation process is guided by certain input conditions. This allows for controlled image synthesis.

[Describe what the conditioning variables are (e.g., clinical parameters, partial images, noise levels) and how they are incorporated into the CNF and/or UNet parts of the architecture. How does this conditioning influence the generated output?]

Component Details
-----------------
* **UNetBlock:** [Describe the structure of your basic UNet building block (e.g., convolutional layers, activation functions, batch normalization). You can link to its API documentation here.]
* **Flow Layers:** [Detail the specific flow layers like Affine Coupling Layers, Invertible 1x1 Convolutions, etc., if applicable.]
* **Loss Function:** [Briefly mention the loss function, typically negative log-likelihood for NFs.]

Model Architecture Diagram
--------------------------
A visual representation of the CNF-UNet architecture helps in understanding the data flow and the interaction between the Normalizing Flow and UNet components.

.. figure:: /_static/model_architecture_diagram.png
   :alt: Diagram of the CNF-UNet model architecture.
   :align: center
   :width: 90%

   Conceptual diagram illustrating the integrated CNF-UNet architecture.

[Remember to replace `model_architecture_diagram.png` with the actual path to your diagram image. You'll need to create this diagram (e.g., using draw.io, Excalidraw, PowerPoint, or a scientific plotting library) and save it in `docs/source/_static/`.]