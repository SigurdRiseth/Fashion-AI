from src.activation_functions.activation_relu import ActivationReLU
from src.activation_functions.activation_softmax import ActivationSoftmax
from src.layers.layer_dense import LayerDense
from src.model.model import Model

# Instantiate the model
model = Model()
# Add layers
model.add(LayerDense(X.shape[1], 128))
model.add(ActivationReLU())
model.add(LayerDense(128, 128))
model.add(ActivationReLU())
model.add(LayerDense(128, 10))
model.add(ActivationSoftmax())
# Set loss, optimizer and accuracy objects
model.set(
 loss=Loss_CategoricalCrossentropy(),
 optimizer=Optimizer_Adam(decay=1e-3),
 accuracy=Accuracy_Categorical()
)
# Finalize the model
model.finalize()
# Train the model
model.train(X, y, validation_data=(X_test, y_test),
 epochs=10, batch_size=128, print_every=100)