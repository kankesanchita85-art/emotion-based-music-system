import matplotlib.pyplot as plt

accuracy = [0.26, 0.35, 0.41, 0.45, 0.47]
val_accuracy = [0.34, 0.42, 0.46, 0.49, 0.52]

loss = [1.77, 1.64, 1.51, 1.43, 1.38]
val_loss = [1.65, 1.46, 1.38, 1.31, 1.26]

plt.figure(figsize=(6,4))
plt.plot(accuracy, marker='o', label="Training Accuracy")
plt.plot(val_accuracy, marker='o', label="Validation Accuracy")
plt.title("CNN Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("assets/accuracy_graph.png")
plt.close()

plt.figure(figsize=(6,4))
plt.plot(loss, marker='o', label="Training Loss")
plt.plot(val_loss, marker='o', label="Validation Loss")
plt.title("CNN Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("assets/loss_graph.png")
plt.close()

print("Graphs created successfully!")