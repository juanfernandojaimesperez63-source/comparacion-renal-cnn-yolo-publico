"""Adaptadores de entrenamiento con las arquitecturas de las fuentes."""


class ReferenceCNN:
    def __init__(self, seed):
        import tensorflow as tf

        self.tf = tf
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
        devices = tf.config.list_physical_devices("GPU")
        self.hardware = tf.config.experimental.get_device_details(devices[0]).get("device_name", "GPU") if devices else "CPU"
        layers = tf.keras.layers
        inputs = layers.Input(shape=(224, 224, 3))
        x = layers.Rescaling(1 / 255)(inputs)
        for i, (filters, kernel) in enumerate(((32, 5), (64, 5), (128, 3), (256, 3)), 1):
            x = layers.Conv2D(filters, kernel, strides=2, padding="same", activation="relu",
                              name=f"conv{i}")(x)
            x = layers.MaxPooling2D()(x)
        x = layers.Dense(128)(layers.Flatten()(x))
        outputs = layers.Dense(4, activation="softmax")(x)
        self.model = tf.keras.Model(inputs, outputs)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-7)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.fit_batch = tf.function(self.fit_batch, reduce_retracing=True)
        self.forward = tf.function(lambda x: self.model(x, training=False), reduce_retracing=True)
        self.parameters = self.model.count_params()

    def fit_batch(self, images, labels):
        with self.tf.GradientTape() as tape:
            probabilities = self.model(images, training=True)
            loss = self.loss(labels, probabilities)
        variables = self.model.trainable_variables
        self.optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
        return loss

    def train(self, images, labels):
        return float(self.fit_batch(images, labels).numpy())

    def predict(self, images):
        return self.forward(images).numpy()

    def save(self, path):
        self.model.save_weights(path / "best.weights.h5")

    def load(self, path):
        self.model.load_weights(path / "best.weights.h5")

    def gradcam(self, image, label):
        tf = self.tf
        model = tf.keras.Model(self.model.input, [self.model.get_layer("conv3").output, self.model.output])
        with tf.GradientTape() as tape:
            feature, scores = model(image[None], training=False)
            score = scores[0, label]
        weights = tf.reduce_mean(tape.gradient(score, feature)[0], axis=(0, 1))
        heatmap = tf.nn.relu(tf.reduce_sum(feature[0] * weights, axis=-1))
        return tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap)).numpy()


class YOLOClassifier:
    def __init__(self, seed):
        import torch
        from ultralytics.nn.tasks import ClassificationModel

        self.torch = torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(4)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hardware = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        self.model = ClassificationModel("yolo26n-cls.yaml", nc=4, verbose=False).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, eps=1e-7)
        self.loss = torch.nn.CrossEntropyLoss()
        self.parameters = sum(parameter.numel() for parameter in self.model.parameters())

    def tensor(self, images):
        return self.torch.from_numpy(images).to(self.device).permute(0, 3, 1, 2).float() / 255

    def train(self, images, labels):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(self.tensor(images))
        loss = self.loss(logits, self.torch.as_tensor(labels, device=self.device, dtype=self.torch.long))
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())

    def predict(self, images):
        self.model.eval()
        with self.torch.inference_mode():
            probabilities, _ = self.model(self.tensor(images))
        return probabilities.cpu().numpy()

    def save(self, path):
        self.torch.save(self.model.state_dict(), path / "best.pt")

    def load(self, path):
        weights = self.torch.load(path / "best.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(weights)
