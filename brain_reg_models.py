import os
import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Optional

# Silence TensorFlow noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# -----------------------
# Differentiable sampler
# -----------------------

class BilinearUpsampling2D(layers.Layer):
    """Custom bilinear upsampling that matches PyTorch's align_corners=True"""
    def __init__(self, size=2, **kwargs):
        super().__init__(**kwargs)
        self.size = size if isinstance(size, (list, tuple)) else (size, size)

    def call(self, inputs):
        h, w = tf.shape(inputs)[1], tf.shape(inputs)[2]
        new_h = h * self.size[0]
        new_w = w * self.size[1]
        # TensorFlow resizing with align_corners=True matches PyTorch
        return tf.image.resize(inputs, [new_h, new_w], method='bilinear', antialias=False)

    def get_config(self):
        config = super().get_config()
        config.update({"size": self.size})
        return config


def bilinear_sampler(img, coords):
    """
    Bilinear sampling on a 2D grid.
    img: (B, H, W, C)
    coords: (B, H, W, 2) with coords[..., 0] = x, coords[..., 1] = y in pixel space
    """
    dtype = coords.dtype
    b = tf.shape(img)[0]
    h = tf.shape(img)[1]
    w = tf.shape(img)[2]

    x = tf.cast(coords[..., 0], dtype)
    y = tf.cast(coords[..., 1], dtype)

    x0 = tf.cast(tf.floor(x), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), tf.int32)
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, 0, w - 1)
    x1 = tf.clip_by_value(x1, 0, w - 1)
    y0 = tf.clip_by_value(y0, 0, h - 1)
    y1 = tf.clip_by_value(y1, 0, h - 1)

    batch_idx = tf.reshape(tf.range(b), (b, 1, 1))
    b_grid = tf.tile(batch_idx, (1, h, w))

    def gather(ix, iy):
        idx = tf.stack([b_grid, iy, ix], axis=-1)
        return tf.gather_nd(img, idx)

    Ia = gather(x0, y0)
    Ib = gather(x0, y1)
    Ic = gather(x1, y0)
    Id = gather(x1, y1)

    x0_f = tf.cast(x0, dtype)
    x1_f = tf.cast(x1, dtype)
    y0_f = tf.cast(y0, dtype)
    y1_f = tf.cast(y1, dtype)

    wa = (x1_f - x) * (y1_f - y)
    wb = (x1_f - x) * (y - y0_f)
    wc = (x - x0_f) * (y1_f - y)
    wd = (x - x0_f) * (y - y0_f)

    wa = tf.expand_dims(wa, -1)
    wb = tf.expand_dims(wb, -1)
    wc = tf.expand_dims(wc, -1)
    wd = tf.expand_dims(wd, -1)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


class SpatialTransformer2D(layers.Layer):
    """2D spatial transformer layer"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        src, flow = inputs
        h = tf.shape(src)[1]
        w = tf.shape(src)[2]
        yy, xx = tf.meshgrid(tf.range(h, dtype=tf.float32), tf.range(w, dtype=tf.float32), indexing="ij")
        grid = tf.stack([xx, yy], axis=-1)[None, ...]
        grid = tf.cast(grid, flow.dtype)
        return bilinear_sampler(src, grid + flow)

# -----------------------
# Building Blocks
# -----------------------

def build_vxm_2p5d_dense_core(img_height, img_width, n_channels, window_radius):
    """
    DPU-Compatible 2.5D VoxelMorph Architecture
    
    Vitis AI 2.5 Requirements:
    1. LeakyReLU alpha=0.1 ONLY (not 0.2!)
    2. Strided Conv2D instead of MaxPooling2D
    3. Activation fused with Conv2D (activation='leaky_relu')
    
    Architecture:
    Encoder: [16, 32, 32, 32] with strided conv downsampling
    Bottleneck: 32 filters
    Decoder: mirrors encoder with bilinear upsampling
    Final: [32, 16] to flow (2 channels)
    """
    n_stack = 2 * window_radius + 1
    ch = n_stack * n_channels

    moving = layers.Input(shape=(img_height, img_width, ch), name="moving_stack")
    fixed = layers.Input(shape=(img_height, img_width, ch), name="fixed_stack")
    x = layers.Concatenate(name="concat_moving_fixed")([moving, fixed])

    # Encoder
    enc_feats = [16, 32, 32, 32]
    skips = []

    for i, nf in enumerate(enc_feats):
        x = layers.Conv2D(nf, 3, padding='same', name=f"enc{i}_conv")(x)
        x = layers.LeakyReLU(alpha=0.1, name=f"enc{i}_act")(x)
        skips.append(x)

        # Strided Conv for downsampling
        x = layers.Conv2D(nf, 3, strides=2, padding='same', name=f"enc{i}_ds")(x)
        x = layers.LeakyReLU(alpha=0.1, name=f"enc{i}_ds_act")(x)

    # Bottleneck
    x = layers.Conv2D(32, 3, padding='same', name="bottleneck_conv")(x)
    x = layers.LeakyReLU(alpha=0.1, name="bottleneck_act")(x)

    # Decoder
    for i, skip in enumerate(reversed(skips)):
        # Bilinear upsampling
        x = layers.UpSampling2D(size=2, interpolation='bilinear', name=f"dec{i}_upsample")(x)
        x = layers.Concatenate(name=f"dec{i}_concat")([x, skip])
        nf = skip.shape[-1]
        x = layers.Conv2D(nf, 3, padding='same', name=f"dec{i}_conv")(x)
        x = layers.LeakyReLU(alpha=0.1, name=f"dec{i}_act")(x)

    # Final convolutions
    x = layers.Conv2D(32, 3, padding='same', name="final_conv0")(x)
    x = layers.LeakyReLU(alpha=0.1, name="final_conv0_act")(x)

    x = layers.Conv2D(16, 3, padding='same', name="final_conv1")(x)
    x = layers.LeakyReLU(alpha=0.1, name="final_conv1_act")(x)

    # Flow prediction (initialized to zero)
    flow_unscaled = layers.Conv2D(2, 3, padding='same', name="flow_unscaled",
                                   kernel_initializer='zeros',
                                   bias_initializer='zeros')(x)

    # FPGA-compatible flow scaling layer (0.1x for INT8 quantization)
    flow_scaled = layers.Conv2D(2, 1, padding='same', use_bias=False,
                                trainable=False,
                                name="flow_scale")(flow_unscaled)

    model = models.Model(inputs=[moving, fixed], outputs=flow_scaled, name="VoxelMorphDense2p5D")

    # Set scaling layer weights and freeze
    import numpy as np
    scale_layer = model.get_layer('flow_scale')
    scale_weights = np.zeros((1, 1, 2, 2), dtype=np.float32)
    scale_weights[0, 0, 0, 0] = 0.1
    scale_weights[0, 0, 1, 1] = 0.1
    scale_layer.set_weights([scale_weights])
    scale_layer.trainable = False

    return model


# -----------------------
# Unified Entry Point
# -----------------------

def build_model(weights_path: str, **kwargs) -> tf.keras.Model:
    """
    Load trained model directly or build from scratch.
    """
    weights_path_str = str(weights_path).lower()

    if os.path.exists(weights_path):
        try:
            print(f"[MODEL] Loading model from {os.path.basename(weights_path)}")
            model = tf.keras.models.load_model(weights_path, compile=False)
            print(f"[MODEL] Loaded model with {len(model.weights)} weight variables")

            try:
                scale_layer = model.get_layer('flow_scale')
                scale_weights = scale_layer.get_weights()[0]
                scale_value = scale_weights[0, 0, 0, 0]
                print(f"[MODEL] flow_scale layer found (scale={scale_value})")
            except:
                print(f"[MODEL] No flow_scale layer found")

            return model

        except Exception as e:
            print(f"[MODEL] Failed to load complete model: {e}")
            print(f"[MODEL] Falling back to building architecture and loading weights...")

    # Fallback to building architecture manually
    img_h = kwargs.get('img_height', 112)
    img_w = kwargs.get('img_width', 96)
    n_ch = kwargs.get('n_channels', 1)

    if "2p5d_dense" in weights_path_str or "2.5d" in weights_path_str:
        wr = kwargs.get('window_radius', 3)
        model = build_vxm_2p5d_dense_core(img_h, img_w, n_ch, wr)
        print(f"[MODEL] Built 2.5D Dense architecture: {img_h}x{img_w}, window_radius={wr}")
    else:
        raise ValueError(f"Unknown model type in {weights_path_str}. Expected '2p5d_dense' in filename.")

    # Load weights
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path, by_name=True, skip_mismatch=True)
            loaded_vars = len(model.weights)
            print(f"[MODEL] Loaded {loaded_vars} weight variables from {os.path.basename(weights_path)}")
        except Exception as e:
            print(f"[MODEL] Weight loading failed: {e}")
            print(f"[MODEL] Continuing with random weights...")
    else:
        print(f"[MODEL] Weights file not found: {weights_path}")

    return model


def get_custom_objects():
    """Return custom layers for model loading"""
    return {
        "SpatialTransformer2D": SpatialTransformer2D,
        "SpatialTransformer": SpatialTransformer2D,
        "BilinearUpsampling2D": BilinearUpsampling2D
    }


def get_calib_dataset(config):
    """
    Uses correct dimensions (112x96) and loads calibration data.
    """
    import numpy as np
    from pathlib import Path
    import cv2

    calib_dir = Path(config.calib_dir)

    if hasattr(config, 'img_height') and hasattr(config, 'img_width'):
        target_h, target_w = config.img_height, config.img_width
    else:
        target_h, target_w = 112, 96

    print(f"[CALIB] Using target size: {target_h}x{target_w}")

    stacks_dir = calib_dir / "stacks"
    if stacks_dir.exists():
        moving_files = sorted(list(stacks_dir.glob("moving_stack_*.npy")))
        fixed_files = sorted(list(stacks_dir.glob("fixed_stack_*.npy")))

        if not moving_files or not fixed_files:
            raise RuntimeError(f"No moving/fixed stack files found in {stacks_dir}")

        if config.calib_samples:
            moving_files = moving_files[:config.calib_samples]
            fixed_files = fixed_files[:config.calib_samples]

        target_ch = 7

        def gen():
            for mv_path, fx_path in zip(moving_files, fixed_files):
                mv = np.load(mv_path).astype("float32")
                fx = np.load(fx_path).astype("float32")
                mv = np.transpose(mv, (1, 2, 0))
                fx = np.transpose(fx, (1, 2, 0))

                if mv.shape[:2] != (target_h, target_w):
                    mv = cv2.resize(mv, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                if fx.shape[:2] != (target_h, target_w):
                    fx = cv2.resize(fx, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                yield ((mv, fx),)

        ds = tf.data.Dataset.from_generator(gen, output_signature=((
            tf.TensorSpec(shape=(target_h, target_w, target_ch), dtype=tf.float32),
            tf.TensorSpec(shape=(target_h, target_w, target_ch), dtype=tf.float32),
        ),))
        return ds.batch(config.calib_batch_size)

    else:
        input_dir = calib_dir / "inputs"
        if not input_dir.exists():
            raise RuntimeError(f"Neither {stacks_dir} nor {input_dir} found!")

        files = sorted(list(input_dir.glob("input_*.npy")))
        if config.calib_samples:
            files = files[:config.calib_samples]

        target_ch = 7

        def gen():
            for path in files:
                combined = np.load(path).astype("float32")
                mv = combined[0:7]
                fx = combined[7:14]
                mv = np.transpose(mv, (1, 2, 0))
                fx = np.transpose(fx, (1, 2, 0))

                if mv.shape[:2] != (target_h, target_w):
                    mv = cv2.resize(mv, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                if fx.shape[:2] != (target_h, target_w):
                    fx = cv2.resize(fx, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                yield ((mv, fx),)

        ds = tf.data.Dataset.from_generator(gen, output_signature=((
            tf.TensorSpec(shape=(target_h, target_w, target_ch), dtype=tf.float32),
            tf.TensorSpec(shape=(target_h, target_w, target_ch), dtype=tf.float32),
        ),))
        return ds.batch(config.calib_batch_size)
