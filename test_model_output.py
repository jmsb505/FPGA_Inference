#!/usr/bin/env python3
"""
Test what the trained model actually outputs (with and without flow_scale layer).
This will help us understand if we need the 10x scaling or not.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

# Silence TensorFlow
tf.get_logger().setLevel('ERROR')

print("Testing Model Output to Understand Flow Scaling")

# Load the model
model_path = "Voxelmorph/trained_weights/2p5d_dense_tf_best.h5"
print(f"\nLoading model: {model_path}")
model = tf.keras.models.load_model(model_path, compile=False)

# Check for flow_scale layer
try:
    flow_scale_layer = model.get_layer('flow_scale')
    weights = flow_scale_layer.get_weights()[0]
    scale_value = weights[0, 0, 0, 0]
    print(f"Model has flow_scale layer (scale={scale_value})")
    has_flow_scale = True
except:
    print(f"Model does not have flow_scale layer")
    has_flow_scale = False

# Create dummy input
print(f"\nCreating dummy input (112x96x7)...")
np.random.seed(42)
moving = np.random.randn(1, 112, 96, 7).astype(np.float32)
fixed = np.random.randn(1, 112, 96, 7).astype(np.float32)

# Run inference
print(f"Running inference...")
flow_output = model.predict([moving, fixed], verbose=0)

print(f"\nModel Output Analysis")
print(f"Flow shape: {flow_output.shape}")
print(f"Flow range: [{flow_output.min():.6f}, {flow_output.max():.6f}]")
print(f"Flow mean: {flow_output.mean():.6f}")
print(f"Flow std: {flow_output.std():.6f}")

# Check unscaled flow
if has_flow_scale:
    print(f"\nChecking unscaled flow (before flow_scale layer)")

    flow_unscaled_layer = model.get_layer('flow_unscaled')
    intermediate_model = tf.keras.Model(inputs=model.inputs, outputs=flow_unscaled_layer.output)
    flow_unscaled = intermediate_model.predict([moving, fixed], verbose=0)

    print(f"Unscaled flow range: [{flow_unscaled.min():.6f}, {flow_unscaled.max():.6f}]")
    print(f"Unscaled flow mean: {flow_unscaled.mean():.6f}")
    print(f"Unscaled flow std: {flow_unscaled.std():.6f}")

    # Verify scaling
    ratio = flow_output.max() / flow_unscaled.max() if flow_unscaled.max() != 0 else 0
    print(f"\nRatio (scaled / unscaled): {ratio:.6f}")
    print(f"Expected ratio: 0.1")

    if abs(ratio - 0.1) < 0.01:
        print("Flow scaling is working correctly")
    else:
        print(f"Unexpected ratio: should be ~0.1")

print(f"\nConclusion")

if has_flow_scale:
    print("The model has a flow_scale layer with a 0.1 multiplier")
    print(f"Current output range: [{flow_output.min():.2f}, {flow_output.max():.2f}]")

    if abs(flow_output.max()) < 0.5 and abs(flow_output.min()) < 0.5:
        print("\nWarning: Model output is small for registration")
        print("Flow values of +/- 0.5 pixels provide minimal deformation.")
        print("Expected range for brain registration is typically +/- 5 to +/- 20 pixels.")
        print("\nSolution: Use a FLOW_SCALE_FACTOR of 10.0 during inference.")
        print(f"Real flow range after scaling: [{flow_output.min()*10:.2f}, {flow_output.max()*10:.2f}]")
    else:
        print("\nModel output range looks reasonable")
else:
    print("The model does not have a flow_scale layer")
    print("Model output is the raw flow prediction")
