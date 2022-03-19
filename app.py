import random
from pathlib import Path

import numpy as np
import shapely.geometry as geo
import shapely.ops as ops
import streamlit as st
from poisson_disc import Bridson_sampling
from pyfastnoiselite.pyfastnoiselite import FastNoiseLite, NoiseType
from setuptools import sandbox

if not list(Path().glob("*.so")):
    sandbox.run_setup("setup.py", ["build_ext", "--inplace"])

from flowfieldcy import FlowGrid, Particle, run_flowfield

st.set_page_config(layout="wide")

noise_options = [
    NoiseType.NoiseType_OpenSimplex2,
    NoiseType.NoiseType_OpenSimplex2S,
    NoiseType.NoiseType_Cellular,
    NoiseType.NoiseType_Perlin,
    NoiseType.NoiseType_ValueCubic,
    NoiseType.NoiseType_Value,
]

col1, col2, col3, col4 = st.columns(4)

seed = col1.number_input("seed", step=1, value=1234)
width = col2.number_input("width", min_value=100, value=600)
height = col3.number_input("height", min_value=100, value=400)
resolution = col4.number_input("resolution", min_value=0.01, max_value=1.0, value=1.0)
noise_type = col1.selectbox("Noise Type", options=noise_options)

steps = col2.number_input("steps", min_value=1, max_value=10000, value=500)


frequency = col3.number_input(
    "frequency",
    min_value=0.0001,
    max_value=0.1,
    value=0.001,
    step=0.0001,
    format="%.4f",
)
octaves = col4.slider("octaves", min_value=1, max_value=20, step=1, value=3)

sample_radius = col1.number_input("Point Sample Radius", value=10)


flowgrid = FlowGrid(width, height, resolution)

result = np.empty(flowgrid.shape, dtype=float)
noise = FastNoiseLite()
noise.seed = seed
noise.noise_type = noise_type
noise.frequency = frequency
noise.fractal_octaves = octaves
for x in range(flowgrid.shape[0]):
    for y in range(flowgrid.shape[1]):
        result[x, y] = np.interp(noise.get_noise(x, y), [-1, 1], [0, 2 * np.pi])

flowgrid.initialize_angles(result)

RADIUS = (1 / flowgrid.resolution_factor) * 10

starting = Bridson_sampling(dims=np.array([width, height]), radius=sample_radius, k=15)

# starting = [(random.random() * width, random.random() * height) for _ in range(1000)]
parts = [Particle(x_, y_, flowgrid, steps) for x_, y_ in starting]

particles = run_flowfield(flowgrid, steps, parts)


mls = []
debug = []
for h in particles:
    if len(h) > 1:
        ls = geo.LineString(h)
        if not ls.is_empty:
            if isinstance(ls, geo.LineString):
                mls.append(ls)
            elif isinstance(ls, geo.MultiLineString):
                mls.extend(list(ls.geoms))
        debug.append(type(ls))
mls = geo.MultiLineString(mls)
merged = ops.linemerge(mls)

st.image(
    merged._repr_svg_()
    .replace('width="300"', f'width="{width}"')
    .replace('height="300"', f'height="{height}"'),
    use_column_width="always",
)
