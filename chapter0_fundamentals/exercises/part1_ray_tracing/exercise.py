# %%
# Setup code
import os
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import einops
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from IPython.display import display
from ipywidgets import interact
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part1_ray_tracing"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_ray_tracing.tests as tests
from part1_ray_tracing.utils import (
    render_lines_with_plotly,
    setup_widget_fig_ray,
    setup_widget_fig_triangle,
)
from plotly_utils import imshow

MAIN = __name__ == "__main__"

# %%
def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is
        also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains
        (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    matrix = t.zeros(num_pixels,2,3)
    matrix[:,1,0] = 1
    matrix[:,1,1] = t.linspace(-y_limit,y_limit,num_pixels)
    return matrix


rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)

# Note that we use t.linspace() if we know the number of points rather than step size
# Or: we can use t.linspace(-y_limit,y_limit,num_pixels,out=matrix[:,1,1])

# %%
fig: go.FigureWidget = setup_widget_fig_ray()
display(fig)


@interact(v=(0.0, 6.0, 0.01), seed=(0, 10, 1))
def update(v=0.0, seed=0):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(0), P(6))
    with fig.batch_update():
        fig.update_traces({"x": x, "y": y}, 0)
        fig.update_traces({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}, 1)
        fig.update_traces({"x": [P(v)[0]], "y": [P(v)[1]]}, 2)

# %%
def intersect_ray_1d(ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    ray = ray[:,:2]
    segment = segment[:,:2]
    assert ray.shape == segment.shape
    A = t.stack((ray[1,:],segment[0,:] - segment[1,:]), dim=1)
    B = einops.rearrange(segment[0,:] - ray[0,:],"i -> i 1")
    try:
        u, v = t.linalg.solve(A, B)
        return (u >= 0) and (0 <= v) and (v <= 1)
    except:
        return False


tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)

# Note the meaning of (u >= 0) and (0 <= v) and (v <= 1) condition

# %%
def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """

    rays = rays[:, :, :2]
    rays = einops.repeat(rays,"i j k -> i l j k", l=segments.shape[0])
    segments = segments[:, :, :2]
    segments = einops.repeat(segments,"i j k -> l i j k", l=rays.shape[0])
    assert rays.shape == segments.shape
    A = t.stack((rays[:,:,1,:], segments[:,:,0,:] - segments[:,:,1,:]), dim=3)
    B = einops.rearrange(segments[:,:,0,:] - rays[:,:,0,:], "a b c -> a b c 1")
    A_singular = t.abs(t.linalg.det(A)) < 1e-8
    A[A_singular] = t.eye(2)
    uv = t.linalg.solve(A, B).squeeze(-1)
    return t.any((uv[:,:,0] >= 0) & (0 <= uv[:,:,1]) & (uv[:,:,1] <= 1) & (~A_singular), dim=-1)


tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)

# %%
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    results = t.zeros(num_pixels_y * num_pixels_z, 2, 3)
    points_y = t.linspace(-y_limit,y_limit,num_pixels_y)
    points_z = t.linspace(-z_limit,z_limit,num_pixels_z)

    results[:, 1, 0] = 1
    results[:, 1, 1] = einops.repeat(points_y, "i -> (i j)", j=num_pixels_z)
    results[:, 1, 2] = einops.repeat(points_z, "j -> (i j)", i=num_pixels_y)
    return results


rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
render_lines_with_plotly(rays_2d)

# %%
Point = Float[Tensor, "points=3"]


def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    """
    left = t.stack((-D, B-A, C-A), dim=-1)
    right = (O-A).unsqueeze(-1)
    if t.abs(t.linalg.det(left)) < 1e-8:
        return False
    else:
        s, u, v = t.linalg.solve(left, right)
        return ((s >= 0) & (u >= 0) & (v >= 0) & ((u + v) <= 1)).item()


tests.test_triangle_ray_intersects(triangle_ray_intersects)


# %%
def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"],
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    assert rays.shape[-1] == triangle.shape[-1]
    nrays, dims = rays.shape[0], rays.shape[-1]
    triangle = einops.repeat(triangle, "t d -> n t d", n=nrays)
    left = t.stack((-rays[:,1], triangle[:,1] - triangle[:,0], triangle[:,2] - triangle[:,0]), dim=-1)
    right = (rays[:,0] - triangle[:,0]).unsqueeze(-1)
    left_singular = t.abs(t.linalg.det(left)) < 1e-8
    left[left_singular] = t.eye(dims)
    sol = t.linalg.solve(left, right).squeeze(-1)
    s, u, v = sol.unbind(dim=-1)
    return (~left_singular) & (s >= 0) & (u >= 0) & (v >= 0) & ((u + v) <= 1)


A = t.tensor([1, 0.0, -0.5])
B = t.tensor([1, -0.5, 0.0])
C = t.tensor([1, 0.5, 0.5])
num_pixels_y = num_pixels_z = 15
y_limit = z_limit = 0.5

# Plot triangle & rays
test_triangle = t.stack([A, B, C], dim=0)
rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
render_lines_with_plotly(rays2d, triangle_lines)

# Calculate and display intersections
intersects = raytrace_triangle(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")

# %%
triangles = t.load(section_dir / "pikachu.pt", weights_only=True)


# %%
def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, " nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    assert rays.shape[-1] == triangles.shape[-1]
    nr, dims, nt = rays.shape[0], rays.shape[-1], triangles.shape[0]
    triangles = einops.repeat(triangles, "nt t d -> nr nt t d", nr=nr)
    rays = einops.repeat(rays, "nr r d -> nr nt r d", nt=nt)
    A, B, C = triangles.unbind(dim=2)
    O, D = rays.unbind(dim=2)
    left = t.stack((-D, B-A, C-A), dim=-1)
    right = (O-A).unsqueeze(-1)
    left_singular = t.abs(t.linalg.det(left)) < 1e-8
    left[left_singular] = t.eye(dims)
    sol = t.linalg.solve(left, right).squeeze(-1)
    s, u, v = sol.unbind(dim=-1)
    valid = (~left_singular) & (s >= 0) & (u >= 0) & (v >= 0) & ((u + v) <= 1)
    s[~valid] = float('inf')
    return s.min(dim=-1)[0]


num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-2, 0.0, 0.0])
dists = raytrace_mesh(rays, triangles)
intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
dists_square = dists.view(num_pixels_y, num_pixels_z)
img = t.stack([intersects, dists_square], dim=0)

fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
fig.update_layout(coloraxis_showscale=False)
for i, text in enumerate(["Intersects", "Distance"]):
    fig.layout.annotations[i]["text"] = text
fig.show()

# %%
# Video & Lighting
def rotation_matrix(theta: Float[Tensor, ""]) -> Float[Tensor, "rows cols"]:
    """
    Creates a rotation matrix representing a counterclockwise rotation of `theta` around the y-axis.
    """
    mat = t.zeros(3,3)
    mat[0,0] = mat[2,2] = t.cos(theta)
    mat[0,2] = t.sin(theta)
    mat[2,0] = -t.sin(theta)
    mat[1,1] = 1
    return mat


tests.test_rotation_matrix(rotation_matrix)

# %%
def raytrace_mesh_video(
    rays: Float[Tensor, "nrays points dim"],
    triangles: Float[Tensor, "ntriangles points dims"],
    rotation_matrix: Callable[[float], Float[Tensor, "rows cols"]],
    raytrace_function: Callable,
    num_frames: int,
) -> Bool[Tensor, "nframes nrays"]:
    """
    Creates a stack of raytracing results, rotating the triangles by `rotation_matrix` each frame.
    """
    result = []
    theta = t.tensor(2 * t.pi) / num_frames
    R = rotation_matrix(theta)
    for theta in tqdm(range(num_frames)):
        triangles = triangles @ R
        result.append(raytrace_function(rays, triangles))
        t.cuda.empty_cache()  # clears GPU memory (this line will be more important later on!)
    return t.stack(result, dim=0)


def display_video(distances: Float[Tensor, "frames y z"]):
    """
    Displays video of raytracing results, using Plotly. `distances` is a tensor where the [i, y, z]
    element is distance to the closest triangle for the i-th frame & the [y, z]-th ray in our 2D
    grid of rays.
    """
    px.imshow(
        distances,
        animation_frame=0,
        origin="lower",
        zmin=0.0,
        zmax=distances[distances.isfinite()].quantile(0.99).item(),
        color_continuous_scale="viridis_r",  # "Brwnyl"
    ).update_layout(coloraxis_showscale=False, width=550, height=600, title="Raytrace mesh video").show()


num_pixels_y = 250
num_pixels_z = 250
y_limit = z_limit = 0.8
num_frames = 50

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-3.0, 0.0, 0.0])
dists = raytrace_mesh_video(rays, triangles, rotation_matrix, raytrace_mesh, num_frames)
dists = einops.rearrange(dists, "frames (y z) -> frames y z", y=num_pixels_y)

display_video(dists)
# %%
# Skip raytrace_mesh_gpu(), raytrace_mesh_lambert()