from setuptools import setup, find_packages

setup(
    name="ink3d-render",
    version="0.1.0",
    description="Multi-pass 3D model rendering with Blender (H/V orbit cameras)",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10,<3.11",
    install_requires=[
        "numpy",
        "imageio[ffmpeg]",
        "Pillow",
        "scipy",
        "tqdm",
    ],
)
