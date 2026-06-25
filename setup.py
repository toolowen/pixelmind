"""
PixelMind: Train a Tiny VLM from Scratch
==========================================
A minimal, educational Vision-Language Model (~65M params) trained from scratch.
Trainable in ~2h for ~$3 on a single RTX 3090.

Pipeline: LLM Pretrain → LLM SFT → VLM Pretrain → VLM SFT → VLM GRPO

Installation:
    pip install -e .            # minimal (for inference only)
    pip install -e ".[train]"   # with training dependencies
    pip install -e ".[all]"     # everything

Quick Start:
    # LLM Chat
    pixelmind-chat-llm --weight sft

    # VLM Chat (evaluates images in a directory)
    pixelmind-chat-vlm --weight sft_vlm --image_dir ./dataset/eval_images/

    # Web Demo
    pixelmind-web-demo --weight sft_vlm
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Additional optional dependencies grouped by use case
extras_require = {
    "train": [
        "torch>=2.0",
        "transformers>=4.45",
        "datasets",
        "pyarrow",
        "Pillow",
        "swanlab",
        "numpy",
        "accelerate",
    ],
    "eval": [
        "torch>=2.0",
        "transformers>=4.45",
        "Pillow",
        "numpy",
        "tqdm",
    ],
    "web": [
        "gradio>=4.0",
    ],
    "dev": [
        "pytest",
        "black",
        "ruff",
        "ipython",
    ],
}
extras_require["all"] = sorted(set(sum(extras_require.values(), [])))

setup(
    name="pixelmind",
    version="0.1.0",
    author="",
    author_email="",
    description="PixelMind: Train a Tiny VLM from Scratch — 65M params, 2h, $3, single GPU",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        "Source": "https://github.com/your/pixelmind",
        "Tracker": "https://github.com/your/pixelmind/issues",
    },
    packages=find_packages(include=["pixelmind", "pixelmind.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "pixelmind-chat-llm=pixelmind.eval.chat_llm:main",
            "pixelmind-chat-vlm=pixelmind.eval.chat_vlm:main",
            "pixelmind-web-demo=pixelmind.scripts.web_demo:main",
            "pixelmind-convert=pixelmind.scripts.convert:main",
        ],
    },
    include_package_data=True,
    package_data={
        "pixelmind": ["py.typed"],
    },
    zip_safe=False,
)
