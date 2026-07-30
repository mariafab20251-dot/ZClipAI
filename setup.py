from setuptools import setup, find_packages

setup(
    name="ai-viral-clipper",
    version="1.0.0",
    description="AI-powered viral short-form clip generator for social media platforms",
    author="AI Viral Clipper Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "PySide6>=6.5.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "faster-whisper>=1.0.0",
        "librosa>=0.10.0",
        "webrtcvad>=2.0.10",
        "pyyaml>=6.0",
        "pydantic>=2.5.0",
        "structlog>=24.1.0",
        "textblob>=0.17.1",
        "vaderSentiment>=3.3.2",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "ai-viral-clipper=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Video",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)