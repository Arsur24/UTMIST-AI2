"""
Quick test script to verify CUDA setup is working correctly.
Run this before starting training to ensure GPU acceleration is available.
"""

import torch
import sys

def test_cuda_setup():
    """Test CUDA availability and configuration"""
    print("="*60)
    print("CUDA Setup Verification")
    print("="*60)

    # Check PyTorch version
    print(f"\nPyTorch version: {torch.__version__}")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")

        # Test tensor operations on GPU
        print("\n" + "="*60)
        print("Testing GPU tensor operations...")
        print("="*60)

        try:
            # Create tensors on GPU
            device = torch.device("cuda")
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)

            # Perform operation
            z = torch.matmul(x, y)

            print("✓ Successfully created tensors on GPU")
            print("✓ Successfully performed matrix multiplication on GPU")
            print(f"✓ Result tensor shape: {z.shape}")
            print(f"✓ Result tensor device: {z.device}")

            # Check memory usage
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
            print(f"\nGPU Memory allocated: {memory_allocated:.2f} MB")
            print(f"GPU Memory reserved: {memory_reserved:.2f} MB")

            print("\n" + "="*60)
            print("✓ CUDA setup is working correctly!")
            print("✓ Your system is ready for GPU-accelerated training")
            print("="*60)

            return True

        except Exception as e:
            print(f"\n✗ Error testing GPU operations: {e}")
            print("Please check your CUDA installation.")
            return False
    else:
        print("\n" + "="*60)
        print("✗ CUDA is not available")
        print("Training will use CPU (much slower)")
        print("\nTo enable CUDA:")
        print("1. Install NVIDIA GPU drivers")
        print("2. Install CUDA Toolkit from NVIDIA")
        print("3. Install PyTorch with CUDA support:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("="*60)
        return False

if __name__ == "__main__":
    success = test_cuda_setup()
    sys.exit(0 if success else 1)

