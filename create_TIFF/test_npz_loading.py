#!/usr/bin/env python3
"""
Test script to verify NPZ loading functionality
"""

import numpy as np
import os

def create_test_npz():
    """Create a test NPZ file to verify loading"""

    # Create test data
    test_images = np.random.rand(10, 512, 512, 1).astype(np.float32)

    # Create test metadata (list of dictionaries)
    test_metadata = []
    for i in range(10):
        test_metadata.append({
            'original_file': f'test_image_{i}.tif',
            'patch_index': i % 5,  # Simulate 5 patches per image
            'total_patches': 5,
            'timestamp': f'20240101T{i:02d}0000'
        })

    # Save test file
    test_file = 'test_sliding_window.npz'
    np.savez(test_file, test_images, test_metadata)

    print(f"Created test file: {test_file}")
    print(f"Images shape: {test_images.shape}")
    print(f"Metadata length: {len(test_metadata)}")

    return test_file

def test_loading(test_file):
    """Test loading the NPZ file"""

    print(f"\nTesting load with allow_pickle=False:")
    try:
        data = np.load(test_file, allow_pickle=False)
        print("SUCCESS: Loaded without allow_pickle")
    except Exception as e:
        print(f"ERROR: {e}")

    print(f"\nTesting load with allow_pickle=True:")
    try:
        data = np.load(test_file, allow_pickle=True)
        images = data['arr_0']
        metadata = data['arr_1']

        print("SUCCESS: Loaded with allow_pickle=True")
        print(f"Images shape: {images.shape}")
        print(f"Metadata type: {type(metadata)}")
        print(f"First metadata item: {metadata[0]}")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Main test function"""

    print("🧪 TESTING NPZ LOADING FUNCTIONALITY")
    print("=" * 50)

    # Create test file
    test_file = create_test_npz()

    # Test loading
    success = test_loading(test_file)

    if success:
        print("\n✅ All tests passed!")
        print(f"📁 Test file: {test_file}")
        print("🚀 You can now run: python visualize_npz_data_training.py --npz_file test_sliding_window.npz")
    else:
        print("\n❌ Tests failed!")
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"🗑️  Cleaned up test file: {test_file}")

if __name__ == "__main__":
    main()
