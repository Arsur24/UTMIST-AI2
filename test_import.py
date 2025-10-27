"""
Quick test to verify imports work correctly
"""
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

print(f"Project root: {project_root}")
print(f"sys.path includes project_root: {project_root in sys.path}")

# Test import
try:
    from environment.agent import train_parallel
    print("✓ Successfully imported train_parallel from environment.agent")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    import traceback
    traceback.print_exc()

# Test user module import
try:
    from user import train_agent
    print("✓ Successfully imported user.train_agent")
except Exception as e:
    print(f"✗ Failed to import user module: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ All imports successful!")

