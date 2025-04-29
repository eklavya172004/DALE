import os
import sys

# Ensure we're in the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
print(f"Changed working directory to: {project_root}")

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))
print(f"Added {os.path.abspath('.')} to Python path")

# List contents of model directory to verify it exists
if os.path.exists('model'):
    print(f"model directory found. Contents: {os.listdir('model')}")
else:
    print("model directory not found!")

try:
    # Import and run the test script
    print("Attempting to import test.test_DALE...")
    from test.test_DALE import main
    print("Successfully imported test.test_DALE")
    
    if __name__ == "__main__":
        print("Calling main() function...")
        main()
        print("main() function completed")
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()