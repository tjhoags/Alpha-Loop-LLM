import os
import sys
import platform
import subprocess
from datetime import datetime

def run_tests():
    """
    Run pytest with machine-isolated report generation.
    Ensures test results from Lenovo and MacBook are not overwritten.
    """
    # Get machine identifier
    machine_id = platform.node().replace(" ", "_").replace(".", "_")
    
    # Create reports directory if not exists
    os.makedirs("test-reports", exist_ok=True)
    
    # Define unique report filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"test-reports/report_{machine_id}_{timestamp}.xml"
    
    print(f"================================================================")
    print(f"ALC-ALGO ISOLATED TEST RUNNER")
    print(f"Machine: {machine_id}")
    print(f"Report:  {report_file}")
    print(f"================================================================")
    
    # Construct pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        f"--junitxml={report_file}",
        "-v"
    ]
    
    # Run tests
    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()

